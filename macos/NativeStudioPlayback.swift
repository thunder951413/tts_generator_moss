import AppKit
import AVFoundation
import CoreMedia
import Foundation
import ScreenCaptureKit
import SwiftUI
import UniformTypeIdentifiers

/// Streams 16-bit little-endian interleaved PCM from the service into an AVAudioPlayerNode.
///
/// Network-side state (`pendingPCM`, `hasScheduledAudio`) is confined to the session's serial
/// delegate queue. State shared with AVAudioPlayerNode completion callbacks is guarded by `lock`.
/// Small TCP reads are coalesced: playback starts once 0.35 s is buffered (or the stream ended),
/// later audio is scheduled in >= 100 ms buffers, and after an underrun 0.25 s is re-buffered.
final class NativePCMStreamPlayer: NSObject, URLSessionDataDelegate {
    private let engine = AVAudioEngine()
    private let node = AVAudioPlayerNode()
    /// Time-stretch stage between the player and the mixer: changes speed, keeps pitch.
    private let timePitch = AVAudioUnitTimePitch()
    private let format: AVAudioFormat
    private let request: URLRequest
    private let lock = NSLock()
    private let prebufferFrames: Int
    private let rebufferFrames: Int
    private let minimumChunkFrames: Int
    private var session: URLSession?
    private var delegateQueue: OperationQueue?
    private var configurationObserver: NSObjectProtocol?
    // Delegate-queue confined.
    private var pendingPCM = Data()
    private var hasScheduledAudio = false
    // Guarded by lock.
    private var pendingBuffers = 0
    private var queuedFrames = 0
    private var completedBuffers = 0
    private var networkFinished = false
    private var stopped = false
    private var receivedFirstAudio = false
    private var configurationChanged = false
    private var responseError: Error?

    /// Playback speed, 0.5–2.0. Pitch is preserved.
    var rate: Float {
        get { timePitch.rate }
        set {
            let clamped = min(2.0, max(0.5, newValue))
            timePitch.rate = clamped
            // Pass audio through untouched at normal speed.
            timePitch.bypass = abs(clamped - 1.0) < 0.001
        }
    }

    var onFirstAudio: (() -> Void)?
    var onComplete: (() -> Void)?
    var onError: ((Error) -> Void)?

    init(request: URLRequest, sampleRate: Double, channels: AVAudioChannelCount) throws {
        guard let format = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: sampleRate,
            channels: max(1, channels),
            interleaved: false
        ) else {
            throw NativeStudioError.server("无法创建 PCM 播放格式。")
        }
        self.request = request
        self.format = format
        self.prebufferFrames = max(1, Int(sampleRate * 0.35))
        self.rebufferFrames = max(1, Int(sampleRate * 0.25))
        self.minimumChunkFrames = max(1, Int(sampleRate * 0.1))
        super.init()
        engine.attach(node)
        engine.attach(timePitch)
        #if compiler(>=6.4)
        if #available(macOS 27.0, *) {
            try engine.connectNode(node, to: timePitch, format: format)
            try engine.connectNode(timePitch, to: engine.mainMixerNode, format: format)
        } else {
            engine.connect(node, to: timePitch, format: format)
            engine.connect(timePitch, to: engine.mainMixerNode, format: format)
        }
        #else
        engine.connect(node, to: timePitch, format: format)
        engine.connect(timePitch, to: engine.mainMixerNode, format: format)
        #endif
        try engine.start()
        try startNode()
    }

    private func startNode() throws {
        #if compiler(>=6.4)
        if #available(macOS 27.0, *) { try node.playAudio() } else { node.play() }
        #else
        node.play()
        #endif
    }

    func start() {
        let queue = OperationQueue()
        queue.name = "qwen-native-pcm-stream"
        queue.maxConcurrentOperationCount = 1
        delegateQueue = queue
        // Output device / sample-rate changes stop the engine; handle them on the same serial
        // queue that schedules audio.
        let observer = NotificationCenter.default.addObserver(
            forName: .AVAudioEngineConfigurationChange,
            object: engine,
            queue: queue
        ) { [weak self] _ in
            self?.handleConfigurationChange()
        }
        lock.lock()
        configurationObserver = observer
        lock.unlock()
        let session = URLSession(
            configuration: .ephemeral,
            delegate: self,
            delegateQueue: queue
        )
        self.session = session
        session.dataTask(with: request).resume()
    }

    func stop() {
        lock.lock()
        let shouldStop = !stopped
        stopped = true
        lock.unlock()
        guard shouldStop else { return }
        removeConfigurationObserver()
        session?.invalidateAndCancel()
        node.stop()
        engine.stop()
    }

    func urlSession(
        _ session: URLSession,
        dataTask: URLSessionDataTask,
        didReceive response: URLResponse,
        completionHandler: @escaping (URLSession.ResponseDisposition) -> Void
    ) {
        guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
            lock.lock()
            responseError = NativeStudioError.server("PCM 音频流连接失败。")
            lock.unlock()
            completionHandler(.cancel)
            return
        }
        completionHandler(.allow)
    }

    func urlSession(
        _ session: URLSession,
        dataTask: URLSessionDataTask,
        didReceive data: Data
    ) {
        lock.lock()
        let isStopped = stopped
        lock.unlock()
        guard !isStopped else { return }
        pendingPCM.append(data)
        scheduleBufferedAudio(streamEnded: false)
    }

    func urlSession(
        _ session: URLSession,
        task: URLSessionTask,
        didCompleteWithError error: Error?
    ) {
        lock.lock()
        let isStopped = stopped
        let failure = responseError ?? error
        lock.unlock()
        if let failure, !isStopped {
            lock.lock()
            networkFinished = true
            lock.unlock()
            DispatchQueue.main.async { [weak self] in self?.onError?(failure) }
            stop()
            return
        }
        guard !isStopped else {
            lock.lock()
            networkFinished = true
            lock.unlock()
            return
        }
        // Flush everything that is left (whole frames only; a trailing partial frame is
        // dropped). Mark the network finished only after scheduling so a buffer finishing in
        // between cannot complete playback early.
        scheduleBufferedAudio(streamEnded: true)
        pendingPCM = Data()
        lock.lock()
        networkFinished = true
        let pending = pendingBuffers
        let watchStall = configurationChanged
        lock.unlock()
        if pending == 0 {
            finishPlayback()
        } else if watchStall {
            armStallWatchdog()
        }
    }

    /// Delegate queue only.
    private func scheduleBufferedAudio(streamEnded: Bool) {
        let frameBytes = Int(format.channelCount) * 2
        let availableFrames = pendingPCM.count / frameBytes
        guard availableFrames > 0 else { return }
        lock.lock()
        let queued = queuedFrames
        lock.unlock()
        let threshold: Int
        if !hasScheduledAudio {
            threshold = prebufferFrames
        } else if queued == 0 {
            threshold = rebufferFrames
        } else {
            threshold = minimumChunkFrames
        }
        guard streamEnded || availableFrames >= threshold else { return }
        scheduleFrames(availableFrames, frameBytes: frameBytes)
    }

    /// Delegate queue only. Converts the first `frameCount` whole frames of `pendingPCM`.
    private func scheduleFrames(_ frameCount: Int, frameBytes: Int) {
        guard let buffer = AVAudioPCMBuffer(
            pcmFormat: format,
            frameCapacity: AVAudioFrameCount(frameCount)
        ), let channels = buffer.floatChannelData else {
            return
        }
        buffer.frameLength = AVAudioFrameCount(frameCount)
        let channelCount = Int(format.channelCount)
        pendingPCM.withUnsafeBytes { raw in
            let source = raw.bindMemory(to: UInt8.self)
            for frame in 0 ..< frameCount {
                for channel in 0 ..< channelCount {
                    let offset = (frame * channelCount + channel) * 2
                    let bits = UInt16(source[offset]) | (UInt16(source[offset + 1]) << 8)
                    channels[channel][frame] = Float(Int16(bitPattern: bits)) / 32768.0
                }
            }
        }
        // Keep the unconsumed tail (including any partial frame) as a fresh zero-based Data.
        pendingPCM = Data(pendingPCM.dropFirst(frameCount * frameBytes))
        hasScheduledAudio = true

        lock.lock()
        pendingBuffers += 1
        queuedFrames += frameCount
        let announceFirstAudio = !receivedFirstAudio
        receivedFirstAudio = true
        lock.unlock()
        if announceFirstAudio {
            DispatchQueue.main.async { [weak self] in self?.onFirstAudio?() }
        }
        node.scheduleBuffer(buffer, completionCallbackType: .dataPlayedBack) { [weak self] _ in
            self?.bufferDidFinish(frameCount)
        }
    }

    private func bufferDidFinish(_ frameCount: Int) {
        lock.lock()
        pendingBuffers = max(0, pendingBuffers - 1)
        queuedFrames = max(0, queuedFrames - frameCount)
        completedBuffers += 1
        let shouldFinish = networkFinished && pendingBuffers == 0 && !stopped
        lock.unlock()
        if shouldFinish { finishPlayback() }
    }

    /// Delegate queue. The engine stopped because the output device or its format changed.
    private func handleConfigurationChange() {
        lock.lock()
        let isStopped = stopped
        configurationChanged = true
        lock.unlock()
        guard !isStopped else { return }
        do {
            if !engine.isRunning {
                try engine.start()
            }
            try startNode()
            armStallWatchdog()
        } catch {
            let failure = NativeStudioError.server("音频输出设备已变化，无法继续播放：\(error.localizedDescription)")
            DispatchQueue.main.async { [weak self] in self?.onError?(failure) }
            stop()
        }
    }

    /// After a device change, buffers scheduled before the change may never report completion.
    /// If no buffer completes for the remaining queued duration (+3 s) once the network is done,
    /// finish instead of leaving the UI waiting forever.
    private func armStallWatchdog() {
        lock.lock()
        let isStopped = stopped
        let frames = queuedFrames
        let completedAtArm = completedBuffers
        lock.unlock()
        guard !isStopped else { return }
        // Queued audio lasts up to 2x its length at the slowest supported speed (0.5x).
        let delay = Double(frames) / format.sampleRate * 2 + 3
        DispatchQueue.global(qos: .utility).asyncAfter(deadline: .now() + delay) { [weak self] in
            guard let self else { return }
            self.lock.lock()
            let active = !self.stopped && self.pendingBuffers > 0
            let progressed = self.completedBuffers != completedAtArm
            let finished = self.networkFinished
            self.lock.unlock()
            guard active else { return }
            if finished && !progressed {
                self.finishPlayback()
            } else {
                self.armStallWatchdog()
            }
        }
    }

    private func removeConfigurationObserver() {
        lock.lock()
        let observer = configurationObserver
        configurationObserver = nil
        lock.unlock()
        if let observer {
            NotificationCenter.default.removeObserver(observer)
        }
    }

    private func finishPlayback() {
        lock.lock()
        guard !stopped else {
            lock.unlock()
            return
        }
        stopped = true
        lock.unlock()
        removeConfigurationObserver()
        session?.finishTasksAndInvalidate()
        // This can run inside an AVAudioPlayerNode completion handler, where stopping the
        // node may block on the render thread. Report completion first, then tear the
        // engine down on the main queue, outside the audio callback.
        DispatchQueue.main.async { [self] in
            self.onComplete?()
            self.node.stop()
            self.engine.stop()
        }
    }
}

// Mutable capture/writer state is confined to sampleQueue. The public callback
// property uses the same queue, and completion handlers return to the main queue.
final class SystemAudioRecorder: NSObject, SCStreamOutput, SCStreamDelegate, @unchecked Sendable {
    private let sampleQueue = DispatchQueue(label: "qwen-system-audio-capture")
    private var stream: SCStream?
    private var writer: AVAssetWriter?
    private var writerInput: AVAssetWriterInput?
    private var outputURL: URL?
    private var startedWriting = false
    private var intentionallyStopping = false
    private var completion: ((Result<URL, Error>) -> Void)?
    private var unexpectedErrorHandler: ((Error) -> Void)?
    var onUnexpectedError: ((Error) -> Void)? {
        get { sampleQueue.sync { unexpectedErrorHandler } }
        set { sampleQueue.sync { unexpectedErrorHandler = newValue } }
    }

    func start(completion: @escaping (Result<Void, Error>) -> Void) {
        Task {
            do {
                let content = try await SCShareableContent.excludingDesktopWindows(
                    false,
                    onScreenWindowsOnly: true
                )
                guard let display = content.displays.first else {
                    throw NativeStudioError.server("没有找到可捕获的显示器。")
                }
                let ownApplications = content.applications.filter {
                    $0.bundleIdentifier == Bundle.main.bundleIdentifier
                }
                let filter = SCContentFilter(
                    display: display,
                    excludingApplications: ownApplications,
                    exceptingWindows: []
                )
                let configuration = SCStreamConfiguration()
                configuration.capturesAudio = true
                configuration.excludesCurrentProcessAudio = true
                configuration.sampleRate = 48_000
                configuration.channelCount = 1
                configuration.width = 2
                configuration.height = 2
                configuration.showsCursor = false

                let outputURL = FileManager.default.temporaryDirectory
                    .appendingPathComponent("qwen-system-audio-\(UUID().uuidString).m4a")
                let writer = try AVAssetWriter(outputURL: outputURL, fileType: .m4a)
                let stream = SCStream(filter: filter, configuration: configuration, delegate: self)
                try stream.addStreamOutput(self, type: .audio, sampleHandlerQueue: sampleQueue)
                sampleQueue.sync {
                    self.outputURL = outputURL
                    self.writer = writer
                    self.stream = stream
                    self.intentionallyStopping = false
                }
                try await stream.startCapture()
                DispatchQueue.main.async { completion(.success(())) }
            } catch {
                sampleQueue.async { [self] in self.reset(removeOutput: true) }
                DispatchQueue.main.async { completion(.failure(error)) }
            }
        }
    }

    func stop(completion: @escaping (Result<URL, Error>) -> Void) {
        sampleQueue.async { [self] in
            intentionallyStopping = true
            self.completion = completion
            guard let stream else {
                finishWriting()
                return
            }
            Task { [self] in
                do {
                    try await stream.stopCapture()
                    self.sampleQueue.async { [self] in self.finishWriting() }
                } catch {
                    self.sampleQueue.async { [self] in self.complete(.failure(error)) }
                }
            }
        }
    }

    func stream(
        _ stream: SCStream,
        didOutputSampleBuffer sampleBuffer: CMSampleBuffer,
        of outputType: SCStreamOutputType
    ) {
        guard outputType == .audio, sampleBuffer.isValid,
              CMSampleBufferDataIsReady(sampleBuffer),
              let writer
        else { return }
        if writerInput == nil {
            let settings: [String: Any] = [
                AVFormatIDKey: kAudioFormatMPEG4AAC,
                AVSampleRateKey: 48_000,
                AVNumberOfChannelsKey: 1,
                AVEncoderBitRateKey: 128_000,
            ]
            let input = AVAssetWriterInput(
                mediaType: .audio,
                outputSettings: settings,
                sourceFormatHint: CMSampleBufferGetFormatDescription(sampleBuffer)
            )
            input.expectsMediaDataInRealTime = true
            guard writer.canAdd(input) else { return }
            writer.add(input)
            writerInput = input
        }
        if !startedWriting {
            guard writer.startWriting() else { return }
            writer.startSession(atSourceTime: CMSampleBufferGetPresentationTimeStamp(sampleBuffer))
            startedWriting = true
        }
        if writerInput?.isReadyForMoreMediaData == true {
            writerInput?.append(sampleBuffer)
        }
    }

    func stream(_ stream: SCStream, didStopWithError error: Error) {
        sampleQueue.async { [weak self] in
            guard let self, !self.intentionallyStopping else { return }
            let handler = self.unexpectedErrorHandler
            self.reset(removeOutput: true)
            DispatchQueue.main.async { handler?(error) }
        }
    }

    private func finishWriting() {
        guard let writer, let outputURL else {
            complete(.failure(NativeStudioError.server("没有捕获到系统声音。请先播放声音再停止录制。")))
            return
        }
        guard startedWriting else {
            complete(.failure(NativeStudioError.server("没有捕获到系统声音。请先播放声音再停止录制。")))
            return
        }
        writerInput?.markAsFinished()
        writer.finishWriting { [weak self] in
            guard let self else { return }
            self.sampleQueue.async { [self] in
                if writer.status == .completed {
                    self.complete(.success(outputURL), removeOutput: false)
                } else {
                    self.complete(.failure(writer.error ?? NativeStudioError.server("系统声音录制写入失败。")))
                }
            }
        }
    }

    private func complete(_ result: Result<URL, Error>, removeOutput: Bool = true) {
        let completion = completion
        reset(removeOutput: removeOutput)
        DispatchQueue.main.async { completion?(result) }
    }

    private func reset(removeOutput: Bool) {
        if removeOutput, let outputURL {
            try? FileManager.default.removeItem(at: outputURL)
        }
        stream = nil
        writer = nil
        writerInput = nil
        outputURL = nil
        startedWriting = false
        intentionallyStopping = false
        completion = nil
    }
}

struct SystemAudioWaveformView: View {
    let samples: [Float]
    let selectionStart: Double
    let selectionEnd: Double

    var body: some View {
        Canvas { context, size in
            let center = size.height / 2
            let count = max(1, samples.count)
            let width = size.width / CGFloat(count)
            for (index, sample) in samples.enumerated() {
                let height = max(2, CGFloat(sample) * size.height * 0.82)
                let rect = CGRect(
                    x: CGFloat(index) * width,
                    y: center - height / 2,
                    width: max(1, width * 0.58),
                    height: height
                )
                context.fill(
                    Path(roundedRect: rect, cornerRadius: 1),
                    with: .color(.accentColor.opacity(0.82))
                )
            }
            if selectionStart > 0 {
                context.fill(
                    Path(CGRect(x: 0, y: 0, width: size.width * selectionStart, height: size.height)),
                    with: .color(.black.opacity(0.42))
                )
            }
            if selectionEnd < 1 {
                context.fill(
                    Path(
                        CGRect(
                            x: size.width * selectionEnd,
                            y: 0,
                            width: size.width * (1 - selectionEnd),
                            height: size.height
                        )
                    ),
                    with: .color(.black.opacity(0.42))
                )
            }
        }
        .background(.black.opacity(0.16), in: RoundedRectangle(cornerRadius: 12))
    }
}

struct SystemAudioTrimView: View {
    @ObservedObject var model: NativeStudioViewModel

    private func time(_ value: TimeInterval) -> String {
        let total = max(0, Int(value.rounded()))
        return String(format: "%02d:%02d", total / 60, total % 60)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: StudioTokens.space4) {
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    Text(model.audioTrimTitle)
                        .font(.system(size: 22, weight: .semibold, design: .rounded))
                    Text(model.audioTrimSubtitle)
                        .foregroundStyle(.secondary)
                }
                Spacer()
                Text("\(time(model.systemAudioTrimStart)) – \(time(model.systemAudioTrimEnd))")
                    .font(.system(.body, design: .monospaced).weight(.medium))
            }

            SystemAudioWaveformView(
                samples: model.systemAudioWaveform,
                selectionStart: model.systemAudioDuration > 0
                    ? model.systemAudioTrimStart / model.systemAudioDuration : 0,
                selectionEnd: model.systemAudioDuration > 0
                    ? model.systemAudioTrimEnd / model.systemAudioDuration : 1
            )
            .frame(height: 150)
            .overlay {
                if model.systemAudioWaveform.isEmpty {
                    ProgressView("正在分析波形…")
                }
            }

            VStack(spacing: StudioTokens.space3) {
                HStack {
                    Text("起点")
                        .frame(width: 44, alignment: .leading)
                    Slider(
                        value: $model.systemAudioTrimStart,
                        in: 0 ... max(0.01, model.systemAudioTrimEnd - 0.25)
                    )
                    Text(time(model.systemAudioTrimStart))
                        .font(.system(.caption, design: .monospaced))
                        .frame(width: 48)
                }
                HStack {
                    Text("终点")
                        .frame(width: 44, alignment: .leading)
                    Slider(
                        value: $model.systemAudioTrimEnd,
                        in: min(model.systemAudioDuration, model.systemAudioTrimStart + 0.25)
                            ... max(model.systemAudioDuration, model.systemAudioTrimStart + 0.26)
                    )
                    Text(time(model.systemAudioTrimEnd))
                        .font(.system(.caption, design: .monospaced))
                        .frame(width: 48)
                }
            }

            TextField("参考音频名称", text: $model.systemAudioName)
                .textFieldStyle(.roundedBorder)

            HStack(spacing: StudioTokens.space2) {
                if model.isExportingSystemAudio || model.isImportingReference {
                    ProgressView()
                        .controlSize(.small)
                }
                Text(model.audioTrimProgress)
                    .font(.caption)
                    .foregroundStyle(
                        model.audioTrimProgress.contains("失败")
                            || model.audioTrimProgress.contains("无法")
                            ? Color.red
                            : Color.secondary
                    )
                Spacer()
            }

            HStack {
                Button("试听选区") { model.previewSystemAudioSelection() }
                    .buttonStyle(StudioSecondaryButtonStyle())
                Button("停止试听") { model.stopPreview() }
                    .buttonStyle(StudioSecondaryButtonStyle())
                Spacer()
                Button("取消") { model.discardSystemAudioRecording() }
                    .buttonStyle(StudioSecondaryButtonStyle())
                    .disabled(model.isExportingSystemAudio || model.isImportingReference)
                Button {
                    model.exportSystemAudioSelection()
                } label: {
                    Label(
                        model.isImportingReference
                            ? "正在导入…"
                            : model.isExportingSystemAudio
                            ? "正在压缩…"
                            : "裁剪并加入参考音频",
                        systemImage: "scissors"
                    )
                }
                .buttonStyle(StudioTintedButtonStyle())
                .disabled(model.isExportingSystemAudio || model.isImportingReference)
            }
        }
        .padding(StudioTokens.space5)
        .frame(width: 720, height: 500)
        .background(StudioBackground())
        .interactiveDismissDisabled(model.isExportingSystemAudio || model.isImportingReference)
    }
}
