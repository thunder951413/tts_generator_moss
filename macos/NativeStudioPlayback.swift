import AppKit
import AVFoundation
import CoreMedia
import Foundation
import ScreenCaptureKit
import SwiftUI
import UniformTypeIdentifiers

final class NativePCMStreamPlayer: NSObject, URLSessionDataDelegate {
    private let engine = AVAudioEngine()
    private let node = AVAudioPlayerNode()
    private let format: AVAudioFormat
    private let request: URLRequest
    private let lock = NSLock()
    private var session: URLSession?
    private var remainder = Data()
    private var pendingBuffers = 0
    private var networkFinished = false
    private var stopped = false
    private var receivedFirstAudio = false
    private var responseError: Error?

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
        super.init()
        engine.attach(node)
        engine.connect(node, to: engine.mainMixerNode, format: format)
        try engine.start()
        node.play()
    }

    func start() {
        let queue = OperationQueue()
        queue.name = "qwen-native-pcm-stream"
        queue.maxConcurrentOperationCount = 1
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
            responseError = NativeStudioError.server("PCM 音频流连接失败。")
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

        var bytes = remainder
        bytes.append(data)
        let frameBytes = Int(format.channelCount) * 2
        let usable = bytes.count - (bytes.count % frameBytes)
        guard usable > 0 else {
            remainder = bytes
            return
        }
        remainder = bytes.subdata(in: usable ..< bytes.count)
        let frameCount = usable / frameBytes
        guard let buffer = AVAudioPCMBuffer(
            pcmFormat: format,
            frameCapacity: AVAudioFrameCount(frameCount)
        ), let channels = buffer.floatChannelData else {
            return
        }
        buffer.frameLength = AVAudioFrameCount(frameCount)
        bytes.withUnsafeBytes { raw in
            let source = raw.bindMemory(to: UInt8.self)
            for frame in 0 ..< frameCount {
                for channel in 0 ..< Int(format.channelCount) {
                    let offset = (frame * Int(format.channelCount) + channel) * 2
                    let bits = UInt16(source[offset]) | (UInt16(source[offset + 1]) << 8)
                    channels[channel][frame] = Float(Int16(bitPattern: bits)) / 32768.0
                }
            }
        }

        lock.lock()
        pendingBuffers += 1
        let announceFirstAudio = !receivedFirstAudio
        receivedFirstAudio = true
        lock.unlock()
        if announceFirstAudio {
            DispatchQueue.main.async { [weak self] in self?.onFirstAudio?() }
        }
        node.scheduleBuffer(buffer, completionCallbackType: .dataPlayedBack) { [weak self] _ in
            self?.bufferDidFinish()
        }
    }

    func urlSession(
        _ session: URLSession,
        task: URLSessionTask,
        didCompleteWithError error: Error?
    ) {
        lock.lock()
        networkFinished = true
        let isStopped = stopped
        let failure = responseError ?? error
        let pending = pendingBuffers
        lock.unlock()
        if let failure, !isStopped {
            DispatchQueue.main.async { [weak self] in self?.onError?(failure) }
            stop()
        } else if !isStopped && pending == 0 {
            finishPlayback()
        }
    }

    private func bufferDidFinish() {
        lock.lock()
        pendingBuffers = max(0, pendingBuffers - 1)
        let shouldFinish = networkFinished && pendingBuffers == 0 && !stopped
        lock.unlock()
        if shouldFinish { finishPlayback() }
    }

    private func finishPlayback() {
        lock.lock()
        guard !stopped else {
            lock.unlock()
            return
        }
        stopped = true
        lock.unlock()
        session?.finishTasksAndInvalidate()
        node.stop()
        engine.stop()
        DispatchQueue.main.async { [weak self] in self?.onComplete?() }
    }
}

final class SystemAudioRecorder: NSObject, SCStreamOutput, SCStreamDelegate {
    private let sampleQueue = DispatchQueue(label: "qwen-system-audio-capture")
    private var stream: SCStream?
    private var writer: AVAssetWriter?
    private var writerInput: AVAssetWriterInput?
    private var outputURL: URL?
    private var startedWriting = false
    private var intentionallyStopping = false
    private var completion: ((Result<URL, Error>) -> Void)?
    var onUnexpectedError: ((Error) -> Void)?

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
                self.outputURL = outputURL
                self.writer = writer
                self.stream = stream
                self.intentionallyStopping = false
                try await stream.startCapture()
                DispatchQueue.main.async { completion(.success(())) }
            } catch {
                reset(removeOutput: true)
                DispatchQueue.main.async { completion(.failure(error)) }
            }
        }
    }

    func stop(completion: @escaping (Result<URL, Error>) -> Void) {
        intentionallyStopping = true
        self.completion = completion
        guard let stream else {
            finishWriting()
            return
        }
        Task {
            do {
                try await stream.stopCapture()
                sampleQueue.async { [weak self] in self?.finishWriting() }
            } catch {
                sampleQueue.async { [weak self] in self?.complete(.failure(error)) }
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
        guard !intentionallyStopping else { return }
        sampleQueue.async { [weak self] in
            guard let self else { return }
            let handler = self.onUnexpectedError
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
            if writer.status == .completed {
                self.complete(.success(outputURL), removeOutput: false)
            } else {
                self.complete(
                    .failure(writer.error ?? NativeStudioError.server("系统声音录制写入失败。"))
                )
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
