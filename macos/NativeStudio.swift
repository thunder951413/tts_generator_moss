import AppKit
import AVFoundation
import CoreMedia
import Foundation
import ScreenCaptureKit
import SwiftUI
import UniformTypeIdentifiers

struct NativeVoice: Identifiable, Hashable {
    let id: String
    let name: String
    let description: String
    let audioPath: String
    let language: String
    let transcript: String
    let transcriptSource: String

    init?(_ value: [String: Any]) {
        guard let audioPath = value["audio_path"] as? String, !audioPath.isEmpty else { return nil }
        self.id = audioPath
        self.name = value["name"] as? String ?? URL(fileURLWithPath: audioPath).deletingPathExtension().lastPathComponent
        self.description = value["description"] as? String ?? ""
        self.audioPath = audioPath
        self.language = value["language"] as? String ?? "Chinese"
        self.transcript = value["transcript"] as? String ?? ""
        self.transcriptSource = value["transcript_source"] as? String ?? ""
    }
}

struct NativeReferenceAudio: Identifiable, Hashable {
    let id: String
    let name: String
    let path: String
    let kind: String
    let hidden: Bool
    let inUse: Bool
    let usages: [String]

    init?(_ value: [String: Any]) {
        guard let id = value["id"] as? String,
              let path = (value["path"] ?? value["audio_path"]) as? String,
              !id.isEmpty, !path.isEmpty
        else { return nil }
        self.id = id
        self.name = value["name"] as? String ?? URL(fileURLWithPath: path).deletingPathExtension().lastPathComponent
        self.path = path
        self.kind = value["kind"] as? String ?? "custom"
        self.hidden = value["hidden"] as? Bool ?? false
        self.inUse = value["in_use"] as? Bool ?? false
        self.usages = value["usages"] as? [String] ?? []
    }
}

struct NativePreset: Identifiable {
    let id: String
    let name: String
    let settings: [String: Any]

    init?(_ value: [String: Any]) {
        guard let id = value["id"] as? String,
              let name = value["name"] as? String,
              let settings = value["settings"] as? [String: Any]
        else { return nil }
        self.id = id
        self.name = name
        self.settings = settings
    }

    var dictionary: [String: Any] {
        ["id": id, "name": name, "settings": settings]
    }
}

enum NativeStudioError: LocalizedError {
    case invalidResponse
    case server(String)
    case missingReference
    case emptyText

    var errorDescription: String? {
        switch self {
        case .invalidResponse:
            return "本地服务返回了无法识别的数据。"
        case .server(let message):
            return message
        case .missingReference:
            return "请先选择、导入或录制参考音频。"
        case .emptyText:
            return "请输入要生成的文本。"
        }
    }
}

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

final class NativeStudioViewModel: ObservableObject {
    @Published var serviceSummary = "正在启动本地服务…"
    @Published var serviceReady = false
    @Published var ttsEnabled = true
    @Published var ttsReady = false
    @Published var sttReady = false
    @Published var isTogglingTTS = false
    @Published var isTogglingSTT = false
    @Published var voices: [NativeVoice] = []
    @Published var presets: [NativePreset] = []
    @Published var selectedPresetID = ""
    @Published var presetName = ""
    @Published var selectedVoicePath = ""
    @Published var referenceAudioPath = ""
    @Published var referenceName = "尚未选择参考音色"
    @Published var text = "欢迎使用 Qwen3-TTS 原生 Mac 音频工作台。"
    @Published var modelProfile = "qwen_0_6b"
    @Published var cloneMode = "xvec"
    @Published var referenceText = ""
    @Published var temperature = 0.9
    @Published var topP = 1.0
    @Published var topK = 50
    @Published var repetitionPenalty = 1.05
    @Published var maxNewTokens = 2048
    @Published var chunkFrames = 8
    @Published var minNewTokens = 2
    @Published var seed = 1234
    @Published var appendSilence = true
    @Published var nonStreamingInput = false
    @Published var streamingGeneration = true
    @Published var isGenerating = false
    @Published var generatedProgress = 0.0
    @Published var generationSummary = "等待生成"
    @Published var serviceSettingsStatus = "尚未应用为服务默认设置"
    @Published var isApplyingServiceSettings = false
    @Published var actualGenerationSeed: Int?
    @Published var submittedSeedSetting: Int?
    @Published var currentJobDisplayID = ""
    @Published var errorMessage = ""
    @Published var outputAudioURL: URL?
    @Published var isImportingReference = false
    @Published var isRecordingReference = false
    @Published var referenceLibrary: [NativeReferenceAudio] = []
    @Published var referenceLibraryPresented = false
    @Published var pendingReferenceDeletion: NativeReferenceAudio?
    @Published var referenceLibraryStatus = ""
    @Published var referenceLibraryTab = "visible"
    @Published var isDeletingReference = false
    @Published var isRecordingSystemAudio = false
    @Published var isPreparingSystemAudioCapture = false
    @Published var systemAudioStatus = "可录制 Mac 正在播放的声音"
    @Published var systemAudioElapsed: TimeInterval = 0
    @Published var systemAudioTrimPresented = false
    @Published var systemAudioDuration: TimeInterval = 0
    @Published var systemAudioTrimStart: TimeInterval = 0
    @Published var systemAudioTrimEnd: TimeInterval = 0
    @Published var systemAudioWaveform: [Float] = []
    @Published var systemAudioName = ""
    @Published var isExportingSystemAudio = false
    @Published var isSavingPreset = false
    @Published var advancedSettingsPresented = false
    @Published var externalAPISettingsPresented = false
    @Published var externalAccessEnabled = false
    @Published var externalPort = 7861
    @Published var externalPassword = ""
    @Published var externalCORSOrigins = ""
    @Published var externalSettingsStatus = ""
    @Published var isApplyingExternalSettings = false
    @Published var isRunningPerformanceTest = false
    @Published var performanceTestStatus = "尚未测试，将使用保守的单通道策略"
    @Published var performanceStreamSummary = "流式：8 帧/块 · 1 路"
    @Published var performanceBlockSummary = "整块：1 路"
    @Published var performanceGainSummary = "并发收益：待测试"

    let service: LocalService
    var onPresetsChanged: (([NativePreset]) -> Void)?
    var onTTSServiceToggleRequested: (() -> Void)?
    var onSTTServiceToggleRequested: (() -> Void)?
    private var pollingTimer: Timer?
    private var currentJobID: String?
    private var stoppedJobIDs: Set<String> = []
    private var pcmStreamPlayer: NativePCMStreamPlayer?
    private var generationEpoch = 0
    private var isForceStopping = false
    private var generationFinished = false
    private var streamPlaybackFinished = false
    private var resultDownloaded = false
    private var player: AVPlayer?
    private var playbackEndObserver: NSObjectProtocol?
    private var playbackFailureObserver: NSObjectProtocol?
    private var playbackJobID: String?
    private var recorder: AVAudioRecorder?
    private var recordingURL: URL?
    private var defaultReferenceAudioPath = ""
    private var systemAudioRecorder: SystemAudioRecorder?
    private var systemAudioRecordingURL: URL?
    private var systemAudioTimer: Timer?
    private var systemAudioStartedAt: Date?

    init(service: LocalService) {
        self.service = service
        self.externalAccessEnabled = service.configuration["HOST"] == "0.0.0.0"
        self.externalPort = service.port
        self.externalPassword = service.configuration["QWEN_TTS_ACCESS_PASSWORD"] ?? ""
        self.externalCORSOrigins = service.configuration["QWEN_TTS_CORS_ORIGINS"] ?? ""
    }

    deinit {
        pollingTimer?.invalidate()
        if let playbackEndObserver {
            NotificationCenter.default.removeObserver(playbackEndObserver)
        }
        if let playbackFailureObserver {
            NotificationCenter.default.removeObserver(playbackFailureObserver)
        }
        pcmStreamPlayer?.stop()
        removeTemporaryOutput()
        if let recordingURL {
            try? FileManager.default.removeItem(at: recordingURL)
        }
        systemAudioTimer?.invalidate()
        if let systemAudioRecordingURL {
            try? FileManager.default.removeItem(at: systemAudioRecordingURL)
        }
    }

    func updateHealth(_ health: [String: Any]) {
        let state = health["state"] as? String ?? "unknown"
        let profile = health["active_profile_label"] as? String ?? "等待模型"
        let scheduler = health["generation_scheduler"] as? [String: Any]
        let stt = health["stt"] as? [String: Any]
        let sttReady = stt?["ready"] as? Bool ?? false
        let active = number(scheduler?["active"]).intValue
        let maximum = number(scheduler?["max_parallel"]).intValue
        ttsEnabled = boolean(health["tts_enabled"], fallback: true)
        ttsReady = state == "ready"
        self.sttReady = sttReady
        serviceReady = ttsEnabled && ttsReady
        serviceSummary = serviceReady
            ? "TTS Metal · STT \(sttReady ? "就绪" : "未就绪") · \(profile) · GPU \(active)/\(maximum)"
            : (ttsEnabled
                ? "TTS 服务状态：\(state)"
                : "HTTP 服务已连接 · TTS 已停止 · STT \(sttReady ? "就绪" : "未就绪")")
    }

    func setTTSService(enabled: Bool, completion: @escaping (Result<[String: Any], Error>) -> Void) {
        guard !isTogglingTTS else { return }
        isTogglingTTS = true
        service.perform("api/tts/\(enabled ? "start" : "stop")", method: "POST", timeout: 120) {
            [weak self] data, response in
            guard let self else { return }
            do {
                let payload = try self.jsonObject(data, response: response)
                self.service.health { health in
                    DispatchQueue.main.async {
                        self.isTogglingTTS = false
                        if let health { self.updateHealth(health) }
                        completion(.success(payload))
                    }
                }
            } catch {
                DispatchQueue.main.async {
                    self.isTogglingTTS = false
                    completion(.failure(error))
                }
            }
        }
    }

    func setSTTService(enabled: Bool, completion: @escaping (Result<[String: Any], Error>) -> Void) {
        guard !isTogglingSTT else { return }
        isTogglingSTT = true
        service.perform("api/stt/\(enabled ? "start" : "stop")", method: "POST", timeout: 60) {
            [weak self] data, response in
            guard let self else { return }
            do {
                let payload = try self.jsonObject(data, response: response)
                self.service.health { health in
                    DispatchQueue.main.async {
                        self.isTogglingSTT = false
                        if let health { self.updateHealth(health) }
                        completion(.success(payload))
                    }
                }
            } catch {
                DispatchQueue.main.async {
                    self.isTogglingSTT = false
                    completion(.failure(error))
                }
            }
        }
    }

    func loadInitialData() {
        loadVoices()
        loadReferenceLibrary()
        refreshPresets()
        loadActiveServiceSettings()
        loadPerformanceProfile()
    }

    func loadPerformanceProfile() {
        service.perform("api/performance") { [weak self] data, response in
            guard let self else { return }
            do {
                let payload = try self.jsonObject(data, response: response)
                let activeProfile = self.string(payload["active_profile"], fallback: self.modelProfile)
                let profiles = payload["profiles"] as? [String: Any] ?? [:]
                guard let profile = profiles[activeProfile] as? [String: Any],
                      let recommendation = profile["recommendation"] as? [String: Any]
                else { return }
                DispatchQueue.main.async {
                    self.applyPerformanceResult(
                        profile: profile,
                        recommendation: recommendation,
                        loaded: true
                    )
                }
            } catch {
                // A missing performance profile is expected before the first test.
            }
        }
    }

    func runPerformanceTest() {
        guard serviceReady, !isGenerating, !isRunningPerformanceTest else { return }
        isRunningPerformanceTest = true
        performanceTestStatus = "正在预热模型并测试流式首包…"
        errorMessage = ""
        let body = try? JSONSerialization.data(
            withJSONObject: ["model_profile": modelProfile]
        )
        service.perform(
            "api/performance/benchmark",
            method: "POST",
            body: body,
            contentType: "application/json",
            timeout: 900
        ) { [weak self] data, response in
            guard let self else { return }
            do {
                let payload = try self.jsonObject(data, response: response)
                guard let recommendation = payload["recommendation"] as? [String: Any] else {
                    throw NativeStudioError.invalidResponse
                }
                DispatchQueue.main.async {
                    self.isRunningPerformanceTest = false
                    self.applyPerformanceResult(
                        profile: payload,
                        recommendation: recommendation,
                        loaded: false
                    )
                }
            } catch {
                DispatchQueue.main.async {
                    self.isRunningPerformanceTest = false
                    self.performanceTestStatus = "测试失败：\(error.localizedDescription)"
                    self.show(error)
                }
            }
        }
    }

    private func applyPerformanceResult(
        profile: [String: Any],
        recommendation: [String: Any],
        loaded: Bool
    ) {
        let chunk = integer(recommendation["stream_chunk_frames"], fallback: 8)
        let streamParallel = integer(recommendation["stream_parallel"], fallback: 1)
        let blockParallel = integer(recommendation["block_parallel"], fallback: 1)
        let block = profile["block_measurements"] as? [String: Any]
        let gain = double(block?["throughput_gain"], fallback: 1.0)
        performanceStreamSummary = "流式：\(chunk) 帧/块 · \(streamParallel) 路"
        performanceBlockSummary = "整块：\(blockParallel) 路并行"
        performanceGainSummary = String(format: "双路吞吐：%.2f×", gain)
        performanceTestStatus = loaded
            ? "已载入本机优化策略，将自动用于阅读器和后台任务"
            : "测试完成，推荐策略已保存并立即应用"
    }

    func loadActiveServiceSettings() {
        service.perform("api/service-settings") { [weak self] data, response in
            guard let self else { return }
            do {
                let payload = try self.jsonObject(data, response: response)
                guard let settings = payload["settings"] as? [String: Any] else {
                    throw NativeStudioError.invalidResponse
                }
                let activePresetID = payload["active_preset_id"] as? String ?? ""
                let name = payload["name"] as? String ?? "服务设置"
                DispatchQueue.main.async {
                    self.modelProfile = self.string(settings["model_profile"], fallback: "qwen_0_6b")
                    self.cloneMode = self.string(settings["qwen_clone_mode"], fallback: "xvec")
                    self.referenceText = self.string(settings["qwen_reference_text"])
                    self.temperature = self.double(settings["qwen_temperature"], fallback: 0.9)
                    self.topP = self.double(settings["qwen_top_p"], fallback: 1.0)
                    self.topK = self.integer(settings["qwen_top_k"], fallback: 50)
                    self.repetitionPenalty = self.double(settings["qwen_repetition_penalty"], fallback: 1.05)
                    self.maxNewTokens = self.integer(settings["qwen_max_new_tokens"], fallback: 2048)
                    self.chunkFrames = self.integer(settings["qwen_chunk_size"], fallback: 8)
                    self.minNewTokens = self.integer(settings["qwen_min_new_tokens"], fallback: 2)
                    self.seed = self.integer(settings["qwen_seed"], fallback: 1234)
                    self.appendSilence = self.boolean(settings["qwen_append_silence"], fallback: true)
                    self.nonStreamingInput = self.boolean(settings["qwen_non_streaming_mode"], fallback: false)
                    self.streamingGeneration = self.boolean(settings["qwen_streaming_generation"], fallback: true)
                    self.referenceAudioPath = self.string(settings["reference_audio_path"])
                    self.referenceName = self.string(settings["voice_name"], fallback: name)
                    self.selectedPresetID = activePresetID
                    self.selectedVoicePath = self.voices.contains(where: { $0.audioPath == self.referenceAudioPath })
                        ? self.referenceAudioPath
                        : ""
                    self.serviceSettingsStatus = "当前服务设置：\(name)"
                }
            } catch {
                self.show(error)
            }
        }
    }

    var localAPIEndpoint: String {
        "http://127.0.0.1:\(externalPort)"
    }

    var networkAPIEndpoint: String {
        let host = ProcessInfo.processInfo.hostName
        return "http://\(host):\(externalPort)"
    }

    var modelDisplayName: String {
        modelProfile == "qwen_1_7b" ? "Qwen3-TTS 1.7B 高质量" : "Qwen3-TTS 0.6B 极速"
    }

    var actualSeedDisplay: String {
        if seed >= 0 {
            return "\(seed)（固定）"
        }
        guard submittedSeedSetting == seed, let actualGenerationSeed else {
            return "随机（生成时确定真实值）"
        }
        return "\(actualGenerationSeed)（本次随机真实值）"
    }

    func copyToPasteboard(_ value: String) {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(value, forType: .string)
        externalSettingsStatus = "已复制到剪贴板"
    }

    func applyExternalSettings() {
        let password = externalPassword.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !password.contains("\n"), !password.contains("\r") else {
            externalSettingsStatus = "访问密码不能包含换行符"
            return
        }
        if externalAccessEnabled && (password.count < 4 || password == "change-me") {
            externalSettingsStatus = "局域网模式需要至少 4 位且不是 change-me 的密码"
            return
        }
        let corsOrigins = externalCORSOrigins.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !corsOrigins.contains("\n"), !corsOrigins.contains("\r") else {
            externalSettingsStatus = "网页来源不能包含换行符"
            return
        }
        isApplyingExternalSettings = true
        externalSettingsStatus = "正在保存并重启语音服务…"
        service.applyExternalSettings(
            exposeToLAN: externalAccessEnabled,
            port: externalPort,
            password: password,
            corsOrigins: corsOrigins
        ) { [weak self] result in
            DispatchQueue.main.async {
                guard let self else { return }
                self.isApplyingExternalSettings = false
                switch result {
                case .success(let health):
                    self.externalPassword = password
                    self.externalCORSOrigins = corsOrigins
                    self.updateHealth(health)
                    self.loadInitialData()
                    self.externalSettingsStatus = "设置已生效，TTS/STT 服务已重新就绪"
                case .failure(let error):
                    self.externalSettingsStatus = error.localizedDescription
                }
            }
        }
    }

    func loadVoices() {
        service.perform("api/voices") { [weak self] data, response in
            guard let self else { return }
            do {
                let payload = try self.jsonObject(data, response: response)
                let rawVoices = payload["voices"] as? [[String: Any]] ?? []
                let voices = rawVoices.compactMap(NativeVoice.init)
                let defaultPath = payload["default_reference_audio_path"] as? String ?? ""
                DispatchQueue.main.async {
                    self.voices = voices
                    self.defaultReferenceAudioPath = defaultPath
                    if self.referenceAudioPath.isEmpty,
                       !defaultPath.isEmpty {
                        self.chooseVoice(defaultPath)
                    } else if voices.contains(where: { $0.audioPath == self.referenceAudioPath }) {
                        self.selectedVoicePath = self.referenceAudioPath
                    }
                }
            } catch {
                self.show(error)
            }
        }
    }

    func loadReferenceLibrary() {
        service.perform("api/reference-audio-library?include_hidden=true") { [weak self] data, response in
            guard let self else { return }
            do {
                let payload = try self.jsonObject(data, response: response)
                let references = (payload["references"] as? [[String: Any]] ?? [])
                    .compactMap(NativeReferenceAudio.init)
                DispatchQueue.main.async { self.referenceLibrary = references }
            } catch {
                self.show(error)
            }
        }
    }

    func useReference(_ reference: NativeReferenceAudio) {
        referenceAudioPath = reference.path
        referenceName = reference.name
        selectedVoicePath = voices.contains(where: { $0.audioPath == reference.path })
            ? reference.path
            : ""
        if cloneMode == "icl",
           let voice = voices.first(where: { $0.audioPath == reference.path }) {
            referenceText = voice.transcript
        } else {
            referenceText = ""
        }
        errorMessage = ""
    }

    func previewReference(_ reference: NativeReferenceAudio) {
        play(URL(fileURLWithPath: reference.path))
    }

    func setReferenceHidden(_ reference: NativeReferenceAudio, hidden: Bool) {
        let body = try? JSONSerialization.data(withJSONObject: ["hidden": hidden])
        service.perform(
            "api/reference-audio-library/\(reference.id)/visibility",
            method: "PUT",
            body: body,
            contentType: "application/json"
        ) { [weak self] data, response in
            guard let self else { return }
            do {
                _ = try self.jsonObject(data, response: response)
                DispatchQueue.main.async {
                    self.loadReferenceLibrary()
                    self.loadVoices()
                }
            } catch {
                self.show(error)
            }
        }
    }

    func deleteReference(_ reference: NativeReferenceAudio) {
        guard reference.kind == "custom" else {
            referenceLibraryStatus = "内置参考音频只能隐藏，不能删除。"
            return
        }
        isDeletingReference = true
        referenceLibraryStatus = "正在删除“\(reference.name)”…"
        let replaceUsages = reference.inUse ? "?replace_usages=true" : ""
        service.perform(
            "api/reference-audio-library/\(reference.id)\(replaceUsages)",
            method: "DELETE"
        ) { [weak self] data, response in
            guard let self else { return }
            do {
                let payload = try self.jsonObject(data, response: response)
                let replacement = payload["replacement"] as? [String: Any]
                let replacementPath = self.string(
                    replacement?["reference_audio_path"],
                    fallback: self.defaultReferenceAudioPath
                )
                let replacementName = self.string(
                    replacement?["name"],
                    fallback: "龙嫱"
                )
                let replacedUsages = payload["replaced_usages"] as? [String] ?? []
                DispatchQueue.main.async {
                    self.isDeletingReference = false
                    if self.referenceAudioPath == reference.path || !replacedUsages.isEmpty {
                        self.referenceAudioPath = replacementPath
                        self.referenceName = replacementName
                        self.selectedVoicePath = replacementPath
                        self.referenceText = ""
                    }
                    self.referenceLibraryStatus = replacedUsages.isEmpty
                        ? "已删除“\(reference.name)”"
                        : "已删除“\(reference.name)”，并将\(replacedUsages.joined(separator: "、"))切换为\(replacementName)"
                    self.loadReferenceLibrary()
                    self.loadVoices()
                    self.refreshPresets()
                    if !replacedUsages.isEmpty {
                        self.loadActiveServiceSettings()
                    }
                }
            } catch {
                DispatchQueue.main.async {
                    self.isDeletingReference = false
                    self.referenceLibraryStatus = error.localizedDescription
                }
            }
        }
    }

    func refreshPresets(selecting presetID: String? = nil) {
        service.perform("api/presets") { [weak self] data, response in
            guard let self else { return }
            do {
                let payload = try self.jsonObject(data, response: response)
                let presets = (payload["presets"] as? [[String: Any]] ?? []).compactMap(NativePreset.init)
                let activePresetID = payload["active_preset_id"] as? String ?? ""
                DispatchQueue.main.async {
                    self.presets = presets
                    if let presetID, presets.contains(where: { $0.id == presetID }) {
                        self.selectedPresetID = presetID
                    } else if !activePresetID.isEmpty,
                              let activePreset = presets.first(where: { $0.id == activePresetID }) {
                        self.applyPreset(activePreset)
                    }
                    self.onPresetsChanged?(presets)
                }
            } catch {
                self.show(error)
            }
        }
    }

    func chooseVoice(_ audioPath: String) {
        selectedVoicePath = audioPath
        guard let voice = voices.first(where: { $0.audioPath == audioPath }) else { return }
        referenceAudioPath = voice.audioPath
        referenceName = voice.name
        if cloneMode == "icl" {
            referenceText = voice.transcript
        }
        errorMessage = ""
    }

    func setCloneMode(_ mode: String) {
        cloneMode = mode == "icl" ? "icl" : "xvec"
        guard cloneMode == "icl",
              referenceText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
              let voice = voices.first(where: { $0.audioPath == selectedVoicePath })
        else { return }
        referenceText = voice.transcript
    }

    func chooseReferenceFile() {
        let panel = NSOpenPanel()
        panel.title = "选择语音克隆参考音频"
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = false
        panel.allowedContentTypes = [.audio]
        guard panel.runModal() == .OK, let url = panel.url else { return }
        importReference(url)
    }

    func importReference(_ url: URL, displayName: String? = nil) {
        guard let fileData = try? Data(contentsOf: url) else {
            show(NativeStudioError.server("无法读取参考音频。"))
            return
        }
        isImportingReference = true
        let multipart = multipartBody(
            fields: [:],
            file: (
                "audio",
                "\(displayName ?? url.deletingPathExtension().lastPathComponent).\(url.pathExtension)",
                "application/octet-stream",
                fileData
            )
        )
        if url.deletingLastPathComponent() == FileManager.default.temporaryDirectory {
            if url.lastPathComponent.hasPrefix("qwen-reference-") {
                recordingURL = nil
                try? FileManager.default.removeItem(at: url)
            } else if url.lastPathComponent.hasPrefix("qwen-system-export-") {
                try? FileManager.default.removeItem(at: url)
            }
        }
        service.perform(
            "api/presets/reference-audio",
            method: "POST",
            body: multipart.data,
            contentType: multipart.contentType,
            timeout: 120
        ) { [weak self] data, response in
            guard let self else { return }
            do {
                let payload = try self.jsonObject(data, response: response)
                guard let path = payload["reference_audio_path"] as? String else {
                    throw NativeStudioError.invalidResponse
                }
                DispatchQueue.main.async {
                    self.isImportingReference = false
                    self.referenceAudioPath = path
                    self.referenceName = displayName ?? url.deletingPathExtension().lastPathComponent
                    self.referenceText = ""
                    self.loadReferenceLibrary()
                    self.loadVoices()
                }
            } catch {
                DispatchQueue.main.async { self.isImportingReference = false }
                self.show(error)
            }
        }
    }

    func toggleReferenceRecording() {
        if isRecordingReference {
            recorder?.stop()
            recorder = nil
            isRecordingReference = false
            if let recordingURL {
                importReference(recordingURL)
            }
            return
        }
        switch AVCaptureDevice.authorizationStatus(for: .audio) {
        case .authorized:
            startReferenceRecording()
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .audio) { [weak self] granted in
                DispatchQueue.main.async {
                    if granted {
                        self?.startReferenceRecording()
                    } else {
                        self?.show(NativeStudioError.server("没有麦克风权限，无法录制参考音频。"))
                    }
                }
            }
        default:
            show(NativeStudioError.server("没有麦克风权限，请在系统设置中允许 Qwen TTS 使用麦克风。"))
        }
    }

    private func startReferenceRecording() {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("qwen-reference-\(UUID().uuidString).wav")
        let settings: [String: Any] = [
            AVFormatIDKey: kAudioFormatLinearPCM,
            AVSampleRateKey: 44_100,
            AVNumberOfChannelsKey: 1,
            AVLinearPCMBitDepthKey: 16,
            AVLinearPCMIsFloatKey: false,
            AVLinearPCMIsBigEndianKey: false,
        ]
        do {
            let recorder = try AVAudioRecorder(url: url, settings: settings)
            recorder.prepareToRecord()
            guard recorder.record() else {
                throw NativeStudioError.server("麦克风录制启动失败。")
            }
            self.recorder = recorder
            recordingURL = url
            isRecordingReference = true
            referenceName = "正在录制参考音频…"
            errorMessage = ""
        } catch {
            show(error)
        }
    }

    func toggleSystemAudioRecording() {
        guard !isPreparingSystemAudioCapture else { return }
        if isRecordingSystemAudio {
            stopSystemAudioRecording()
        } else {
            startSystemAudioRecording()
        }
    }

    private func startSystemAudioRecording() {
        player?.pause()
        let recorder = SystemAudioRecorder()
        systemAudioRecorder = recorder
        isPreparingSystemAudioCapture = true
        recorder.onUnexpectedError = { [weak self] error in
            guard let self else { return }
            self.systemAudioTimer?.invalidate()
            self.systemAudioTimer = nil
            self.systemAudioStartedAt = nil
            self.systemAudioRecorder = nil
            self.isRecordingSystemAudio = false
            self.systemAudioStatus = "系统声音录制意外停止"
            self.show(error)
        }
        systemAudioStatus = "正在请求系统录音权限…"
        errorMessage = ""
        recorder.start { [weak self] result in
            guard let self else { return }
            switch result {
            case .success:
                self.isPreparingSystemAudioCapture = false
                self.isRecordingSystemAudio = true
                self.systemAudioElapsed = 0
                self.systemAudioStartedAt = Date()
                self.systemAudioStatus = "正在录制系统播放声音，点击停止才会结束"
                self.systemAudioTimer?.invalidate()
                self.systemAudioTimer = Timer.scheduledTimer(withTimeInterval: 0.25, repeats: true) {
                    [weak self] _ in
                    guard let self, let startedAt = self.systemAudioStartedAt else { return }
                    self.systemAudioElapsed = Date().timeIntervalSince(startedAt)
                }
            case .failure(let error):
                self.isPreparingSystemAudioCapture = false
                self.systemAudioRecorder = nil
                self.systemAudioStatus = "系统声音录制未启动"
                self.show(
                    NativeStudioError.server(
                        "\(error.localizedDescription)\n请在“系统设置 → 隐私与安全性 → 屏幕与系统音频录制”中允许 Qwen TTS。首次授权后可能需要重新打开应用。"
                    )
                )
            }
        }
    }

    private func stopSystemAudioRecording() {
        guard let recorder = systemAudioRecorder else { return }
        isRecordingSystemAudio = false
        systemAudioTimer?.invalidate()
        systemAudioTimer = nil
        systemAudioStartedAt = nil
        systemAudioStatus = "正在整理录音…"
        recorder.stop { [weak self] result in
            guard let self else { return }
            self.systemAudioRecorder = nil
            switch result {
            case .success(let url):
                self.prepareSystemAudioForTrimming(url)
            case .failure(let error):
                self.systemAudioStatus = "录制失败"
                self.show(error)
            }
        }
    }

    private func prepareSystemAudioForTrimming(_ url: URL) {
        systemAudioRecordingURL = url
        let asset = AVURLAsset(url: url)
        let duration = CMTimeGetSeconds(asset.duration)
        guard duration.isFinite, duration > 0.1 else {
            try? FileManager.default.removeItem(at: url)
            systemAudioRecordingURL = nil
            systemAudioStatus = "没有捕获到可用声音"
            show(NativeStudioError.server("录音太短或没有捕获到系统声音。"))
            return
        }
        systemAudioDuration = duration
        systemAudioTrimStart = 0
        systemAudioTrimEnd = duration
        systemAudioName = "系统录音 \(Self.recordingDateFormatter.string(from: Date()))"
        systemAudioWaveform = []
        systemAudioStatus = "录制完成，可裁剪后加入参考音频"
        systemAudioTrimPresented = true
        analyzeWaveform(url)
    }

    private func analyzeWaveform(_ url: URL) {
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            do {
                let file = try AVAudioFile(
                    forReading: url,
                    commonFormat: .pcmFormatFloat32,
                    interleaved: false
                )
                let bucketCount = 180
                let framesPerBucket = max(1, Int(file.length) / bucketCount)
                var peaks: [Float] = []
                while file.framePosition < file.length && peaks.count < bucketCount {
                    let remaining = Int(file.length - file.framePosition)
                    let count = min(framesPerBucket, remaining)
                    guard let buffer = AVAudioPCMBuffer(
                        pcmFormat: file.processingFormat,
                        frameCapacity: AVAudioFrameCount(count)
                    ) else { break }
                    try file.read(into: buffer, frameCount: AVAudioFrameCount(count))
                    guard let channels = buffer.floatChannelData else { break }
                    var peak: Float = 0
                    for channel in 0 ..< Int(buffer.format.channelCount) {
                        for frame in 0 ..< Int(buffer.frameLength) {
                            peak = max(peak, abs(channels[channel][frame]))
                        }
                    }
                    peaks.append(min(1, peak))
                }
                DispatchQueue.main.async { self?.systemAudioWaveform = peaks }
            } catch {
                DispatchQueue.main.async { self?.systemAudioWaveform = [] }
            }
        }
    }

    func previewSystemAudioSelection() {
        guard let url = systemAudioRecordingURL else { return }
        let item = AVPlayerItem(url: url)
        item.forwardPlaybackEndTime = CMTime(seconds: systemAudioTrimEnd, preferredTimescale: 600)
        player = AVPlayer(playerItem: item)
        player?.seek(to: CMTime(seconds: systemAudioTrimStart, preferredTimescale: 600))
        player?.play()
    }

    func stopPreview() {
        if let playbackEndObserver {
            NotificationCenter.default.removeObserver(playbackEndObserver)
            self.playbackEndObserver = nil
        }
        if let playbackFailureObserver {
            NotificationCenter.default.removeObserver(playbackFailureObserver)
            self.playbackFailureObserver = nil
        }
        player?.pause()
        player?.replaceCurrentItem(with: nil)
        player = nil
        if let playbackJobID {
            service.perform("api/generate-stream/\(playbackJobID)/close", method: "POST") { _, _ in }
            self.playbackJobID = nil
        }
    }

    func discardSystemAudioRecording() {
        stopPreview()
        if let systemAudioRecordingURL {
            try? FileManager.default.removeItem(at: systemAudioRecordingURL)
        }
        systemAudioRecordingURL = nil
        systemAudioTrimPresented = false
        systemAudioStatus = "可录制 Mac 正在播放的声音"
        systemAudioWaveform = []
    }

    func exportSystemAudioSelection() {
        guard let sourceURL = systemAudioRecordingURL else { return }
        let start = max(0, systemAudioTrimStart)
        let duration = max(0, systemAudioTrimEnd - start)
        guard duration >= 0.25 else {
            show(NativeStudioError.server("请至少保留 0.25 秒音频。"))
            return
        }
        let safeName = systemAudioName
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .replacingOccurrences(of: "/", with: "-")
        let outputURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("qwen-system-export-\(UUID().uuidString).m4a")
        let asset = AVURLAsset(url: sourceURL)
        guard let exporter = AVAssetExportSession(
            asset: asset,
            presetName: AVAssetExportPresetAppleM4A
        ) else {
            show(NativeStudioError.server("无法创建音频裁剪任务。"))
            return
        }
        exporter.outputURL = outputURL
        exporter.outputFileType = .m4a
        exporter.timeRange = CMTimeRange(
            start: CMTime(seconds: start, preferredTimescale: 600),
            duration: CMTime(seconds: duration, preferredTimescale: 600)
        )
        isExportingSystemAudio = true
        exporter.exportAsynchronously { [weak self] in
            DispatchQueue.main.async {
                guard let self else { return }
                self.isExportingSystemAudio = false
                guard exporter.status == .completed else {
                    try? FileManager.default.removeItem(at: outputURL)
                    self.show(
                        exporter.error ?? NativeStudioError.server("裁剪后的音频导出失败。")
                    )
                    return
                }
                self.stopPreview()
                if let source = self.systemAudioRecordingURL {
                    try? FileManager.default.removeItem(at: source)
                }
                self.systemAudioRecordingURL = nil
                self.systemAudioTrimPresented = false
                self.systemAudioStatus = "已导出并加入参考音频"
                self.importReference(
                    outputURL,
                    displayName: safeName.isEmpty ? "系统录音" : safeName
                )
            }
        }
    }

    private static let recordingDateFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd HH-mm-ss"
        return formatter
    }()

    func playReference() {
        guard !referenceAudioPath.isEmpty else {
            show(NativeStudioError.missingReference)
            return
        }
        play(URL(fileURLWithPath: referenceAudioPath))
    }

    func playOutput() {
        guard let outputAudioURL else { return }
        play(outputAudioURL)
    }

    private func play(_ url: URL, jobID: String? = nil) {
        stopPreview()
        let item = AVPlayerItem(url: url)
        player = AVPlayer(playerItem: item)
        playbackJobID = jobID
        let finishPlayback: () -> Void = { [weak self] in
            guard let self else { return }
            if let jobID = self.playbackJobID {
                self.service.perform("api/generate-stream/\(jobID)/close", method: "POST") { _, _ in }
            }
            self.playbackJobID = nil
            if let observer = self.playbackEndObserver {
                NotificationCenter.default.removeObserver(observer)
                self.playbackEndObserver = nil
            }
            if let observer = self.playbackFailureObserver {
                NotificationCenter.default.removeObserver(observer)
                self.playbackFailureObserver = nil
            }
        }
        playbackEndObserver = NotificationCenter.default.addObserver(
            forName: .AVPlayerItemDidPlayToEndTime,
            object: item,
            queue: .main
        ) { _ in finishPlayback() }
        playbackFailureObserver = NotificationCenter.default.addObserver(
            forName: .AVPlayerItemFailedToPlayToEndTime,
            object: item,
            queue: .main
        ) { _ in finishPlayback() }
        player?.play()
    }

    func generate() {
        guard !isGenerating, !isForceStopping else { return }
        guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            show(NativeStudioError.emptyText)
            return
        }
        guard !referenceAudioPath.isEmpty else {
            show(NativeStudioError.missingReference)
            return
        }
        if cloneMode == "icl" && referenceText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            show(NativeStudioError.server("ICL 模式需要填写与参考音频逐字对应的文字。"))
            return
        }
        stopPolling()
        pcmStreamPlayer?.stop()
        pcmStreamPlayer = nil
        removeTemporaryOutput()
        outputAudioURL = nil
        isGenerating = true
        generatedProgress = 0
        generationSummary = "正在提交生成任务…"
        errorMessage = ""
        actualGenerationSeed = nil
        submittedSeedSetting = nil
        currentJobDisplayID = ""
        generationFinished = false
        streamPlaybackFinished = false
        resultDownloaded = false
        let requestEpoch = generationEpoch
        let requestedSeed = seed
        let fields: [String: String] = [
            "mode": "voice_clone",
            "language": "Chinese",
            "text": text,
            "prompt_text": "",
            "max_new_tokens": String(maxNewTokens),
            "codec_chunk_frames": String(chunkFrames),
            "seed": String(seed),
            "temperature": String(temperature),
            "top_p": String(topP),
            "top_k": String(topK),
            "repetition_penalty": String(repetitionPenalty),
            "model_profile": modelProfile,
            "voice_name": referenceName,
            "qwen_clone_mode": cloneMode,
            "qwen_reference_text": referenceText,
            "qwen_non_streaming_mode": nonStreamingInput ? "1" : "0",
            "qwen_append_silence": appendSilence ? "1" : "0",
            "qwen_min_new_tokens": String(minNewTokens),
            "streaming_generation": streamingGeneration ? "1" : "0",
            "example_audio_path": referenceAudioPath,
            "use_service_settings": "0",
        ]
        let multipart = multipartBody(fields: fields, file: nil)
        service.perform(
            "api/generate-stream/start",
            method: "POST",
            body: multipart.data,
            contentType: multipart.contentType,
            timeout: 30
        ) { [weak self] data, response in
            guard let self else { return }
            do {
                let payload = try self.jsonObject(data, response: response)
                guard let jobID = payload["job_id"] as? String else {
                    throw NativeStudioError.invalidResponse
                }
                let resolvedSeed = self.number(payload["seed"]).intValue
                let sampleRate = self.number(payload["sample_rate"]).doubleValue
                let channels = self.number(payload["channels"]).intValue
                DispatchQueue.main.async {
                    guard self.generationEpoch == requestEpoch, !self.isForceStopping else { return }
                    self.currentJobID = jobID
                    self.stoppedJobIDs.remove(jobID)
                    self.currentJobDisplayID = String(jobID.prefix(8))
                    self.submittedSeedSetting = requestedSeed
                    self.actualGenerationSeed = resolvedSeed
                    self.generationSummary = "任务已排队"
                    if self.streamingGeneration {
                        self.startPCMStream(
                            jobID,
                            sampleRate: max(8_000, sampleRate),
                            channels: max(1, channels)
                        )
                    }
                    self.startPolling(jobID)
                }
            } catch {
                DispatchQueue.main.async {
                    guard self.generationEpoch == requestEpoch, !self.isForceStopping else { return }
                    self.isGenerating = false
                    self.show(error)
                }
            }
        }
    }

    func stopGeneration() {
        guard let currentJobID else { return }
        stoppedJobIDs.insert(currentJobID)
        service.perform("api/generate-stream/\(currentJobID)/close", method: "POST") { _, _ in }
        pcmStreamPlayer?.stop()
        pcmStreamPlayer = nil
        stopPolling()
        isGenerating = false
        self.currentJobID = nil
        generationSummary = "已停止当前生成"
    }

    func forceStopAllAudioAndComputation(
        completion: @escaping (Result<Void, Error>) -> Void
    ) {
        guard !isForceStopping else { return }
        isForceStopping = true

        // Stop native playback and invalidate every pending local generation callback first.
        // The backend request is deliberately global, so no per-job close request is sent here.
        generationEpoch &+= 1
        if let currentJobID {
            stoppedJobIDs.insert(currentJobID)
        }
        pcmStreamPlayer?.stop()
        pcmStreamPlayer = nil
        stopPreview()
        stopPolling()
        currentJobID = nil
        currentJobDisplayID = ""
        isGenerating = false
        generatedProgress = 0
        generationFinished = false
        streamPlaybackFinished = false
        resultDownloaded = false
        errorMessage = ""
        generationSummary = "已停止本地音频，正在停止所有运算…"

        service.perform("api/service/stop-all", method: "POST") { [weak self] data, response in
            guard let self else { return }
            do {
                _ = try self.jsonObject(data, response: response)
                DispatchQueue.main.async {
                    self.isForceStopping = false
                    self.generationSummary = "已强制停止所有音频与运算"
                    completion(.success(()))
                }
            } catch {
                DispatchQueue.main.async {
                    self.isForceStopping = false
                    self.generationSummary = "本地音频已停止，但后台停止请求失败"
                    self.errorMessage = error.localizedDescription
                    completion(.failure(error))
                }
            }
        }
    }

    private func startPCMStream(_ jobID: String, sampleRate: Double, channels: Int) {
        do {
            let request = service.authorizedRequest(
                "api/generate-stream/\(jobID)/audio",
                timeout: 600
            )
            let stream = try NativePCMStreamPlayer(
                request: request,
                sampleRate: sampleRate,
                channels: AVAudioChannelCount(channels)
            )
            stream.onFirstAudio = { [weak self] in
                guard let self, self.currentJobID == jobID else { return }
                self.generationSummary = "正在边生成边播放 PCM 音频…"
            }
            stream.onComplete = { [weak self] in
                guard let self, self.currentJobID == jobID else { return }
                self.streamPlaybackFinished = true
                self.finishStreamingGenerationIfReady()
            }
            stream.onError = { [weak self] error in
                guard let self, self.currentJobID == jobID,
                      !self.stoppedJobIDs.contains(jobID)
                else { return }
                self.service.perform("api/generate-stream/\(jobID)/close", method: "POST") { _, _ in }
                self.stopPolling()
                self.isGenerating = false
                self.currentJobID = nil
                self.show(error)
            }
            pcmStreamPlayer = stream
            stream.start()
        } catch {
            service.perform("api/generate-stream/\(jobID)/close", method: "POST") { _, _ in }
            stopPolling()
            isGenerating = false
            currentJobID = nil
            show(error)
        }
    }

    private func startPolling(_ jobID: String) {
        pollingTimer = Timer.scheduledTimer(withTimeInterval: 0.5, repeats: true) { [weak self] _ in
            self?.poll(jobID)
        }
        pollingTimer?.fire()
    }

    private func poll(_ jobID: String) {
        service.perform("api/generate-stream/\(jobID)/status") { [weak self] data, response in
            guard let self else { return }
            do {
                let payload = try self.jsonObject(data, response: response)
                let state = payload["state"] as? String ?? "unknown"
                let generated = self.number(payload["generated_frames"]).doubleValue
                let maximum = max(1, self.number(payload["max_new_tokens"]).doubleValue)
                let audioSeconds = self.number(payload["emitted_audio_seconds"]).doubleValue
                let resolvedSeed = payload["seed"] == nil ? nil : self.number(payload["seed"]).intValue
                DispatchQueue.main.async {
                    guard self.currentJobID == jobID else { return }
                    if let resolvedSeed {
                        self.actualGenerationSeed = resolvedSeed
                    }
                    self.generatedProgress = min(1, generated / maximum)
                    self.generationSummary = "\(self.nativeStateLabel(state)) · \(Int(generated))/\(Int(maximum)) 帧 · \(String(format: "%.1f", audioSeconds)) 秒音频"
                    if state == "finished" {
                        self.stopPolling()
                        self.generationFinished = true
                        self.downloadResult(jobID, autoPlay: !self.streamingGeneration)
                    } else if state == "error" || state == "closed" {
                        self.stopPolling()
                        self.isGenerating = false
                        self.pcmStreamPlayer?.stop()
                        self.pcmStreamPlayer = nil
                        if state == "closed" && self.stoppedJobIDs.contains(jobID) {
                            self.generationSummary = "已停止当前生成"
                        } else {
                            self.errorMessage = payload["error"] as? String ?? "生成任务失败。"
                        }
                        self.currentJobID = nil
                    }
                }
            } catch {
                self.show(error)
            }
        }
    }

    private func downloadResult(_ jobID: String, autoPlay: Bool) {
        generationSummary = "正在读取生成结果…"
        let resultPath = "api/generate-stream/\(jobID)/result-audio" + (autoPlay ? "?playback=1" : "")
        service.perform(resultPath, timeout: 120) { [weak self] data, response in
            guard let self else { return }
            guard response?.statusCode == 200, let data else {
                DispatchQueue.main.async {
                    guard self.currentJobID == jobID else { return }
                    self.isGenerating = false
                    self.show(NativeStudioError.server("生成完成，但无法读取输出音频。"))
                }
                return
            }
            let url = FileManager.default.temporaryDirectory
                .appendingPathComponent("qwen-tts-native-\(jobID).wav")
            do {
                try data.write(to: url, options: .atomic)
                DispatchQueue.main.async {
                    guard self.currentJobID == jobID else {
                        try? FileManager.default.removeItem(at: url)
                        return
                    }
                    self.outputAudioURL = url
                    self.resultDownloaded = true
                    self.generatedProgress = 1
                    if autoPlay {
                        self.isGenerating = false
                        self.generationSummary = "生成完成，可以试听"
                        self.play(url, jobID: jobID)
                        self.currentJobID = nil
                    } else {
                        self.generationSummary = self.streamPlaybackFinished
                            ? "流式播放完成，可以再次试听"
                            : "生成完成，正在播放剩余音频…"
                        self.finishStreamingGenerationIfReady()
                    }
                }
            } catch {
                DispatchQueue.main.async {
                    guard self.currentJobID == jobID else { return }
                    self.isGenerating = false
                    self.show(error)
                }
            }
        }
    }

    private func finishStreamingGenerationIfReady() {
        guard generationFinished, streamPlaybackFinished, resultDownloaded else { return }
        let finishedJobID = currentJobID
        isGenerating = false
        currentJobID = nil
        pcmStreamPlayer = nil
        generatedProgress = 1
        generationSummary = "流式播放完成，可以再次试听"
        if let finishedJobID {
            service.perform("api/generate-stream/\(finishedJobID)/close", method: "POST") { _, _ in }
        }
    }

    private func removeTemporaryOutput() {
        guard let outputAudioURL,
              outputAudioURL.deletingLastPathComponent() == FileManager.default.temporaryDirectory,
              outputAudioURL.lastPathComponent.hasPrefix("qwen-tts-native-")
        else { return }
        try? FileManager.default.removeItem(at: outputAudioURL)
    }

    func applySelectedPreset() {
        guard let preset = presets.first(where: { $0.id == selectedPresetID }) else { return }
        applyPreset(preset)
    }

    func applyPreset(_ preset: NativePreset) {
        let settings = preset.settings
        modelProfile = string(settings["model_profile"], fallback: "qwen_0_6b")
        cloneMode = string(settings["qwen_clone_mode"], fallback: "xvec")
        referenceText = string(settings["qwen_reference_text"])
        temperature = double(settings["qwen_temperature"], fallback: 0.9)
        topP = double(settings["qwen_top_p"], fallback: 1.0)
        topK = integer(settings["qwen_top_k"], fallback: 50)
        repetitionPenalty = double(settings["qwen_repetition_penalty"], fallback: 1.05)
        maxNewTokens = integer(settings["qwen_max_new_tokens"], fallback: 2048)
        chunkFrames = integer(settings["qwen_chunk_size"], fallback: 8)
        minNewTokens = integer(settings["qwen_min_new_tokens"], fallback: 2)
        seed = integer(settings["qwen_seed"], fallback: 1234)
        appendSilence = boolean(settings["qwen_append_silence"], fallback: true)
        nonStreamingInput = boolean(settings["qwen_non_streaming_mode"], fallback: false)
        streamingGeneration = boolean(settings["qwen_streaming_generation"], fallback: true)
        let path = string(settings["reference_audio_path"])
        referenceAudioPath = path
        if let voice = voices.first(where: { $0.audioPath == path }) {
            selectedVoicePath = path
            referenceName = voice.name
            if cloneMode == "icl",
               referenceText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                referenceText = voice.transcript
            }
        } else {
            selectedVoicePath = ""
            referenceName = string(settings["voice_name"], fallback: preset.name)
        }
        selectedPresetID = preset.id
        presetName = preset.name
        errorMessage = ""
        activatePresetForService(preset.id)
    }

    private func activatePresetForService(_ presetID: String) {
        guard let data = try? JSONSerialization.data(withJSONObject: ["preset_id": presetID]) else { return }
        service.perform(
            "api/service-settings/active-preset",
            method: "PUT",
            body: data,
            contentType: "application/json"
        ) { [weak self] data, response in
            guard let self else { return }
            do {
                let payload = try self.jsonObject(data, response: response)
                let name = payload["name"] as? String ?? "当前预设"
                DispatchQueue.main.async {
                    self.serviceSettingsStatus = "服务已应用：\(name)"
                }
            } catch {
                self.show(error)
            }
        }
    }

    func applyCurrentAsServiceSettings() {
        guard !referenceAudioPath.isEmpty else {
            show(NativeStudioError.missingReference)
            return
        }
        isApplyingServiceSettings = true
        serviceSettingsStatus = "正在应用到后台服务…"
        let settings: [String: Any] = [
            "model_profile": modelProfile,
            "voice_name": referenceName,
            "reference_audio_path": referenceAudioPath,
            "qwen_clone_mode": cloneMode,
            "qwen_reference_text": referenceText,
            "qwen_temperature": temperature,
            "qwen_top_p": topP,
            "qwen_top_k": topK,
            "qwen_repetition_penalty": repetitionPenalty,
            "qwen_max_new_tokens": maxNewTokens,
            "qwen_chunk_size": chunkFrames,
            "qwen_min_new_tokens": minNewTokens,
            "qwen_seed": seed,
            "qwen_append_silence": appendSilence,
            "qwen_non_streaming_mode": nonStreamingInput,
            "qwen_streaming_generation": streamingGeneration,
        ]
        let payload: [String: Any] = [
            "name": "工作台 · \(referenceName)",
            "settings": settings,
        ]
        guard let data = try? JSONSerialization.data(withJSONObject: payload) else {
            isApplyingServiceSettings = false
            show(NativeStudioError.invalidResponse)
            return
        }
        service.perform(
            "api/service-settings",
            method: "PUT",
            body: data,
            contentType: "application/json"
        ) { [weak self] data, response in
            guard let self else { return }
            do {
                let result = try self.jsonObject(data, response: response)
                let name = result["name"] as? String ?? "音频工作台设置"
                DispatchQueue.main.async {
                    self.isApplyingServiceSettings = false
                    self.selectedPresetID = ""
                    self.serviceSettingsStatus = "已应用：\(name)；后续调用将使用此配置"
                }
            } catch {
                DispatchQueue.main.async { self.isApplyingServiceSettings = false }
                self.show(error)
            }
        }
    }

    func savePreset(update: Bool) {
        let name = presetName.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !name.isEmpty else {
            show(NativeStudioError.server("请填写预设名称。"))
            return
        }
        guard !referenceAudioPath.isEmpty else {
            show(NativeStudioError.missingReference)
            return
        }
        isSavingPreset = true
        let settings: [String: Any] = [
            "model_profile": modelProfile,
            "voice_name": referenceName,
            "reference_audio_path": referenceAudioPath,
            "qwen_clone_mode": cloneMode,
            "qwen_reference_text": referenceText,
            "qwen_temperature": temperature,
            "qwen_top_p": topP,
            "qwen_top_k": topK,
            "qwen_repetition_penalty": repetitionPenalty,
            "qwen_max_new_tokens": maxNewTokens,
            "qwen_chunk_size": chunkFrames,
            "qwen_min_new_tokens": minNewTokens,
            "qwen_seed": seed,
            "qwen_append_silence": appendSilence,
            "qwen_non_streaming_mode": nonStreamingInput,
            "qwen_streaming_generation": streamingGeneration,
        ]
        let payload: [String: Any] = ["name": name, "settings": settings]
        guard let data = try? JSONSerialization.data(withJSONObject: payload) else {
            show(NativeStudioError.invalidResponse)
            isSavingPreset = false
            return
        }
        let canUpdate = update && !selectedPresetID.isEmpty
        let path = canUpdate ? "api/presets/\(selectedPresetID)" : "api/presets"
        service.perform(
            path,
            method: canUpdate ? "PUT" : "POST",
            body: data,
            contentType: "application/json"
        ) { [weak self] data, response in
            guard let self else { return }
            do {
                let payload = try self.jsonObject(data, response: response)
                guard let preset = NativePreset(payload) else { throw NativeStudioError.invalidResponse }
                DispatchQueue.main.async {
                    self.isSavingPreset = false
                    self.applyPreset(preset)
                    self.refreshPresets(selecting: preset.id)
                    self.generationSummary = "预设已保存：\(preset.name)"
                }
            } catch {
                DispatchQueue.main.async { self.isSavingPreset = false }
                self.show(error)
            }
        }
    }

    func deleteSelectedPreset() {
        guard !selectedPresetID.isEmpty else { return }
        let deletingID = selectedPresetID
        service.perform("api/presets/\(deletingID)", method: "DELETE") { [weak self] data, response in
            guard let self else { return }
            do {
                _ = try self.jsonObject(data, response: response)
                DispatchQueue.main.async {
                    self.selectedPresetID = ""
                    self.presetName = ""
                    self.refreshPresets()
                }
            } catch {
                self.show(error)
            }
        }
    }

    private func stopPolling() {
        pollingTimer?.invalidate()
        pollingTimer = nil
    }

    private func jsonObject(_ data: Data?, response: HTTPURLResponse?) throws -> [String: Any] {
        guard let response, let data else { throw NativeStudioError.invalidResponse }
        if !(200 ... 299).contains(response.statusCode) {
            let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
            let detail = object?["detail"] as? String ?? String(data: data, encoding: .utf8) ?? "HTTP \(response.statusCode)"
            throw NativeStudioError.server(detail)
        }
        guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw NativeStudioError.invalidResponse
        }
        return object
    }

    private func show(_ error: Error) {
        DispatchQueue.main.async {
            self.errorMessage = error.localizedDescription
        }
    }

    private func number(_ value: Any?) -> NSNumber {
        if let value = value as? NSNumber { return value }
        if let value = value as? String, let number = Double(value) { return NSNumber(value: number) }
        return 0
    }

    private func string(_ value: Any?, fallback: String = "") -> String {
        if let value = value as? String { return value }
        if let value { return String(describing: value) }
        return fallback
    }

    private func double(_ value: Any?, fallback: Double) -> Double {
        let parsed = number(value).doubleValue
        return value == nil ? fallback : parsed
    }

    private func integer(_ value: Any?, fallback: Int) -> Int {
        let parsed = number(value).intValue
        return value == nil ? fallback : parsed
    }

    private func boolean(_ value: Any?, fallback: Bool) -> Bool {
        if let value = value as? Bool { return value }
        if let value = value as? NSNumber { return value.boolValue }
        if let value = value as? String {
            return ["true", "1", "yes"].contains(value.lowercased())
        }
        return fallback
    }

    private func nativeStateLabel(_ state: String) -> String {
        [
            "queued": "排队",
            "loading_runtime": "加载模型",
            "running": "生成中",
            "finished": "已完成",
            "closed": "已停止",
            "error": "错误",
        ][state] ?? state
    }

    private func multipartBody(
        fields: [String: String],
        file: (field: String, filename: String, mime: String, data: Data)?
    ) -> (data: Data, contentType: String) {
        let boundary = "QwenTTSNative-\(UUID().uuidString)"
        var body = Data()
        for (key, value) in fields.sorted(by: { $0.key < $1.key }) {
            body.appendUTF8("--\(boundary)\r\n")
            body.appendUTF8("Content-Disposition: form-data; name=\"\(key)\"\r\n\r\n")
            body.appendUTF8(value)
            body.appendUTF8("\r\n")
        }
        if let file {
            body.appendUTF8("--\(boundary)\r\n")
            body.appendUTF8("Content-Disposition: form-data; name=\"\(file.field)\"; filename=\"\(file.filename)\"\r\n")
            body.appendUTF8("Content-Type: \(file.mime)\r\n\r\n")
            body.append(file.data)
            body.appendUTF8("\r\n")
        }
        body.appendUTF8("--\(boundary)--\r\n")
        return (body, "multipart/form-data; boundary=\(boundary)")
    }
}

private extension Data {
    mutating func appendUTF8(_ text: String) {
        append(text.data(using: .utf8) ?? Data())
    }
}

private struct AdvancedSettingsView: View {
    @ObservedObject var model: NativeStudioViewModel

    var body: some View {
        VStack(spacing: 0) {
            StudioModalHeader(
                title: "高级生成设置",
                subtitle: "控制语音稳定性、表现力与流式延迟",
                close: { model.advancedSettingsPresented = false }
            )
            ScrollView {
                VStack(alignment: .leading, spacing: StudioTokens.space4) {
                    GroupBox("采样参数") {
                        VStack(spacing: 15) {
                            slider("Temperature", value: $model.temperature, range: 0.1 ... 2.0)
                            slider("Top P", value: $model.topP, range: 0.1 ... 1.0)
                            slider("Repetition penalty", value: $model.repetitionPenalty, range: 0.8 ... 2.0)
                            integerField("Top K", value: $model.topK, range: 1 ... 200)
                        }
                    }
                    GroupBox("生成与流式") {
                        VStack(alignment: .leading, spacing: 13) {
                            Toggle("启用 PCM 流式输出", isOn: $model.streamingGeneration)
                            Text("开启后原生播放器会边生成边播放 PCM；关闭后等待最终 WAV 再试听。")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                            HStack(spacing: StudioTokens.space4) {
                                integerField("Seed（-1 为随机）", value: $model.seed, range: -1 ... 999_999)
                                integerField("最大生成帧", value: $model.maxNewTokens, range: 24 ... 2048)
                            }
                            HStack(spacing: StudioTokens.space4) {
                                integerField("流式块帧数", value: $model.chunkFrames, range: 1 ... 24)
                                    .disabled(!model.streamingGeneration)
                                integerField("最小生成帧", value: $model.minNewTokens, range: 2 ... 256)
                            }
                        }
                    }
                    GroupBox("输入与参考音频") {
                        VStack(alignment: .leading, spacing: 11) {
                            Toggle("参考音频末尾自动补静音", isOn: $model.appendSilence)
                            Toggle("一次性输入完整文本", isOn: $model.nonStreamingInput)
                            Text("小流式块会更快听到首段声音；较大的流式块通常更稳定。")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                    }
                }
                .padding(StudioTokens.space5)
            }
            StudioFooterBar {
                Button("恢复推荐值") {
                    model.temperature = 0.9
                    model.topP = 1.0
                    model.topK = 50
                    model.repetitionPenalty = 1.05
                    model.maxNewTokens = 2048
                    model.chunkFrames = 8
                    model.minNewTokens = 2
                    model.seed = 1234
                    model.streamingGeneration = true
                }
                .buttonStyle(StudioSecondaryButtonStyle())
                .frame(width: 138)
                Spacer()
                Button("完成") { model.advancedSettingsPresented = false }
                    .buttonStyle(StudioTintedButtonStyle())
                    .frame(width: 108)
                    .keyboardShortcut(.defaultAction)
            }
        }
        .frame(width: 680, height: 650)
        .background(StudioBackground())
        .groupBoxStyle(StudioGlassGroupBoxStyle())
    }

    private func slider(
        _ title: String,
        value: Binding<Double>,
        range: ClosedRange<Double>
    ) -> some View {
        HStack(spacing: StudioTokens.space3) {
            Text(title).frame(width: 155, alignment: .leading)
            Slider(value: value, in: range)
            Text(String(format: "%.2f", value.wrappedValue))
                .monospacedDigit()
                .frame(width: 55, alignment: .trailing)
        }
    }

    private func integerField(
        _ title: String,
        value: Binding<Int>,
        range: ClosedRange<Int>
    ) -> some View {
        let clamped = Binding(
            get: { value.wrappedValue },
            set: { value.wrappedValue = min(range.upperBound, max(range.lowerBound, $0)) }
        )
        return VStack(alignment: .leading, spacing: StudioTokens.space1) {
            Text(title).font(.caption).foregroundStyle(.secondary)
            TextField(title, value: clamped, format: .number)
                .textFieldStyle(.roundedBorder)
        }
        .frame(maxWidth: .infinity)
    }
}

private struct ExternalAPISettingsView: View {
    @ObservedObject var model: NativeStudioViewModel

    var body: some View {
        VStack(spacing: 0) {
            StudioModalHeader(
                title: "对外接口设置",
                subtitle: "供本机、局域网设备与自动化工具调用 TTS / STT",
                close: { model.externalAPISettingsPresented = false }
            )

            ScrollView {
                VStack(alignment: .leading, spacing: StudioTokens.space4) {
                    GroupBox("访问范围") {
                        VStack(alignment: .leading, spacing: StudioTokens.space3) {
                            Toggle("允许局域网设备访问", isOn: $model.externalAccessEnabled)
                            Text(
                                model.externalAccessEnabled
                                    ? "主服务将监听所有网络接口；内部 STT 服务仍只监听本机。"
                                    : "仅允许这台 Mac 上的应用访问。"
                            )
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            HStack(spacing: StudioTokens.space3) {
                                Text("端口")
                                    .frame(width: 80, alignment: .leading)
                                TextField("7861", value: $model.externalPort, format: .number)
                                    .textFieldStyle(.roundedBorder)
                                    .frame(width: 130)
                                    .frame(height: StudioTokens.compactControlHeight)
                            }
                        }
                    }

                    GroupBox("认证") {
                        VStack(alignment: .leading, spacing: StudioTokens.space2) {
                            SecureField("访问密码 / API Key", text: $model.externalPassword)
                                .textFieldStyle(.roundedBorder)
                                .frame(height: StudioTokens.compactControlHeight)
                            Text("客户端可使用 Authorization: Bearer 或 X-API-Key 请求头。")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                            if model.externalPassword.isEmpty
                                || (model.externalAccessEnabled && model.externalPassword.count < 4)
                                || model.externalPassword == "change-me" {
                                Label("局域网模式必须使用至少 4 位且不是 change-me 的密码。", systemImage: "exclamationmark.triangle.fill")
                                    .font(.caption)
                                    .foregroundStyle(.orange)
                            }
                        }
                    }

                    GroupBox("浏览器跨域") {
                        VStack(alignment: .leading, spacing: StudioTokens.space2) {
                            TextField(
                                "例如 https://reader.example；多个来源用逗号分隔",
                                text: $model.externalCORSOrigins
                            )
                            .textFieldStyle(.roundedBorder)
                            .frame(height: StudioTokens.compactControlHeight)
                            Text("留空仅允许同源网页；原生应用和命令行调用不受影响。输入 * 会允许任意网页来源。")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                    }

                    GroupBox("调用地址") {
                        VStack(spacing: StudioTokens.space3) {
                            endpointRow("本机", value: model.localAPIEndpoint)
                            endpointRow("局域网", value: model.networkAPIEndpoint)
                            HStack {
                                Text("STT")
                                    .font(.caption.weight(.semibold))
                                    .frame(width: 60, alignment: .leading)
                                Text("/v1/audio/transcriptions")
                                    .font(.system(.body, design: .monospaced))
                                    .textSelection(.enabled)
                                Spacer()
                            }
                            HStack {
                                Text("TTS")
                                    .font(.caption.weight(.semibold))
                                    .frame(width: 60, alignment: .leading)
                                Text("/api/generate-stream/start")
                                    .font(.system(.body, design: .monospaced))
                                    .textSelection(.enabled)
                                Spacer()
                            }
                        }
                    }

                    if !model.externalSettingsStatus.isEmpty {
                        Text(model.externalSettingsStatus)
                            .font(.callout)
                            .foregroundStyle(
                                model.externalSettingsStatus.contains("已生效") ? Color.green : Color.secondary
                            )
                            .textSelection(.enabled)
                    }
                }
                .padding(StudioTokens.space5)
            }

            StudioFooterBar {
                Button("取消") { model.externalAPISettingsPresented = false }
                    .buttonStyle(StudioSecondaryButtonStyle())
                    .frame(width: 100)
                Spacer()
                Button {
                    model.applyExternalSettings()
                } label: {
                    Label(
                        model.isApplyingExternalSettings ? "正在重启服务…" : "保存并重启服务",
                        systemImage: "arrow.triangle.2.circlepath"
                    )
                }
                .buttonStyle(StudioTintedButtonStyle())
                .frame(width: 178)
                .disabled(model.isApplyingExternalSettings)
            }
        }
        .frame(width: 650, height: 660)
        .background(StudioBackground())
        .groupBoxStyle(StudioGlassGroupBoxStyle())
    }

    private func endpointRow(_ title: String, value: String) -> some View {
        HStack {
            Text(title)
                .font(.caption.weight(.semibold))
                .frame(width: 60, alignment: .leading)
            Text(value)
                .font(.system(size: 12, design: .monospaced))
                .textSelection(.enabled)
            Spacer()
            Button("复制") { model.copyToPasteboard(value) }
                .buttonStyle(StudioToolbarButtonStyle())
        }
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
                    Text("裁剪系统录音")
                        .font(.system(size: 22, weight: .semibold, design: .rounded))
                    Text("只保留声音稳定、背景干净的一段作为克隆参考。")
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

            HStack {
                Button("试听选区") { model.previewSystemAudioSelection() }
                    .buttonStyle(StudioSecondaryButtonStyle())
                Button("停止试听") { model.stopPreview() }
                    .buttonStyle(StudioSecondaryButtonStyle())
                Spacer()
                Button("丢弃", role: .destructive) { model.discardSystemAudioRecording() }
                    .buttonStyle(StudioSecondaryButtonStyle(destructive: true))
                    .disabled(model.isExportingSystemAudio)
                Button {
                    model.exportSystemAudioSelection()
                } label: {
                    Label(
                        model.isExportingSystemAudio ? "正在导出…" : "导出并加入参考音频",
                        systemImage: "square.and.arrow.down"
                    )
                }
                .buttonStyle(StudioTintedButtonStyle())
                .disabled(model.isExportingSystemAudio)
            }
        }
        .padding(StudioTokens.space5)
        .frame(width: 720, height: 500)
        .background(StudioBackground())
        .interactiveDismissDisabled(model.isExportingSystemAudio)
    }
}

struct ReferenceAudioLibraryView: View {
    @ObservedObject var model: NativeStudioViewModel

    private var visibleReferences: [NativeReferenceAudio] {
        model.referenceLibrary.filter { !$0.hidden }
    }

    private var hiddenReferences: [NativeReferenceAudio] {
        model.referenceLibrary.filter(\.hidden)
    }

    private var displayedReferences: [NativeReferenceAudio] {
        model.referenceLibraryTab == "hidden" ? hiddenReferences : visibleReferences
    }

    var body: some View {
        VStack(alignment: .leading, spacing: StudioTokens.space4) {
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    Text("参考音频管理")
                        .font(.system(size: 22, weight: .semibold, design: .rounded))
                    Text("隐藏的音频不会出现在音色选择菜单中，但不会影响已有预设。")
                        .foregroundStyle(.secondary)
                }
                Spacer()
                Button("完成") { model.referenceLibraryPresented = false }
                    .buttonStyle(StudioTintedButtonStyle())
            }
            Divider()
            Picker("参考音频分类", selection: $model.referenceLibraryTab) {
                Text("可用音频 · \(visibleReferences.count)").tag("visible")
                Text("已隐藏 · \(hiddenReferences.count)").tag("hidden")
            }
            .pickerStyle(.segmented)
            .labelsHidden()
            .frame(maxWidth: 360)
            if !model.referenceLibraryStatus.isEmpty {
                Text(model.referenceLibraryStatus)
                    .font(.callout)
                    .foregroundStyle(
                        model.referenceLibraryStatus.contains("无法")
                            || model.referenceLibraryStatus.contains("不能")
                            ? Color.red
                            : Color.secondary
                    )
                    .padding(.horizontal, StudioTokens.space2)
            }
            ScrollView {
                LazyVStack(spacing: StudioTokens.space2) {
                    if displayedReferences.isEmpty {
                        VStack(spacing: StudioTokens.space3) {
                            Image(
                                systemName: model.referenceLibraryTab == "hidden"
                                    ? "eye.slash"
                                    : "waveform"
                            )
                            .font(.system(size: 30))
                            .foregroundStyle(.secondary)
                            Text(
                                model.referenceLibraryTab == "hidden"
                                    ? "没有隐藏的参考音频"
                                    : "没有可用的参考音频"
                            )
                            .font(.headline)
                            Text(
                                model.referenceLibraryTab == "hidden"
                                    ? "在“可用音频”中点击隐藏后，会集中显示在这里。"
                                    : "可以从工作台导入、录制或恢复隐藏的参考音频。"
                            )
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        }
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 80)
                    }
                    ForEach(displayedReferences) { reference in
                        HStack(spacing: StudioTokens.space3) {
                            Image(systemName: reference.kind == "builtin" ? "waveform.badge.plus" : "waveform")
                                .foregroundStyle(reference.hidden ? Color.secondary : Color.accentColor)
                                .frame(width: 24)
                            VStack(alignment: .leading, spacing: 3) {
                                HStack(spacing: 6) {
                                    Text(reference.name)
                                        .font(.headline)
                                    Text(reference.kind == "builtin" ? "内置" : "用户")
                                        .font(.caption2.weight(.semibold))
                                        .padding(.horizontal, 6)
                                        .padding(.vertical, 2)
                                        .background(.secondary.opacity(0.14), in: Capsule())
                                    if reference.hidden {
                                        Text("已隐藏")
                                            .font(.caption2)
                                            .foregroundStyle(.secondary)
                                    }
                                }
                                if reference.inUse {
                                    Text("使用中 · \(reference.usages.joined(separator: "、"))")
                                        .font(.caption)
                                        .foregroundStyle(.secondary)
                                }
                            }
                            Spacer()
                            HStack(spacing: StudioTokens.space2) {
                                Button {
                                    model.previewReference(reference)
                                } label: {
                                    Label("试听", systemImage: "play.fill")
                                }
                                .buttonStyle(StudioCompactActionButtonStyle())
                                Button {
                                    model.useReference(reference)
                                    model.referenceLibraryPresented = false
                                } label: {
                                    Label("使用", systemImage: "checkmark.circle")
                                }
                                .buttonStyle(StudioCompactActionButtonStyle())
                                Button {
                                    model.setReferenceHidden(reference, hidden: !reference.hidden)
                                } label: {
                                    Label(
                                        reference.hidden ? "显示" : "隐藏",
                                        systemImage: reference.hidden ? "eye" : "eye.slash"
                                    )
                                }
                                .buttonStyle(StudioCompactActionButtonStyle())
                                if reference.kind == "custom" {
                                    Button(role: .destructive) {
                                        model.referenceLibraryStatus = ""
                                        model.pendingReferenceDeletion = reference
                                    } label: {
                                        Label("删除", systemImage: "trash")
                                    }
                                    .buttonStyle(StudioCompactActionButtonStyle(destructive: true))
                                    .disabled(model.isDeletingReference)
                                } else {
                                    Button { } label: {
                                        Label("内置", systemImage: "lock.fill")
                                    }
                                    .buttonStyle(StudioCompactActionButtonStyle())
                                    .disabled(true)
                                }
                            }
                        }
                        .padding(StudioTokens.space3)
                        .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 12))
                    }
                }
            }
        }
        .padding(StudioTokens.space5)
        .frame(width: 920, height: 640)
        .background(StudioBackground())
        .confirmationDialog(
            "删除“\(model.pendingReferenceDeletion?.name ?? "")”？",
            isPresented: Binding(
                get: { model.pendingReferenceDeletion != nil },
                set: { if !$0 { model.pendingReferenceDeletion = nil } }
            ),
            titleVisibility: .visible
        ) {
            Button(
                model.pendingReferenceDeletion?.inUse == true
                    ? "切换为龙嫱并删除"
                    : "删除音频文件",
                role: .destructive
            ) {
                if let reference = model.pendingReferenceDeletion {
                    model.deleteReference(reference)
                }
                model.pendingReferenceDeletion = nil
            }
            Button("取消", role: .cancel) { model.pendingReferenceDeletion = nil }
        } message: {
            if let reference = model.pendingReferenceDeletion, reference.inUse {
                Text(
                    "该音频仍被\(reference.usages.joined(separator: "、"))使用。"
                        + "继续删除会把这些设置自动切换为默认音色“龙嫱”，音频文件删除后无法恢复。"
                )
            } else {
                Text("音频文件删除后无法恢复。")
            }
        }
    }
}

struct NativeStudioView: View {
    @ObservedObject var model: NativeStudioViewModel

    var body: some View {
        ZStack {
            StudioBackground()

            VStack(spacing: 0) {
                header
                HSplitView {
                    sidebar
                        .frame(minWidth: 320, idealWidth: 340, maxWidth: 370)
                    mainWorkspace
                        .frame(minWidth: 680)
                }
            }
        }
        .frame(minWidth: 1020, minHeight: 720)
        .groupBoxStyle(StudioGlassGroupBoxStyle())
        .controlSize(.regular)
        .sheet(isPresented: $model.advancedSettingsPresented) {
            AdvancedSettingsView(model: model)
        }
        .sheet(isPresented: $model.externalAPISettingsPresented) {
            ExternalAPISettingsView(model: model)
        }
        .sheet(
            isPresented: $model.systemAudioTrimPresented,
            onDismiss: {
                if !model.isExportingSystemAudio {
                    model.discardSystemAudioRecording()
                }
            }
        ) {
            SystemAudioTrimView(model: model)
        }
        .sheet(isPresented: $model.referenceLibraryPresented) {
            ReferenceAudioLibraryView(model: model)
        }
    }

    private var header: some View {
        HStack(spacing: StudioTokens.space3) {
            VStack(alignment: .leading, spacing: StudioTokens.space1) {
                Text("Qwen 语音工作室")
                    .font(.system(size: 20, weight: .semibold, design: .rounded))
                Text("本地生成 · 参考音色 · 流式试听")
                    .font(.system(size: 11))
                    .foregroundStyle(.secondary)
            }
            Spacer()
            Button { model.onTTSServiceToggleRequested?() } label: {
                StudioStatusPill(
                    title: model.ttsEnabled
                        ? (model.ttsReady ? "TTS Metal 已就绪" : "TTS Metal 正在启动")
                        : "TTS Metal 已停止",
                    ready: model.ttsEnabled && model.ttsReady
                )
            }
            .buttonStyle(.plain)
            .help(model.ttsEnabled ? "点击停止 TTS 服务" : "点击启动 TTS 服务")
            .disabled(model.isTogglingTTS)
            Button { model.onSTTServiceToggleRequested?() } label: {
                StudioStatusPill(
                    title: model.sttReady ? "STT Metal 已就绪" : "STT Metal 已停止",
                    ready: model.sttReady
                )
            }
            .buttonStyle(.plain)
            .help(model.sttReady ? "点击停止 STT 服务" : "点击启动 STT 服务")
            .disabled(model.isTogglingSTT)
            Button {
                model.externalSettingsStatus = ""
                model.externalAPISettingsPresented = true
            } label: {
                Label("对外接口", systemImage: "network")
            }
            .buttonStyle(StudioToolbarButtonStyle())
            Picker("模型", selection: $model.modelProfile) {
                Text("0.6B 极速").tag("qwen_0_6b")
                Text("1.7B 高质量").tag("qwen_1_7b")
            }
            .pickerStyle(.segmented)
            .frame(width: 236)
        }
        .padding(.horizontal, StudioTokens.space5)
        .frame(height: 72)
        .background(.ultraThinMaterial)
        .overlay(alignment: .bottom) {
            Rectangle().fill(.white.opacity(0.25)).frame(height: 0.7)
        }
    }

    private var sidebar: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: StudioTokens.space4) {
                GroupBox("语音预设") {
                    VStack(alignment: .leading, spacing: StudioTokens.space2) {
                        Picker("已保存", selection: $model.selectedPresetID) {
                            Text("选择预设…").tag("")
                            ForEach(model.presets) { preset in
                                Text(preset.name).tag(preset.id)
                            }
                        }
                        TextField("预设名称", text: $model.presetName)
                            .textFieldStyle(.roundedBorder)
                            .frame(height: StudioTokens.compactControlHeight)
                        HStack(spacing: StudioTokens.space2) {
                            Button("应用") { model.applySelectedPreset() }
                                .buttonStyle(StudioSecondaryButtonStyle())
                                .disabled(model.selectedPresetID.isEmpty)
                            Button("保存新预设") { model.savePreset(update: false) }
                                .buttonStyle(StudioTintedButtonStyle())
                                .disabled(model.isSavingPreset)
                            Button("更新") { model.savePreset(update: true) }
                                .buttonStyle(StudioSecondaryButtonStyle())
                                .disabled(model.selectedPresetID.isEmpty || model.isSavingPreset)
                        }
                        Button("删除选中预设", role: .destructive) { model.deleteSelectedPreset() }
                            .buttonStyle(
                                StudioSecondaryButtonStyle(
                                    destructive: true,
                                    height: StudioTokens.compactControlHeight
                                )
                            )
                            .disabled(model.selectedPresetID.isEmpty)
                    }
                }

                GroupBox("参考音色") {
                    VStack(alignment: .leading, spacing: StudioTokens.space2) {
                        Picker(
                            "内置音色",
                            selection: Binding(
                                get: { model.selectedVoicePath },
                                set: { model.chooseVoice($0) }
                            )
                        ) {
                            Text("自定义参考音频").tag("")
                            ForEach(model.voices) { voice in
                                Text(voice.name).tag(voice.audioPath)
                            }
                        }
                        Text(model.referenceName)
                            .font(.headline)
                        if let voice = model.voices.first(where: { $0.audioPath == model.selectedVoicePath }),
                           !voice.description.isEmpty {
                            Text(voice.description)
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                        Button {
                            model.applyCurrentAsServiceSettings()
                        } label: {
                            Label(
                                model.isApplyingServiceSettings ? "正在应用…" : "应用当前设置",
                                systemImage: "checkmark.seal.fill"
                            )
                            .frame(maxWidth: .infinity)
                        }
                        .buttonStyle(StudioTintedButtonStyle())
                        .disabled(model.referenceAudioPath.isEmpty || model.isApplyingServiceSettings)
                        Text(model.serviceSettingsStatus)
                            .font(.caption)
                            .foregroundStyle(
                                model.serviceSettingsStatus.contains("已应用")
                                    ? Color.green
                                    : Color.secondary
                            )
                        VStack(spacing: StudioTokens.space2) {
                            HStack(spacing: StudioTokens.space2) {
                                Button {
                                    model.playReference()
                                } label: {
                                    Label("试听参考", systemImage: "play.fill")
                                        .frame(maxWidth: .infinity)
                                }
                                .buttonStyle(StudioSecondaryButtonStyle())
                            Button {
                                model.chooseReferenceFile()
                                } label: {
                                    Label(model.isImportingReference ? "导入中…" : "导入音频", systemImage: "folder")
                                        .frame(maxWidth: .infinity)
                                }
                                .buttonStyle(StudioSecondaryButtonStyle())
                                .disabled(
                                    model.isImportingReference
                                        || model.isRecordingSystemAudio
                                        || model.isPreparingSystemAudioCapture
                                )
                            }
                            Button {
                                model.toggleReferenceRecording()
                            } label: {
                                Label(
                                    model.isRecordingReference ? "停止录音" : "录制参考",
                                    systemImage: model.isRecordingReference ? "stop.circle.fill" : "mic.circle"
                                )
                                .frame(maxWidth: .infinity)
                            }
                            .buttonStyle(
                                StudioTintedButtonStyle(
                                    destructive: model.isRecordingReference
                                )
                            )
                            .disabled(
                                model.isRecordingSystemAudio
                                    || model.isPreparingSystemAudioCapture
                            )
                            Button {
                                model.toggleSystemAudioRecording()
                            } label: {
                                Label(
                                    model.isPreparingSystemAudioCapture
                                        ? "正在请求录音权限…"
                                        : model.isRecordingSystemAudio
                                        ? "停止系统声音录制 · \(Int(model.systemAudioElapsed)) 秒"
                                        : "录制 Mac 播放声音",
                                    systemImage: model.isRecordingSystemAudio
                                        ? "stop.circle.fill"
                                        : "macbook.and.iphone"
                                )
                                .frame(maxWidth: .infinity)
                            }
                            .buttonStyle(
                                StudioTintedButtonStyle(
                                    destructive: model.isRecordingSystemAudio
                                )
                            )
                            .disabled(
                                model.isRecordingReference
                                    || model.isImportingReference
                                    || model.isPreparingSystemAudioCapture
                            )
                            Text(model.systemAudioStatus)
                                .font(.caption)
                                .foregroundStyle(model.isRecordingSystemAudio ? .red : .secondary)
                            Button {
                                model.referenceLibraryStatus = ""
                                model.loadReferenceLibrary()
                                model.referenceLibraryPresented = true
                            } label: {
                                Label("管理参考音频", systemImage: "music.note.list")
                                    .frame(maxWidth: .infinity)
                            }
                            .buttonStyle(StudioSecondaryButtonStyle())
                        }
                    }
                }

                GroupBox("克隆设置") {
                    VStack(alignment: .leading, spacing: StudioTokens.space3) {
                        Picker(
                            "克隆方式",
                            selection: Binding(
                                get: { model.cloneMode },
                                set: { model.setCloneMode($0) }
                            )
                        ) {
                            Text("X-vector").tag("xvec")
                            Text("ICL 高相似度").tag("icl")
                        }
                        .pickerStyle(.segmented)
                        if model.cloneMode == "icl" {
                            Text("参考音频逐字文本")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                            TextEditor(text: $model.referenceText)
                                .font(.body)
                                .frame(minHeight: 110)
                                .padding(StudioTokens.space2)
                                .background(
                                    Color(nsColor: .textBackgroundColor).opacity(0.65),
                                    in: RoundedRectangle(
                                        cornerRadius: StudioTokens.innerRadius,
                                        style: .continuous
                                    )
                                )
                                .overlay {
                                    RoundedRectangle(
                                        cornerRadius: StudioTokens.innerRadius,
                                        style: .continuous
                                    )
                                    .strokeBorder(Color.primary.opacity(0.10), lineWidth: 0.7)
                                }
                            if let voice = model.voices.first(where: {
                                $0.audioPath == model.selectedVoicePath
                            }), !voice.transcript.isEmpty {
                                Label(
                                    voice.transcriptSource.isEmpty
                                        ? "已匹配预置 ICL 逐字稿"
                                        : "已匹配 ICL 逐字稿 · \(voice.transcriptSource)",
                                    systemImage: "checkmark.seal.fill"
                                )
                                .font(.caption)
                                .foregroundStyle(.green)
                            } else {
                                Label(
                                    "自定义参考音频需要填写准确的逐字稿",
                                    systemImage: "exclamationmark.triangle.fill"
                                )
                                .font(.caption)
                                .foregroundStyle(.orange)
                            }
                        }
                        Toggle("参考音频末尾自动补静音", isOn: $model.appendSilence)
                        Toggle("一次性输入完整文本", isOn: $model.nonStreamingInput)
                    }
                }
            }
            .padding(StudioTokens.space4)
        }
        .background(.ultraThinMaterial)
        .overlay(alignment: .trailing) {
            Rectangle().fill(.white.opacity(0.22)).frame(width: 0.7)
        }
    }

    private var mainWorkspace: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: StudioTokens.space4) {
                GroupBox("文本试听") {
                    TextEditor(text: $model.text)
                        .font(.system(size: 17))
                        .scrollContentBackground(.hidden)
                        .frame(minHeight: 205)
                        .padding(StudioTokens.space3)
                        .background(
                            Color(nsColor: .textBackgroundColor).opacity(0.68),
                            in: RoundedRectangle(
                                cornerRadius: StudioTokens.elementRadius,
                                style: .continuous
                            )
                        )
                        .overlay {
                            RoundedRectangle(
                                cornerRadius: StudioTokens.elementRadius,
                                style: .continuous
                            )
                                .strokeBorder(.primary.opacity(0.08), lineWidth: 0.7)
                        }
                }

                GroupBox {
                    Button {
                        model.advancedSettingsPresented = true
                    } label: {
                        HStack {
                            VStack(alignment: .leading, spacing: 3) {
                                Label("高级参数", systemImage: "slider.horizontal.3")
                                    .font(.system(size: 13, weight: .semibold))
                                Text("采样、随机种子、生成帧和流式块设置")
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            }
                            Spacer()
                            Text("调整")
                                .font(.caption.weight(.semibold))
                                .foregroundStyle(Color.accentColor)
                            Image(systemName: "chevron.right")
                                .font(.caption.weight(.semibold))
                                .foregroundStyle(.secondary)
                        }
                        .contentShape(Rectangle())
                    }
                    .buttonStyle(StudioNavigationButtonStyle())
                    .accessibilityLabel("打开高级参数设置")
                }

                GroupBox("性能优化") {
                    VStack(alignment: .leading, spacing: StudioTokens.space3) {
                        HStack(spacing: StudioTokens.space4) {
                            VStack(alignment: .leading, spacing: 4) {
                                Label("为这台 Mac 自动选择生成策略", systemImage: "gauge.with.dots.needle.67percent")
                                    .font(.system(size: 13, weight: .semibold))
                                Text("分别测试流式首包、生成速度和双路吞吐；完成后自动应用到阅读器与整书任务。")
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            }
                            Spacer()
                            Button {
                                model.runPerformanceTest()
                            } label: {
                                Label(
                                    model.isRunningPerformanceTest ? "测试中…" : "测试性能",
                                    systemImage: "speedometer"
                                )
                            }
                            .buttonStyle(
                                StudioTintedButtonStyle(
                                    height: StudioTokens.compactControlHeight
                                )
                            )
                            .frame(width: 132)
                            .disabled(
                                !model.serviceReady
                                    || model.isGenerating
                                    || model.isRunningPerformanceTest
                            )
                        }
                        if model.isRunningPerformanceTest {
                            ProgressView()
                                .progressViewStyle(.linear)
                                .tint(StudioTokens.accent)
                        }
                        HStack(spacing: StudioTokens.space2) {
                            performanceTile(
                                title: "实时流式",
                                value: model.performanceStreamSummary
                            )
                            performanceTile(
                                title: "整块与整书",
                                value: model.performanceBlockSummary
                            )
                            performanceTile(
                                title: "实测收益",
                                value: model.performanceGainSummary
                            )
                        }
                        Text(model.performanceTestStatus)
                            .font(.caption)
                            .foregroundStyle(
                                model.performanceTestStatus.contains("失败")
                                    ? Color.red
                                    : Color.secondary
                            )
                    }
                }

                GroupBox("当前生成参数") {
                    LazyVGrid(
                        columns: [
                            GridItem(.flexible(), spacing: StudioTokens.space2),
                            GridItem(.flexible(), spacing: StudioTokens.space2),
                            GridItem(.flexible(), spacing: StudioTokens.space2),
                        ],
                        alignment: .leading,
                        spacing: StudioTokens.space2
                    ) {
                        parameterTile("模型", model.modelDisplayName)
                        parameterTile(
                            "音色与克隆",
                            "\(model.referenceName) · \(model.cloneMode == "icl" ? "ICL" : "X-vector")"
                        )
                        parameterTile(
                            "Seed",
                            model.actualSeedDisplay,
                            emphasized: model.seed < 0 && model.actualGenerationSeed != nil
                        )
                        parameterTile(
                            "采样",
                            String(
                                format: "T %.2f · P %.2f · K %d",
                                model.temperature,
                                model.topP,
                                model.topK
                            )
                        )
                        parameterTile(
                            "生成约束",
                            String(
                                format: "RP %.2f · 最大 %d · 最小 %d",
                                model.repetitionPenalty,
                                model.maxNewTokens,
                                model.minNewTokens
                            )
                        )
                        parameterTile(
                            "流式输出",
                            model.streamingGeneration
                                ? "PCM 开启 · \(model.chunkFrames) 帧/块"
                                : "关闭 · 仅最终 WAV"
                        )
                        parameterTile(
                            "输入处理",
                            "\(model.nonStreamingInput ? "完整文本" : "流式输入") · 补静音\(model.appendSilence ? "开启" : "关闭")"
                        )
                        parameterTile(
                            "当前任务",
                            model.currentJobDisplayID.isEmpty
                                ? "尚未提交"
                                : "\(model.currentJobDisplayID) · \(model.isGenerating ? "进行中" : "已结束")"
                        )
                    }
                }

                GroupBox("生成与试听") {
                    VStack(alignment: .leading, spacing: StudioTokens.space3) {
                        HStack(spacing: StudioTokens.space2) {
                            Button {
                                model.generate()
                            } label: {
                                Label(model.isGenerating ? "生成中…" : "生成语音", systemImage: "waveform.badge.plus")
                            }
                            .buttonStyle(StudioPrimaryButtonStyle())
                            .frame(maxWidth: 300)
                            .disabled(model.isGenerating || !model.serviceReady)
                            Button("停止") { model.stopGeneration() }
                                .buttonStyle(
                                    StudioTintedButtonStyle(
                                        destructive: true,
                                        height: StudioTokens.primaryControlHeight
                                    )
                                )
                                .frame(width: 110)
                                .disabled(!model.isGenerating)
                            Button {
                                model.playOutput()
                            } label: {
                                Label("试听结果", systemImage: "play.circle.fill")
                            }
                            .buttonStyle(
                                StudioSecondaryButtonStyle(
                                    height: StudioTokens.primaryControlHeight
                                )
                            )
                            .frame(width: 142)
                            .disabled(model.outputAudioURL == nil)
                        }
                        ProgressView(value: model.generatedProgress)
                            .tint(StudioTokens.accent)
                        Text(model.generationSummary)
                            .font(.callout)
                            .foregroundStyle(.secondary)
                        if !model.errorMessage.isEmpty {
                            Text(model.errorMessage)
                                .font(.callout)
                                .foregroundStyle(.red)
                                .textSelection(.enabled)
                        }
                    }
                }
            }
            .padding(StudioTokens.space5)
        }
    }

    private func parameterTile(
        _ title: String,
        _ value: String,
        emphasized: Bool = false
    ) -> some View {
        VStack(alignment: .leading, spacing: StudioTokens.space1) {
            Text(title)
                .font(.system(size: 10, weight: .semibold))
                .foregroundStyle(.secondary)
            Text(value)
                .font(.system(size: 12, weight: emphasized ? .semibold : .medium))
                .foregroundStyle(emphasized ? StudioTokens.accent : Color.primary)
                .lineLimit(2)
                .fixedSize(horizontal: false, vertical: true)
        }
        .frame(maxWidth: .infinity, minHeight: 44, alignment: .topLeading)
        .padding(.horizontal, StudioTokens.space2)
        .padding(.vertical, 7)
        .background(
            (emphasized ? StudioTokens.accent.opacity(0.09) : Color.primary.opacity(0.045)),
            in: RoundedRectangle(cornerRadius: StudioTokens.innerRadius, style: .continuous)
        )
    }

    private func performanceTile(title: String, value: String) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title)
                .font(.caption2.weight(.semibold))
                .foregroundStyle(.secondary)
            Text(value)
                .font(.system(size: 12, weight: .semibold))
                .lineLimit(1)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.horizontal, StudioTokens.space3)
        .frame(height: 52)
        .background(
            Color.primary.opacity(0.045),
            in: RoundedRectangle(cornerRadius: StudioTokens.innerRadius, style: .continuous)
        )
    }
}
