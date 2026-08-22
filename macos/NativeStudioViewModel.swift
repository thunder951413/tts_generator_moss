import AppKit
import AVFoundation
import CoreMedia
import Foundation
import ScreenCaptureKit
import SwiftUI
import UniformTypeIdentifiers

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
    @Published var audioTrimTitle = "裁剪参考音频"
    @Published var audioTrimSubtitle = "只保留声音稳定、背景干净的一段作为克隆参考。"
    @Published var audioTrimProgress = ""
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
    private var trimSourceIsTemporary = false
    private var reopenReferenceLibraryAfterTrim = false
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
        if trimSourceIsTemporary, let systemAudioRecordingURL {
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

    func trimReference(_ reference: NativeReferenceAudio) {
        stopPreview()
        referenceLibraryStatus = "正在打开“\(reference.name)”的裁剪工具…"
        referenceLibraryPresented = false
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.2) { [weak self] in
            self?.prepareReferenceAudioForTrimming(
                URL(fileURLWithPath: reference.path),
                name: "\(reference.name) · 裁剪",
                title: "裁剪已有参考音频",
                subtitle: "原音频会保留；确认后创建一个新的裁剪副本。",
                sourceIsTemporary: false,
                reopenLibrary: true
            )
        }
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
        prepareReferenceAudioForTrimming(
            url,
            name: url.deletingPathExtension().lastPathComponent,
            title: "裁剪导入音频",
            subtitle: "先选择需要的声音片段，再压缩并加入参考音频。",
            sourceIsTemporary: false
        )
    }

    func importReference(
        _ url: URL,
        displayName: String? = nil,
        completion: ((Bool) -> Void)? = nil
    ) {
        isImportingReference = true
        audioTrimProgress = "正在读取并准备上传…"
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self else { return }
            guard let fileData = try? Data(contentsOf: url) else {
                DispatchQueue.main.async {
                    self.isImportingReference = false
                    self.audioTrimProgress = "无法读取参考音频"
                    completion?(false)
                }
                self.show(NativeStudioError.server("无法读取参考音频。"))
                return
            }
            let multipart = self.multipartBody(
                fields: [:],
                file: (
                    "audio",
                    "\(displayName ?? url.deletingPathExtension().lastPathComponent).\(url.pathExtension)",
                    "application/octet-stream",
                    fileData
                )
            )
            DispatchQueue.main.async { self.audioTrimProgress = "正在上传并标准化音频…" }
            self.service.perform(
                "api/presets/reference-audio",
                method: "POST",
                body: multipart.data,
                contentType: multipart.contentType,
                timeout: 180
            ) { [weak self] data, response in
                guard let self else { return }
                if url.deletingLastPathComponent() == FileManager.default.temporaryDirectory,
                   url.lastPathComponent.hasPrefix("qwen-system-export-") {
                    try? FileManager.default.removeItem(at: url)
                }
                do {
                    let payload = try self.jsonObject(data, response: response)
                    guard let path = payload["reference_audio_path"] as? String else {
                        throw NativeStudioError.invalidResponse
                    }
                    DispatchQueue.main.async {
                        self.isImportingReference = false
                        self.audioTrimProgress = "导入完成"
                        self.referenceAudioPath = path
                        self.referenceName = displayName ?? url.deletingPathExtension().lastPathComponent
                        self.referenceText = ""
                        self.loadReferenceLibrary()
                        self.loadVoices()
                        completion?(true)
                    }
                } catch {
                    DispatchQueue.main.async {
                        self.isImportingReference = false
                        self.audioTrimProgress = "导入失败：\(error.localizedDescription)"
                        completion?(false)
                    }
                    self.show(error)
                }
            }
        }
    }

    func toggleReferenceRecording() {
        if isRecordingReference {
            recorder?.stop()
            recorder = nil
            isRecordingReference = false
            if let recordingURL {
                self.recordingURL = nil
                prepareReferenceAudioForTrimming(
                    recordingURL,
                    name: "麦克风录音 \(Self.recordingDateFormatter.string(from: Date()))",
                    title: "裁剪麦克风录音",
                    subtitle: "选择发音稳定、没有停顿和杂音的一段作为参考。",
                    sourceIsTemporary: true
                )
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
                self.prepareReferenceAudioForTrimming(
                    url,
                    name: "系统录音 \(Self.recordingDateFormatter.string(from: Date()))",
                    title: "裁剪系统录音",
                    subtitle: "只保留声音稳定、背景干净的一段作为克隆参考。",
                    sourceIsTemporary: true
                )
            case .failure(let error):
                self.systemAudioStatus = "录制失败"
                self.show(error)
            }
        }
    }

    private func prepareReferenceAudioForTrimming(
        _ url: URL,
        name: String,
        title: String,
        subtitle: String,
        sourceIsTemporary: Bool,
        reopenLibrary: Bool = false
    ) {
        if trimSourceIsTemporary, let previous = systemAudioRecordingURL, previous != url {
            try? FileManager.default.removeItem(at: previous)
        }
        systemAudioRecordingURL = url
        trimSourceIsTemporary = sourceIsTemporary
        reopenReferenceLibraryAfterTrim = reopenLibrary
        audioTrimTitle = title
        audioTrimSubtitle = subtitle
        audioTrimProgress = "正在分析音频…"
        let asset = AVURLAsset(url: url)
        let duration = CMTimeGetSeconds(asset.duration)
        guard duration.isFinite, duration > 0.1 else {
            if sourceIsTemporary { try? FileManager.default.removeItem(at: url) }
            systemAudioRecordingURL = nil
            trimSourceIsTemporary = false
            systemAudioStatus = "没有捕获到可用声音"
            show(NativeStudioError.server("音频太短、无法读取或没有可用声音。"))
            return
        }
        systemAudioDuration = duration
        systemAudioTrimStart = 0
        systemAudioTrimEnd = duration
        systemAudioName = name
        systemAudioWaveform = []
        systemAudioStatus = "音频已准备，可裁剪后加入参考音频"
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
                DispatchQueue.main.async {
                    self?.systemAudioWaveform = peaks
                    self?.audioTrimProgress = "拖动起点和终点选择需要保留的片段"
                }
            } catch {
                DispatchQueue.main.async {
                    self?.systemAudioWaveform = []
                    self?.audioTrimProgress = "无法生成波形，但仍可按时间裁剪"
                }
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
        guard systemAudioRecordingURL != nil || systemAudioTrimPresented else { return }
        stopPreview()
        let shouldReopenLibrary = reopenReferenceLibraryAfterTrim
        if trimSourceIsTemporary, let systemAudioRecordingURL {
            try? FileManager.default.removeItem(at: systemAudioRecordingURL)
        }
        systemAudioRecordingURL = nil
        trimSourceIsTemporary = false
        reopenReferenceLibraryAfterTrim = false
        systemAudioTrimPresented = false
        systemAudioStatus = "可录制 Mac 正在播放的声音"
        audioTrimProgress = ""
        systemAudioWaveform = []
        if shouldReopenLibrary {
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.2) { [weak self] in
                self?.loadReferenceLibrary()
                self?.referenceLibraryPresented = true
            }
        }
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
        audioTrimProgress = "正在导出并压缩选区…"
        exporter.exportAsynchronously { [weak self] in
            DispatchQueue.main.async {
                guard let self else { return }
                guard exporter.status == .completed else {
                    self.isExportingSystemAudio = false
                    try? FileManager.default.removeItem(at: outputURL)
                    self.audioTrimProgress = "导出失败，可调整选区后重试"
                    self.show(
                        exporter.error ?? NativeStudioError.server("裁剪后的音频导出失败。")
                    )
                    return
                }
                self.stopPreview()
                self.importReference(
                    outputURL,
                    displayName: safeName.isEmpty ? "系统录音" : safeName
                ) { [weak self] success in
                    guard let self else { return }
                    self.isExportingSystemAudio = false
                    guard success else { return }
                    let shouldReopenLibrary = self.reopenReferenceLibraryAfterTrim
                    if self.trimSourceIsTemporary, let source = self.systemAudioRecordingURL {
                        try? FileManager.default.removeItem(at: source)
                    }
                    self.systemAudioRecordingURL = nil
                    self.trimSourceIsTemporary = false
                    self.reopenReferenceLibraryAfterTrim = false
                    self.systemAudioTrimPresented = false
                    self.systemAudioStatus = "已裁剪并加入参考音频"
                    if shouldReopenLibrary {
                        DispatchQueue.main.asyncAfter(deadline: .now() + 0.2) { [weak self] in
                            self?.loadReferenceLibrary()
                            self?.referenceLibraryPresented = true
                        }
                    }
                }
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

struct AdvancedSettingsView: View {
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

struct ExternalAPISettingsView: View {
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
