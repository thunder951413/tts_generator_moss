import AppKit
import AVFoundation
import Foundation
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

final class NativeStudioViewModel: ObservableObject {
    @Published var serviceSummary = "正在启动本地服务…"
    @Published var serviceReady = false
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
    @Published var isSavingPreset = false
    @Published var advancedSettingsPresented = false
    @Published var externalAPISettingsPresented = false
    @Published var externalAccessEnabled = false
    @Published var externalPort = 7861
    @Published var externalPassword = ""
    @Published var externalSettingsStatus = ""
    @Published var isApplyingExternalSettings = false

    let service: LocalService
    var onPresetsChanged: (([NativePreset]) -> Void)?
    private var pollingTimer: Timer?
    private var currentJobID: String?
    private var player: AVPlayer?
    private var recorder: AVAudioRecorder?
    private var recordingURL: URL?

    init(service: LocalService) {
        self.service = service
        self.externalAccessEnabled = service.configuration["HOST"] == "0.0.0.0"
        self.externalPort = service.port
        self.externalPassword = service.configuration["QWEN_TTS_ACCESS_PASSWORD"] ?? ""
    }

    deinit {
        pollingTimer?.invalidate()
    }

    func updateHealth(_ health: [String: Any]) {
        let state = health["state"] as? String ?? "unknown"
        let profile = health["active_profile_label"] as? String ?? "等待模型"
        let scheduler = health["generation_scheduler"] as? [String: Any]
        let stt = health["stt"] as? [String: Any]
        let sttReady = stt?["ready"] as? Bool ?? false
        let active = number(scheduler?["active"]).intValue
        let maximum = number(scheduler?["max_parallel"]).intValue
        serviceReady = state == "ready"
        serviceSummary = serviceReady
            ? "TTS Metal · STT \(sttReady ? "就绪" : "未就绪") · \(profile) · GPU \(active)/\(maximum)"
            : "服务状态：\(state)"
    }

    func loadInitialData() {
        loadVoices()
        refreshPresets()
        loadActiveServiceSettings()
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
        isApplyingExternalSettings = true
        externalSettingsStatus = "正在保存并重启语音服务…"
        service.applyExternalSettings(
            exposeToLAN: externalAccessEnabled,
            port: externalPort,
            password: password
        ) { [weak self] result in
            DispatchQueue.main.async {
                guard let self else { return }
                self.isApplyingExternalSettings = false
                switch result {
                case .success(let health):
                    self.externalPassword = password
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
                DispatchQueue.main.async {
                    self.voices = voices
                    if self.referenceAudioPath.isEmpty,
                       let defaultPath = payload["default_reference_audio_path"] as? String {
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

    func importReference(_ url: URL) {
        guard let fileData = try? Data(contentsOf: url) else {
            show(NativeStudioError.server("无法读取参考音频。"))
            return
        }
        isImportingReference = true
        let multipart = multipartBody(
            fields: [:],
            file: ("audio", url.lastPathComponent, "application/octet-stream", fileData)
        )
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
                    self.selectedVoicePath = ""
                    self.referenceAudioPath = path
                    self.referenceName = url.deletingPathExtension().lastPathComponent
                    self.referenceText = ""
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

    private func play(_ url: URL) {
        player = AVPlayer(url: url)
        player?.play()
    }

    func generate() {
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
        outputAudioURL = nil
        isGenerating = true
        generatedProgress = 0
        generationSummary = "正在提交生成任务…"
        errorMessage = ""
        actualGenerationSeed = nil
        submittedSeedSetting = nil
        currentJobDisplayID = ""
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
                DispatchQueue.main.async {
                    self.currentJobID = jobID
                    self.currentJobDisplayID = String(jobID.prefix(8))
                    self.submittedSeedSetting = requestedSeed
                    self.actualGenerationSeed = resolvedSeed
                    self.generationSummary = "任务已排队"
                    self.startPolling(jobID)
                }
            } catch {
                DispatchQueue.main.async { self.isGenerating = false }
                self.show(error)
            }
        }
    }

    func stopGeneration() {
        guard let currentJobID else { return }
        service.perform("api/generate-stream/\(currentJobID)/close", method: "POST") { _, _ in }
        stopPolling()
        isGenerating = false
        generationSummary = "已停止当前生成"
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
                    if let resolvedSeed {
                        self.actualGenerationSeed = resolvedSeed
                    }
                    self.generatedProgress = min(1, generated / maximum)
                    self.generationSummary = "\(self.nativeStateLabel(state)) · \(Int(generated))/\(Int(maximum)) 帧 · \(String(format: "%.1f", audioSeconds)) 秒音频"
                    if state == "finished" {
                        self.stopPolling()
                        self.downloadResult(jobID)
                    } else if state == "error" || state == "closed" {
                        self.stopPolling()
                        self.isGenerating = false
                        self.errorMessage = payload["error"] as? String ?? "生成任务失败。"
                    }
                }
            } catch {
                self.show(error)
            }
        }
    }

    private func downloadResult(_ jobID: String) {
        generationSummary = "正在读取生成结果…"
        service.perform("api/generate-stream/\(jobID)/result-audio", timeout: 120) { [weak self] data, response in
            guard let self else { return }
            guard response?.statusCode == 200, let data else {
                self.show(NativeStudioError.server("生成完成，但无法读取输出音频。"))
                DispatchQueue.main.async { self.isGenerating = false }
                return
            }
            let url = FileManager.default.temporaryDirectory
                .appendingPathComponent("qwen-tts-native-\(jobID).wav")
            do {
                try data.write(to: url, options: .atomic)
                DispatchQueue.main.async {
                    self.outputAudioURL = url
                    self.isGenerating = false
                    self.generatedProgress = 1
                    self.generationSummary = "生成完成，可以试听"
                    self.playOutput()
                }
            } catch {
                self.show(error)
                DispatchQueue.main.async { self.isGenerating = false }
            }
        }
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
                            Text("开启后，服务会在生成过程中持续提供音频块；关闭后只保留最终 WAV。")
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
                            if model.externalPassword.isEmpty {
                                Label("密码为空会关闭 API 认证，不建议在局域网模式使用。", systemImage: "exclamationmark.triangle.fill")
                                    .font(.caption)
                                    .foregroundStyle(.orange)
                            }
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
            StudioStatusPill(title: "TTS Metal 已就绪", ready: model.serviceReady)
            StudioStatusPill(
                title: model.serviceSummary.contains("STT 就绪") ? "STT Metal 已就绪" : "STT 未就绪",
                ready: model.serviceSummary.contains("STT 就绪")
            )
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
                                .disabled(model.isImportingReference)
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
                            Button {
                                model.applyCurrentAsServiceSettings()
                            } label: {
                                Label(
                                    model.isApplyingServiceSettings ? "正在应用…" : "应用为服务设置",
                                    systemImage: "checkmark.seal.fill"
                                )
                            }
                            .buttonStyle(StudioTintedButtonStyle())
                            .frame(width: 178)
                            .disabled(model.outputAudioURL == nil || model.isApplyingServiceSettings)
                        }
                        ProgressView(value: model.generatedProgress)
                            .tint(StudioTokens.accent)
                        Text(model.generationSummary)
                            .font(.callout)
                            .foregroundStyle(.secondary)
                        Text(model.serviceSettingsStatus)
                            .font(.caption)
                            .foregroundStyle(
                                model.serviceSettingsStatus.contains("已应用")
                                    ? Color.green
                                    : Color.secondary
                            )
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
}
