import AppKit
import Foundation
import SwiftUI
import UniformTypeIdentifiers

enum STTMediaKind: String, CaseIterable, Identifiable {
    case audio
    case video

    var id: String { rawValue }
    var title: String { self == .audio ? "音频" : "视频" }
    var icon: String { self == .audio ? "waveform" : "film" }
    var extensions: [String] {
        switch self {
        case .audio:
            return ["wav", "mp3", "m4a", "aac", "flac", "ogg", "opus", "aiff", "caf"]
        case .video:
            return ["mp4", "mov", "m4v", "mkv", "webm", "avi"]
        }
    }
}

enum STTSubtitleFormat: String, CaseIterable, Identifiable {
    case srt
    case vtt

    var id: String { rawValue }
    var title: String { rawValue.uppercased() }
}

final class STTWorkbenchViewModel: ObservableObject {
    @Published var mediaKind: STTMediaKind = .audio
    @Published var selectedFileURL: URL?
    @Published var includeTimestamps = false
    @Published var subtitleFormat: STTSubtitleFormat = .srt
    @Published var language = "auto"
    @Published var prompt = ""
    @Published var resultText = ""
    @Published var statusText = "选择音频或视频文件开始转写"
    @Published var errorText = ""
    @Published var isProcessing = false
    @Published var serviceReachable = false
    @Published var sttReady = false

    let service: LocalService
    private var uploadTask: URLSessionUploadTask?

    init(service: LocalService) {
        self.service = service
    }

    var selectedFileName: String {
        selectedFileURL?.lastPathComponent ?? "尚未选择文件"
    }

    var selectedFileDetail: String {
        guard let url = selectedFileURL else {
            return mediaKind == .audio ? "支持 WAV、MP3、M4A、FLAC 等格式" : "支持 MP4、MOV、MKV、WebM 等格式"
        }
        let values = try? url.resourceValues(forKeys: [.fileSizeKey])
        let size = ByteCountFormatter.string(
            fromByteCount: Int64(values?.fileSize ?? 0),
            countStyle: .file
        )
        return "\(size) · \(url.pathExtension.uppercased())"
    }

    var outputDescription: String {
        includeTimestamps ? "生成 \(subtitleFormat.title) 字幕，可直接保存使用" : "生成可编辑的纯文字稿"
    }

    func refreshHealth() {
        service.health { [weak self] payload in
            DispatchQueue.main.async {
                guard let self else { return }
                self.serviceReachable = payload != nil
                let stt = payload?["stt"] as? [String: Any]
                self.sttReady = stt?["ready"] as? Bool ?? false
            }
        }
    }

    func chooseFile() {
        let panel = NSOpenPanel()
        panel.title = mediaKind == .audio ? "选择要转写的音频" : "选择要提取字幕的视频"
        panel.prompt = "选择"
        panel.canChooseDirectories = false
        panel.canChooseFiles = true
        panel.allowsMultipleSelection = false
        panel.allowedContentTypes = mediaKind.extensions.compactMap { UTType(filenameExtension: $0) }
        guard panel.runModal() == .OK, let url = panel.url else { return }
        selectedFileURL = url
        resultText = ""
        errorText = ""
        statusText = "已选择 \(url.lastPathComponent)"
    }

    func clearSelection() {
        selectedFileURL = nil
        resultText = ""
        errorText = ""
        statusText = "选择音频或视频文件开始转写"
    }

    func transcribe() {
        guard let source = selectedFileURL else {
            errorText = "请先选择要处理的\(mediaKind.title)文件。"
            return
        }
        guard serviceReachable else {
            errorText = "本地服务尚未连接，请先从菜单栏启动服务。"
            return
        }
        guard !isProcessing else { return }

        let responseFormat = includeTimestamps ? subtitleFormat.rawValue : "text"
        let languageValue = language
        let promptValue = prompt
        let boundary = "----QwenSTT\(UUID().uuidString.replacingOccurrences(of: "-", with: ""))"
        let bodyURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("qwen-stt-upload-\(UUID().uuidString).multipart")
        isProcessing = true
        errorText = ""
        resultText = ""
        statusText = "正在准备 \(source.lastPathComponent)…"

        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            do {
                try Self.buildMultipartBody(
                    at: bodyURL,
                    source: source,
                    boundary: boundary,
                    responseFormat: responseFormat,
                    language: languageValue,
                    prompt: promptValue
                )
            } catch {
                DispatchQueue.main.async {
                    guard let self else { return }
                    self.isProcessing = false
                    self.errorText = "无法读取所选文件：\(error.localizedDescription)"
                    self.statusText = "处理失败"
                }
                return
            }
            DispatchQueue.main.async {
                guard let self, self.isProcessing else {
                    try? FileManager.default.removeItem(at: bodyURL)
                    return
                }
                self.startUpload(
                    bodyURL: bodyURL,
                    boundary: boundary,
                    sourceKind: self.mediaKind
                )
            }
        }
    }

    private func startUpload(bodyURL: URL, boundary: String, sourceKind: STTMediaKind) {
        var request = service.authorizedRequest(
            "v1/audio/transcriptions",
            method: "POST",
            contentType: "multipart/form-data; boundary=\(boundary)",
            timeout: 3_600
        )
        request.setValue("text/plain, application/json", forHTTPHeaderField: "Accept")
        statusText = sourceKind == .video ? "正在提取声音并识别…" : "正在识别语音…"

        let task = URLSession.shared.uploadTask(with: request, fromFile: bodyURL) { [weak self] data, response, error in
            try? FileManager.default.removeItem(at: bodyURL)
            DispatchQueue.main.async {
                guard let self else { return }
                self.isProcessing = false
                self.uploadTask = nil
                if let error = error as? URLError, error.code == .cancelled {
                    self.statusText = "已停止当前转写"
                    return
                }
                if let error {
                    self.errorText = "转写失败：\(error.localizedDescription)"
                    self.statusText = "处理失败"
                    return
                }
                guard let http = response as? HTTPURLResponse, let data else {
                    self.errorText = "服务没有返回有效结果。"
                    self.statusText = "处理失败"
                    return
                }
                guard (200 ..< 300).contains(http.statusCode) else {
                    self.errorText = Self.serverError(from: data, status: http.statusCode)
                    self.statusText = "处理失败"
                    self.refreshHealth()
                    return
                }
                self.resultText = String(data: data, encoding: .utf8) ?? ""
                self.statusText = self.includeTimestamps ? "字幕生成完成" : "文字转写完成"
                self.sttReady = true
            }
        }
        uploadTask = task
        task.resume()
    }

    func stop() {
        uploadTask?.cancel()
        uploadTask = nil
        isProcessing = false
        statusText = "正在停止…"
    }

    func copyResult() {
        guard !resultText.isEmpty else { return }
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(resultText, forType: .string)
        statusText = "结果已复制到剪贴板"
    }

    func saveResult() {
        guard !resultText.isEmpty else { return }
        let outputExtension = includeTimestamps ? subtitleFormat.rawValue : "txt"
        let baseName = selectedFileURL?.deletingPathExtension().lastPathComponent ?? "转写结果"
        let panel = NSSavePanel()
        panel.title = includeTimestamps ? "保存字幕" : "保存文字稿"
        panel.nameFieldStringValue = "\(baseName).\(outputExtension)"
        if let outputType = UTType(filenameExtension: outputExtension) {
            panel.allowedContentTypes = [outputType]
        }
        guard panel.runModal() == .OK, let url = panel.url else { return }
        do {
            try resultText.write(to: url, atomically: true, encoding: .utf8)
            statusText = "已保存到 \(url.lastPathComponent)"
        } catch {
            errorText = "保存失败：\(error.localizedDescription)"
        }
    }

    private static func buildMultipartBody(
        at target: URL,
        source: URL,
        boundary: String,
        responseFormat: String,
        language: String,
        prompt: String
    ) throws {
        FileManager.default.createFile(atPath: target.path, contents: nil)
        let output = try FileHandle(forWritingTo: target)
        defer { try? output.close() }

        func write(_ value: String) throws {
            guard let data = value.data(using: .utf8) else { return }
            try output.write(contentsOf: data)
        }
        func field(_ name: String, _ value: String) throws {
            try write("--\(boundary)\r\n")
            try write("Content-Disposition: form-data; name=\"\(name)\"\r\n\r\n")
            try write("\(value)\r\n")
        }

        try field("model", "whisper-small")
        try field("language", language)
        try field("prompt", prompt)
        try field("response_format", responseFormat)
        let safeName = source.lastPathComponent.replacingOccurrences(of: "\"", with: "")
        try write("--\(boundary)\r\n")
        try write("Content-Disposition: form-data; name=\"file\"; filename=\"\(safeName)\"\r\n")
        try write("Content-Type: \(mimeType(for: source))\r\n\r\n")
        let input = try FileHandle(forReadingFrom: source)
        defer { try? input.close() }
        while let chunk = try input.read(upToCount: 1_048_576), !chunk.isEmpty {
            try output.write(contentsOf: chunk)
        }
        try write("\r\n--\(boundary)--\r\n")
    }

    private static func mimeType(for url: URL) -> String {
        let ext = url.pathExtension.lowercased()
        let values = [
            "wav": "audio/wav", "mp3": "audio/mpeg", "m4a": "audio/mp4",
            "aac": "audio/aac", "flac": "audio/flac", "ogg": "audio/ogg",
            "opus": "audio/opus", "aiff": "audio/aiff", "caf": "audio/x-caf",
            "mp4": "video/mp4", "mov": "video/quicktime", "m4v": "video/x-m4v",
            "mkv": "video/x-matroska", "webm": "video/webm", "avi": "video/x-msvideo",
        ]
        return values[ext] ?? "application/octet-stream"
    }

    private static func serverError(from data: Data, status: Int) -> String {
        if let payload = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let detail = payload["detail"] as? String {
            return "转写失败（\(status)）：\(detail)"
        }
        let text = String(data: data, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines)
        return "转写失败（\(status)）\(text?.isEmpty == false ? "：\(text!)" : "")"
    }
}

struct STTWorkbenchView: View {
    @ObservedObject var model: STTWorkbenchViewModel

    var body: some View {
        ZStack {
            StudioBackground()
            VStack(spacing: 0) {
                header
                HSplitView {
                    controls
                        .frame(minWidth: 380, idealWidth: 420, maxWidth: 470)
                    result
                        .frame(minWidth: 520)
                }
                .padding(StudioTokens.space5)
            }
        }
        .frame(minWidth: 980, minHeight: 680)
        .onAppear { model.refreshHealth() }
    }

    private var header: some View {
        HStack(spacing: StudioTokens.space4) {
            ZStack {
                RoundedRectangle(cornerRadius: 14, style: .continuous)
                    .fill(StudioTokens.accent.opacity(0.12))
                Image(systemName: "captions.bubble.fill")
                    .font(.system(size: 22, weight: .semibold))
                    .foregroundStyle(StudioTokens.accent)
            }
            .frame(width: 48, height: 48)
            VStack(alignment: .leading, spacing: 3) {
                Text("语音转文字")
                    .font(.system(size: 24, weight: .bold, design: .rounded))
                Text("本地 Metal 转写 · 音频、视频与字幕生成")
                    .font(.system(size: 13))
                    .foregroundStyle(.secondary)
            }
            Spacer()
            StudioStatusPill(
                title: model.sttReady ? "STT Metal 已就绪" : (model.serviceReachable ? "按需启动 STT" : "服务未连接"),
                ready: model.sttReady
            )
        }
        .padding(.horizontal, StudioTokens.space5)
        .frame(height: 82)
        .background(.ultraThinMaterial)
        .overlay(alignment: .bottom) { Divider().opacity(0.45) }
    }

    private var controls: some View {
        ScrollView {
            VStack(spacing: StudioTokens.space4) {
                GroupBox("输入文件") {
                    VStack(alignment: .leading, spacing: StudioTokens.space3) {
                        Picker("媒体类型", selection: $model.mediaKind) {
                            ForEach(STTMediaKind.allCases) { kind in
                                Label(kind.title, systemImage: kind.icon).tag(kind)
                            }
                        }
                        .pickerStyle(.segmented)
                        .onChange(of: model.mediaKind) { _ in model.clearSelection() }

                        Button(action: model.chooseFile) {
                            HStack(spacing: StudioTokens.space3) {
                                Image(systemName: model.mediaKind == .audio ? "waveform.badge.plus" : "film.stack")
                                    .font(.system(size: 22, weight: .medium))
                                    .foregroundStyle(StudioTokens.accent)
                                VStack(alignment: .leading, spacing: 3) {
                                    Text(model.selectedFileName)
                                        .font(.system(size: 13, weight: .semibold))
                                        .lineLimit(1)
                                    Text(model.selectedFileDetail)
                                        .font(.caption)
                                        .foregroundStyle(.secondary)
                                        .lineLimit(1)
                                }
                                Spacer()
                                Image(systemName: "chevron.right")
                                    .foregroundStyle(.tertiary)
                            }
                            .padding(StudioTokens.space3)
                            .background(Color.primary.opacity(0.035), in: RoundedRectangle(cornerRadius: 12))
                        }
                        .buttonStyle(.plain)
                    }
                }
                .groupBoxStyle(StudioGlassGroupBoxStyle())

                GroupBox("输出方式") {
                    VStack(alignment: .leading, spacing: StudioTokens.space3) {
                        Toggle("生成带时间戳的字幕", isOn: $model.includeTimestamps)
                            .toggleStyle(.switch)
                        if model.includeTimestamps {
                            Picker("字幕格式", selection: $model.subtitleFormat) {
                                ForEach(STTSubtitleFormat.allCases) { format in
                                    Text(format.title).tag(format)
                                }
                            }
                            .pickerStyle(.segmented)
                        }
                        Text(model.outputDescription)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
                .groupBoxStyle(StudioGlassGroupBoxStyle())

                GroupBox("识别设置") {
                    VStack(alignment: .leading, spacing: StudioTokens.space3) {
                        Picker("语言", selection: $model.language) {
                            Text("自动识别").tag("auto")
                            Text("中文").tag("zh")
                            Text("英语").tag("en")
                            Text("日语").tag("ja")
                            Text("韩语").tag("ko")
                        }
                        .pickerStyle(.menu)
                        TextField("提示词（可选，例如人名或专业术语）", text: $model.prompt)
                            .textFieldStyle(.roundedBorder)
                    }
                }
                .groupBoxStyle(StudioGlassGroupBoxStyle())

                if !model.errorText.isEmpty {
                    Label(model.errorText, systemImage: "exclamationmark.triangle.fill")
                        .font(.caption)
                        .foregroundStyle(.red)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(StudioTokens.space3)
                        .background(Color.red.opacity(0.08), in: RoundedRectangle(cornerRadius: 12))
                }

                Button(action: model.transcribe) {
                    HStack {
                        if model.isProcessing { ProgressView().controlSize(.small) }
                        Label(
                            model.isProcessing ? "正在处理…" : (model.includeTimestamps ? "生成字幕" : "转写为文字"),
                            systemImage: "sparkles"
                        )
                    }
                }
                .buttonStyle(StudioPrimaryButtonStyle())
                .disabled(model.selectedFileURL == nil || model.isProcessing || !model.serviceReachable)

                if model.isProcessing {
                    Button(action: model.stop) {
                        Label("停止当前转写", systemImage: "stop.fill")
                    }
                    .buttonStyle(StudioSecondaryButtonStyle(destructive: true))
                }
            }
            .padding(.trailing, StudioTokens.space2)
        }
    }

    private var result: some View {
        GroupBox {
            VStack(spacing: 0) {
                HStack {
                    VStack(alignment: .leading, spacing: 2) {
                        Text(model.includeTimestamps ? "字幕结果" : "文字结果")
                            .font(.system(size: 15, weight: .semibold))
                        Text(model.statusText)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    Spacer()
                    Button(action: model.copyResult) {
                        Label("复制", systemImage: "doc.on.doc")
                    }
                    .buttonStyle(StudioToolbarButtonStyle())
                    .disabled(model.resultText.isEmpty)
                    Button(action: model.saveResult) {
                        Label("保存", systemImage: "square.and.arrow.down")
                    }
                    .buttonStyle(StudioToolbarButtonStyle())
                    .disabled(model.resultText.isEmpty)
                }
                .padding(.bottom, StudioTokens.space3)

                ZStack {
                    RoundedRectangle(cornerRadius: 13, style: .continuous)
                        .fill(Color(nsColor: .textBackgroundColor).opacity(0.72))
                    if model.resultText.isEmpty {
                        VStack(spacing: StudioTokens.space3) {
                            Image(systemName: model.includeTimestamps ? "captions.bubble" : "text.alignleft")
                                .font(.system(size: 34, weight: .light))
                                .foregroundStyle(StudioTokens.accent.opacity(0.6))
                            Text("处理完成后，结果会显示在这里")
                                .font(.system(size: 13, weight: .medium))
                                .foregroundStyle(.secondary)
                            Text("结果可以直接编辑、复制或保存到本地")
                                .font(.caption)
                                .foregroundStyle(.tertiary)
                        }
                    }
                    TextEditor(text: $model.resultText)
                        .font(.system(size: 14, design: .monospaced))
                        .scrollContentBackground(.hidden)
                        .padding(StudioTokens.space3)
                        .opacity(model.resultText.isEmpty ? 0 : 1)
                }
                .overlay {
                    RoundedRectangle(cornerRadius: 13, style: .continuous)
                        .strokeBorder(Color.primary.opacity(0.08), lineWidth: 0.8)
                }
            }
        }
        .groupBoxStyle(StudioGlassGroupBoxStyle())
    }
}
