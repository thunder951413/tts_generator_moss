import AppKit
import CryptoKit
import Foundation
import SwiftUI

private let serviceSessionCookie = "qwen_tts_service_session"

final class LocalService {
    let repositoryRoot: URL
    private(set) var configuration: [String: String]
    private(set) var port: Int
    private(set) var ownsProcess = false
    private var process: Process?
    private var hasReachedReady = false
    private var lastExitStatus: Int32?
    var onUnexpectedTermination: ((Int32) -> Void)?

    var baseURL: URL {
        URL(string: "http://127.0.0.1:\(port)/")!
    }

    init(repositoryRoot: URL) {
        self.repositoryRoot = repositoryRoot
        self.configuration = Self.readEnvironmentFile(repositoryRoot.appendingPathComponent(".env.macos"))
        self.port = Int(self.configuration["PORT"] ?? "7861") ?? 7861
        log("app initialized; root=\(repositoryRoot.path); port=\(self.port)")
    }

    deinit {
        stopIfOwned()
    }

    var cookieValue: String? {
        let password = configuration["QWEN_TTS_ACCESS_PASSWORD"] ?? ""
        guard !password.isEmpty else { return nil }
        let key = SymmetricKey(data: Data(password.utf8))
        let signature = HMAC<SHA256>.authenticationCode(
            for: Data("qwen-tts-service-session".utf8), using: key
        )
        return signature.map { String(format: "%02x", $0) }.joined()
    }

    func request(_ path: String, completion: @escaping (Data?, HTTPURLResponse?) -> Void) {
        perform(path, completion: completion)
    }

    func authorizedRequest(
        _ path: String,
        method: String = "GET",
        body: Data? = nil,
        contentType: String? = nil,
        timeout: TimeInterval = 30
    ) -> URLRequest {
        let url = URL(string: path, relativeTo: baseURL)!
        var request = URLRequest(url: url)
        request.httpMethod = method
        request.httpBody = body
        request.timeoutInterval = timeout
        if let contentType {
            request.setValue(contentType, forHTTPHeaderField: "Content-Type")
        }
        if let cookieValue {
            request.setValue("\(serviceSessionCookie)=\(cookieValue)", forHTTPHeaderField: "Cookie")
        }
        return request
    }

    func perform(
        _ path: String,
        method: String = "GET",
        body: Data? = nil,
        contentType: String? = nil,
        timeout: TimeInterval = 30,
        completion: @escaping (Data?, HTTPURLResponse?) -> Void
    ) {
        let request = authorizedRequest(
            path,
            method: method,
            body: body,
            contentType: contentType,
            timeout: timeout
        )
        URLSession.shared.dataTask(with: request) { data, response, _ in
            completion(data, response as? HTTPURLResponse)
        }.resume()
    }

    func health(completion: @escaping ([String: Any]?) -> Void) {
        request("api/health") { data, response in
            guard response?.statusCode == 200, let data,
                  let payload = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
            else {
                completion(nil)
                return
            }
            completion(payload)
        }
    }

    func startIfNeeded(completion: @escaping (Result<[String: Any], Error>) -> Void) {
        log("checking local service")
        health { [weak self] status in
            if let status {
                self?.log("using existing local service")
                completion(.success(status))
                return
            }
            self?.log("local service unavailable; starting owned process")
            self?.start(completion: completion)
        }
    }

    func applyExternalSettings(
        exposeToLAN: Bool,
        port newPort: Int,
        password: String,
        corsOrigins: String,
        completion: @escaping (Result<[String: Any], Error>) -> Void
    ) {
        guard (1024 ... 65_535).contains(newPort) else {
            completion(.failure(ServiceError.invalidPort))
            return
        }
        if exposeToLAN && (password.count < 4 || password == "change-me") {
            completion(.failure(ServiceError.insecureLANPassword))
            return
        }
        let environmentURL = repositoryRoot.appendingPathComponent(".env.macos")
        do {
            try Self.updateEnvironmentFile(
                environmentURL,
                values: [
                    "HOST": exposeToLAN ? "0.0.0.0" : "127.0.0.1",
                    "PORT": String(newPort),
                    "QWEN_TTS_ACCESS_PASSWORD": password,
                    "QWEN_TTS_CORS_ORIGINS": corsOrigins,
                ]
            )
            configuration = Self.readEnvironmentFile(environmentURL)
            port = newPort
        } catch {
            completion(.failure(error))
            return
        }

        guard ownsProcess, let runningProcess = process, runningProcess.isRunning else {
            completion(.failure(ServiceError.restartRequired))
            return
        }
        ownsProcess = false
        process = nil
        runningProcess.terminate()
        log("restarting owned service after external API settings changed")
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            runningProcess.waitUntilExit()
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.35) {
                self?.start(completion: completion)
            }
        }
    }

    private func start(completion: @escaping (Result<[String: Any], Error>) -> Void) {
        let python = repositoryRoot.appendingPathComponent(".venv/bin/python")
        let script = repositoryRoot.appendingPathComponent("clis/qwen_tts_app.py")
        let library = repositoryRoot.appendingPathComponent(".runtime/qwentts.cpp/build-metal/libqwen.dylib")
        guard FileManager.default.isExecutableFile(atPath: python.path),
              FileManager.default.fileExists(atPath: script.path),
              FileManager.default.fileExists(atPath: library.path)
        else {
            log("runtime missing; python=\(python.path) script=\(script.path) library=\(library.path)")
            completion(.failure(ServiceError.runtimeMissing(repositoryRoot.path)))
            return
        }
        let process = Process()
        process.executableURL = python
        process.arguments = [
            script.path,
            "--host", configuration["HOST"] ?? "127.0.0.1",
            "--port", String(port),
            "--qwen-backend", "ggml",
            "--qwen-quant", configuration["QWEN_TTS_QUANT"] ?? "Q4_K_M",
            "--qwentts-library", library.path,
        ]
        process.currentDirectoryURL = repositoryRoot
        var environment = ProcessInfo.processInfo.environment
        for (key, value) in configuration { environment[key] = value }
        environment["QWEN_TTS_PYTHON"] = python.path
        environment["QWEN_TTS_BACKEND"] = "ggml"
        environment["QWENTTS_CPP_LIBRARY"] = library.path
        let libraryDirectory = library.deletingLastPathComponent().path
        environment["DYLD_LIBRARY_PATH"] = [libraryDirectory, environment["DYLD_LIBRARY_PATH"]]
            .compactMap { $0 }
            .joined(separator: ":")
        process.environment = environment
        let logsDirectory = repositoryRoot.appendingPathComponent("logs")
        try? FileManager.default.createDirectory(at: logsDirectory, withIntermediateDirectories: true)
        let logURL = logsDirectory.appendingPathComponent("macos-app-service.log")
        FileManager.default.createFile(atPath: logURL.path, contents: nil)
        let logHandle = try? FileHandle(forWritingTo: logURL)
        logHandle?.seekToEndOfFile()
        process.standardOutput = logHandle
        process.standardError = logHandle
        process.terminationHandler = { [weak self] terminated in
            DispatchQueue.main.async {
                guard let self, self.ownsProcess, self.process === terminated else { return }
                let shouldRecover = self.hasReachedReady
                self.lastExitStatus = terminated.terminationStatus
                self.ownsProcess = false
                self.process = nil
                self.log("owned service exited unexpectedly with status \(terminated.terminationStatus)")
                if shouldRecover {
                    self.onUnexpectedTermination?(terminated.terminationStatus)
                }
            }
        }
        do {
            hasReachedReady = false
            lastExitStatus = nil
            try process.run()
            log("started service process pid=\(process.processIdentifier)")
            self.process = process
            ownsProcess = true
            waitUntilReady(remainingAttempts: 80, completion: completion)
        } catch {
            log("could not start service: \(error.localizedDescription)")
            completion(.failure(error))
        }
    }

    private func waitUntilReady(
        remainingAttempts: Int,
        completion: @escaping (Result<[String: Any], Error>) -> Void
    ) {
        health { [weak self] status in
            if let status, status["state"] as? String == "ready" {
                self?.hasReachedReady = true
                completion(.success(status))
                return
            }
            if let status = self?.lastExitStatus {
                completion(.failure(ServiceError.serviceExited(status)))
                return
            }
            guard remainingAttempts > 0 else {
                self?.log("timed out while waiting for service health")
                completion(.failure(ServiceError.startTimedOut))
                return
            }
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                self?.waitUntilReady(remainingAttempts: remainingAttempts - 1, completion: completion)
            }
        }
    }

    func stopIfOwned() {
        guard ownsProcess, let process, process.isRunning else { return }
        ownsProcess = false
        self.process = nil
        process.terminate()
        log("stopped owned service process")
    }

    private func log(_ message: String) {
        let logsDirectory = repositoryRoot.appendingPathComponent("logs")
        try? FileManager.default.createDirectory(at: logsDirectory, withIntermediateDirectories: true)
        let logURL = logsDirectory.appendingPathComponent("macos-app.log")
        let line = "[\(ISO8601DateFormatter().string(from: Date()))] \(message)\n"
        if let data = line.data(using: .utf8) {
            if FileManager.default.fileExists(atPath: logURL.path),
               let handle = try? FileHandle(forWritingTo: logURL) {
                handle.seekToEndOfFile()
                try? handle.write(contentsOf: data)
                try? handle.close()
            } else {
                try? data.write(to: logURL, options: .atomic)
            }
        }
    }

    private static func readEnvironmentFile(_ url: URL) -> [String: String] {
        guard let text = try? String(contentsOf: url, encoding: .utf8) else { return [:] }
        var values: [String: String] = [:]
        for rawLine in text.split(whereSeparator: \.isNewline) {
            let line = rawLine.trimmingCharacters(in: .whitespaces)
            guard !line.isEmpty, !line.hasPrefix("#"), let separator = line.firstIndex(of: "=") else { continue }
            let key = String(line[..<separator]).trimmingCharacters(in: .whitespaces)
            var value = String(line[line.index(after: separator)...]).trimmingCharacters(in: .whitespaces)
            if value.count >= 2,
               ((value.hasPrefix("\"") && value.hasSuffix("\"")) || (value.hasPrefix("'") && value.hasSuffix("'"))) {
                value.removeFirst()
                value.removeLast()
            }
            if !key.isEmpty { values[key] = value }
        }
        return values
    }

    private static func updateEnvironmentFile(_ url: URL, values: [String: String]) throws {
        let original = (try? String(contentsOf: url, encoding: .utf8)) ?? ""
        var remaining = values
        var lines: [String] = []
        for rawLine in original.split(separator: "\n", omittingEmptySubsequences: false) {
            let line = String(rawLine)
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            guard !trimmed.hasPrefix("#"), let separator = trimmed.firstIndex(of: "=") else {
                lines.append(line)
                continue
            }
            let key = String(trimmed[..<separator]).trimmingCharacters(in: .whitespaces)
            if let value = remaining.removeValue(forKey: key) {
                lines.append("\(key)=\(value)")
            } else {
                lines.append(line)
            }
        }
        if !remaining.isEmpty {
            if lines.last?.isEmpty == false { lines.append("") }
            for key in remaining.keys.sorted() {
                lines.append("\(key)=\(remaining[key] ?? "")")
            }
        }
        try (lines.joined(separator: "\n") + "\n").write(to: url, atomically: true, encoding: .utf8)
    }
}

private enum ServiceError: LocalizedError {
    case runtimeMissing(String)
    case startTimedOut
    case invalidPort
    case insecureLANPassword
    case restartRequired
    case serviceExited(Int32)

    var errorDescription: String? {
        switch self {
        case .runtimeMissing(let root):
            return "找不到 Metal 运行环境。请先在以下目录运行 ./setup-macos.sh：\n\(root)"
        case .startTimedOut:
            return "本地语音服务在 40 秒内没有就绪。请查看 logs/macos-app-service.log。"
        case .invalidPort:
            return "端口必须在 1024 到 65535 之间。"
        case .insecureLANPassword:
            return "允许局域网访问时，密码至少需要 4 个字符，且不能使用 change-me。"
        case .restartRequired:
            return "设置已经保存，但当前服务不是由本应用启动的。请手动重启后台服务后生效。"
        case .serviceExited(let status):
            return "本地语音服务启动后立即退出（状态 \(status)）。请检查端口是否被占用及 logs/macos-app-service.log。"
        }
    }
}

final class AppDelegate: NSObject, NSApplicationDelegate, NSWindowDelegate, NSMenuDelegate {
    private let service: LocalService
    private let viewModel: NativeStudioViewModel
    private var statusItem: NSStatusItem!
    private var serviceStatusItem: NSMenuItem!
    private var modelStatusItem: NSMenuItem!
    private var voiceStatusItem: NSMenuItem!
    private var taskStatusItem: NSMenuItem!
    private var sttStatusItem: NSMenuItem!
    private var presetsMenu: NSMenu!
    private var window: NSWindow!
    private var statusTimer: Timer?
    private var isTerminating = false
    private var isForceStopping = false
    private let launchInBackground = CommandLine.arguments.contains("--background")

    override init() {
        let service = LocalService(repositoryRoot: AppDelegate.findRepositoryRoot())
        self.service = service
        self.viewModel = NativeStudioViewModel(service: service)
        super.init()
        viewModel.onPresetsChanged = { [weak self] presets in
            self?.renderPresets(presets)
        }
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        service.onUnexpectedTermination = { [weak self] status in
            guard let self, !self.isTerminating else { return }
            self.updateMenuStatus("服务异常退出，正在恢复…")
            self.viewModel.serviceReady = false
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.8) { [weak self] in
                guard let self, !self.isTerminating else { return }
                self.service.startIfNeeded { [weak self] result in
                    DispatchQueue.main.async {
                        switch result {
                        case .success(let health):
                            self?.applyServiceHealth(health)
                            self?.viewModel.updateHealth(health)
                            self?.viewModel.loadInitialData()
                        case .failure(let error):
                            self?.updateMenuStatus("自动恢复失败（\(status)）")
                            self?.presentError(error)
                        }
                    }
                }
            }
        }
        buildWindow()
        buildApplicationMenu()
        buildStatusMenu()
        updateMenuStatus("正在启动本地 Metal 服务…")
        service.startIfNeeded { [weak self] result in
            DispatchQueue.main.async {
                switch result {
                case .success(let health):
                    self?.applyServiceHealth(health)
                    self?.viewModel.updateHealth(health)
                    self?.viewModel.loadInitialData()
                    if self?.launchInBackground == false {
                        self?.showStudio()
                    }
                case .failure(let error):
                    self?.updateMenuStatus("服务启动失败")
                    self?.presentError(error)
                }
            }
        }
        statusTimer = Timer.scheduledTimer(withTimeInterval: 2, repeats: true) { [weak self] _ in
            self?.refreshHealth()
        }
    }

    func applicationWillTerminate(_ notification: Notification) {
        isTerminating = true
        statusTimer?.invalidate()
        service.stopIfOwned()
    }

    func windowShouldClose(_ sender: NSWindow) -> Bool {
        window.orderOut(nil)
        return false
    }

    private func buildWindow() {
        let content = NativeStudioView(model: viewModel)
        let hostingView = NSHostingView(rootView: content)
        window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 1280, height: 860),
            styleMask: [.titled, .closable, .miniaturizable, .resizable],
            backing: .buffered,
            defer: false
        )
        window.title = "Qwen TTS Audio Studio"
        window.contentView = hostingView
        window.contentMinSize = NSSize(width: 1080, height: 740)
        window.delegate = self
        window.setFrameAutosaveName("QwenTTSStudioWindow")
        if let visibleFrame = NSScreen.main?.visibleFrame {
            let targetSize = NSSize(
                width: min(1280, visibleFrame.width - 32),
                height: min(860, visibleFrame.height - 32)
            )
            let currentSize = window.contentLayoutRect.size
            let adjustedSize = NSSize(
                width: min(max(currentSize.width, targetSize.width), visibleFrame.width - 32),
                height: min(max(currentSize.height, targetSize.height), visibleFrame.height - 32)
            )
            if abs(currentSize.width - adjustedSize.width) > 1
                || abs(currentSize.height - adjustedSize.height) > 1 {
                window.setContentSize(adjustedSize)
            }
        }
        window.center()
    }

    private func buildApplicationMenu() {
        let mainMenu = NSMenu()

        let appMenuItem = NSMenuItem()
        let appMenu = NSMenu(title: "Qwen TTS")
        let quitItem = NSMenuItem(
            title: "退出 Qwen TTS",
            action: #selector(quit(_:)),
            keyEquivalent: "q"
        )
        quitItem.target = self
        appMenu.addItem(quitItem)
        appMenuItem.submenu = appMenu
        mainMenu.addItem(appMenuItem)

        let windowMenuItem = NSMenuItem()
        let windowMenu = NSMenu(title: "窗口")
        let closeItem = NSMenuItem(
            title: "关闭音频控制台",
            action: #selector(closeStudio(_:)),
            keyEquivalent: "w"
        )
        closeItem.target = self
        windowMenu.addItem(closeItem)
        windowMenuItem.submenu = windowMenu
        mainMenu.addItem(windowMenuItem)
        NSApp.mainMenu = mainMenu
        NSApp.windowsMenu = windowMenu
    }

    private func buildStatusMenu() {
        statusItem = NSStatusBar.system.statusItem(withLength: 31)
        if let button = statusItem.button {
            button.image = makeTTSStatusIcon()
            button.imagePosition = .imageOnly
            button.title = ""
            button.toolTip = "Qwen TTS · 正在连接"
            button.setAccessibilityLabel("Qwen TTS")
        }
        let menu = NSMenu()
        menu.delegate = self
        serviceStatusItem = makeStatusMenuItem("服务：正在连接")
        modelStatusItem = makeStatusMenuItem("当前模型：正在读取")
        voiceStatusItem = makeStatusMenuItem("当前音色：正在读取")
        taskStatusItem = makeStatusMenuItem("任务：等待服务")
        sttStatusItem = makeStatusMenuItem("转写：正在读取")
        menu.addItem(serviceStatusItem)
        menu.addItem(modelStatusItem)
        menu.addItem(voiceStatusItem)
        menu.addItem(taskStatusItem)
        menu.addItem(sttStatusItem)
        menu.addItem(.separator())
        let openItem = NSMenuItem(title: "打开音频工作台", action: #selector(openStudio(_:)), keyEquivalent: "o")
        openItem.target = self
        menu.addItem(openItem)
        let browserItem = NSMenuItem(title: "打开 Qwen 声阅", action: #selector(openNovelReader(_:)), keyEquivalent: "")
        browserItem.target = self
        menu.addItem(browserItem)
        let apiSettingsItem = NSMenuItem(
            title: "对外接口设置…",
            action: #selector(openAPISettings(_:)),
            keyEquivalent: ""
        )
        apiSettingsItem.target = self
        menu.addItem(apiSettingsItem)
        let presetItem = NSMenuItem(title: "切换语音预设", action: nil, keyEquivalent: "")
        presetsMenu = NSMenu(title: "切换语音预设")
        presetItem.submenu = presetsMenu
        menu.addItem(presetItem)
        let refreshItem = NSMenuItem(title: "刷新状态和预设", action: #selector(refreshFromMenu(_:)), keyEquivalent: "r")
        refreshItem.target = self
        menu.addItem(refreshItem)
        menu.addItem(.separator())
        let forceStopItem = NSMenuItem(
            title: "强制停止所有音频与运算",
            action: #selector(forceStopAllAudioAndComputation(_:)),
            keyEquivalent: ""
        )
        forceStopItem.target = self
        menu.addItem(forceStopItem)
        menu.addItem(.separator())
        let quitItem = NSMenuItem(title: "退出 Qwen TTS", action: #selector(quit(_:)), keyEquivalent: "q")
        quitItem.target = self
        menu.addItem(quitItem)
        statusItem.menu = menu
    }

    private func makeStatusMenuItem(_ title: String) -> NSMenuItem {
        let item = NSMenuItem(title: title, action: nil, keyEquivalent: "")
        item.isEnabled = false
        return item
    }

    private func makeTTSStatusIcon() -> NSImage {
        let size = NSSize(width: 27, height: 16)
        let image = NSImage(size: size, flipped: false) { rect in
            let style = NSMutableParagraphStyle()
            style.alignment = .center
            let attributes: [NSAttributedString.Key: Any] = [
                .font: NSFont.monospacedSystemFont(ofSize: 9.7, weight: .bold),
                .foregroundColor: NSColor.labelColor,
                .kern: -0.45,
                .paragraphStyle: style,
            ]
            let text = NSAttributedString(string: "TTS", attributes: attributes)
            let textSize = text.size()
            let textRect = NSRect(
                x: rect.midX - textSize.width / 2,
                y: rect.midY - textSize.height / 2 + 0.5,
                width: textSize.width,
                height: textSize.height
            )
            text.draw(in: textRect)
            return true
        }
        image.isTemplate = true
        image.accessibilityDescription = "TTS"
        return image
    }

    func menuWillOpen(_ menu: NSMenu) {
        refreshHealth()
        refreshPresets()
        viewModel.loadActiveServiceSettings()
        refreshVoiceStatus()
    }

    private func refreshHealth() {
        service.health { [weak self] health in
            DispatchQueue.main.async {
                if let health {
                    self?.applyServiceHealth(health)
                    self?.viewModel.updateHealth(health)
                }
                else { self?.updateMenuStatus("服务未连接") }
            }
        }
    }

    private func applyServiceHealth(_ health: [String: Any]) {
        let state = health["state"] as? String ?? "unknown"
        let profile = health["active_profile_label"] as? String ?? "等待模型"
        let scheduler = health["generation_scheduler"] as? [String: Any]
        let stt = health["stt"] as? [String: Any]
        let sttReady = stt?["ready"] as? Bool ?? false
        let active = scheduler?["active"] as? Int ?? 0
        let maximum = scheduler?["max_parallel"] as? Int ?? 0
        if state == "ready" {
            serviceStatusItem?.title = "服务：Metal 已连接"
            modelStatusItem?.title = "当前模型：\(profile.isEmpty ? viewModel.modelDisplayName : profile)"
            taskStatusItem?.title = active > 0
                ? "任务：正在生成 · GPU \(active)/\(maximum)"
                : "任务：空闲 · GPU \(active)/\(maximum)"
            sttStatusItem?.title = sttReady ? "转写：STT 已就绪" : "转写：STT 未就绪"
            statusItem.button?.toolTip = active > 0 ? "Qwen TTS · 正在生成" : "Qwen TTS · 服务就绪"
            refreshVoiceStatus()
        } else {
            serviceStatusItem?.title = "服务：\(state)"
            modelStatusItem?.title = "当前模型：\(viewModel.modelDisplayName)"
            taskStatusItem?.title = "任务：等待服务"
            sttStatusItem?.title = "转写：等待服务"
            statusItem.button?.toolTip = "Qwen TTS · \(state)"
            refreshVoiceStatus()
        }
    }

    private func updateMenuStatus(_ text: String) {
        serviceStatusItem?.title = "服务：\(text)"
    }

    private func refreshVoiceStatus() {
        let voice = viewModel.referenceName.trimmingCharacters(in: .whitespacesAndNewlines)
        voiceStatusItem?.title = "当前音色：\(voice.isEmpty ? "尚未设置" : voice)"
    }

    private func refreshPresets() {
        viewModel.refreshPresets()
    }

    private func renderPresets(_ presets: [NativePreset]) {
        presetsMenu.removeAllItems()
        if presets.isEmpty {
            let empty = NSMenuItem(title: "尚无预设，请在工作台中保存", action: nil, keyEquivalent: "")
            empty.isEnabled = false
            presetsMenu.addItem(empty)
            return
        }
        for preset in presets {
            let item = NSMenuItem(title: preset.name, action: #selector(selectPreset(_:)), keyEquivalent: "")
            item.target = self
            item.representedObject = preset.id
            item.state = preset.id == viewModel.selectedPresetID ? .on : .off
            presetsMenu.addItem(item)
        }
    }

    @objc private func selectPreset(_ sender: NSMenuItem) {
        guard let presetID = sender.representedObject as? String,
              let preset = viewModel.presets.first(where: { $0.id == presetID })
        else { return }
        viewModel.applyPreset(preset)
        showStudio()
    }

    @objc private func openStudio(_ sender: Any?) { showStudio() }

    @objc private func closeStudio(_ sender: Any?) {
        guard window.isVisible else { return }
        window.performClose(sender)
    }

    private func showStudio() {
        NSApp.activate(ignoringOtherApps: true)
        window.makeKeyAndOrderFront(nil)
    }

    @objc private func openNovelReader(_ sender: Any?) {
        let candidates = [
            URL(fileURLWithPath: "/Applications/QwenReader.app"),
            Bundle.main.bundleURL
                .deletingLastPathComponent()
                .appendingPathComponent("QwenReader.app"),
            service.repositoryRoot
                .appendingPathComponent("dist/QwenReader.app"),
        ]
        if let readerApp = candidates.first(where: {
            FileManager.default.fileExists(atPath: $0.path)
        }) {
            let configuration = NSWorkspace.OpenConfiguration()
            configuration.activates = true
            NSWorkspace.shared.openApplication(at: readerApp, configuration: configuration)
            return
        }
        NSWorkspace.shared.open(service.baseURL.appendingPathComponent("reader"))
    }

    @objc private func openAPISettings(_ sender: Any?) {
        showStudio()
        viewModel.externalSettingsStatus = ""
        viewModel.externalAPISettingsPresented = true
    }

    @objc private func refreshFromMenu(_ sender: Any?) {
        refreshHealth()
        refreshPresets()
    }

    @objc private func forceStopAllAudioAndComputation(_ sender: Any?) {
        guard !isForceStopping else { return }
        isForceStopping = true
        updateMenuStatus("正在强制停止所有音频与运算…")
        viewModel.forceStopAllAudioAndComputation { [weak self] result in
            DispatchQueue.main.async {
                self?.isForceStopping = false
                switch result {
                case .success:
                    self?.updateMenuStatus("已强制停止所有音频与运算")
                    self?.statusItem.button?.toolTip = "Qwen TTS · 已停止所有任务"
                case .failure:
                    self?.updateMenuStatus("强制停止请求失败")
                    self?.statusItem.button?.toolTip = "Qwen TTS · 强制停止请求失败"
                }
            }
        }
    }

    @objc private func quit(_ sender: Any?) { NSApp.terminate(nil) }

    private func presentError(_ error: Error) {
        let alert = NSAlert(error: error)
        alert.runModal()
    }

    private static func findRepositoryRoot() -> URL {
        let arguments = CommandLine.arguments
        if let index = arguments.firstIndex(of: "--repo-root"), index + 1 < arguments.count {
            let root = URL(fileURLWithPath: arguments[index + 1]).standardizedFileURL
            if isRepositoryRoot(root) { return root }
        }
        if let root = ProcessInfo.processInfo.environment["QWEN_TTS_REPO_ROOT"] {
            let url = URL(fileURLWithPath: root).standardizedFileURL
            if isRepositoryRoot(url) { return url }
        }
        let bundle = Bundle.main.bundleURL
        if let embeddedRoot = Bundle.main.url(
            forResource: "repository-root",
            withExtension: "txt"
        ), let path = try? String(contentsOf: embeddedRoot, encoding: .utf8)
            .trimmingCharacters(in: .whitespacesAndNewlines) {
            let url = URL(fileURLWithPath: path).standardizedFileURL
            if isRepositoryRoot(url) { return url }
        }
        let candidates = [
            URL(fileURLWithPath: FileManager.default.currentDirectoryPath),
            bundle.deletingLastPathComponent().deletingLastPathComponent(),
            bundle.deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent(),
        ]
        for candidate in candidates where isRepositoryRoot(candidate) { return candidate }
        return URL(fileURLWithPath: FileManager.default.currentDirectoryPath).standardizedFileURL
    }

    private static func isRepositoryRoot(_ url: URL) -> Bool {
        FileManager.default.fileExists(atPath: url.appendingPathComponent("clis/qwen_tts_app.py").path)
    }
}

@main
enum QwenTTSApplication {
    static func main() {
        let app = NSApplication.shared
        app.setActivationPolicy(.accessory)
        let delegate = AppDelegate()
        app.delegate = delegate
        app.run()
    }
}
