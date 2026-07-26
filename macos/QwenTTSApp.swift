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

    func perform(
        _ path: String,
        method: String = "GET",
        body: Data? = nil,
        contentType: String? = nil,
        timeout: TimeInterval = 30,
        completion: @escaping (Data?, HTTPURLResponse?) -> Void
    ) {
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
        completion: @escaping (Result<[String: Any], Error>) -> Void
    ) {
        guard (1024 ... 65_535).contains(newPort) else {
            completion(.failure(ServiceError.invalidPort))
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
        do {
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
                completion(.success(status))
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
        process.terminate()
        log("stopped owned service process")
        ownsProcess = false
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
    case restartRequired

    var errorDescription: String? {
        switch self {
        case .runtimeMissing(let root):
            return "找不到 Metal 运行环境。请先在以下目录运行 ./setup-macos.sh：\n\(root)"
        case .startTimedOut:
            return "本地语音服务在 40 秒内没有就绪。请查看 logs/macos-app-service.log。"
        case .invalidPort:
            return "端口必须在 1024 到 65535 之间。"
        case .restartRequired:
            return "设置已经保存，但当前服务不是由本应用启动的。请手动重启后台服务后生效。"
        }
    }
}

final class AppDelegate: NSObject, NSApplicationDelegate, NSWindowDelegate, NSMenuDelegate {
    private let service: LocalService
    private let viewModel: NativeStudioViewModel
    private var statusItem: NSStatusItem!
    private var statusLabel: NSMenuItem!
    private var presetsMenu: NSMenu!
    private var window: NSWindow!
    private var statusTimer: Timer?

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
        buildWindow()
        buildStatusMenu()
        updateMenuStatus("正在启动本地 Metal 服务…")
        service.startIfNeeded { [weak self] result in
            DispatchQueue.main.async {
                switch result {
                case .success(let health):
                    self?.applyServiceHealth(health)
                    self?.viewModel.updateHealth(health)
                    self?.viewModel.loadInitialData()
                    self?.showStudio()
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

    private func buildStatusMenu() {
        statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
        statusItem.button?.title = "TTS …"
        let menu = NSMenu()
        menu.delegate = self
        statusLabel = NSMenuItem(title: "正在连接本地服务…", action: nil, keyEquivalent: "")
        statusLabel.isEnabled = false
        menu.addItem(statusLabel)
        menu.addItem(.separator())
        let openItem = NSMenuItem(title: "打开音频工作台", action: #selector(openStudio(_:)), keyEquivalent: "o")
        openItem.target = self
        menu.addItem(openItem)
        let browserItem = NSMenuItem(title: "打开小说阅读器", action: #selector(openNovelReader(_:)), keyEquivalent: "")
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
        let quitItem = NSMenuItem(title: "退出 Qwen TTS", action: #selector(quit(_:)), keyEquivalent: "q")
        quitItem.target = self
        menu.addItem(quitItem)
        statusItem.menu = menu
    }

    func menuWillOpen(_ menu: NSMenu) {
        refreshHealth()
        refreshPresets()
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
            statusItem.button?.title = active > 0 ? "语音 ●" : "语音 ✓"
            let sttLabel = sttReady ? "STT ✓" : "STT 未就绪"
            updateMenuStatus("TTS Metal · \(sttLabel) · \(profile) · GPU \(active)/\(maximum)")
        } else {
            statusItem.button?.title = "语音 …"
            updateMenuStatus("服务状态：\(state)")
        }
    }

    private func updateMenuStatus(_ text: String) {
        statusLabel?.title = text
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

    private func showStudio() {
        NSApp.activate(ignoringOtherApps: true)
        window.makeKeyAndOrderFront(nil)
    }

    @objc private func openNovelReader(_ sender: Any?) {
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
