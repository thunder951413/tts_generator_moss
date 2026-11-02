import AppKit
import CryptoKit
import Foundation
import SwiftUI

private let serviceSessionCookie = "qwen_tts_service_session"

final class LocalService {
    /// Lifecycle of the service this app manages. Mutated only on the main queue.
    enum LifecycleState {
        case idle
        case starting
        case running
        case stopping
    }

    let repositoryRoot: URL
    private(set) var configuration: [String: String]
    private(set) var port: Int
    private(set) var ownsProcess = false
    private var process: Process?
    // hasReachedReady / lastExitStatus / lifecycle state are confined to the main queue.
    private var hasReachedReady = false
    private var lastExitStatus: Int32?
    private(set) var lifecycleState: LifecycleState = .idle
    private var pendingStartCompletions: [(Result<[String: Any], Error>) -> Void] = []
    var onUnexpectedTermination: ((Int32) -> Void)?

    private static let logQueue = DispatchQueue(label: "qwen-local-service-log")
    // Both are only touched on logQueue.
    private static let logTimestampFormatter = ISO8601DateFormatter()
    private static var logHandles: [String: FileHandle] = [:]

    /// True while a start (or a restart after changed settings) is in progress.
    var isStarting: Bool {
        lifecycleState == .starting || lifecycleState == .stopping
    }

    var baseURL: URL {
        URL(string: "http://127.0.0.1:\(port)/")!
    }

    var ownsRunningProcess: Bool {
        ownsProcess && (process?.isRunning == true)
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

    /// Variant used by long-running imports where a missing HTTP response must
    /// remain distinguishable from a server-side JSON error.
    func performResult(
        _ path: String,
        method: String = "GET",
        body: Data? = nil,
        contentType: String? = nil,
        timeout: TimeInterval = 30,
        completion: @escaping (Data?, HTTPURLResponse?, Error?) -> Void
    ) {
        let request = authorizedRequest(
            path,
            method: method,
            body: body,
            contentType: contentType,
            timeout: timeout
        )
        URLSession.shared.dataTask(with: request) { data, response, error in
            completion(data, response as? HTTPURLResponse, error)
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

    static func sourceBuildID(repositoryRoot: URL) -> String? {
        let repositoryRoot = repositoryRoot.resolvingSymlinksInPath().standardizedFileURL
        var files: [(relative: String, url: URL)] = []
        for directory in ["qwen_tts_service", "web/novel_reader"] {
            let directoryURL = repositoryRoot.appendingPathComponent(directory)
            guard let entries = FileManager.default.enumerator(atPath: directoryURL.path) else { return nil }
            for case let entry as String in entries {
                let relative = "\(directory)/\(entry)"
                let file = repositoryRoot.appendingPathComponent(relative)
                if ["py", "js", "html", "css"].contains(file.pathExtension),
                   (try? file.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile) == true {
                    files.append((relative, file))
                }
            }
        }
        var digest = SHA256()
        for file in files.sorted(by: { $0.relative < $1.relative }) {
            guard let data = try? Data(contentsOf: file.url) else { return nil }
            digest.update(data: Data(file.relative.utf8))
            digest.update(data: Data([0]))
            digest.update(data: data)
            digest.update(data: Data([0]))
        }
        return digest.finalize().map { String(format: "%02x", $0) }.joined()
    }

    func compatibilityError(_ status: [String: Any]) -> Error? {
        guard let identity = status["service"] as? [String: Any],
              identity["name"] as? String == "qwen-local-speech",
              identity["api_version"] as? Int == 2,
              let capabilities = identity["capabilities"] as? [String],
              Set(["worker-auth", "bounded-worker-recovery", "resumable-reader", "storage-policy"]).isSubset(of: Set(capabilities)),
              let buildID = Self.sourceBuildID(repositoryRoot: repositoryRoot),
              identity["build_id"] as? String == buildID else {
            return ServiceError.incompatibleService
        }
        return nil
    }

    /// Starts (or attaches to) the local service. Concurrent callers are coalesced onto the
    /// one in-flight start so a second process can never replace and orphan the first.
    /// Completions are delivered on the main queue.
    func startIfNeeded(completion: @escaping (Result<[String: Any], Error>) -> Void) {
        guard Thread.isMainThread else {
            DispatchQueue.main.async { [weak self] in
                self?.startIfNeeded(completion: completion)
            }
            return
        }
        pendingStartCompletions.append(completion)
        guard pendingStartCompletions.count == 1 else {
            log("service start already in progress; waiting for it")
            return
        }
        lifecycleState = .starting
        log("checking local service")
        health { [weak self] status in
            guard let self else { return }
            // Hashing the source tree is comparatively slow, keep it off the main queue.
            let incompatibility: Error? = status.flatMap { self.compatibilityError($0) }
            DispatchQueue.main.async {
                if let status {
                    if let incompatibility {
                        self.finishStart(.failure(incompatibility))
                        return
                    }
                    self.log("using existing local service")
                    self.finishStart(.success(status))
                    return
                }
                if self.ownsProcess, self.process?.isRunning == true {
                    // Our own process is alive but not answering yet; wait for it instead of
                    // launching a second one on the same port.
                    self.log("owned service process still running; waiting for health")
                    self.waitUntilReady(remainingAttempts: 660) { [weak self] result in
                        self?.finishStart(result)
                    }
                    return
                }
                self.log("local service unavailable; starting owned process")
                self.start { [weak self] result in
                    self?.finishStart(result)
                }
            }
        }
    }

    private func finishStart(_ result: Result<[String: Any], Error>) {
        guard Thread.isMainThread else {
            DispatchQueue.main.async { [weak self] in self?.finishStart(result) }
            return
        }
        switch result {
        case .success:
            lifecycleState = .running
        case .failure:
            lifecycleState = ownsRunningProcess ? .running : .idle
        }
        let completions = pendingStartCompletions
        pendingStartCompletions = []
        for completion in completions {
            completion(result)
        }
    }

    func applyExternalSettings(
        exposeToLAN: Bool,
        port newPort: Int,
        password: String,
        corsOrigins: String,
        completion: @escaping (Result<[String: Any], Error>) -> Void
    ) {
        guard pendingStartCompletions.isEmpty, lifecycleState != .stopping else {
            completion(.failure(ServiceError.serviceBusy))
            return
        }
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
        // Any start requested while the old process shuts down joins this restart.
        lifecycleState = .stopping
        pendingStartCompletions.append(completion)
        runningProcess.terminate()
        log("restarting owned service after external API settings changed")
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            runningProcess.waitUntilExit()
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.35) {
                guard let self else { return }
                self.lifecycleState = .starting
                self.start { [weak self] result in
                    self?.finishStart(result)
                }
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
        Self.rotateServiceLogIfNeeded(logURL)
        if !FileManager.default.fileExists(atPath: logURL.path) {
            FileManager.default.createFile(atPath: logURL.path, contents: nil)
        }
        let logHandle = try? FileHandle(forWritingTo: logURL)
        logHandle?.seekToEndOfFile()
        process.standardOutput = logHandle
        process.standardError = logHandle
        process.terminationHandler = { [weak self] terminated in
            // The child keeps its own descriptor; release the app's copy.
            try? logHandle?.close()
            DispatchQueue.main.async {
                guard let self, self.ownsProcess, self.process === terminated else { return }
                let shouldRecover = self.hasReachedReady
                self.lastExitStatus = terminated.terminationStatus
                self.ownsProcess = false
                self.process = nil
                if self.lifecycleState == .running {
                    self.lifecycleState = .idle
                }
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
            waitUntilReady(remainingAttempts: 660, completion: completion)
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
            guard let self else { return }
            let readyStatus: [String: Any]?
            if let status, ["ready", "degraded"].contains(status["state"] as? String ?? "") {
                readyStatus = status
            } else {
                readyStatus = nil
            }
            let incompatibility: Error? = readyStatus.flatMap { self.compatibilityError($0) }
            // Readiness bookkeeping is shared with the termination handler; keep it on main.
            DispatchQueue.main.async {
                if let readyStatus {
                    if let incompatibility {
                        completion(.failure(incompatibility))
                        return
                    }
                    self.hasReachedReady = true
                    completion(.success(readyStatus))
                    return
                }
                if let exitStatus = self.lastExitStatus {
                    completion(.failure(ServiceError.serviceExited(exitStatus)))
                    return
                }
                guard remainingAttempts > 0 else {
                    self.log("timed out while waiting for service health")
                    completion(.failure(ServiceError.startTimedOut))
                    return
                }
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) { [weak self] in
                    self?.waitUntilReady(remainingAttempts: remainingAttempts - 1, completion: completion)
                }
            }
        }
    }

    func stopIfOwned() {
        guard ownsProcess, let process, process.isRunning else { return }
        ownsProcess = false
        self.process = nil
        lifecycleState = .idle
        if !pendingStartCompletions.isEmpty {
            // The termination handler ignores processes we no longer own, so tell a pending
            // waitUntilReady loop directly instead of letting it poll for minutes.
            lastExitStatus = SIGTERM
        }
        process.terminate()
        log("stopped owned service process")
    }

    /// Keeps one previous generation of the service log once it grows past 10 MB.
    private static func rotateServiceLogIfNeeded(_ logURL: URL) {
        guard let attributes = try? FileManager.default.attributesOfItem(atPath: logURL.path),
              let size = attributes[.size] as? NSNumber,
              size.int64Value > 10 * 1024 * 1024
        else { return }
        let rotatedURL = logURL.appendingPathExtension("1")
        try? FileManager.default.removeItem(at: rotatedURL)
        try? FileManager.default.moveItem(at: logURL, to: rotatedURL)
    }

    private func log(_ message: String) {
        let logsDirectory = repositoryRoot.appendingPathComponent("logs")
        let logURL = logsDirectory.appendingPathComponent("macos-app.log")
        let date = Date()
        LocalService.logQueue.async {
            let line = "[\(LocalService.logTimestampFormatter.string(from: date))] \(message)\n"
            guard let data = line.data(using: .utf8) else { return }
            let key = logURL.path
            var openHandle = LocalService.logHandles[key]
            if openHandle == nil {
                try? FileManager.default.createDirectory(at: logsDirectory, withIntermediateDirectories: true)
                if !FileManager.default.fileExists(atPath: key) {
                    FileManager.default.createFile(atPath: key, contents: nil)
                }
                if let created = try? FileHandle(forWritingTo: logURL) {
                    created.seekToEndOfFile()
                    LocalService.logHandles[key] = created
                    openHandle = created
                }
            }
            guard let handle = openHandle else { return }
            try? handle.write(contentsOf: data)
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

/// Single adaptive poller for `api/health`. Every consumer (menu bar, studio, STT workbench)
/// observes the latest snapshot instead of issuing its own requests. Main-queue only.
final class ServiceHealthMonitor {
    private let service: LocalService
    private var observers: [([String: Any]?) -> Void] = []
    private var timer: Timer?
    private var requestInFlight = false
    private var refreshQueuedWhileInFlight = false
    private var consecutiveFailures = 0
    private var running = false
    private(set) var latest: [String: Any]?

    /// True while something the user is waiting on is changing (start/stop/toggle/generation).
    var isTransitioning: () -> Bool = { false }
    /// True while a window is visible or the status menu is open.
    var isForeground: () -> Bool = { false }

    init(service: LocalService) {
        self.service = service
    }

    func addObserver(_ observer: @escaping ([String: Any]?) -> Void) {
        observers.append(observer)
    }

    func start() {
        guard !running else { return }
        running = true
        schedule(after: 0.5)
    }

    func stop() {
        running = false
        timer?.invalidate()
        timer = nil
    }

    /// Requests one immediate refresh. Bursts collapse into the request already in flight
    /// plus at most one follow-up.
    func refreshNow() {
        guard running else { return }
        if requestInFlight {
            refreshQueuedWhileInFlight = true
            return
        }
        timer?.invalidate()
        timer = nil
        poll()
    }

    /// Pulls the next tick forward when activity starts (never pushes it back).
    func refreshSoon() {
        guard running, !requestInFlight else { return }
        let soon = Date().addingTimeInterval(1.5)
        if let fireDate = timer?.fireDate, fireDate <= soon { return }
        schedule(after: 1.5)
    }

    private func schedule(after interval: TimeInterval) {
        timer?.invalidate()
        let next = Timer(timeInterval: interval, repeats: false) { [weak self] _ in
            guard let self else { return }
            self.timer = nil
            if !self.requestInFlight {
                self.poll()
            }
        }
        next.tolerance = max(0.1, interval * 0.2)
        RunLoop.main.add(next, forMode: .common)
        timer = next
    }

    private func poll() {
        requestInFlight = true
        service.health { [weak self] payload in
            DispatchQueue.main.async {
                guard let self else { return }
                self.requestInFlight = false
                self.latest = payload
                self.consecutiveFailures = payload == nil ? self.consecutiveFailures + 1 : 0
                for observer in self.observers {
                    observer(payload)
                }
                guard self.running else { return }
                if self.refreshQueuedWhileInFlight {
                    self.refreshQueuedWhileInFlight = false
                    self.poll()
                    return
                }
                self.schedule(after: self.nextInterval(payload))
            }
        }
    }

    private func nextInterval(_ payload: [String: Any]?) -> TimeInterval {
        let foreground = isForeground()
        if isTransitioning() {
            return 1.5
        }
        guard let payload else {
            // Unreachable: back off 2 → 4 → 8 → 30 s.
            let backoff: [TimeInterval] = [2, 4, 8, 30]
            return backoff[min(max(consecutiveFailures, 1), backoff.count) - 1]
        }
        let state = payload["state"] as? String ?? "unknown"
        let scheduler = payload["generation_scheduler"] as? [String: Any]
        let active = (scheduler?["active"] as? NSNumber)?.intValue ?? 0
        var waiting = 0
        for key in ["waiting_interactive", "waiting_document", "waiting_external", "waiting_stt"] {
            let count = (scheduler?[key] as? NSNumber)?.intValue ?? 0
            waiting += count
        }
        let sttActive = (scheduler?["stt_active"] as? NSNumber)?.boolValue ?? false
        let ttsEnabled = (payload["tts_enabled"] as? NSNumber)?.boolValue ?? true
        let settling = ttsEnabled && ["loading", "starting", "recovering", "restarting"].contains(state)
        if active > 0 || waiting > 0 || sttActive || settling {
            return foreground ? 1.5 : 5
        }
        return foreground ? 5 : 30
    }
}

private enum ServiceError: LocalizedError {
    case incompatibleService
    case runtimeMissing(String)
    case startTimedOut
    case invalidPort
    case insecureLANPassword
    case restartRequired
    case serviceNotOwned
    case serviceExited(Int32)
    case serviceBusy

    var errorDescription: String? {
        switch self {
        case .incompatibleService:
            return "当前端口的服务版本与此应用目录不一致。请重启该目录的后台服务，再启动应用。"
        case .runtimeMissing(let root):
            return "找不到 Metal 运行环境。请先在以下目录运行 ./setup-macos.sh：\n\(root)"
        case .startTimedOut:
            return "本地语音服务在 330 秒内没有就绪。请查看模型下载进度及 logs/macos-app-service.log。"
        case .invalidPort:
            return "端口必须在 1024 到 65535 之间。"
        case .insecureLANPassword:
            return "允许局域网访问时，密码至少需要 4 个字符，且不能使用 change-me。"
        case .restartRequired:
            return "设置已经保存，但当前服务不是由本应用启动的。请手动重启后台服务后生效。"
        case .serviceNotOwned:
            return "当前服务不是由本应用启动，不能从菜单栏停止。"
        case .serviceExited(let status):
            return "本地语音服务启动后立即退出（状态 \(status)）。请检查端口是否被占用及 logs/macos-app-service.log。"
        case .serviceBusy:
            return "本地语音服务正在启动或重启，请稍后再试。"
        }
    }
}

final class AppDelegate: NSObject, NSApplicationDelegate, NSWindowDelegate, NSMenuDelegate {
    private let service: LocalService
    private let viewModel: NativeStudioViewModel
    private let sttViewModel: STTWorkbenchViewModel
    private var statusItem: NSStatusItem!
    private var serviceStatusItem: NSMenuItem!
    private var modelStatusItem: NSMenuItem!
    private var voiceStatusItem: NSMenuItem!
    private var taskStatusItem: NSMenuItem!
    private var sttStatusItem: NSMenuItem!
    private var startServiceMenuItem: NSMenuItem!
    private var stopServiceMenuItem: NSMenuItem!
    private var presetsMenu: NSMenu!
    private var window: NSWindow!
    private var sttWindow: NSWindow!
    private let healthMonitor: ServiceHealthMonitor
    private var statusMenuIsOpen = false
    private var lastHealthReachable = false
    private var lastMenuMetadataRefresh: Date?
    private var isTerminating = false
    private var isForceStopping = false
    private let launchInBackground = CommandLine.arguments.contains("--background")

    override init() {
        let service = LocalService(repositoryRoot: AppDelegate.findRepositoryRoot())
        self.service = service
        self.viewModel = NativeStudioViewModel(service: service)
        self.sttViewModel = STTWorkbenchViewModel(service: service)
        self.healthMonitor = ServiceHealthMonitor(service: service)
        super.init()
        healthMonitor.addObserver { [weak self] health in
            self?.handleHealthSnapshot(health)
        }
        healthMonitor.isTransitioning = { [weak self] in
            guard let self else { return false }
            return self.service.isStarting
                || self.isForceStopping
                || self.viewModel.isTogglingTTS
                || self.viewModel.isTogglingSTT
                || self.viewModel.isGenerating
                || self.sttViewModel.isProcessing
        }
        healthMonitor.isForeground = { [weak self] in
            guard let self else { return false }
            return self.statusMenuIsOpen
                || (self.window?.isVisible ?? false)
                || (self.sttWindow?.isVisible ?? false)
        }
        sttViewModel.requestHealthRefresh = { [weak self] in
            self?.healthMonitor.refreshNow()
        }
        viewModel.onTTSServiceToggleRequested = { [weak self] in
            self?.requestTTSRuntimeToggle()
        }
        viewModel.onSTTServiceToggleRequested = { [weak self] in
            self?.requestSTTRuntimeToggle()
        }
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
                            // Keep unsaved editor settings; only refresh metadata.
                            self?.viewModel.reloadAfterServiceRestart()
                        case .failure(let error):
                            self?.updateMenuStatus("自动恢复失败（\(status)）")
                            self?.presentError(error)
                        }
                    }
                }
            }
        }
        buildWindow()
        buildSTTWindow()
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
        healthMonitor.start()
        NSWorkspace.shared.notificationCenter.addObserver(
            self,
            selector: #selector(systemDidWake(_:)),
            name: NSWorkspace.didWakeNotification,
            object: nil
        )
    }

    @objc private func systemDidWake(_ notification: Notification) {
        healthMonitor.refreshNow()
    }

    func applicationWillTerminate(_ notification: Notification) {
        isTerminating = true
        healthMonitor.stop()
        NSWorkspace.shared.notificationCenter.removeObserver(self)
        service.stopIfOwned()
    }

    func windowShouldClose(_ sender: NSWindow) -> Bool {
        sender.orderOut(nil)
        return false
    }

    func windowDidBecomeKey(_ notification: Notification) {
        healthMonitor.refreshNow()
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

    private func buildSTTWindow() {
        let content = STTWorkbenchView(model: sttViewModel)
        let hostingView = NSHostingView(rootView: content)
        sttWindow = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 1120, height: 760),
            styleMask: [.titled, .closable, .miniaturizable, .resizable],
            backing: .buffered,
            defer: false
        )
        sttWindow.title = "Qwen STT 语音转文字"
        sttWindow.contentView = hostingView
        sttWindow.contentMinSize = NSSize(width: 980, height: 680)
        sttWindow.delegate = self
        sttWindow.setFrameAutosaveName("QwenSTTWorkbenchWindow")
        sttWindow.center()
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

        let workspaceMenuItem = NSMenuItem()
        let workspaceMenu = NSMenu(title: "工作台")
        let openAudioItem = NSMenuItem(
            title: "打开音频工作台",
            action: #selector(openStudio(_:)),
            keyEquivalent: "o"
        )
        openAudioItem.target = self
        workspaceMenu.addItem(openAudioItem)
        let openSTTItem = NSMenuItem(
            title: "打开语音转文字工作台",
            action: #selector(openSTTWorkbench(_:)),
            keyEquivalent: "t"
        )
        openSTTItem.keyEquivalentModifierMask = [.command, .shift]
        openSTTItem.target = self
        workspaceMenu.addItem(openSTTItem)
        workspaceMenuItem.submenu = workspaceMenu
        mainMenu.addItem(workspaceMenuItem)

        let windowMenuItem = NSMenuItem()
        let windowMenu = NSMenu(title: "窗口")
        let closeItem = NSMenuItem(
            title: "关闭当前窗口",
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
        let sttItem = NSMenuItem(
            title: "打开语音转文字工作台",
            action: #selector(openSTTWorkbench(_:)),
            keyEquivalent: "t"
        )
        sttItem.keyEquivalentModifierMask = [.command, .shift]
        sttItem.target = self
        menu.addItem(sttItem)
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
        let presetItem = NSMenuItem(title: "服务默认音色", action: nil, keyEquivalent: "")
        presetsMenu = NSMenu(title: "服务默认音色")
        presetItem.submenu = presetsMenu
        menu.addItem(presetItem)
        let refreshItem = NSMenuItem(title: "刷新状态和预设", action: #selector(refreshFromMenu(_:)), keyEquivalent: "r")
        refreshItem.target = self
        menu.addItem(refreshItem)
        menu.addItem(.separator())
        startServiceMenuItem = NSMenuItem(
            title: "启动本地服务",
            action: #selector(startLocalService(_:)),
            keyEquivalent: "s"
        )
        startServiceMenuItem.keyEquivalentModifierMask = [.command, .option]
        startServiceMenuItem.target = self
        menu.addItem(startServiceMenuItem)
        stopServiceMenuItem = NSMenuItem(
            title: "停止本地服务",
            action: #selector(stopLocalService(_:)),
            keyEquivalent: "s"
        )
        stopServiceMenuItem.keyEquivalentModifierMask = [.command, .shift]
        stopServiceMenuItem.target = self
        menu.addItem(stopServiceMenuItem)
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
        updateServiceMenuActions(isReachable: false)
    }

    private func makeStatusMenuItem(_ title: String) -> NSMenuItem {
        let item = NSMenuItem(title: title, action: nil, keyEquivalent: "")
        item.isEnabled = false
        return item
    }

    private func makeTTSStatusIcon() -> NSImage {
        // A speech bubble with a waveform stays legible at menu-bar scale.
        // Draw in points so AppKit can rasterize the template at any display scale.
        let size = NSSize(width: 20, height: 18)
        let image = NSImage(size: size, flipped: false) { _ in
            NSColor.black.setStroke()
            let bubble = NSBezierPath()
            bubble.lineWidth = 1.5
            bubble.lineCapStyle = .round
            bubble.lineJoinStyle = .round
            bubble.move(to: NSPoint(x: 5.25, y: 16.25))
            bubble.line(to: NSPoint(x: 14.75, y: 16.25))
            bubble.curve(to: NSPoint(x: 18.25, y: 12.75),
                         controlPoint1: NSPoint(x: 16.75, y: 16.25),
                         controlPoint2: NSPoint(x: 18.25, y: 14.75))
            bubble.line(to: NSPoint(x: 18.25, y: 8.5))
            bubble.curve(to: NSPoint(x: 14.75, y: 5),
                         controlPoint1: NSPoint(x: 18.25, y: 6.5),
                         controlPoint2: NSPoint(x: 16.75, y: 5))
            bubble.line(to: NSPoint(x: 8.5, y: 5))
            bubble.line(to: NSPoint(x: 4.75, y: 2.25))
            bubble.line(to: NSPoint(x: 4.75, y: 5.05))
            bubble.curve(to: NSPoint(x: 1.75, y: 8.5),
                         controlPoint1: NSPoint(x: 2.95, y: 5.25),
                         controlPoint2: NSPoint(x: 1.75, y: 6.65))
            bubble.line(to: NSPoint(x: 1.75, y: 12.75))
            bubble.curve(to: NSPoint(x: 5.25, y: 16.25),
                         controlPoint1: NSPoint(x: 1.75, y: 14.75),
                         controlPoint2: NSPoint(x: 3.25, y: 16.25))
            bubble.close()
            bubble.stroke()

            let waveform = NSBezierPath()
            waveform.lineWidth = 1.75
            waveform.lineCapStyle = .round
            for (x, height): (CGFloat, CGFloat) in [(6.5, 3), (10, 6), (13.5, 4)] {
                waveform.move(to: NSPoint(x: x, y: 10.5 - height / 2))
                waveform.line(to: NSPoint(x: x, y: 10.5 + height / 2))
            }
            waveform.stroke()
            return true
        }
        image.isTemplate = true
        image.accessibilityDescription = "Qwen TTS 语音工作台"
        return image
    }

    func menuWillOpen(_ menu: NSMenu) {
        statusMenuIsOpen = true
        healthMonitor.refreshNow()
        refreshVoiceStatus()
        // Presets and the applied voice change rarely; avoid a request burst on every open.
        let now = Date()
        if let last = lastMenuMetadataRefresh, now.timeIntervalSince(last) < 15 { return }
        lastMenuMetadataRefresh = now
        refreshPresets()
        viewModel.loadActiveServiceSettings(applyingToEditor: false)
    }

    func menuDidClose(_ menu: NSMenu) {
        statusMenuIsOpen = false
    }

    private func refreshHealth() {
        healthMonitor.refreshNow()
    }

    private func handleHealthSnapshot(_ health: [String: Any]?) {
        sttViewModel.applyHealth(health)
        if let health {
            applyServiceHealth(health)
            viewModel.updateHealth(health)
        } else {
            viewModel.markServiceUnreachable()
            updateMenuStatus(service.isStarting ? "正在启动本地服务…" : "服务未连接")
            updateServiceMenuActions(isReachable: false)
        }
    }

    private func applyServiceHealth(_ health: [String: Any]) {
        let state = health["state"] as? String ?? "unknown"
        let profile = health["active_profile_label"] as? String ?? "等待模型"
        let scheduler = health["generation_scheduler"] as? [String: Any]
        let stt = health["stt"] as? [String: Any]
        let ttsEnabled = health["tts_enabled"] as? Bool ?? true
        let sttReady = stt?["ready"] as? Bool ?? false
        let active = scheduler?["active"] as? Int ?? 0
        let maximum = scheduler?["max_parallel"] as? Int ?? 0
        if state == "ready" || state == "degraded" || !ttsEnabled {
            serviceStatusItem?.title = ttsEnabled ? "服务：Metal 已连接" : "服务：HTTP 已连接 · TTS 已停止"
            modelStatusItem?.title = "当前模型：\(profile.isEmpty ? viewModel.modelDisplayName : profile)"
            taskStatusItem?.title = ttsEnabled && active > 0
                ? "任务：正在生成 · GPU \(active)/\(maximum)"
                : (ttsEnabled ? "任务：空闲 · GPU \(active)/\(maximum)" : "任务：TTS 已停止")
            let waitingSTT = scheduler?["waiting_stt"] as? Int ?? 0
            let activeSTT = scheduler?["stt_active"] as? Bool ?? false
            sttStatusItem?.title = activeSTT ? "转写：正在处理"
                : (waitingSTT > 0 ? "转写：等待计算 · \(waitingSTT) 个请求"
                    : (sttReady ? "转写：STT 已就绪" : "转写：STT 未就绪"))
            statusItem.button?.toolTip = ttsEnabled
                ? (active > 0 ? "Qwen TTS · 正在生成" : "Qwen TTS · 服务就绪")
                : "Qwen TTS · TTS 已停止"
            refreshVoiceStatus()
        } else {
            serviceStatusItem?.title = "服务：\(state)"
            modelStatusItem?.title = "当前模型：\(viewModel.modelDisplayName)"
            taskStatusItem?.title = "任务：等待服务"
            sttStatusItem?.title = "转写：等待服务"
            statusItem.button?.toolTip = "Qwen TTS · \(state)"
            refreshVoiceStatus()
        }
        updateServiceMenuActions(isReachable: true)
    }

    private func updateServiceMenuActions(isReachable: Bool) {
        lastHealthReachable = isReachable
        startServiceMenuItem?.isEnabled = !isReachable && !service.isStarting
        stopServiceMenuItem?.isEnabled = isReachable && service.ownsRunningProcess
    }

    /// The status menu auto-enables items, so the explicit isEnabled flags above only take
    /// effect through this validation hook.
    @objc func validateMenuItem(_ menuItem: NSMenuItem) -> Bool {
        if menuItem === startServiceMenuItem {
            return !lastHealthReachable && !service.isStarting
        }
        if menuItem === stopServiceMenuItem {
            return lastHealthReachable && service.ownsRunningProcess && !service.isStarting
        }
        return true
    }

    private func updateMenuStatus(_ text: String) {
        serviceStatusItem?.title = "服务：\(text)"
    }

    private func presentConfirmation(
        title: String,
        message: String,
        confirmTitle: String,
        destructive: Bool = false,
        action: @escaping () -> Void
    ) {
        let alert = NSAlert()
        alert.messageText = title
        alert.informativeText = message
        let confirm = alert.addButton(withTitle: confirmTitle)
        if destructive { confirm.hasDestructiveAction = true }
        alert.addButton(withTitle: "取消")
        guard alert.runModal() == .alertFirstButtonReturn else { return }
        action()
    }

    private func requestTTSRuntimeToggle() {
        let shouldStart = !viewModel.ttsEnabled
        presentConfirmation(
            title: shouldStart ? "启动 TTS 服务？" : "停止 TTS 服务？",
            message: shouldStart
                ? "将预热当前模型，之后可以继续生成语音。"
                : "会停止当前语音生成与播放，并释放 TTS 模型占用的内存。STT 仍可继续使用。",
            confirmTitle: shouldStart ? "启动 TTS" : "停止 TTS",
            destructive: !shouldStart
        ) { [weak self] in
            guard let self else { return }
            self.updateMenuStatus(shouldStart ? "正在启动 TTS…" : "正在停止 TTS…")
            self.healthMonitor.refreshSoon()
            self.viewModel.setTTSService(enabled: shouldStart) { [weak self] result in
                switch result {
                case .success:
                    self?.updateMenuStatus(shouldStart ? "TTS 已启动" : "TTS 已停止")
                case .failure(let error):
                    self?.updateMenuStatus("TTS 操作失败")
                    self?.presentError(error)
                }
            }
        }
    }

    private func requestSTTRuntimeToggle() {
        let shouldStart = !viewModel.sttReady
        presentConfirmation(
            title: shouldStart ? "启动 STT 服务？" : "停止 STT 服务？",
            message: shouldStart
                ? "将加载本地转写模型。"
                : "会中断正在进行的音频转文字任务，并释放 STT 模型占用的内存。TTS 不会停止。",
            confirmTitle: shouldStart ? "启动 STT" : "停止 STT",
            destructive: !shouldStart
        ) { [weak self] in
            guard let self else { return }
            self.updateMenuStatus(shouldStart ? "正在启动 STT…" : "正在停止 STT…")
            self.healthMonitor.refreshSoon()
            self.viewModel.setSTTService(enabled: shouldStart) { [weak self] result in
                switch result {
                case .success:
                    self?.updateMenuStatus(shouldStart ? "STT 已启动" : "STT 已停止")
                case .failure(let error):
                    self?.updateMenuStatus("STT 操作失败")
                    self?.presentError(error)
                }
            }
        }
    }

    private func refreshVoiceStatus() {
        let voice = viewModel.appliedServiceVoiceName.trimmingCharacters(in: .whitespacesAndNewlines)
        voiceStatusItem?.title = "当前音色：\(voice.isEmpty ? "尚未设置" : voice)"
    }

    private func refreshPresets() {
        viewModel.refreshPresets()
    }

    private func renderPresets(_ presets: [NativePreset]) {
        guard presetsMenu != nil else { return }
        presetsMenu.removeAllItems()
        if presets.isEmpty {
            let empty = NSMenuItem(title: "尚无预设，请在工作台中保存", action: nil, keyEquivalent: "")
            empty.isEnabled = false
            presetsMenu.addItem(empty)
            return
        }
        let hint = NSMenuItem(title: "用于 API 调用与新导入的书；已导入的书不受影响", action: nil, keyEquivalent: "")
        hint.isEnabled = false
        presetsMenu.addItem(hint)
        if viewModel.serviceActivePresetID.isEmpty {
            let custom = NSMenuItem(title: "工作台自定义：\(viewModel.appliedServiceVoiceName)", action: nil, keyEquivalent: "")
            custom.state = .on
            custom.isEnabled = false
            presetsMenu.addItem(custom)
        }
        presetsMenu.addItem(.separator())
        for preset in presets {
            let item = NSMenuItem(title: preset.name, action: #selector(selectPreset(_:)), keyEquivalent: "")
            item.target = self
            item.representedObject = preset.id
            item.state = preset.id == viewModel.serviceActivePresetID ? .on : .off
            presetsMenu.addItem(item)
        }
    }

    @objc private func selectPreset(_ sender: NSMenuItem) {
        guard let presetID = sender.representedObject as? String,
              let preset = viewModel.presets.first(where: { $0.id == presetID })
        else { return }
        viewModel.makePresetServiceDefault(preset)
    }

    @objc private func openStudio(_ sender: Any?) { showStudio() }

    @objc private func openSTTWorkbench(_ sender: Any?) { showSTTWorkbench() }

    @objc private func closeStudio(_ sender: Any?) {
        if let keyWindow = NSApp.keyWindow, keyWindow.isVisible {
            keyWindow.performClose(sender)
        } else if window.isVisible {
            window.performClose(sender)
        }
    }

    private func showStudio() {
        NSApp.activate(ignoringOtherApps: true)
        window.makeKeyAndOrderFront(nil)
    }

    private func showSTTWorkbench() {
        healthMonitor.refreshNow()
        NSApp.activate(ignoringOtherApps: true)
        sttWindow.makeKeyAndOrderFront(nil)
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

    @objc private func startLocalService(_ sender: Any?) {
        presentConfirmation(
            title: "启动本地服务？",
            message: "将启动本机的 TTS 与 STT 服务，并恢复菜单栏状态监控。",
            confirmTitle: "启动服务"
        ) { [weak self] in
            guard let self else { return }
            guard !self.service.isStarting else {
                self.updateMenuStatus("正在启动本地服务…")
                return
            }
            self.updateMenuStatus("正在启动本地服务…")
            self.updateServiceMenuActions(isReachable: false)
            self.healthMonitor.refreshSoon()
            self.service.startIfNeeded { [weak self] result in
                DispatchQueue.main.async {
                    switch result {
                    case .success(let health):
                        self?.applyServiceHealth(health)
                        self?.viewModel.updateHealth(health)
                        self?.viewModel.reloadAfterServiceRestart()
                    case .failure(let error):
                        self?.updateMenuStatus("服务启动失败")
                        self?.updateServiceMenuActions(isReachable: false)
                        self?.presentError(error)
                    }
                }
            }
        }
    }

    @objc private func stopLocalService(_ sender: Any?) {
        guard service.ownsRunningProcess else {
            presentError(ServiceError.serviceNotOwned)
            return
        }
        presentConfirmation(
            title: "停止本地服务？",
            message: "会中断所有 TTS / STT 请求、音频播放和后台生成。之后可用菜单栏“启动本地服务”恢复。",
            confirmTitle: "停止服务",
            destructive: true
        ) { [weak self] in
            guard let self else { return }
            self.updateMenuStatus("正在停止本地服务…")
            self.viewModel.forceStopAllAudioAndComputation { [weak self] _ in
                guard let self else { return }
                self.service.stopIfOwned()
                self.viewModel.markServiceUnreachable()
                self.updateMenuStatus("本地服务已停止")
                self.updateServiceMenuActions(isReachable: false)
                self.healthMonitor.refreshSoon()
            }
        }
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

#if !QWEN_STUDIO_TESTS
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
#endif
