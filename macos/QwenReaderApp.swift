import AppKit
import CryptoKit
import SwiftUI
import UniformTypeIdentifiers
import WebKit

private let readerServiceURLKey = "QwenReaderServiceURL"
private let serviceSessionCookie = "qwen_tts_service_session"

private func openReaderSettings() {
    NSApp.sendAction(Selector(("showSettingsWindow:")), to: nil, from: nil)
}

@MainActor
final class ReaderAppModel: NSObject, ObservableObject, WKNavigationDelegate, WKUIDelegate {
    @Published var serviceURLText: String
    @Published var statusText = "正在连接语音服务…"
    @Published var isLoading = true
    @Published var loadError = false

    let webView: WKWebView
    private var serviceURL: URL
    private var launchAttempted = false

    override init() {
        let initial = Self.savedOrDefaultServiceURL()
        serviceURL = initial
        serviceURLText = initial.absoluteString

        let configuration = WKWebViewConfiguration()
        configuration.websiteDataStore = .default()
        configuration.allowsAirPlayForMediaPlayback = true
        configuration.mediaTypesRequiringUserActionForPlayback = []
        webView = WKWebView(frame: .zero, configuration: configuration)

        super.init()
        webView.navigationDelegate = self
        webView.uiDelegate = self
        webView.allowsMagnification = true
        webView.setValue(false, forKey: "drawsBackground")
    }

    func start() {
        connectAndLoad(allowServiceLaunch: true)
    }

    func reload() {
        if webView.url == nil || loadError {
            connectAndLoad(allowServiceLaunch: true)
        } else {
            loadError = false
            isLoading = true
            webView.reload()
        }
    }

    func applyServiceURL() {
        guard let normalized = Self.normalizedServiceURL(serviceURLText) else {
            statusText = "请输入有效的 HTTP 或 HTTPS 服务地址"
            loadError = true
            return
        }
        serviceURL = normalized
        serviceURLText = normalized.absoluteString
        UserDefaults.standard.set(normalized.absoluteString, forKey: readerServiceURLKey)
        launchAttempted = false
        webView.configuration.websiteDataStore.httpCookieStore.getAllCookies { [weak self] cookies in
            guard let self else { return }
            let matching = cookies.filter { $0.name == serviceSessionCookie }
            let group = DispatchGroup()
            for cookie in matching {
                group.enter()
                self.webView.configuration.websiteDataStore.httpCookieStore.delete(cookie) {
                    group.leave()
                }
            }
            group.notify(queue: .main) {
                self.connectAndLoad(allowServiceLaunch: true)
            }
        }
    }

    func useLocalService() {
        serviceURL = Self.defaultLocalServiceURL()
        serviceURLText = serviceURL.absoluteString
        UserDefaults.standard.set(serviceURL.absoluteString, forKey: readerServiceURLKey)
        launchAttempted = false
        connectAndLoad(allowServiceLaunch: true)
    }

    func openExternally() {
        NSWorkspace.shared.open(readerURL)
    }

    private var readerURL: URL {
        serviceURL.appendingPathComponent("reader")
    }

    private var healthURL: URL {
        serviceURL.appendingPathComponent("api/health")
    }

    private var isLocalService: Bool {
        guard let host = serviceURL.host?.lowercased() else { return false }
        return host == "127.0.0.1" || host == "localhost" || host == "::1"
    }

    private func connectAndLoad(allowServiceLaunch: Bool) {
        loadError = false
        isLoading = true
        statusText = "正在连接 \(serviceURL.host ?? "语音服务")…"
        var request = URLRequest(url: healthURL)
        request.timeoutInterval = 2.5
        URLSession.shared.dataTask(with: request) { [weak self] _, response, _ in
            Task { @MainActor in
                guard let self else { return }
                if let response = response as? HTTPURLResponse,
                   (200 ... 399).contains(response.statusCode) {
                    self.installLocalSessionCookieAndLoad()
                } else if allowServiceLaunch && self.isLocalService && !self.launchAttempted {
                    self.launchAttempted = true
                    self.statusText = "正在启动本地 Metal 语音服务…"
                    self.launchAudioService()
                    try? await Task.sleep(for: .milliseconds(900))
                    self.waitForLocalService(remainingAttempts: 30)
                } else {
                    self.presentConnectionError()
                }
            }
        }.resume()
    }

    private func waitForLocalService(remainingAttempts: Int) {
        guard remainingAttempts > 0 else {
            presentConnectionError()
            return
        }
        var request = URLRequest(url: healthURL)
        request.timeoutInterval = 1.2
        URLSession.shared.dataTask(with: request) { [weak self] _, response, _ in
            Task { @MainActor in
                guard let self else { return }
                if let response = response as? HTTPURLResponse,
                   (200 ... 399).contains(response.statusCode) {
                    self.installLocalSessionCookieAndLoad()
                } else {
                    try? await Task.sleep(for: .milliseconds(450))
                    self.waitForLocalService(remainingAttempts: remainingAttempts - 1)
                }
            }
        }.resume()
    }

    private func installLocalSessionCookieAndLoad() {
        guard isLocalService,
              let password = Self.localEnvironment()["QWEN_TTS_ACCESS_PASSWORD"],
              !password.isEmpty,
              let host = serviceURL.host
        else {
            loadReader()
            return
        }
        let key = SymmetricKey(data: Data(password.utf8))
        let signature = HMAC<SHA256>.authenticationCode(
            for: Data("qwen-tts-service-session".utf8),
            using: key
        ).map { String(format: "%02x", $0) }.joined()
        let properties: [HTTPCookiePropertyKey: Any] = [
            .domain: host,
            .path: "/",
            .name: serviceSessionCookie,
            .value: signature,
            .secure: serviceURL.scheme?.lowercased() == "https",
        ]
        guard let cookie = HTTPCookie(properties: properties) else {
            loadReader()
            return
        }
        webView.configuration.websiteDataStore.httpCookieStore.setCookie(cookie) { [weak self] in
            Task { @MainActor in self?.loadReader() }
        }
    }

    private func loadReader() {
        loadError = false
        isLoading = true
        statusText = "正在载入小说阅读器…"
        webView.load(URLRequest(url: readerURL, cachePolicy: .reloadRevalidatingCacheData))
    }

    private func launchAudioService() {
        let candidates = [
            URL(fileURLWithPath: "/Applications/QwenTTS.app"),
            Bundle.main.bundleURL
                .deletingLastPathComponent()
                .appendingPathComponent("QwenTTS.app"),
        ]
        guard let appURL = candidates.first(where: {
            FileManager.default.fileExists(atPath: $0.path)
        }) else {
            return
        }
        let configuration = NSWorkspace.OpenConfiguration()
        configuration.activates = false
        configuration.arguments = ["--background"]
        NSWorkspace.shared.openApplication(at: appURL, configuration: configuration)
    }

    private func presentConnectionError() {
        isLoading = false
        loadError = true
        statusText = isLocalService
            ? "无法连接本地语音服务，请确认 Qwen TTS 已安装并能够启动。"
            : "无法连接此网络服务，请检查地址、网络和服务兼容性。"
    }

    func webView(_ webView: WKWebView, didStartProvisionalNavigation navigation: WKNavigation!) {
        isLoading = true
        loadError = false
    }

    func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
        isLoading = false
        loadError = false
        statusText = "已连接"
    }

    func webView(
        _ webView: WKWebView,
        didFailProvisionalNavigation navigation: WKNavigation!,
        withError error: Error
    ) {
        isLoading = false
        loadError = true
        statusText = error.localizedDescription
    }

    func webView(
        _ webView: WKWebView,
        didFail navigation: WKNavigation!,
        withError error: Error
    ) {
        isLoading = false
        loadError = true
        statusText = error.localizedDescription
    }

    func webView(
        _ webView: WKWebView,
        runOpenPanelWith parameters: WKOpenPanelParameters,
        initiatedByFrame frame: WKFrameInfo,
        completionHandler: @escaping ([URL]?) -> Void
    ) {
        let panel = NSOpenPanel()
        panel.title = "导入小说"
        panel.prompt = "导入"
        panel.message = "选择 TXT、Markdown 或 DOCX 小说文件"
        panel.canChooseFiles = true
        panel.canChooseDirectories = parameters.allowsDirectories
        panel.allowsMultipleSelection = parameters.allowsMultipleSelection
        panel.allowedContentTypes = [
            .plainText,
            UTType(filenameExtension: "md") ?? .plainText,
            UTType(filenameExtension: "markdown") ?? .plainText,
            UTType(filenameExtension: "docx") ?? .data,
        ]
        panel.begin { response in
            completionHandler(response == .OK ? panel.urls : nil)
        }
    }

    static func normalizedServiceURL(_ raw: String) -> URL? {
        var value = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !value.isEmpty else { return nil }
        if !value.contains("://") {
            value = "http://\(value)"
        }
        guard var components = URLComponents(string: value),
              ["http", "https"].contains(components.scheme?.lowercased() ?? ""),
              components.host != nil
        else {
            return nil
        }
        components.path = ""
        components.query = nil
        components.fragment = nil
        guard var url = components.url else { return nil }
        if !url.absoluteString.hasSuffix("/") {
            url.appendPathComponent("")
        }
        return url
    }

    static func savedOrDefaultServiceURL() -> URL {
        if let saved = UserDefaults.standard.string(forKey: readerServiceURLKey),
           let url = normalizedServiceURL(saved) {
            return url
        }
        return defaultLocalServiceURL()
    }

    static func defaultLocalServiceURL() -> URL {
        let port = Int(localEnvironment()["PORT"] ?? "7866") ?? 7866
        return URL(string: "http://127.0.0.1:\(port)/")!
    }

    static func repositoryRoot() -> URL? {
        if let embeddedRoot = Bundle.main.url(
            forResource: "repository-root",
            withExtension: "txt"
        ), let path = try? String(contentsOf: embeddedRoot, encoding: .utf8)
            .trimmingCharacters(in: .whitespacesAndNewlines) {
            return URL(fileURLWithPath: path).standardizedFileURL
        }
        return nil
    }

    static func localEnvironment() -> [String: String] {
        guard let root = repositoryRoot(),
              let contents = try? String(
                contentsOf: root.appendingPathComponent(".env.macos"),
                encoding: .utf8
              )
        else {
            return [:]
        }
        var values: [String: String] = [:]
        for rawLine in contents.split(whereSeparator: \.isNewline) {
            let line = rawLine.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !line.isEmpty, !line.hasPrefix("#"),
                  let separator = line.firstIndex(of: "=")
            else { continue }
            let key = String(line[..<separator]).trimmingCharacters(in: .whitespaces)
            let value = String(line[line.index(after: separator)...])
                .trimmingCharacters(in: .whitespacesAndNewlines)
                .trimmingCharacters(in: CharacterSet(charactersIn: "\"'"))
            values[key] = value
        }
        return values
    }
}

struct ReaderWebView: NSViewRepresentable {
    @ObservedObject var model: ReaderAppModel

    func makeNSView(context: Context) -> WKWebView {
        model.webView
    }

    func updateNSView(_ nsView: WKWebView, context: Context) {}
}

struct ReaderRootView: View {
    @ObservedObject var model: ReaderAppModel

    var body: some View {
        ZStack {
            ReaderWebView(model: model)

            if model.loadError {
                connectionError
            } else if model.isLoading {
                loadingIndicator
            }
        }
        .frame(minWidth: 940, minHeight: 680)
        .background(Color(nsColor: .windowBackgroundColor))
        .onAppear { model.start() }
        .toolbar {
            ToolbarItemGroup {
                Button(action: model.reload) {
                    Label("重新载入", systemImage: "arrow.clockwise")
                }
                .help("重新载入小说阅读器")

                Button(action: openReaderSettings) {
                    Label("服务设置", systemImage: "network")
                }
                .help("设置本地或网络语音服务")
            }
        }
    }

    private var loadingIndicator: some View {
        VStack(spacing: 12) {
            ProgressView()
                .controlSize(.large)
            Text(model.statusText)
                .font(.system(size: 13, weight: .medium))
                .foregroundStyle(.secondary)
        }
        .padding(24)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 16))
    }

    private var connectionError: some View {
        VStack(spacing: 14) {
            Image(systemName: "waveform.badge.exclamationmark")
                .font(.system(size: 32, weight: .medium))
                .foregroundStyle(.secondary)
            Text("暂时无法打开阅读器")
                .font(.system(size: 19, weight: .semibold))
            Text(model.statusText)
                .multilineTextAlignment(.center)
                .foregroundStyle(.secondary)
                .frame(maxWidth: 420)
            HStack(spacing: 10) {
                Button("重试", action: model.reload)
                    .buttonStyle(.borderedProminent)
                Button(action: openReaderSettings) {
                    Text("服务设置")
                }
                .buttonStyle(.bordered)
            }
        }
        .padding(30)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 20))
    }
}

struct ReaderSettingsView: View {
    @ObservedObject var model: ReaderAppModel

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            VStack(alignment: .leading, spacing: 5) {
                Text("语音与阅读服务")
                    .font(.system(size: 20, weight: .semibold))
                Text("可连接本机、局域网或提供兼容阅读器接口的网络服务。")
                    .foregroundStyle(.secondary)
            }

            VStack(alignment: .leading, spacing: 7) {
                Text("服务地址")
                    .font(.system(size: 12, weight: .semibold))
                TextField("http://127.0.0.1:7866", text: $model.serviceURLText)
                    .textFieldStyle(.roundedBorder)
                Text("服务需要提供 /reader 页面及同源的 /api 接口。HTTPS 网络服务可直接填写域名。")
                    .font(.system(size: 11))
                    .foregroundStyle(.secondary)
            }

            HStack {
                Button("恢复本地服务", action: model.useLocalService)
                Spacer()
                Button("在浏览器中打开", action: model.openExternally)
                Button("应用并连接", action: model.applyServiceURL)
                    .buttonStyle(.borderedProminent)
            }
        }
        .padding(24)
        .frame(width: 520)
    }
}

final class ReaderApplicationDelegate: NSObject, NSApplicationDelegate {
    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        true
    }
}

@main
struct QwenReaderApplication: App {
    @NSApplicationDelegateAdaptor(ReaderApplicationDelegate.self) private var appDelegate
    @StateObject private var model = ReaderAppModel()

    var body: some Scene {
        Window("Qwen 声阅", id: "reader") {
            ReaderRootView(model: model)
        }
        .defaultSize(width: 1280, height: 860)
        .windowResizability(.contentMinSize)
        .commands {
            CommandGroup(after: .toolbar) {
                Button("重新载入阅读器") {
                    model.reload()
                }
                .keyboardShortcut("r", modifiers: .command)
            }
        }

        Settings {
            ReaderSettingsView(model: model)
        }
    }
}
