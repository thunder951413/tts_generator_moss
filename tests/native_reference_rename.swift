import Foundation

// Exercise the production view model and HTTP client without launching models or touching user data.
final class RenameProtocol: URLProtocol {
    static var status = 200
    static var savedName = "新名字"
    static var requests: [String] = []
    static let referenceID = "01234567890123456789012345678901"
    static let referencePath = "/tmp/qwen-rename-test.wav"
    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    override func startLoading() {
        let path = request.url!.path
        Self.requests.append("\(request.httpMethod ?? "GET") \(path)")
        let isRename = request.httpMethod == "PUT"
        let status = isRename ? Self.status : 200
        let payload: [String: Any] = isRename
            ? (status == 200
                ? ["ok": true, "reference": ["id": Self.referenceID, "name": Self.savedName, "path": Self.referencePath]]
                : ["detail": "Method Not Allowed"])
            : (path == "/api/service-settings"
                ? ["name": "后台音色", "settings": ["voice_name": "后台音色", "reference_audio_path": "/tmp/other.wav", "qwen_temperature": 0.3]]
                : ["presets": [["id": "active", "name": "已保存预设", "settings": ["reference_audio_path": "/tmp/other.wav", "qwen_temperature": 0.3]]], "active_preset_id": "active"])
        let response = HTTPURLResponse(url: request.url!, statusCode: status, httpVersion: nil, headerFields: ["Content-Type": "application/json"])!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: try! JSONSerialization.data(withJSONObject: payload))
        client?.urlProtocolDidFinishLoading(self)
    }
    override func stopLoading() {}
}

@main
enum RenameTests {
    static func waitUntil(_ condition: () -> Bool) {
        let deadline = Date().addingTimeInterval(5)
        while !condition() && Date() < deadline {
            RunLoop.current.run(until: Date().addingTimeInterval(0.01))
        }
        precondition(condition(), "Timed out")
    }

    static func main() {
        URLProtocol.registerClass(RenameProtocol.self)
        let service = LocalService(repositoryRoot: FileManager.default.temporaryDirectory)
        let model = NativeStudioViewModel(service: service)
        let reference = NativeReferenceAudio(["id": RenameProtocol.referenceID, "name": "旧名字", "path": RenameProtocol.referencePath, "kind": "custom"])!
        model.referenceLibrary = [reference]
        model.voices = [NativeVoice(["name": reference.name, "audio_path": reference.path])!]
        model.referenceAudioPath = reference.path
        model.referenceName = reference.name
        model.temperature = 1.37
        model.seed = 5678

        model.beginRenamingReference(reference)
        // A dismissed presentation must not erase the identity captured by the save action.
        model.pendingReferenceRename = nil
        model.renameReference(reference, name: "新名字")
        waitUntil { !model.isRenamingReference && !model.presets.isEmpty }
        precondition(model.referenceLibrary[0].name == "新名字")
        precondition(model.voices[0].name == "新名字")
        precondition(model.referenceName == "新名字")
        precondition(model.referenceAudioPath == reference.path)
        precondition(model.temperature == 1.37 && model.seed == 5678)
        precondition(!RenameProtocol.requests.contains { $0.contains("service-settings") })
        model.loadActiveServiceSettings(applyingToEditor: false)
        waitUntil { model.appliedServiceVoiceName == "后台音色" }
        precondition(model.referenceName == "新名字")
        precondition(model.temperature == 1.37 && model.seed == 5678)
        precondition(model.referenceAudioPath == reference.path)

        RenameProtocol.status = 405
        model.beginRenamingReference(reference)
        model.renameReference(reference, name: "应当失败")
        waitUntil { !model.isRenamingReference }
        precondition(model.referenceLibraryStatus.contains("旧版本"))
        precondition(model.pendingReferenceRename != nil, "Failed saves must remain editable")
        precondition(model.referenceLibrary[0].name == "新名字")

        RenameProtocol.status = 200
        RenameProtocol.savedName = "服务端规范化名称"
        model.renameReference(reference, name: "客户端名称")
        waitUntil { !model.isRenamingReference }
        precondition(model.referenceName == "服务端规范化名称")
        precondition(model.pendingReferenceRename == nil)
        print("Native rename: save identity, persisted name, preset isolation, failure and retry passed")
    }
}
