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
