import AppKit
import AVFoundation
import CoreMedia
import Foundation
import ScreenCaptureKit
import SwiftUI
import UniformTypeIdentifiers

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
                                    model.trimReference(reference)
                                } label: {
                                    Label("裁剪", systemImage: "scissors")
                                }
                                .buttonStyle(StudioCompactActionButtonStyle())
                                if reference.kind == "custom" {
                                    Button {
                                        model.beginRenamingReference(reference)
                                    } label: {
                                        Label("改名", systemImage: "pencil")
                                    }
                                    .buttonStyle(StudioCompactActionButtonStyle())
                                    .disabled(model.isRenamingReference)
                                }
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
                                        if reference.usages.contains(where: { $0.hasPrefix("书籍") || $0.hasPrefix("生成任务") }) {
                                            model.referenceLibraryStatus = "无法删除：该音频仍被书籍或生成任务引用，请先移除书籍或等待任务结束。"
                                        } else {
                                            model.pendingReferenceDeletion = reference
                                        }
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
        .sheet(item: $model.pendingReferenceRename) { reference in
            ReferenceAudioRenameView(model: model, reference: reference)
        }
    }
}

struct ReferenceAudioRenameView: View {
    @ObservedObject var model: NativeStudioViewModel
    let reference: NativeReferenceAudio
    @FocusState private var nameFocused: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: StudioTokens.space4) {
            Label("重命名参考音频", systemImage: "pencil")
                .font(.title3.weight(.semibold))
            Text("更改显示名称，音频内容和引用关系会保留。")
                .font(.callout)
                .foregroundStyle(.secondary)
            TextField("音频名称", text: $model.pendingReferenceRenameName)
                .textFieldStyle(.roundedBorder)
                .focused($nameFocused)
                .disabled(model.isRenamingReference)
            if !model.referenceLibraryStatus.isEmpty {
                Text(model.referenceLibraryStatus)
                    .font(.callout)
                    .foregroundStyle(model.isRenamingReference ? Color.secondary : Color.red)
            }
            HStack {
                Spacer()
                Button("取消") { model.pendingReferenceRename = nil }
                    .keyboardShortcut(.cancelAction)
                    .buttonStyle(StudioCompactActionButtonStyle())
                    .disabled(model.isRenamingReference)
                Button(model.isRenamingReference ? "保存中…" : "保存") {
                    model.renameReference(reference, name: model.pendingReferenceRenameName)
                }
                .keyboardShortcut(.defaultAction)
                .buttonStyle(StudioTintedButtonStyle())
                .disabled(model.isRenamingReference || model.pendingReferenceRenameName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            }
        }
        .padding(StudioTokens.space5)
        .frame(width: 420)
        .background(StudioBackground())
        .interactiveDismissDisabled(model.isRenamingReference)
        .onAppear { nameFocused = true }
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
                                    Label(model.isImportingReference ? "导入中…" : "导入音频/视频", systemImage: "folder")
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
