# Qwen macOS 音频服务检查（2026-09-06）

## 本次修复：参考音频改名不生效

运行实例的直接证据：7866 端口后台进程仍是 9 月 5 日启动的旧服务；其 OpenAPI 在 `/api/reference-audio-library/{reference_id}` 下只有 DELETE，没有 PUT。向不存在的测试 ID 发送改名请求返回 405，证明不是名称缓存或音频解码问题。上次替换 App 后，`LocalService.startIfNeeded` 复用了旧服务，单纯 health=ready 没有验证新接口已部署。

本次同时更新并重启 App 和后台，运行中的 OpenAPI 已包含 PUT。用新建的短静音测试音频完成导入 → 改名 → 重新读取音频库 → 重新读取音色菜单，名称均一致；当前服务设置未改变，测试音频已清理。

原生交互同步修正：

- 改用独立编辑 sheet，保存直接携带该行的音频对象；不再依赖 alert 关闭时可能已清空的 pending 状态。
- 请求失败保留输入和窗口；旧服务返回 405 时明确提示需要重启服务。
- 只在服务器确认保存后更新列表、音色菜单和当前音频名称，使用服务器返回的最终名称。
- 刷新预设列表不再激活预设，不再重载工作台整套服务参数，保留尚未应用的模型、种子、温度等选择。
- 音频 ID、物理路径、音频内容保持稳定，预设和当前服务中的 voice_name 仍由后端同步。

## 检查范围和整体判断

覆盖正在使用的 macOS 音频工作台/菜单栏、独立阅读器和网页逻辑、FastAPI 路由、预设/参考音频库、生成与播放协调、文档项目、AAC 和 STT 生命周期。旧 MOSS 模型训练/实验目录、第三方 qwentts.cpp/whisper.cpp 推理内核未做逐行审计，也没有重新进行模型质量或多路吞吐基准测试。

项目已有明确分层：Swift 原生 UI → 本地 HTTP 服务 → 调度/播放协调 → Metal TTS 与 Whisper STT；小说项目独立持久化，网页阅读器通过接口消费。这一结构可以继续维护。本轮暴露的主要问题集中在部署版本一致性、异步状态与异常收尾，而不是是否使用 Metal。已有测试多验证成功路径，一部分 UI 检查只是查源码字符串，无法证明实际点击链路正常。

## 待处理发现（按优先级）

下面是检查发现，不表示本轮已修复。P1 为会导致服务阻塞、任务丢失或常见操作失败的问题；P2 为行为/性能不一致。

### P1：AAC 转码同步阻塞 HTTP 服务

位置：`qwen_tts_service/webapp/routers/generation.py:560-602`。

异步 `result-audio-aac` 路由直接调用 `subprocess.run`，未设置超时。转换期间 Uvicorn 主事件循环不能处理其他异步请求，状态刷新、停止、播放租约续期/释放也会延迟；ffmpeg 若挂住，影响不限于这一个音频。代码路径已确认，未在实际服务上制造挂死。建议移至有超时的后台执行，并对相同 job/码率的输出做单次转换与原子发布。

### P1：整书合并失败后永久显示生成中

位置：`qwen_tts_service/document_projects.py:787-801`，`web/novel_reader/reader.js:1107-1122`。

段落全部完成后 `_merge_final_audio` 抛错没有被转换为项目失败状态。finally 移除线程和停止事件，但磁盘 manifest 仍为 running，阅读器继续轮询。隔离临时项目已复现：强制合并异常后 manifest 为 running，final_audio 为空，两个工作线程登记表已清空。建议持久化合并失败和错误详情，保留已完成段落并允许仅重试合并。

### P1：STT 冷启动并发会启动两个进程

位置：`qwen_tts_service/stt_runtime.py:93-150`、`:238`。

start 创建进程后释放锁再等待就绪；另一请求发现进程存在但未就绪时再次创建 whisper-server。隔离 mock 线程测试复现：两个并发 start 产生两个进程，最终 `_process` 只指向第二个。实际会导致端口竞争或遗留进程。建议用启动中状态和条件等待，让所有调用共享一次启动结果，并使 stop 能取消这次启动。

### P1：删除参考音频遗漏书籍和排队任务的引用

位置：`qwen_tts_service/presets.py:396-418`、`:469-473`；`qwen_tts_service/document_projects.py:348-356`、`:898-939`。

当前引用检查只覆盖预设和 active_service；书籍项目保存的 reference_audio_path 及已创建任务的请求也可能继续引用该文件。更换服务音色后，旧音频可能显示为未使用并允许删除，使旧书续作或排队任务失败。建议统一登记项目/任务引用，或按任务生命周期持有参考副本。仅做代码追踪，没有删除用户数据验证。

### P2：打开菜单栏会覆盖工作台未应用的参数

位置：`macos/QwenTTSApp.swift:692-696`、`macos/NativeStudioViewModel.swift` 的 `refreshPresets` 和 `loadActiveServiceSettings`。

每次展开菜单都会刷新并应用活动预设、再载入服务参数。用户在工作台调整参数后只打开菜单查看状态，也会被旧服务配置覆盖；两个异步请求还可能互相覆盖。改名内部已改为只刷新元数据，但菜单这一独立路径仍存在。建议将“查询当前服务信息”和“把服务设置载入编辑器”分开，只在明确选择预设或加载动作时覆盖草稿。

### P2：整书切换模型时先应用了旧模型的性能配置

位置：`qwen_tts_service/webapp/routers/documents.py:255-272`、`web/novel_reader/reader.js:839-867`。

start-selection 先根据项目旧 settings 应用调度配置，随后才解析新的 settings_json 并重置项目。阅读器“生成全书”正是调用此接口，模型变更时会附带新设置。因此模型实际使用新值，调度/文档并行参数却可能仍用旧值。普通 `/start` 固定使用保存配置，不受此问题影响。建议验证并确定最终配置后，再应用对应性能策略。

### P2：上传的临时参考音频没有完整生命周期

位置：`qwen_tts_service/webapp/routers/generation.py:292-297`、`:629` 附近。

允许独立参数的生成请求上传 prompt_audio 后，全量读入内存并写入 uploads，未限制该分支大小，也未登记给任务关闭清理。反复调用会累积文件，异常早退同样没有清理。默认使用已应用音色时此分支被跳过。建议流式限量上传、记录临时文件所有权，在异常/关闭/过期时回收。

### P2：播放进度没有保存段内位置

位置：`web/novel_reader/reader.js:1038`、`:1180`、`:1230`、`:1511`；打开书籍逻辑 `:417-426`。

后端虽支持 offset_seconds，前端各保存点固定传 0，恢复只使用段落序号。所以重新打开后会从该块开头播放。若产品只要求块级续听，这是当前能力限制；若要精确续听，需要区分 AAC currentTime 与流式已播放时长后再保存/恢复。

### P2：独立阅读器的默认端口不一致

位置：`macos/QwenReaderApp.swift:303` 对比 `macos/QwenTTSApp.swift:29` 和 `qwen_tts_service/webapp/cli.py:55`。

读取不到环境文件时，阅读器回退 7866，服务回退 7861。当前机器环境文件有效，不受影响；迁移安装或路径解析失败时会出现服务运行、阅读器连不上的情况。建议统一缺省值或由服务描述文件提供连接信息。

## 验证与边界

- `.venv/bin/python -m pytest -q tests`：27 passed，1 条 Starlette/httpx 弃用警告。
- `tests/native_reference_rename.swift`：编译实际 ViewModel/HTTP 客户端，URLProtocol 隔离网络；验证保存目标不依赖弹窗状态、列表和菜单更新、保留草稿参数、不触发服务设置写入、405 失败保留编辑及重试、使用服务器最终名称。全部通过。
- `./scripts/build_macos_app.sh`：成功；应用签名验证成功，已更新 `/Applications/QwenTTS.app` 并重启后台。
- `node --check web/novel_reader/reader.js`、`git diff --check`：通过。
- 运行服务的改名闭环检查：通过；测试音频已删除，原有参考库及当前服务设置保留。
- 编译器提示现有 AVFoundation 旧接口和 Sendable 捕获警告，当前不阻塞构建；需要独立兼容性整理。
- 原生界面自动化连接两次均报 `Sky Computer Use native pipe closed before response`，因此本次没有声称已自动点击窗口进行视觉验收；验证来自生产 ViewModel 行为测试、真实 HTTP 与成功构建。

建议优先顺序：AAC 非阻塞转码与失败收尾 → STT 单次启动 → 引用生命周期 → UI 草稿隔离与模型策略同步 → 续听/端口一致性。

## 协作与复核记录

按 cost-aware-coding 策略，主代理实现并验证改名；两个 Terra 子代理只读检查后台和阅读器。主代理复核关键代码，要求隔离复现 STT 双进程与整书合并失败；将“所有启动都使用旧性能配置”的初始表述修正为 start-selection 路径。未提供任务级 token/费用统计，未估算费用。
