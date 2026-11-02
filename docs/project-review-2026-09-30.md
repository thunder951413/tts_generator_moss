# Qwen macOS 音频服务全面评估 2026 09 30

> 修复状态更新（2026-10-01）：本报告中确认的代码缺陷已修复，最终干净环境回归为 86 项 Python 和 9 项阅读器测试全部通过，并完成原生构建和真实服务、浏览器验收。详见 [修复与验收报告](/Users/surfing/tools/tts_generator_moss/docs/repair-validation-2026-10-01.md)。以下保留修复前的评估、复现和建议，文中的“当前”指 2026-09-30 评估时的代码状态。

本次评估覆盖当前使用的 Qwen TTS/STT 后台、原生工作台、小说阅读器、文档生成、部署脚本和测试。总体判断：功能已较完整，服务分层可以继续演进；当前更适合作为个人本地工具。长期后台运行的可靠性、内部 worker 的访问边界、部署版本校验和自动验收仍需补齐。

评估基于 `mac-qwen-metal` 分支的 `035f11d`，包含工作区已有的菜单栏图标修改。该修改未被覆盖。本次只新增评估报告，没有实施代码修复、重启运行服务或调整用户配置。

## 评估范围与验证结果

代码检查覆盖 FastAPI 鉴权和各路由组、RuntimeManager、Qwen worker 池、Whisper 生命周期、GPU 调度、播放租约、任务持久化、整书分段与合并，以及 Swift 和阅读器的关键调用链。旧 MOSS 训练与实验目录、第三方 GGML/Metal 和 Whisper 推理内核未做逐行审计。

| 验证 | 结果 | 可以说明的范围 |
| --- | --- | --- |
| `.venv/bin/python -m pytest tests/ -q` | 67 passed，1 failed，26.76 秒 | 后台回归大部分通过，当前工作区未达到全部测试通过 |
| 原生工作台 Swift 类型检查 | 通过，有警告 | 当前源码可以通过类型检查，不能替代链接、签名和实际界面验收 |
| 独立阅读器 Swift 类型检查 | 通过 | 阅读器当前源码可以通过类型检查 |
| 原生参考音频改名行为测试 | 编译并运行通过 | 保存对象身份、服务端名称、草稿隔离、失败保留和重试行为正常 |
| JavaScript 与 Shell 语法检查 | 通过 | `reader.js` 和主要 macOS 启动、构建脚本语法有效 |
| `pip check` 与 `git diff --check` | 通过 | 当前已安装环境没有声明依赖冲突，现有差异没有空白格式错误 |
| 当前运行服务的只读 health | HTTP 200 | 1.7B TTS 与 STT 均 ready，GPU 并发 2，文档并发 2，检查时无活动计算 |
| 独立临时目录和 mock 复现 | 见下文 | 不依赖真实模型，可验证状态和协议问题；不代表模型音质测试 |

唯一 pytest 失败位于 [test_macos_qwen_service.py](/Users/surfing/tools/tts_generator_moss/tests/test_macos_qwen_service.py:164)：测试要求源码中出现 `NSAttributedString(string: "TTS"`，而工作区已经改为绘制气泡波形图标。该失败表明断言和当前设计不一致，不能据此判断菜单栏图标不能运行。这个大测试在首次失败后也不会继续执行后面的断言。

未进行真实模型的音质盲测、长书生成压力测试、24 小时运行测试、全新机器安装、GUI 自动点击验收或依赖漏洞数据库扫描。报告中的性能数据来自仓库已有记录，不能视为本轮重新测量。

## 已确认的主要问题

P1 表示核心使用流程可能错误、停滞，或需要优先收紧的访问边界；P2 表示特定配置、交付或体验问题。下列问题均为当前代码的发现，不表示本轮已经修复。

### P1 内部 worker 可以绕过主服务鉴权并写出指定目录

位置：[qwen_worker.py](/Users/surfing/tools/tts_generator_moss/qwen_tts_service/qwen_worker.py:384)，以及同文件的 `WorkerState.generate` 第 106 至 114 行。

主服务检查密码，但 worker 的 `/generate` 没有独立凭证检查，接收请求后直接解析 JSON；它接受调用方传入的 `ref_audio`、`output_dir` 和 `run_id`，没有目录白名单或规范化后的边界检查。

隔离验证：在随机 loopback 端口启动生产 WorkerHandler，使用 fake 模型、临时参考文件，发送没有鉴权头且 Content-Type 为 text/plain 的请求，返回 HTTP 200。传入 `run_id=../escaped` 后，临时测试目录中生成了配置输出目录之外的 `generated.wav`。

影响限定在能访问本机 worker 端口的调用方；默认 worker 绑定 loopback，因此没有据此声称局域网用户可直接访问。主 API 的密码、生成队列和路径校验不能约束这条内部调用路径。

建议为主服务与 worker 建立每次启动生成的私有凭证，或使用 Unix domain socket；worker 自身固定可写目录、生成 run_id，并校验参考文件根目录和请求大小。验收应覆盖未认证请求、绝对路径、`..` 和超量请求。

### P1 worker 全部恢复失败后生成会无限等待

位置：[qwen_runtime.py](/Users/surfing/tools/tts_generator_moss/qwen_tts_service/qwen_runtime.py:262)，以及 `synthesize` 第 412 至 419 行。

worker 恢复最多尝试三次。全部失败后只记录日志，没有向等待方发布终止故障。`synthesize` 持续对空池执行带一秒超时的 get，超时后继续循环，没有总等待上限或故障状态判断。

隔离验证：令三次 worker 创建均失败；生成线程仍在等待，只有显式关闭整个 runtime 才退出。代码中的等待没有总期限，因此并非仅在测试的观察窗口内比较慢。

影响：当所有 lane 都不可用时，生成任务可能长期占用调度槽和 runtime session，并阻止后续任务、模型切换或 STT 获得计算窗口。主 health 的 ready 来源也不会自动反映池已永久耗尽。

建议发布 lane 恢复状态和最终错误，对 worker 获取设置总期限并支持任务取消；连续恢复失败应使任务明确失败，同时允许之后重新启动。应增加“部分恢复失败”和“所有恢复失败”的回归测试。

### P1 服务重启后已完成任务不能通过正常播放入口重放

位置：[app_factory.py](/Users/surfing/tools/tts_generator_moss/qwen_tts_service/webapp/app_factory.py:125)、[streaming_jobs.py](/Users/surfing/tools/tts_generator_moss/qwen_tts_service/streaming_jobs.py:183)、[generation.py](/Users/surfing/tools/tts_generator_moss/qwen_tts_service/webapp/routers/generation.py:509)。

新服务以当前时间初始化 playback epoch，恢复任务时保留旧任务的 playback_epoch。播放入口要求两者相同，因而把已经保存的历史结果也当成过期播放会话。

隔离验证：同一完成任务重启前 `result-audio?playback=1` 返回 200；使用恢复后的 manifest 和新服务 epoch 后返回 409，detail 为 playback session is stale；`playback=0` 下载仍返回 200。

影响：文件和任务恢复成功，但原生重放或外部客户端默认播放入口仍可能失败。应区分“停止操作撤销正在进行的会话”和“用户明确重新播放历史结果”。历史结果重放应新建播放租约，继续保留全局停止对旧请求的撤销语义。

### P1 逐块失控重试导致 PCM 流和最终 WAV 不一致

位置：[qwen_worker.py](/Users/surfing/tools/tts_generator_moss/qwen_tts_service/qwen_worker.py:221)。

worker 检测到失控后清空本地 audio_chunks，再用新种子重新生成；但检测前已 yield 出去的音频无法从主服务队列和客户端播放缓冲中撤回。当前协议只发出 progress，没有用于切换或撤销音频的消息。

隔离验证：把失控预算缩短为两秒，首轮逐块输出一秒音频，越界后重试生成 0.8 秒。所有 audio 事件合计为 2.8 秒，保存的最终 WAV 只有 0.8 秒。现有重试测试让首块直接超过预算，恰好没有首轮残留，因此未覆盖这一情况。

影响：用户可能先听到失控内容，再听到重新开始的正文；实时听到的内容、长度与最终保存文件不一致。原始 PCM 协议无法撤回已播放声音。建议实时流在已发音频后明确中止，由客户端决定重试；完整块生成可保留透明重试。若保留流式重试，需要增加客户端可识别的流代次和缓冲清理协议。

### P1 阅读器把生成错误当成当前块播放完成

位置：[reader.js](/Users/surfing/tools/tts_generator_moss/web/novel_reader/reader.js:1426)，以及第 1286 至 1289 行。

`playStreamingAudio` 在流结束后读取最终状态，只对 closed 抛出错误，没有处理 error 或 interrupted。上层随后保存该块的播放进度并继续下一块。

隔离验证：运行生产 JavaScript 函数，mock 一个正常结束的空 PCM 流和 `state=error` 的最终状态，函数成功返回。这个验证使用 Node mock 环境，没有声称已经通过浏览器点击重现。

影响：模型错误、worker 崩溃或失控终止后，当前块可能只播放部分内容，阅读器仍推进进度。建议只有明确 finished 且完整消费音频时才推进；其他状态显示后端错误并保留当前位置。网络断流和状态查询失败也需要明确定义。

### P2 包含中文字符的服务密码使登录和鉴权报错

位置：[auth_pages.py](/Users/surfing/tools/tts_generator_moss/qwen_tts_service/webapp/routers/auth_pages.py:64)、[app_factory.py](/Users/surfing/tools/tts_generator_moss/qwen_tts_service/webapp/app_factory.py:250)。

密码以字符串传入 hmac.compare_digest；该比较方式不能处理非 ASCII 字符串。界面和配置没有要求密码只能使用 ASCII。

隔离验证：在临时 app 中设置含中文的审计示例密码，提交相同密码登录返回 500，未认证访问 `/api/runtime` 也返回 500。当前用户密码未被修改或输出。

建议将比较双方统一编码为 UTF-8 bytes，或在设置入口提供明确的一致性校验；增加中文和 Unicode 密码测试。

## 交付与体验问题

| 优先级 | 问题和证据 | 建议 |
| --- | --- | --- |
| P2 | CI 依赖不完整。pyproject 的 ci 额外依赖只声明 torch 和 pytest，基础依赖也没有启用 fastapi/Starlette 的 HTTP 测试客户端额外依赖。模拟没有 httpx/httpx2 时，导入 TestClient 报 requires the httpx2 package。 | 显式声明兼容的 HTTP 测试客户端；在干净环境安装并运行 CI，避免依赖本机历史安装。 |
| P2 | 应用仍只凭 health 响应复用后台。`LocalService.startIfNeeded` 没有检查服务版本或 API 能力；当前运行 health 也没有 version/build/API 版本字段。 | 发布服务构建标识、API 版本和必要能力；应用连接时验证兼容性并提示需要更新的具体组件。不能据此直接认定本机后台已落后。 |
| P2 | 精确续听仍未实现。reader.js 中三个 savePlayback 调用都传 offset=0。 | 保存 AAC currentTime 或流式已播放时长，并明确暂停、退出、下一块和重新打开时的恢复规则。 |
| P2 | README 的安装命令漏了 whisper-cpp，setup-macos.sh 却要求 whisper-server；README 描述单 worker，环境模板与本机配置使用两路。README 称安装不下载模型，setup 实际下载 Whisper small。LOCAL_DEPLOYMENT 中还有写死的 7866 阅读器地址。 | 统一安装清单、默认并发、模型下载范围和端口说明，区分机器建议与通用默认值。 |
| P2 | 历史任务和生成文件缺少统一保留期限及回收策略。StreamingJobManager 只限制活动任务，完成任务长期留在内存和磁盘；未消费的完成 PCM spool 也不会按时自动回收。 | 对临时音频、历史任务和可保留结果分别设置 TTL、总容量限制和启动清理，避免直接清理书籍或用户参考库。 |

检查时 outputs 共 10,937 个文件，约 1.715 GiB。该统计包含正常书籍和音频数据，不能全部算作垃圾或泄漏；它说明维护策略应覆盖长期累积，而不能只依赖正常关闭网页时的清理请求。

还需关注两个尚未做负载验证的边界：DOCX 只限制压缩文件大小，没有限制解压后的 XML 大小；文档解析和许多全量 manifest 读写发生在异步路由中。大书、高压缩率文档和多个项目同时生成时，应实测事件循环响应、磁盘 I/O、转码队列长度和峰值内存，再决定是否拆分状态存储及增加背压。

## 各维度判断

| 维度 | 当前判断 | 判断依据与限制 |
| --- | --- | --- |
| 架构 | 可继续演进 | 原生 UI、HTTP、调度、运行时和持久化已有分层；复杂度主要集中在跨层状态和生命周期。 |
| 功能 | 本地场景较完整 | 已有克隆、流式与完整块播放、参考库、预设、录音裁剪、文档项目、AAC、阅读器和 STT；精确续听与部署协商不足。 |
| 稳定性 | 需要优先修复 | 重启重放、worker 池耗尽、流式重试和错误进度均有隔离复现。 |
| 安全边界 | 主服务已有保护，内部路径需收紧 | 主 API 有密码、loopback 默认、跨域限制和路径白名单；worker 能绕过这些边界。 |
| 性能 | 两路配置有历史依据 | 现有短热启动基准支持这台 M4 Max 的两路选择，不能代表冷启动、长文本、音质、P95 延迟或其他 Mac。 |
| 测试 | 后台回归进步明显，发布验证不足 | 已覆盖多项并发和失败生命周期；源码字符串断言脆弱，Swift 行为测试未纳入当前 pytest CI。 |
| 部署 | 仍依赖本机仓库和环境 | App 构建写入 repository-root，运行依赖外部 venv、动态库和工具；并非独立可搬移的应用包。 |
| 可维护性 | 尚可，需要持续收敛 | ViewModel 超过 2,000 行、文档管理超过 1,200 行；路由组重复解包上下文字段，跨模块大量顶层导入依赖启动时 sys.path。 |

仓库 [qwen-performance.json](/Users/surfing/tools/tts_generator_moss/qwen-performance.json) 的基准日期为 2026-07-26，使用 M4 Max 128 GB、Q4_K_M、热模型与参考缓存、每任务 96 帧音频。记录中的 0.6B 聚合实时倍速从单路 1.9307 到两路 3.1132，1.7B 从 1.8873 到 2.8692；这里的 realtime_factor 实际表示音频时长除以运行时长，越大越快。

这组数据能支撑该机器上两路并发的初步选择。它没有足够的重复轮次、冷启动样本、延迟分布或音质评价，不能据此承诺稳定实时表现。后续性能验收应固定文本、音色和种子，分别测冷启动、短句交互、长段落、整书、TTS/STT 交替和单路故障恢复，并记录首音延迟、完成时间、内存、错误率和实际播放连续性。

## 与上次报告的比较

2026-09-06 报告里的 AAC 同步阻塞、整书合并失败状态、STT 重复启动、参考音频引用遗漏、菜单覆盖草稿、start-selection 性能策略顺序、临时参考上传限量与所有权、默认端口不一致，在当前实现中均已有对应修正。其中 AAC、整书、STT、引用和原生改名等路径还得到本轮回归支持。

版本一致性检查和精确续听仍需处理。不能将旧报告的其他发现直接作为当前尚未修复的问题，也不能由这次单元测试通过推导所有实机场景已经验证。

## 建议的实施顺序与验收

1. 收紧 worker 通信和文件路径边界，并保证池耗尽时任务有确定的失败结果。验收未认证调用被拒绝、越界目录无写入、恢复失败后调度槽能够释放。
2. 修正流式失败与重试协议。验收失败块不推进进度，逐块失控不会把不同生成轮次拼入同一成功流，实时播放与保存结果的语义一致。
3. 修正重启后的历史结果播放，补齐 Unicode 密码和干净 CI 安装。验收历史结果能重新申请新播放租约，同时旧会话仍被全局停止撤销。
4. 增加 App 与服务的版本协商，更新安装文档和部署默认值，再补精确续听和资源保留策略。
5. 在上述功能稳定后，执行长书和持续运行测试，以及真实 macOS 的录音权限、裁剪、播放、停启、更新和重连验收，再判断是否适合更广泛交付。

2026-09-30 评估时未实施修复；随后修复工作已完成，当前状态以 [2026-10-01 修复与验收报告](/Users/surfing/tools/tts_generator_moss/docs/repair-validation-2026-10-01.md) 为准。
