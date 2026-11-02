# Qwen3-TTS Generator for Apple Silicon

这是 `tts_generator_moss` 的 macOS ARM 分支。应用只提供 Qwen3-TTS：

- Qwen3-TTS 0.6B Base
- Qwen3-TTS 1.7B Base
- X-vector 与 ICL 音色克隆
- Apple Metal GPU 加速
- 文本试听、文档分段生成、AAC 合并
- 后台任务、断点续做、跨浏览器查看任务

模型文件不会提交到 Git。首次使用相应档位时，GGUF 权重由
Hugging Face 下载到本机缓存。

## 技术路线

macOS 不支持 NVIDIA CUDA，因此本分支不使用 Faster Qwen3-TTS 的
Torch/CUDA Graph 后端，而使用它的实验性 GGML 适配层，并从源码构建
`qwentts.cpp` 的 Metal 后端。

相关项目：

- [Faster Qwen3-TTS](https://github.com/andimarafioti/faster-qwen3-tts)
- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)
- [qwentts.cpp](https://github.com/ServeurpersoCom/qwentts.cpp)
- [qwentts-cpp-python](https://github.com/andimarafioti/qwentts-cpp-python)

## 系统要求

- Apple Silicon：M1、M2、M3、M4 或后续 ARM Mac
- macOS 14 或更高版本
- Homebrew
- 0.6B 建议至少 16 GB 统一内存
- 1.7B 建议至少 24 GB 统一内存
- 模型缓存和构建文件需要约 15–30 GB 空间

安装依赖：

```bash
xcode-select --install
brew install python@3.12 cmake ninja ffmpeg libsndfile whisper-cpp
```

## 安装

```bash
git clone --branch mac-qwen-metal \
  https://github.com/thunder951413/tts_generator_moss.git
cd tts_generator_moss
chmod +x setup-macos.sh start-macos.sh stop-macos.sh
./setup-macos.sh
```

安装脚本会：

1. 创建 `.venv`。
2. 拉取固定版本的 Faster Qwen3-TTS、qwentts Python 包和 qwentts.cpp。
3. 使用 `GGML_METAL=ON` 编译原生 arm64 Metal 动态库。
4. 安装网页服务依赖。
5. 下载并校验 Whisper small 的 STT 权重，生成本地 `.env.macos`。Qwen TTS 权重在首次加载时下载，所有权重均不提交到 Git。

## 配置

编辑 `.env.macos`：

```dotenv
QWEN_TTS_ACCESS_PASSWORD=请改成强密码
HOST=127.0.0.1
PORT=7861
QWEN_TTS_QUANT=Q4_K_M
QWEN_TTS_0_6B_LANES=1
QWEN_TTS_1_7B_LANES=1
QWEN_TTS_MAX_PARALLEL_GENERATIONS=1
QWEN_TTS_DOCUMENT_PARALLEL_GENERATIONS=1
```

以上单路配置适用于通用机器。环境模板使用 M4 Max 128 GB 已测的两路配置；小内存 Mac 请按上例改为单路。已有性能基准会优先应用该模型保存的推荐值，可在原生工作台重新测试并更新。

默认仅允许本机访问。需要让局域网设备调用时，再把 `HOST` 改为
`0.0.0.0`；启动脚本会拒绝“空密码 / change-me / 少于 4 位密码”的
局域网暴露配置。浏览器跨域调用还需显式设置
`QWEN_TTS_CORS_ORIGINS`（多个来源用逗号分隔）。

量化选择：

| 量化 | 特点 | 建议 |
|---|---|---|
| `Q4_K_M` | 内存最少、通常最快 | 默认 |
| `Q8_0` | 质量与占用折中 | 统一内存充足时测试 |
| `BF16` | 体积和内存最大 | 只建议大内存机器对比 |

如需把模型放在外置盘，在 `.env.macos` 设置：

```dotenv
HF_HOME=/Volumes/Models/huggingface
```

## 启动和停止

后台启动：

```bash
./start-macos.sh
```

访问：

```text
http://127.0.0.1:7861/reader
```

地址中的端口以 `.env.macos` 的 `PORT` 为准。启用局域网访问后可把 `127.0.0.1` 换为本机局域网 IP。

网页关闭后任务继续执行。重新登录后可以查看文本任务、文档项目、已完成
段落和当前进度。

停止服务：

```bash
./stop-macos.sh
```

日志：

```text
logs/service.log        # 服务日志，5 MB 轮转保留 3 份；health/轮询类成功请求不记录
logs/service.out.log
logs/service.err.log
logs/qwen-workers/
```

日志目录可用 `QWEN_TTS_LOG_DIR` 修改（设为 `none` 关闭文件日志）；`QWEN_TTS_LOG_STDERR=0` 可关闭向标准错误重复输出。

## macOS 应用

项目包含两个职责分离的原生应用：

- `QwenTTS.app`：菜单栏音频服务、音色预设、性能测试与服务设置。
- `QwenReader.app`：独立小说阅读器，使用 WebKit 加载本地阅读页面。

构建：

```bash
./scripts/build_macos_app.sh
./scripts/build_macos_reader.sh
```

阅读器默认连接 `.env.macos` 中的本地端口；本地服务尚未运行时，会在
后台启动 `QwenTTS.app`。在“Qwen 声阅 → 设置”中也可以填写局域网或
兼容网络服务地址。目标服务需要同时提供 `/reader` 页面和同源 `/api`
接口。

## 模型与输出目录

以下内容均被 `.gitignore` 排除：

- `.runtime/`：源码和 Metal 构建产物
- `.venv/`：Python 环境
- `models/` 和 Hugging Face 模型缓存
- `outputs/`：项目状态、分段 WAV、AAC 和任务记录
- `.env.macos`：密码和本机配置
- `logs/`、PID 文件

不要把 `outputs/` 放在自动清理的临时目录中；断点续做依赖其中的项目状态。

## 性能说明

默认只启动一个 worker。每增加一路都会额外加载一份模型和 codec，明显增加
统一内存占用。Metal 下多路是否提速与具体 M 系列芯片、内存带宽和量化有关，
应先用单路生成固定文本，再逐步测试两路；不要直接沿用 NVIDIA 上的并行结论。

状态中的 Seed 是实际传给 qwentts.cpp 的 Seed。设置 `-1` 时，服务先生成一个
确定的随机数，再将实际数值传给后端并保存到任务状态。

## 更新、数据保留与验证

本应用依赖当前仓库的 `.venv`、Metal 动态库、FFmpeg 和模型缓存。构建的 App 记录仓库路径，不能单独复制到另一台机器使用。搬迁时先在新路径完成 `./setup-macos.sh`，再重新构建 App。

更新源码后重启后台服务，再运行 `./scripts/build_macos_app.sh` 和 `./scripts/build_macos_reader.sh`。原生工作台会检查 API 版本、必要能力和服务源码构建标识；端口上是旧服务时会明确提示重启。主服务 API 为版本 2；worker 使用随机本机端口与每次启动生成的私有凭证。

### 书架、分卷与音色

- **多文件长篇**：导入时可一次多选或把文件拖进窗口。多个文件可“合并为一本书”（按文件名顺序，每个文件一卷，“第十卷”排在“第二卷”之后），或“分别成书并归入同一系列”，也可“追加到已有书籍”作为新的一卷。正文里的“第 N 卷 / 卷 N / Volume N”会被识别为卷。
- **书架**：同一“系列”的书折叠为一组；支持按书名、系列搜索，按最近阅读、最近导入或书名排序；卡片显示“读到第几章”。书名与系列可在“管理”中修改。
- **目录**：按卷分组可折叠；没有分卷且超过 150 章时按每 100 章分组；可搜索章节标题或直接输入章号回车跳转。
- **音色互不干扰**：每本书保存自己的朗读设置（音色、模型、Seed、AAC 码率）。服务默认音色（QwenTTS App 的“设为服务默认”）只用于 API 调用和新导入的书，修改它不会改变已导入的书；实时流式、在线整块与整书生成都使用书本身的设置。在声阅中通过“听书 → 更改”或“设置 → 本书朗读”修改；更换音色/模型/Seed 会先提示将清除的已生成音频，只改码率不需要重新合成。
- 阅读外观（字号、行距、字体、版心宽度、主题、播放时自动滚动）只保存在当前浏览器。

阅读器每两秒以及暂停、停止、离开页面时记录播放位置。已保存 AAC 从实际 `currentTime` 恢复；实时播放会用保存的种子重新生成当前块，再跳过已听样本。在目录中切换章节也会更新书签。更换模型、音色或生成设置后，阅读位置保留，但重新生成音频的时长可能变化，建议从块首重听。失败或截断的流不会推进到下一块。

服务每分钟执行维护：完成后 5 分钟回收未消费 PCM，1 小时后回收临时听书结果，24 小时后回收失联的上传和 AAC 缓存。普通生成结果、书籍和参考库不自动删除。历史任务保留在磁盘，默认缓存最近 1000 个任务到内存，较早结果可按任务 ID 加载。容量检查使用最近一分钟的统计，磁盘剩余空间在每次接纳任务时检查。

可在 `.env.macos` 设置以下策略（单位均为字节或秒）：

```dotenv
QWEN_TTS_STORAGE_MAX_BYTES=10737418240
QWEN_TTS_STORAGE_MIN_FREE_BYTES=1073741824
QWEN_TTS_JOB_CACHE_LIMIT=1000
QWEN_TTS_JOB_HISTORY_LIMIT=20000
QWEN_TTS_PCM_TTL_SECONDS=300
QWEN_TTS_EPHEMERAL_TTL_SECONDS=3600
QWEN_TTS_WORKER_WAIT_TIMEOUT=330
```

容量达到上限或剩余空间不足时返回 HTTP 507，活动生成队列满时返回 429。DOCX 的上传和解压正文均限制为 30 MB，每次导入最多 500 万字；文档解析在线程池运行，整书生成对转码积压施加背压。

运行后台、阅读器和原生行为回归：

```bash
.venv/bin/python -m pip install -e '.[ci]'
.venv/bin/python -m pytest tests/ -q
node --test tests/test_reader_playback.cjs tests/test_reader_player.cjs
./scripts/test_native_macos.sh
```

也可以一次跑完全部检查并构建两个 App：`./scripts/verify_all.sh`（加 `--no-build` 只跑测试）。
