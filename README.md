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
- 建议 macOS 14 或更高版本
- Homebrew
- 0.6B 建议至少 16 GB 统一内存
- 1.7B 建议至少 24 GB 统一内存
- 模型缓存和构建文件需要约 15–30 GB 空间

安装依赖：

```bash
xcode-select --install
brew install python@3.12 cmake ninja ffmpeg libsndfile
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
5. 生成本地 `.env.macos`，但不会下载或提交模型权重。

## 配置

编辑 `.env.macos`：

```dotenv
QWEN_TTS_ACCESS_PASSWORD=请改成强密码
HOST=0.0.0.0
PORT=7861
QWEN_TTS_QUANT=Q4_K_M
QWEN_TTS_0_6B_LANES=1
QWEN_TTS_1_7B_LANES=1
QWEN_TTS_MAX_PARALLEL_GENERATIONS=1
```

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
http://Mac的局域网IP:7861
```

网页关闭后任务继续执行。重新登录后可以查看文本任务、文档项目、已完成
段落和当前进度。

停止服务：

```bash
./stop-macos.sh
```

日志：

```text
logs/service.out.log
logs/service.err.log
logs/qwen-workers/
```

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
