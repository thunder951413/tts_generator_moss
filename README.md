# MOSS TTS Generator

一个面向 Windows + NVIDIA GPU 的本地文本转语音工具，基于 [OpenMOSS/MOSS-TTS](https://github.com/OpenMOSS/MOSS-TTS) 和 `MOSS-TTS-Local-Transformer-v1.5` 开发。

本仓库主要增加了本地 Web 界面、流式试听、音色克隆、长文档分段生成、AAC 导出、项目断点续作和 Windows 环境配置。模型架构、推理能力及基础代码来自 MOSS-TTS。

## 主要功能

- 本地文字转语音与实时流式试听。
- 自适应播放预缓冲，降低流式播放卡顿。
- 随时停止当前流式生成。
- 使用本地参考音频克隆音色。
- 内置音色收藏、隐藏和试听管理。
- 支持 TXT、Markdown 和 DOCX 文档项目。
- 长文档自动分段，按段保存生成状态并支持停止、继续。
- 已完成段落可立即播放，并能连续播放后续已生成内容。
- 段落转换为 AAC-LC，最终合并为 M4A 文件。
- 模型和音频编解码器常驻 CUDA。
- 共享 GPU 调度器支持交互生成和文档生成，默认最多并行两路。
- 页面设置、音色状态和文档项目状态持久化保存。

## 运行要求

- Windows 10/11
- NVIDIA GPU，建议至少 16 GB 显存
- 最新 NVIDIA 驱动
- Python 3.12
- PowerShell
- [uv](https://docs.astral.sh/uv/)
- conda 或 Miniforge

模型权重、Python 虚拟环境、FFmpeg 运行库、生成结果和项目数据不会存入 Git 仓库。

## 安装

克隆仓库：

```powershell
git clone --recurse-submodules https://github.com/thunder951413/tts_generator_moss.git
cd tts_generator_moss
```

安装 `uv` 和 Miniforge：

```powershell
winget install --id astral-sh.uv -e
winget install --id CondaForge.Miniforge3 -e
```

安装完成后重新打开 PowerShell，然后执行自动配置：

```powershell
powershell -ExecutionPolicy Bypass -File .\setup-moss-tts.ps1
```

配置脚本会自动：

1. 创建 `.venv` Python 3.12 环境。
2. 安装 CUDA 12.8 版 PyTorch 及项目依赖。
3. 创建项目专用的 FFmpeg 7 共享库环境。
4. 从 Hugging Face 下载 `MOSS-TTS-Local-Transformer-v1.5`。
5. 下载 `MOSS-Audio-Tokenizer-v2`。
6. 验证 CUDA、TorchCodec、FFmpeg 和模型文件。

如果模型已经放入 `models/`，可以跳过下载：

```powershell
powershell -ExecutionPolicy Bypass -File .\setup-moss-tts.ps1 -SkipModelDownload
```

只检查现有环境：

```powershell
powershell -ExecutionPolicy Bypass -File .\setup-moss-tts.ps1 -CheckOnly
```

## 启动与停止

启动：

```powershell
powershell -ExecutionPolicy Bypass -File .\start-moss-tts.ps1
```

打开：<http://127.0.0.1:7861>

停止：

```powershell
powershell -ExecutionPolicy Bypass -File .\stop-moss-tts.ps1
```

健康状态：<http://127.0.0.1:7861/api/health>

## 文档项目

文档项目保存在：

```text
outputs/moss_tts_document_projects/<project-id>/
```

其中包含：

- `manifest.json`：项目参数、段落状态、进度、错误和播放位置。
- `sources/`：加入项目的原始文档。
- `segments/`：已完成的 AAC/M4A 段落。
- `final/complete.m4a`：全部完成后合并的最终音频。

文档生成固定使用项目创建时的音色和采样参数。如果修改这些参数，项目会清除旧输出并从头生成，以避免同一个项目中出现不同音色或风格。

## 本地数据

以下内容默认被 Git 忽略：

```text
.venv/
.ffmpeg-runtime/
models/
logs/
outputs/
moss-tts.pid
```

请在升级、清理或迁移前自行备份 `outputs/moss_tts_document_projects/`。

## 上游项目与许可

本项目使用并修改了 [OpenMOSS/MOSS-TTS](https://github.com/OpenMOSS/MOSS-TTS)。MOSS-TTS 的模型、源码及相关组件应遵循其上游许可证和模型许可。本仓库继续保留上游 [LICENSE](LICENSE)。

内置参考音频仅用于本地音色克隆测试。使用相关音频、生成内容或对外发布前，请确认你拥有必要授权，并遵守音频来源平台及适用法律的要求。
