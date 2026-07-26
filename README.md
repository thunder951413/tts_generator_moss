# TTS Generator

一个运行在 Windows + NVIDIA GPU 上的本地语音生成服务。它通过统一的 Web 界面提供短文本试听、音色克隆、长文档分段生成、后台任务、断点续作和 AAC/M4A 导出。

网页只负责提交与查看任务，实际生成由常驻后台服务完成。关闭网页、刷新页面或换一台设备登录，不会中断已经提交的任务。

## 当前应用状态

- 支持本地文本测试和文件生成两个工作区。
- 提供四个可切换的生成档位，模型切换时安全换载显存。
- 服务端任务独立于浏览器运行，局域网内的其他设备登录后可继续查看。
- 文本任务记录状态、模型、音色、Seed、生成帧数和最终音频。
- 文件项目保存来源文档、固定参数、分段进度、错误、播放位置和已生成音频。
- 支持停止和继续文件项目；失败后从未完成段落恢复，不会废弃此前结果。
- 每完成一个段落即可点击试听，并能连续播放后续已经完成的段落。
- 文档段落转换为 48 kHz 双声道 AAC-LC，全部完成后合并为 M4A。
- 支持流式生成、边生成边试听、可调预缓冲和随时停止当前生成。
- 模型与音频编解码器常驻 GPU，减少重复加载开销。
- 使用共享 GPU 调度器协调文本任务和文件项目。
- 页面控件、展开状态、音色收藏/隐藏和项目选择均持久保存。
- 使用持久服务密码保护局域网访问。

## 模型与克隆方式

应用当前提供以下档位：

| 档位 | 输出 | 主要用途 |
| --- | --- | --- |
| 高质量 4B | 48 kHz 立体声 | 高质量中文生成 |
| 轻量 1.7B | 24 kHz 单声道 | 较低显存占用 |
| Qwen 0.6B | 24 kHz 单声道 | 默认极速克隆 |
| Qwen 1.7B | 24 kHz 单声道 | 更高质量克隆 |

Qwen 档位支持两种克隆方式：

- **X-vector**：不需要参考文字，适合快速测试和跨语言克隆。
- **ICL**：同时使用参考音频和逐字对应的参考文字，优先保证音色相似度。

内置 40 个本地参考音色，包含中文、粤语、闽南语、英语、日语、韩语和印尼语。每个音色已经预置 ICL 参考文案；选择音色后自动填入，也可以在浏览器中按音色校正并持久保存。

当前推理后端基于：

- [OpenMOSS/MOSS-TTS](https://github.com/OpenMOSS/MOSS-TTS)
- [andimarafioti/faster-qwen3-tts](https://github.com/andimarafioti/faster-qwen3-tts)

## 运行要求

- Windows 10/11
- NVIDIA GPU，建议至少 16 GB 显存
- 支持 CUDA 12.8 的 NVIDIA 驱动
- PowerShell
- [uv](https://docs.astral.sh/uv/)
- conda 或 Miniforge

安装脚本会创建应用所需的 Python 3.12 主环境、独立的 Qwen Python 3.10 环境以及 FFmpeg 7 运行库。模型权重、虚拟环境、日志和生成结果不会提交到 Git。

## 安装

```powershell
git clone --recurse-submodules https://github.com/thunder951413/tts_generator_moss.git
cd tts_generator_moss
```

安装基础工具：

```powershell
winget install --id astral-sh.uv -e
winget install --id CondaForge.Miniforge3 -e
```

重新打开 PowerShell，然后执行：

```powershell
powershell -ExecutionPolicy Bypass -File .\setup-moss-tts.ps1
```

配置脚本会自动完成：

1. 创建主 Python 3.12 环境并安装 CUDA 版 PyTorch。
2. 创建项目专用 FFmpeg 7 运行库。
3. 下载高质量和轻量模型及对应音频 Tokenizer。
4. 在相邻目录安装 Faster Qwen3-TTS 运行时。
5. 创建独立的 Qwen Python 3.10 + CUDA 环境。
6. 下载 Qwen 0.6B 和 1.7B Base 模型。
7. 验证 CUDA、FFmpeg、Python 环境和全部模型文件。

已有模型时可以跳过重复下载：

```powershell
powershell -ExecutionPolicy Bypass -File .\setup-moss-tts.ps1 -SkipModelDownload
```

只检查环境：

```powershell
powershell -ExecutionPolicy Bypass -File .\setup-moss-tts.ps1 -CheckOnly
```

## 启动与访问

启动服务：

```powershell
powershell -ExecutionPolicy Bypass -File .\start-moss-tts.ps1
```

本机访问：<http://127.0.0.1:7861>

启动脚本会输出可用的局域网地址。首次启动自动生成访问密码并保存在 `.moss-tts-password`；该文件不会提交到 Git。也可以在启动前指定密码：

```powershell
$env:MOSS_TTS_ACCESS_PASSWORD = "请换成足够长的密码"
powershell -ExecutionPolicy Bypass -File .\start-moss-tts.ps1
```

不要把 7861 端口直接暴露到互联网。需要公网访问时，应使用带 HTTPS 和访问控制的反向代理。

其他局域网设备无法连接时，可在管理员 PowerShell 中放行专用端口：

```powershell
New-NetFirewallRule -DisplayName "TTS Generator Service" -Direction Inbound -Protocol TCP -LocalPort 7861 -Action Allow -Profile Private
```

停止服务：

```powershell
powershell -ExecutionPolicy Bypass -File .\stop-moss-tts.ps1
```

健康检查：<http://127.0.0.1:7861/api/health>

## 文本生成

文本生成支持：

- 选择模型、参考音色和对应的采样参数。
- 固定 Seed 或使用 `-1` 让服务端随机生成实际 Seed。
- 在 Status、状态摘要和任务中心查看本次推理真正使用的 Seed；随机 Seed 也可以据此复现。
- 流式生成及播放，或关闭流式输出后等待完整音频。
- 关闭网页后继续生成。
- 从任务中心重新打开运行中或已完成的任务。
- 完成后直接播放或下载 WAV。

## 文件项目

支持导入 TXT、Markdown 和 DOCX。拖入第一个文件时创建项目，之后可以继续向同一项目追加文档。

项目创建时会锁定：

- 模型档位
- 参考音色与音频
- 克隆方式及 ICL 参考文案
- Seed
- 采样参数
- 分段长度

切换页面上的模型或参数不会改变已有项目，也不会使已完成段落失效。要使用另一套参数，应创建新项目。

项目使用 `-1` 创建时，服务端会生成一个实际 Seed 并将它锁定到项目；后续所有分段和断点续作都使用同一个实际值。

项目目录：

```text
outputs/moss_tts_document_projects/<project-id>/
```

主要内容：

- `manifest.json`：项目配置、状态、分段、进度、错误和播放位置。
- `sources/`：加入项目的原始文档。
- `segments/`：已经完成的 AAC/M4A 段落。
- `final/complete.m4a`：全部段落完成后的最终音频。

## 性能配置

Qwen 的本机并发实测结果保存在 `qwen-performance.json`，启动服务时会读取推荐路数。当前配置针对 RTX 4090 Laptop GPU 16 GB：

- Qwen 0.6B：1 路
- Qwen 1.7B：1 路

可以临时覆盖：

```powershell
$env:QWEN_TTS_0_6B_LANES = "2"
$env:QWEN_TTS_1_7B_LANES = "1"
powershell -ExecutionPolicy Bypass -File .\start-moss-tts.ps1
```

重新测试并发：

```powershell
.\.venv\Scripts\python.exe .\scripts\benchmark_qwen_concurrency.py
```

## 本地数据与备份

以下内容默认不进入 Git：

```text
.venv/
.ffmpeg-runtime/
models/
logs/
outputs/
.moss-tts-password
moss-tts.pid
```

升级、清理或迁移前，至少备份：

```text
outputs/moss_tts_document_projects/
outputs/moss_tts_service_jobs/
.moss-tts-password
```

## 许可与音频使用

本仓库保留上游 [LICENSE](LICENSE)。推理代码、模型权重及相关组件分别遵循其对应上游项目和模型许可。

内置参考音频仅用于本地音色克隆测试。使用、分发参考音频或发布生成内容前，请确认拥有必要授权，并遵守音频来源平台条款及适用法律。
