# Local Qwen3-TTS service

## macOS Apple Silicon / Metal

The macOS service uses Faster Qwen3-TTS with a native `qwentts.cpp` GGML
backend. Both the talker and audio codec run on Metal; model weights are
downloaded to the Hugging Face cache on first use.

Prerequisites:

- Apple Silicon Mac
- Python 3.12, CMake, Ninja, FFmpeg, libsndfile, and whisper.cpp

Install Homebrew prerequisites and build the pinned Metal runtime:

```bash
brew install python@3.12 cmake ninja ffmpeg libsndfile whisper-cpp
./setup-macos.sh
```

Edit `.env.macos`, especially `QWEN_TTS_ACCESS_PASSWORD`, then run:

```bash
./start-macos-app.sh
curl http://127.0.0.1:7861/api/health
```

`start-macos-app.sh` builds and opens `dist/QwenTTS.app`. This is the normal
desktop entry point: it lives in the macOS menu bar, starts the local service
when necessary, shows Metal/model/GPU-slot status, and opens a fully native
SwiftUI/AppKit audio-test workspace. It does not embed the web page or link
WebKit. The native controls call the same password-protected localhost API used
by the separately available web interface.

The **切换语音预设** menu is backed by the same persistent presets that appear
at the top of the text-test workspace. A preset saves the selected Qwen model,
reference audio, ICL transcript, seed, sampling values, stream settings and
other Qwen controls. Saving a custom uploaded/recorded reference copies it to
`outputs/qwen_tts_presets/reference_audio/`, so it continues to work from the
menu bar after the browser window is closed. The workbench can apply, update,
preview and delete those presets; menu-bar selection applies the exact same
settings in the native window.

The native workspace includes model and clone-mode selection, the built-in
voice catalog, reference-audio preview, local audio import, microphone
recording, ICL transcript editing, seed and sampling controls, generation
progress, stop control, and native playback of the generated WAV.

For a headless/local-network service without the macOS window, keep using:

```bash
./start-macos.sh
```

To rebuild the application explicitly after source changes:

```bash
./scripts/build_macos_app.sh
open ./dist/QwenTTS.app
```

The first start downloads the 0.6B Q4_K_M checkpoint and tokenizer. Selecting
the 1.7B profile downloads its talker checkpoint. Stop with
`./stop-macos.sh`.

## Unified TTS + STT background API

The main service listens on `0.0.0.0:7861` by default, so any application on
this Mac or the local network can use the API. The internal whisper.cpp server
continues to listen only on `127.0.0.1:7890`; external clients always go
through the authenticated main service.

Use either HTTP header with the password from `.env.macos`:

```text
Authorization: Bearer 1234
X-API-Key: 1234
```

Browser CORS preflight is enabled for all origins. This makes the service easy
to call from local web apps, command-line programs, automation tools and native
applications without removing authentication. Do not expose port 7861 directly
to the public internet with the example password.

TTS uses the existing two-stage streaming protocol. Start a job, then consume
the returned raw PCM endpoint while generation is still running:

```bash
curl http://127.0.0.1:7861/api/generate-stream/start \
  -H 'Authorization: Bearer 1234' \
  -F 'text=这是流式语音测试。' \
  -F 'model_profile=qwen_0_6b' \
  -F 'streaming_generation=1'

curl -N http://127.0.0.1:7861/api/generate-stream/JOB_ID/audio \
  -H 'Authorization: Bearer 1234' \
  --output speech.pcm
```

The stream response headers describe `pcm_s16le`, sample rate and channel
count. Poll `/api/generate-stream/JOB_ID/status` for progress or download the
final WAV from `/api/generate-stream/JOB_ID/result-audio`.

STT is compatible with the common OpenAI audio-transcription request shape:

```bash
curl http://127.0.0.1:7861/v1/audio/transcriptions \
  -H 'Authorization: Bearer 1234' \
  -F file=@sample.wav \
  -F model=whisper-small \
  -F language=zh \
  -F response_format=verbose_json
```

Supported response formats are `json`, `verbose_json`, `text`, `srt`, and
`vtt`. `/api/health`, `/api/runtime`, and `/api/stt/status` report the two
resident Metal runtimes separately.

Measured idle resident-set totals on this M4 Max 128 GB machine, after one real
TTS generation and one real STT request:

| Resident configuration | TTS workers | API | STT small | Service total |
| --- | ---: | ---: | ---: | ---: |
| Qwen 0.6B Q4_K_M, 2 lanes + Whisper small | 6,139 MiB | 226 MiB | 705 MiB | 7,071 MiB |
| Qwen 1.7B Q4_K_M, 2 lanes + Whisper small | 8,396 MiB | 227 MiB | 703 MiB | 9,325 MiB |

The menu-bar application adds about 139 MiB. Only the selected Qwen TTS profile
is resident; switching between 0.6B and 1.7B unloads the previous TTS workers.
Whisper small stays resident and normally adds about 0.7 GiB.

On the measured 40-core M4 Max with 128 GB unified memory, two lanes per model
are the recommended mixed/interactive setting. Four lanes have the highest
aggregate batch throughput, but each concurrent request becomes slower than
realtime. Reproduce the measurements with:

```bash
export QWENTTS_CPP_LIBRARY="$PWD/.runtime/qwentts.cpp/build-metal/libqwen.dylib"
export DYLD_LIBRARY_PATH="$PWD/.runtime/qwentts.cpp/build-metal"
.venv/bin/python scripts/benchmark_qwen_concurrency.py \
  --profile-id qwen_0_6b \
  --model-dir Qwen/Qwen3-TTS-12Hz-0.6B-Base \
  --qwen-python "$PWD/.venv/bin/python" \
  --reference-audio "$PWD/assets/audio/reference_zh_1.wav" \
  --max-lanes 4 --frames 96 \
  --output "$PWD/outputs/qwen_benchmark_m4max_06.json"
```

Measured summaries and the active recommendation are stored in
`qwen-performance.json`.

The document workspace is optimized for full books. One project keeps two
Metal lanes full with different text segments while AAC encoding runs in a
separate two-worker CPU pool. Segment files may finish out of order, but their
zero-padded indices and the final concat manifest always preserve source order.
Interactive text/API requests retain priority for the next available GPU slot;
already-running book segments are allowed to finish. Configure this independently
with `QWEN_TTS_DOCUMENT_PARALLEL_GENERATIONS` (default `2`).

### Local novel reader

Open `http://127.0.0.1:7866/reader` (or choose **打开小说阅读器** from the
menu-bar app). The reader accepts TXT, Markdown, and DOCX novels, recognizes
common Chinese and English chapter headings, and maps every chapter to
approximately 120–260 character display/audio blocks.

- **实时听本章** submits blocks in reading order to the PCM streaming endpoint,
  keeps one resolved Seed across the listening session, and highlights the
  block currently being played.
- **高质量生成本章** runs only the selected chapter through the persistent
  two-lane document pipeline. Finished blocks are stored as mono AAC-LC in M4A
  containers and become playable immediately. The reader offers 48, 64, 80,
  96, 128, and 192 kbps; 80 kbps is the default voice-quality recommendation.
- Once every block is complete, the ordered AAC segments are concatenated into
  `final/complete.m4a`. Intermediate WAV files are removed after AAC encoding.

## Legacy Windows MOSS-TTS deployment

This deployment uses MOSS-TTS-Local-Transformer-v1.5 and MOSS-Audio-Tokenizer-v2 on CUDA.

## Windows environment setup

Prerequisites:

- Windows 10/11 with an NVIDIA GPU and a current NVIDIA driver.
- PowerShell 7 or Windows PowerShell 5.1.
- `uv` and `conda` available in `PATH`. They can be installed with `winget install --id astral-sh.uv -e` and `winget install --id CondaForge.Miniforge3 -e`.
- Git LFS is recommended for the upstream repository, but model weights are downloaded directly from Hugging Face and are not committed to Git.

From the repository root, create the Python environment, install the CUDA runtime, prepare FFmpeg 7 shared libraries, and download both models:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup-moss-tts.ps1
```

The setup is idempotent. To verify an existing installation without changing or downloading anything:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup-moss-tts.ps1 -CheckOnly
```

Use `-SkipModelDownload` when the two model directories have already been copied into `models/`. Model weights, `.venv`, `.ffmpeg-runtime`, logs, generated documents, and service PID files are intentionally ignored by Git.

The required model directories are:

- `models/MOSS-TTS-Local-Transformer-v1.5`
- `models/MOSS-Audio-Tokenizer-v2`

After setup:

- Web UI: http://127.0.0.1:7861
- Health API: http://127.0.0.1:7861/api/health
- Start: `powershell -ExecutionPolicy Bypass -File .\start-moss-tts.ps1`
- Stop: `powershell -ExecutionPolicy Bypass -File .\stop-moss-tts.ps1`
- Logs: `logs/moss-tts.out.log` and `logs/moss-tts.err.log`

The model and codec weights are stored under `models/`, so normal startup works offline after the first setup.
The project-local `.ffmpeg-runtime` contains FFmpeg 7 shared libraries required by TorchCodec on Windows; it does not replace the system FFmpeg installation.

## Document audio projects

The web UI supports persistent TXT, Markdown, and DOCX audio projects. Project data is stored under `outputs/moss_tts_document_projects/<project-id>/`.

- `manifest.json`: project state, fixed generation settings, segment status, progress, errors, and playback position.
- `sources/`: every document added to the project.
- `segments/`: completed AAC-LC segments in M4A containers.
- `final/complete.m4a`: concatenated final audio after all segments finish.

Generation is non-streaming at the document API boundary: a segment becomes playable only after its complete WAV is generated, converted to AAC, validated, and atomically installed. Stop requests finish the current segment and then persist a paused state. Starting with changed voice or sampling settings clears prior segment/final audio and restarts the project from the beginning.

## Generation performance

- The TTS model and audio codec are preloaded, warmed up, and kept resident on CUDA until the service stops.
- Clone reference audio codes are cached by resolved path, modification time, file size, and quantizer count.
- New document projects default to 200 Chinese characters per segment and 16 codec chunk frames.
- GPU synthesis of the next segment overlaps CPU AAC encoding of the previous segment.
- A shared priority scheduler covers both interactive text tests and document projects. Interactive work has queue priority.
- Token generation may run concurrently, while the stateful audio codec is serialized so concurrent jobs cannot corrupt one another's streaming decoder state. AAC encoding uses one CPU thread per segment.
- The RTX 4090 Laptop deployment defaults to two concurrent GPU generations. Override it with `MOSS_TTS_MAX_PARALLEL_GENERATIONS=1` before starting the service if a lower-memory GPU needs serial execution.
- `/api/health` reports scheduler concurrency and reference-cache hit counters.

Safe-concurrency scaling on the installed RTX 4090 Laptop GPU was measured with identical fixed-seed, 30.64-second document outputs. Two-way concurrency reached 1.615x aggregate realtime speed, versus 1.590x for three-way and 1.591x for four-way. Peak GPU memory was about 13.38 GB, 13.67 GB, and 14.07 GB respectively. All completed outputs had identical decoded PCM. Two-way concurrency is therefore the default; higher values add latency and memory use without improving aggregate throughput on this machine. Per-request CUDA random generators keep generated tokens deterministic under concurrency.
