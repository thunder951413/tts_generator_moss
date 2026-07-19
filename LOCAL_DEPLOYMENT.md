# Local MOSS-TTS service

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
