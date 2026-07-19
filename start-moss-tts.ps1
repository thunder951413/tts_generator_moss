$ErrorActionPreference = "Stop"

$repoDir = $PSScriptRoot
$python = Join-Path $repoDir ".venv\Scripts\python.exe"
$app = Join-Path $repoDir "clis\moss_tts_local_v1.5_app.py"
$modelDir = Join-Path $repoDir "models\MOSS-TTS-Local-Transformer-v1.5"
$codecDir = Join-Path $repoDir "models\MOSS-Audio-Tokenizer-v2"
$ffmpegBin = Join-Path $repoDir ".ffmpeg-runtime\Library\bin"
$logDir = Join-Path $repoDir "logs"
$pidFile = Join-Path $repoDir "moss-tts.pid"
$stdoutLog = Join-Path $logDir "moss-tts.out.log"
$stderrLog = Join-Path $logDir "moss-tts.err.log"
$maxParallelGenerations = if ($env:MOSS_TTS_MAX_PARALLEL_GENERATIONS) { [int]$env:MOSS_TTS_MAX_PARALLEL_GENERATIONS } else { 2 }

if (-not (Test-Path -LiteralPath $python)) { throw "Python environment not found: $python" }
if (-not (Test-Path -LiteralPath $modelDir)) { throw "TTS model not found: $modelDir" }
if (-not (Test-Path -LiteralPath $codecDir)) { throw "Audio tokenizer not found: $codecDir" }
if (-not (Test-Path -LiteralPath $ffmpegBin)) { throw "Project FFmpeg 7 runtime not found: $ffmpegBin" }

if (Test-Path -LiteralPath $pidFile) {
    $existingPid = [int](Get-Content -LiteralPath $pidFile -Raw)
    if (Get-Process -Id $existingPid -ErrorAction SilentlyContinue) {
        Write-Output "MOSS-TTS is already running (PID $existingPid): http://127.0.0.1:7861"
        exit 0
    }
    Remove-Item -LiteralPath $pidFile -Force
}

New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$env:PATH = "$ffmpegBin;$env:PATH"
$env:PYTHONPATH = if ($env:PYTHONPATH) { "$repoDir;$env:PYTHONPATH" } else { $repoDir }
$arguments = @(
    $app,
    "--host", "127.0.0.1",
    "--port", "7861",
    "--model-dir", $modelDir,
    "--codec-dir", $codecDir,
    "--dtype", "bfloat16",
    "--attn-implementation", "sdpa",
    "--codec-weight-dtype", "bf16",
    "--codec-compute-dtype", "bf16",
    "--max-parallel-generations", $maxParallelGenerations
)

$process = Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $repoDir -WindowStyle Hidden -RedirectStandardOutput $stdoutLog -RedirectStandardError $stderrLog -PassThru
$serverPid = $null
for ($attempt = 0; $attempt -lt 60; $attempt++) {
    $listener = Get-NetTCPConnection -LocalAddress "127.0.0.1" -LocalPort 7861 -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($listener) {
        $serverPid = $listener.OwningProcess
        break
    }
    Start-Sleep -Milliseconds 500
}
if (-not $serverPid) {
    throw "MOSS-TTS did not open port 7861. See $stderrLog"
}
Set-Content -LiteralPath $pidFile -Value $serverPid
Write-Output "MOSS-TTS started (PID $serverPid). Loading models at http://127.0.0.1:7861"
Write-Output "Health: http://127.0.0.1:7861/api/health"
