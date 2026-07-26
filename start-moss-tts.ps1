$ErrorActionPreference = "Stop"

$repoDir = $PSScriptRoot
$python = Join-Path $repoDir ".venv\Scripts\python.exe"
$app = Join-Path $repoDir "clis\moss_tts_local_v1.5_app.py"
$modelDir = Join-Path $repoDir "models\MOSS-TTS-Local-Transformer-v1.5"
$codecDir = Join-Path $repoDir "models\MOSS-Audio-Tokenizer-v2"
$liteModelDir = Join-Path $repoDir "models\MOSS-TTS-Local-Transformer"
$liteCodecDir = Join-Path $repoDir "models\MOSS-Audio-Tokenizer"
$qwenRoot = Join-Path (Split-Path $repoDir -Parent) "faster-qwen3-tts"
$qwenPython = Join-Path $qwenRoot ".venv\Scripts\python.exe"
$qwen06ModelDir = Join-Path $qwenRoot "models\Qwen3-TTS-12Hz-0.6B-Base"
$qwen17ModelDir = Join-Path $qwenRoot "models\Qwen3-TTS-12Hz-1.7B-Base"
$qwenPerformanceFile = Join-Path $repoDir "qwen-performance.json"
$ffmpegBin = Join-Path $repoDir ".ffmpeg-runtime\Library\bin"
$logDir = Join-Path $repoDir "logs"
$pidFile = Join-Path $repoDir "moss-tts.pid"
$passwordFile = Join-Path $repoDir ".moss-tts-password"
$stdoutLog = Join-Path $logDir "moss-tts.out.log"
$stderrLog = Join-Path $logDir "moss-tts.err.log"
$qwenPerformance = if (Test-Path -LiteralPath $qwenPerformanceFile) {
    Get-Content -LiteralPath $qwenPerformanceFile -Raw | ConvertFrom-Json
} else {
    [pscustomobject]@{ qwen_0_6b_lanes = 1; qwen_1_7b_lanes = 1 }
}
$qwen06Lanes = if ($env:QWEN_TTS_0_6B_LANES) { [int]$env:QWEN_TTS_0_6B_LANES } else { [int]$qwenPerformance.qwen_0_6b_lanes }
$qwen17Lanes = if ($env:QWEN_TTS_1_7B_LANES) { [int]$env:QWEN_TTS_1_7B_LANES } else { [int]$qwenPerformance.qwen_1_7b_lanes }
$maxParallelGenerations = if ($env:MOSS_TTS_MAX_PARALLEL_GENERATIONS) {
    [int]$env:MOSS_TTS_MAX_PARALLEL_GENERATIONS
} else {
    [Math]::Max(2, [Math]::Max($qwen06Lanes, $qwen17Lanes))
}

if (-not (Test-Path -LiteralPath $python)) { throw "Python environment not found: $python" }
if (-not (Test-Path -LiteralPath $modelDir)) { throw "TTS model not found: $modelDir" }
if (-not (Test-Path -LiteralPath $codecDir)) { throw "Audio tokenizer not found: $codecDir" }
if (-not (Test-Path -LiteralPath $liteModelDir)) { throw "1.7B TTS model not found: $liteModelDir" }
if (-not (Test-Path -LiteralPath $liteCodecDir)) { throw "1.7B audio tokenizer not found: $liteCodecDir" }
if (-not (Test-Path -LiteralPath $qwenPython -PathType Leaf)) { throw "Faster Qwen3-TTS Python environment not found: $qwenPython" }
if (-not (Test-Path -LiteralPath (Join-Path $qwen06ModelDir "config.json") -PathType Leaf)) { throw "Qwen3-TTS 0.6B model not found: $qwen06ModelDir" }
if (-not (Test-Path -LiteralPath (Join-Path $qwen17ModelDir "config.json") -PathType Leaf)) { throw "Qwen3-TTS 1.7B model not found: $qwen17ModelDir" }
if (-not (Test-Path -LiteralPath $ffmpegBin)) { throw "Project FFmpeg 7 runtime not found: $ffmpegBin" }

if ($env:MOSS_TTS_ACCESS_PASSWORD) {
    $servicePassword = $env:MOSS_TTS_ACCESS_PASSWORD
} elseif (Test-Path -LiteralPath $passwordFile -PathType Leaf) {
    $servicePassword = (Get-Content -LiteralPath $passwordFile -Raw).Trim()
} else {
    $passwordBytes = New-Object byte[] 18
    $passwordRng = [Security.Cryptography.RandomNumberGenerator]::Create()
    try { $passwordRng.GetBytes($passwordBytes) } finally { $passwordRng.Dispose() }
    $servicePassword = [Convert]::ToBase64String($passwordBytes).TrimEnd('=').Replace('+', '-').Replace('/', '_')
    Set-Content -LiteralPath $passwordFile -Value $servicePassword -Encoding ASCII
}
if (-not $servicePassword) { throw "Service password must not be empty." }

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
$env:MOSS_TTS_ACCESS_PASSWORD = $servicePassword
$arguments = @(
    $app,
    "--host", "0.0.0.0",
    "--port", "7861",
    "--model-dir", $modelDir,
    "--codec-dir", $codecDir,
    "--lite-model-dir", $liteModelDir,
    "--lite-codec-dir", $liteCodecDir,
    "--qwen-python", $qwenPython,
    "--qwen-0-6b-model-dir", $qwen06ModelDir,
    "--qwen-1-7b-model-dir", $qwen17ModelDir,
    "--qwen-0-6b-lanes", $qwen06Lanes,
    "--qwen-1-7b-lanes", $qwen17Lanes,
    "--dtype", "bfloat16",
    "--attn-implementation", "sdpa",
    "--codec-weight-dtype", "bf16",
    "--codec-compute-dtype", "bf16",
    "--max-parallel-generations", $maxParallelGenerations
)

$process = Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $repoDir -WindowStyle Hidden -RedirectStandardOutput $stdoutLog -RedirectStandardError $stderrLog -PassThru
$serverPid = $null
for ($attempt = 0; $attempt -lt 600; $attempt++) {
    $listener = Get-NetTCPConnection -LocalPort 7861 -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
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
Write-Output "Local TTS service started (PID $serverPid). Local: http://127.0.0.1:7861"
Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue |
    Where-Object { $_.IPAddress -notlike '127.*' -and $_.IPAddress -notlike '169.254.*' -and $_.AddressState -eq 'Preferred' } |
    Select-Object -ExpandProperty IPAddress -Unique |
    ForEach-Object { Write-Output "LAN: http://$($_):7861" }
Write-Output "Service password: $servicePassword"
Write-Output "Health: http://127.0.0.1:7861/api/health"
