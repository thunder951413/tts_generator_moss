[CmdletBinding()]
param(
    [switch]$CheckOnly,
    [switch]$SkipModelDownload
)

$ErrorActionPreference = "Stop"

$repoDir = $PSScriptRoot
$venvDir = Join-Path $repoDir ".venv"
$python = Join-Path $venvDir "Scripts\python.exe"
$hfCli = Join-Path $venvDir "Scripts\hf.exe"
$ffmpegRuntime = Join-Path $repoDir ".ffmpeg-runtime"
$ffmpegBin = Join-Path $ffmpegRuntime "Library\bin"
$ffmpeg = Join-Path $ffmpegBin "ffmpeg.exe"
$sox = Join-Path $ffmpegBin "sox.exe"
$modelRoot = Join-Path $repoDir "models"
$ttsModelDir = Join-Path $modelRoot "MOSS-TTS-Local-Transformer-v1.5"
$codecModelDir = Join-Path $modelRoot "MOSS-Audio-Tokenizer-v2"
$liteTtsModelDir = Join-Path $modelRoot "MOSS-TTS-Local-Transformer"
$liteCodecModelDir = Join-Path $modelRoot "MOSS-Audio-Tokenizer"
$qwenRoot = Join-Path (Split-Path $repoDir -Parent) "faster-qwen3-tts"
$qwenPython = Join-Path $qwenRoot ".venv\Scripts\python.exe"
$qwen06ModelDir = Join-Path $qwenRoot "models\Qwen3-TTS-12Hz-0.6B-Base"
$qwen17ModelDir = Join-Path $qwenRoot "models\Qwen3-TTS-12Hz-1.7B-Base"

function Require-Command {
    param([Parameter(Mandatory = $true)][string]$Name, [string]$InstallHint)
    $command = Get-Command $Name -ErrorAction SilentlyContinue
    if (-not $command) {
        throw "Required command '$Name' was not found. $InstallHint Open a new PowerShell window after installation."
    }
    return $command.Source
}

function Assert-File {
    param([Parameter(Mandatory = $true)][string]$Path, [string]$Description)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "$Description was not found: $Path"
    }
}

function Test-Installation {
    Assert-File -Path $python -Description "Project Python runtime"
    Assert-File -Path $ffmpeg -Description "Project FFmpeg runtime"
    Assert-File -Path $sox -Description "Project SoX runtime"
    Assert-File -Path (Join-Path $ttsModelDir "config.json") -Description "MOSS-TTS model"
    Assert-File -Path (Join-Path $codecModelDir "config.json") -Description "MOSS audio tokenizer"
    Assert-File -Path (Join-Path $liteTtsModelDir "model.safetensors.index.json") -Description "MOSS-TTS 1.7B model"
    Assert-File -Path (Join-Path $liteCodecModelDir "model.safetensors.index.json") -Description "MOSS 1.7B audio tokenizer"
    Assert-File -Path $qwenPython -Description "Faster Qwen3-TTS Python runtime"
    Assert-File -Path (Join-Path $qwen06ModelDir "config.json") -Description "Qwen3-TTS 0.6B model"
    Assert-File -Path (Join-Path $qwen17ModelDir "config.json") -Description "Qwen3-TTS 1.7B model"

    $previousPath = $env:PATH
    $previousPythonPath = $env:PYTHONPATH
    try {
        $env:PATH = "$ffmpegBin;$env:PATH"
        $env:PYTHONPATH = if ($env:PYTHONPATH) { "$repoDir;$env:PYTHONPATH" } else { $repoDir }
        & $python -c "import torch, torchcodec, transformers, fastapi, uvicorn, modelscope; assert torch.cuda.is_available(), 'CUDA is not available to PyTorch'; print(f'Python environment OK | torch={torch.__version__} | CUDA={torch.cuda.get_device_name(0)}')"
        if ($LASTEXITCODE -ne 0) { throw "Python/CUDA validation failed." }
        & $qwenPython -c "import torch, transformers, qwen_tts, faster_qwen3_tts; assert torch.cuda.is_available(), 'Qwen CUDA is not available'; print(f'Faster Qwen3-TTS OK | torch={torch.__version__} | transformers={transformers.__version__}')"
        if ($LASTEXITCODE -ne 0) { throw "Faster Qwen3-TTS validation failed." }
        & $ffmpeg -version | Select-Object -First 1
        if ($LASTEXITCODE -ne 0) { throw "FFmpeg validation failed." }
    } finally {
        $env:PATH = $previousPath
        $env:PYTHONPATH = $previousPythonPath
    }
}

if ($CheckOnly) {
    Test-Installation
    Write-Output "MOSS-TTS local environment is ready."
    exit 0
}

$uv = Require-Command -Name "uv" -InstallHint "Install it with: winget install --id astral-sh.uv -e."
$conda = Require-Command -Name "conda" -InstallHint "Install it with: winget install --id CondaForge.Miniforge3 -e."
$git = Require-Command -Name "git" -InstallHint "Install it with: winget install --id Git.Git -e."

Set-Location -LiteralPath $repoDir
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    & $uv venv $venvDir --python 3.12
    if ($LASTEXITCODE -ne 0) { throw "Failed to create the Python environment." }
}

& $uv pip install --python $python --torch-backend cu128 -e ".[torch-runtime,local-app]"
if ($LASTEXITCODE -ne 0) { throw "Failed to install Python dependencies." }

if (-not (Test-Path -LiteralPath $ffmpeg -PathType Leaf)) {
    & $conda create --prefix $ffmpegRuntime --channel conda-forge "ffmpeg=7" --yes
    if ($LASTEXITCODE -ne 0) { throw "Failed to create the FFmpeg 7 runtime." }
}
if (-not (Test-Path -LiteralPath $sox -PathType Leaf)) {
    & $conda install --prefix $ffmpegRuntime --channel conda-forge sox --yes
    if ($LASTEXITCODE -ne 0) { throw "Failed to install the SoX runtime." }
}

if (-not (Test-Path -LiteralPath (Join-Path $qwenRoot ".git") -PathType Container)) {
    & $git clone "https://github.com/andimarafioti/faster-qwen3-tts.git" $qwenRoot
    if ($LASTEXITCODE -ne 0) { throw "Failed to clone Faster Qwen3-TTS." }
}
if (-not (Test-Path -LiteralPath $qwenPython -PathType Leaf)) {
    & $uv python install 3.10
    if ($LASTEXITCODE -ne 0) { throw "Failed to install uv-managed Python 3.10." }
    & $uv venv (Join-Path $qwenRoot ".venv") --python 3.10 --managed-python
    if ($LASTEXITCODE -ne 0) { throw "Failed to create the Faster Qwen3-TTS environment." }
}
& $uv pip install --python $qwenPython --torch-backend cu128 -e $qwenRoot "numba>=0.61" "llvmlite>=0.44" "fastapi>=0.100" "uvicorn[standard]>=0.24" "python-multipart>=0.0.7"
if ($LASTEXITCODE -ne 0) { throw "Failed to install Faster Qwen3-TTS dependencies." }

if (-not $SkipModelDownload) {
    Assert-File -Path $hfCli -Description "Hugging Face CLI"
    New-Item -ItemType Directory -Force -Path $modelRoot | Out-Null
    & $hfCli download "OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5" --local-dir $ttsModelDir
    if ($LASTEXITCODE -ne 0) { throw "Failed to download the MOSS-TTS model." }
    & $hfCli download "OpenMOSS-Team/MOSS-Audio-Tokenizer-v2" --local-dir $codecModelDir
    if ($LASTEXITCODE -ne 0) { throw "Failed to download the audio tokenizer." }
    & $hfCli download "OpenMOSS-Team/MOSS-TTS-Local-Transformer" --local-dir $liteTtsModelDir
    if ($LASTEXITCODE -ne 0) { throw "Failed to download the MOSS-TTS 1.7B model." }
    & $hfCli download "OpenMOSS-Team/MOSS-Audio-Tokenizer" --local-dir $liteCodecModelDir
    if ($LASTEXITCODE -ne 0) { throw "Failed to download the 1.7B audio tokenizer." }
    & $python (Join-Path $repoDir "scripts\download_qwen_models.py")
    if ($LASTEXITCODE -ne 0) { throw "Failed to download Qwen3-TTS 0.6B/1.7B." }
}

Test-Installation
Write-Output "Setup complete. Start the service with: powershell -ExecutionPolicy Bypass -File .\start-moss-tts.ps1"
