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
$modelRoot = Join-Path $repoDir "models"
$ttsModelDir = Join-Path $modelRoot "MOSS-TTS-Local-Transformer-v1.5"
$codecModelDir = Join-Path $modelRoot "MOSS-Audio-Tokenizer-v2"

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
    Assert-File -Path (Join-Path $ttsModelDir "config.json") -Description "MOSS-TTS model"
    Assert-File -Path (Join-Path $codecModelDir "config.json") -Description "MOSS audio tokenizer"

    $previousPath = $env:PATH
    $previousPythonPath = $env:PYTHONPATH
    try {
        $env:PATH = "$ffmpegBin;$env:PATH"
        $env:PYTHONPATH = if ($env:PYTHONPATH) { "$repoDir;$env:PYTHONPATH" } else { $repoDir }
        & $python -c "import torch, torchcodec, transformers, fastapi, uvicorn; assert torch.cuda.is_available(), 'CUDA is not available to PyTorch'; print(f'Python environment OK | torch={torch.__version__} | CUDA={torch.cuda.get_device_name(0)}')"
        if ($LASTEXITCODE -ne 0) { throw "Python/CUDA validation failed." }
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

if (-not $SkipModelDownload) {
    Assert-File -Path $hfCli -Description "Hugging Face CLI"
    New-Item -ItemType Directory -Force -Path $modelRoot | Out-Null
    & $hfCli download "OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5" --local-dir $ttsModelDir
    if ($LASTEXITCODE -ne 0) { throw "Failed to download the MOSS-TTS model." }
    & $hfCli download "OpenMOSS-Team/MOSS-Audio-Tokenizer-v2" --local-dir $codecModelDir
    if ($LASTEXITCODE -ne 0) { throw "Failed to download the audio tokenizer." }
}

Test-Installation
Write-Output "Setup complete. Start the service with: powershell -ExecutionPolicy Bypass -File .\start-moss-tts.ps1"
