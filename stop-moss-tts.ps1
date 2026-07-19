$ErrorActionPreference = "Stop"

$pidFile = Join-Path $PSScriptRoot "moss-tts.pid"
$listener = Get-NetTCPConnection -LocalAddress "127.0.0.1" -LocalPort 7861 -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
$servicePid = if ($listener) { [int]$listener.OwningProcess } elseif (Test-Path -LiteralPath $pidFile) { [int](Get-Content -LiteralPath $pidFile -Raw) } else { 0 }
if (-not $servicePid) {
    Write-Output "MOSS-TTS is not running."
    exit 0
}
$process = Get-Process -Id $servicePid -ErrorAction SilentlyContinue
if ($process) {
    Stop-Process -Id $servicePid
    $process.WaitForExit(10000) | Out-Null
    Write-Output "MOSS-TTS stopped (PID $servicePid)."
} else {
    Write-Output "MOSS-TTS process $servicePid was already stopped."
}
Remove-Item -LiteralPath $pidFile -Force -ErrorAction SilentlyContinue
