$ErrorActionPreference = "Stop"

$pidFile = Join-Path $PSScriptRoot "moss-tts.pid"
$listener = Get-NetTCPConnection -LocalPort 7861 -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
$servicePid = if ($listener) { [int]$listener.OwningProcess } elseif (Test-Path -LiteralPath $pidFile) { [int](Get-Content -LiteralPath $pidFile -Raw) } else { 0 }
if (-not $servicePid) {
    Write-Output "MOSS-TTS is not running."
    exit 0
}
$process = Get-Process -Id $servicePid -ErrorAction SilentlyContinue
if ($process) {
    $workerPids = Get-CimInstance Win32_Process |
        Where-Object {
            $_.ParentProcessId -eq $servicePid -and
            $_.CommandLine -like "*qwen_worker.py*"
        } |
        Select-Object -ExpandProperty ProcessId
    foreach ($workerPid in $workerPids) {
        Stop-Process -Id $workerPid -Force -ErrorAction SilentlyContinue
    }
    Stop-Process -Id $servicePid
    $process.WaitForExit(10000) | Out-Null
    Write-Output "Local TTS service stopped (PID $servicePid)."
} else {
    Write-Output "MOSS-TTS process $servicePid was already stopped."
}
Remove-Item -LiteralPath $pidFile -Force -ErrorAction SilentlyContinue
