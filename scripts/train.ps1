# Start / stop / inspect the self-improvement daemons (see docs/training.md).
#   scripts\train.ps1 start      # league trainer + co-evolution, in the background
#   scripts\train.ps1 stop
#   scripts\train.ps1 status
#   scripts\train.ps1 logs       # follow both logs
param([ValidateSet("start", "stop", "status", "logs")][string]$cmd = "status")

$root = Split-Path -Parent $PSScriptRoot
$py = Join-Path $root ".venv\Scripts\python.exe"
Set-Location $root

function Stop-Trainers {
    Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
        Where-Object { $_.CommandLine -match 'league_train|coevo' } |
        ForEach-Object { Stop-Process -Id $_.ProcessId -Force; "stopped $($_.ProcessId)" }
    Remove-Item runs\league\lock, runs\coevo\lock -ErrorAction SilentlyContinue
}

switch ($cmd) {
    "start" {
        Stop-Trainers | Out-Null
        New-Item -ItemType Directory -Force runs\league, runs\coevo | Out-Null
        & $py harness\params.py --check
        if (-not $?) { Write-Error "parameter registry does not match learned.rs"; exit 1 }
        Start-Process -WindowStyle Hidden -FilePath $py -ArgumentList "harness\league_train.py", "--forever" `
            -RedirectStandardOutput runs\league\train.log -RedirectStandardError runs\league\train.err
        Start-Process -WindowStyle Hidden -FilePath $py -ArgumentList "harness\coevo.py", "--forever" `
            -RedirectStandardOutput runs\coevo\train.log -RedirectStandardError runs\coevo\train.err
        "started; logs in runs\league\train.log and runs\coevo\train.log"
    }
    "stop" { Stop-Trainers }
    "status" { & $py harness\status.py }
    "logs" { Get-Content runs\coevo\train.log, runs\league\train.log -Wait -Tail 20 }
}
