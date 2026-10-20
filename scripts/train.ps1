# Start / stop / inspect the autonomous self-improvement harness.
#   scripts\train.ps1 start      # starts autonomous harness in background with --auto-submit
#   scripts\train.ps1 stop       # stops background trainers and cleans locks
#   scripts\train.ps1 status     # prints champion, rank, benchmark vs top 5, and pending losses
#   scripts\train.ps1 benchmark  # runs immediate benchmark against Top 5 teams
#   scripts\train.ps1 push       # bundles and submits current champion to Battlecode platform
#   scripts\train.ps1 logs       # follows live training log
#   scripts\train.ps1 run-once   # runs 1 cycle in foreground
param([ValidateSet("start", "stop", "status", "benchmark", "push", "logs", "run-once")][string]$cmd = "status")

$root = Split-Path -Parent $PSScriptRoot
$py = Join-Path $root ".venv\Scripts\python.exe"
Set-Location $root

function Stop-Trainers {
    Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
        Where-Object { $_.CommandLine -match 'auto_train|league_train|coevo' } |
        ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue; "stopped $($_.ProcessId)" }
    Remove-Item runs\auto_train\lock, runs\league\lock, runs\coevo\lock -ErrorAction SilentlyContinue
}

switch ($cmd) {
    "start" {
        Stop-Trainers | Out-Null
        New-Item -ItemType Directory -Force (Join-Path $root "runs\auto_train") | Out-Null
        & $py (Join-Path $root "harness\params.py") --check
        if (-not $?) { Write-Error "parameter registry does not match learned.rs"; exit 1 }
        
        $logOut = Join-Path $root "runs\auto_train\train.log"
        $logErr = Join-Path $root "runs\auto_train\train.err"
        $script = Join-Path $root "harness\auto_train.py"

        $procArgs = @("-u", $script, "--forever", "--auto-submit")
        Start-Process -WindowStyle Hidden -FilePath $py -ArgumentList $procArgs -WorkingDirectory $root
        Write-Host "Autonomous harness started in background (auto-submit on)." -ForegroundColor Green
        Write-Host "Logs     : runs\auto_train\train.log"
        Write-Host "Status   : train status"
        Write-Host "Tail logs: train logs"
    }
    "stop" {
        Stop-Trainers
        Write-Host "Trainers stopped and lock files cleaned." -ForegroundColor Yellow
    }
    "status" {
        & $py harness\auto_train.py --status
    }
    "benchmark" {
        & $py harness\auto_train.py --benchmark
    }
    "push" {
        & $py harness\auto_train.py --push
    }
    "run-once" {
        & $py harness\auto_train.py --cycles 1 --auto-submit
    }
    "logs" {
        if (Test-Path runs\auto_train\train.log) {
            Get-Content runs\auto_train\train.log -Wait -Tail 30
        } else {
            Write-Host "No log file found at runs\auto_train\train.log. Start the trainer first." -ForegroundColor Yellow
        }
    }
}
