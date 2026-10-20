# The league daemon (harness/league.py) on Windows. Extra arguments pass through.
#   scripts\train.ps1 start                          # worker: train and push to branch 'league'
#   scripts\train.ps1 start --lead                   # the one PC that reads the ladder + writes reviews
#   scripts\train.ps1 start --lead --auto-submit     # ... and uploads (team-agreed rules, one PC only)
#   scripts\train.ps1 stop | status | logs | review | sync | bench | submit
#   scripts\train.ps1 once [--no-push]               # one cycle in the foreground
param(
    [Parameter(Position = 0)]
    [ValidateSet("start", "stop", "status", "logs", "review", "sync", "maps", "bench", "submit", "replay", "once")]
    [string]$cmd = "status",
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$rest = @()
)

$root = Split-Path -Parent $PSScriptRoot
$py = Join-Path $root ".venv\Scripts\python.exe"
$league = Join-Path $root "harness\league.py"
$scratch = Join-Path $root "runs\league"
Set-Location $root

function Stop-League {
    Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
        Where-Object { $_.CommandLine -match 'league\.py.+run' } |
        ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue; "stopped pid $($_.ProcessId)" }
    Remove-Item (Join-Path $scratch "lock") -ErrorAction SilentlyContinue
}

switch ($cmd) {
    "start" {
        Stop-League | Out-Null
        New-Item -ItemType Directory -Force $scratch | Out-Null
        & $py (Join-Path $root "harness\params.py") --check
        if (-not $?) { Write-Error "harness/params.py does not match bots/main/src/learned.rs"; exit 1 }
        $procArgs = @("-u", $league, "run", "--forever") + $rest
        Start-Process -WindowStyle Hidden -FilePath $py -ArgumentList $procArgs -WorkingDirectory $root `
            -RedirectStandardOutput (Join-Path $scratch "league.log") -RedirectStandardError (Join-Path $scratch "league.err")
        Write-Host "league started ($($rest -join ' ')). Log: runs\league\league.log; status: scripts\train.ps1 status" -ForegroundColor Green
    }
    "stop" { Stop-League; Write-Host "league stopped." -ForegroundColor Yellow }
    "logs" {
        $log = Join-Path $scratch "league.log"
        if (Test-Path $log) { Get-Content $log -Wait -Tail 40 } else { Write-Host "no log yet: start the league first" }
    }
    "once" { & $py -u $league run --cycles 1 @rest }
    default { & $py $league $cmd @rest }
}
