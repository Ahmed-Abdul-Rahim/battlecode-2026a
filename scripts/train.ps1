# The league daemon (harness/league.py) on Windows. Extra arguments pass through.
#   scripts\train.ps1 start                          # worker: train and push to branch 'league'
#   scripts\train.ps1 start --lead                   # the one PC that reads the ladder + writes reviews
#   scripts\train.ps1 start --lead --auto-submit     # ... and uploads (team-agreed rules, one PC only)
#   scripts\train.ps1 stop | status | logs | review | sync | bench | submit
#   scripts\train.ps1 once [--no-push]               # one cycle in the foreground
#   scripts\train.ps1 experiments [list|add|stop]     # experiment autopilot (own worktree)
#   scripts\train.ps1 exp-status                       # what was tried, merged, rejected, next
#   scripts\train.ps1 ml-status                        # neural net: data, trained networks, results
#   scripts\train.ps1 ml [list|stop]                  # neural-net jobs: data games + training
param(
    [Parameter(Position = 0)]
    [ValidateSet("start", "stop", "status", "logs", "review", "sync", "maps", "bench", "submit", "replay", "experiments", "exp-status", "ml", "ml-status", "once")]
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
        # Keep the previous run's log (Start-Process overwrites its output files).
        foreach ($f in "league.log", "league.err") {
            $p = Join-Path $scratch $f
            if (Test-Path $p) { Move-Item -Force $p ($p -replace '\.(log|err)$', '.prev.$1') }
        }
        $procArgs = @("-X", "utf8", "-u", $league, "run", "--forever") + $rest
        Start-Process -WindowStyle Hidden -FilePath $py -ArgumentList $procArgs -WorkingDirectory $root `
            -RedirectStandardOutput (Join-Path $scratch "league.log") -RedirectStandardError (Join-Path $scratch "league.err")
        Write-Host "league started ($($rest -join ' ')). Log: runs\league\league.log; status: scripts\train.ps1 status" -ForegroundColor Green
    }
    "stop" { Stop-League; Write-Host "league stopped." -ForegroundColor Yellow }
    "logs" {
        $log = Join-Path $scratch "league.log"
        if (Test-Path $log) { Get-Content $log -Wait -Tail 40 } else { Write-Host "no log yet: start the league first" }
    }
    "once" { & $py -X utf8 -u $league run --cycles 1 @rest }
    "experiments" {
        # The autopilot needs its own checkout: the daemon rewrites bots/main and pulls mid-run.
        $wt = Join-Path (Split-Path -Parent $root) "battlecode-autopilot"
        if (-not (Test-Path $wt)) {
            git -C $root fetch -q origin
            git -C $root worktree add -q -B autopilot $wt origin/league 2>$null
        }
        $x = Join-Path $wt "harness\experiments.py"
        if ($rest -contains "list") { & $py -X utf8 $x list; return }
        if ($rest.Count -gt 0 -and $rest[0] -eq "add") {
            # Queue an idea (pushed, so every PC's autopilot sees it). See docs/league.pdf.
            Push-Location $wt; & $py -X utf8 $x @rest; Pop-Location; return
        }
        if ($rest -contains "stop") {
            Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
                Where-Object { $_.CommandLine -match 'experiments\.py.+run' } |
                ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue; "stopped pid $($_.ProcessId)" }
            return
        }
        New-Item -ItemType Directory -Force (Join-Path $wt "runs\league") | Out-Null
        $env:PATH = (Join-Path $root ".venv\Scripts") + ";" + $env:PATH
        Start-Process -WindowStyle Hidden -FilePath $py -WorkingDirectory $wt `
            -ArgumentList @("-X", "utf8", "-u", $x, "run", "--forever") `
            -RedirectStandardOutput (Join-Path $wt "runs\league\experiments.log") `
            -RedirectStandardError (Join-Path $wt "runs\league\experiments.err")
        Write-Host "experiment autopilot started in $wt (log: runs\league\experiments.log)" -ForegroundColor Green
    }
    "exp-status" { & $py -X utf8 (Join-Path $root "harness\experiments.py") status }
    "ml-status" { & $py -X utf8 (Join-Path $root "harness\mljobs.py") status }
    "ml" {
        # Data games and training runs (harness/mljobs.py), shared with every PC through git.
        $wt = Join-Path (Split-Path -Parent $root) "battlecode-ml"
        if (-not (Test-Path $wt)) {
            git -C $root fetch -q origin
            git -C $root worktree add -q -B mlpilot $wt origin/league 2>$null
        }
        $x = Join-Path $wt "harness\mljobs.py"
        if ($rest -contains "list") { & $py -X utf8 $x list; return }
        if ($rest -contains "stop") {
            Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
                Where-Object { $_.CommandLine -match 'mljobs\.py.+run' } |
                ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue; "stopped pid $($_.ProcessId)" }
            return
        }
        New-Item -ItemType Directory -Force (Join-Path $wt "runs\league") | Out-Null
        $env:PATH = (Join-Path $root ".venv\Scripts") + ";" + $env:PATH
        Start-Process -WindowStyle Hidden -FilePath $py -WorkingDirectory $wt `
            -ArgumentList @("-X", "utf8", "-u", $x, "run", "--forever") `
            -RedirectStandardOutput (Join-Path $wt "runs\league\ml.log") `
            -RedirectStandardError (Join-Path $wt "runs\league\ml.err")
        Write-Host "ml jobs started in $wt (log: runs\league\ml.log; queue: scripts\train.ps1 ml list)" -ForegroundColor Green
    }
    default { & $py -X utf8 $league $cmd @rest }
}
