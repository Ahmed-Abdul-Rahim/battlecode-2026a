# The league daemon (harness/league.py) on Windows. Linux / macOS: scripts/train.sh (same commands).
#   scripts\train.ps1 train [--lead --auto-submit]   # start; the flags are remembered for the next `train`
#   scripts\train.ps1 stop | status | logs
#   scripts\train.ps1 experiments [list]             # the queue; queued experiments run inside the daemon
#   scripts\train.ps1 experiments add --title "..." --area combat --ref origin/feat/x --knobs K=1
#   scripts\train.ps1 exp-status                       # what was tried, merged, rejected, next
# Without flags `train` starts a worker (trains θ, tests experiments, pushes). Exactly one PC runs
# --lead (reads the ladder, builds mimics, writes reviews); only the lead may add --auto-submit.
param(
    [Parameter(Position = 0)]
    [ValidateSet("train", "stop", "status", "logs", "experiments", "exp-status")]
    [string]$cmd = "status",
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$rest = @()
)

$root = Split-Path -Parent $PSScriptRoot
$py = Join-Path $root ".venv\Scripts\python.exe"
$league = Join-Path $root "harness\league.py"
$experiments = Join-Path $root "harness\experiments.py"
$scratch = Join-Path $root "runs\league"
$saved = Join-Path $scratch "train.args"
Set-Location $root

function Stop-League {
    Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
        Where-Object { $_.CommandLine -match '(league|experiments)\.py.+run' } |
        ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue; "stopped pid $($_.ProcessId)" }
    Remove-Item (Join-Path $scratch "lock") -ErrorAction SilentlyContinue
}

switch ($cmd) {
    "train" {
        New-Item -ItemType Directory -Force $scratch | Out-Null
        if ($rest.Count -gt 0) { Set-Content -Encoding utf8 $saved ($rest -join " ") }
        elseif (Test-Path $saved) { $rest = @((Get-Content $saved -Raw).Trim() -split '\s+' | Where-Object { $_ }) }
        Stop-League | Out-Null
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
        $role = if ($rest -contains "--lead") { "lead" } else { "worker" }
        Write-Host "league started as $role ($($rest -join ' ')). Log: scripts\train.ps1 logs; status: scripts\train.ps1 status" -ForegroundColor Green
    }
    "stop" { Stop-League; Write-Host "league stopped." -ForegroundColor Yellow }
    "logs" {
        $log = Join-Path $scratch "league.log"
        if (Test-Path $log) { Get-Content $log -Wait -Tail 40 } else { Write-Host "no log yet: scripts\train.ps1 train first" }
    }
    "experiments" {
        if ($rest.Count -eq 0) { $rest = @("list") }
        & $py -X utf8 $experiments @rest
    }
    "exp-status" { & $py -X utf8 $experiments status }
    default { & $py -X utf8 $league $cmd @rest }
}
