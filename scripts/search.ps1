# The offline plan search (harness/search), end to end. Every step saves its progress and
# resumes where it stopped, so stopping or shutting down loses at most a few minutes.
#   powershell -ExecutionPolicy Bypass -File scripts\search.ps1 run       # everything, in order (resumable)
#   powershell -ExecutionPolicy Bypass -File scripts\search.ps1 status    # what is done, what runs now
#   powershell -ExecutionPolicy Bypass -File scripts\search.ps1 stop      # stop safely (resume with run)
#   powershell -ExecutionPolicy Bypass -File scripts\search.ps1 results   # per-map results, warm trial, study
#   powershell -ExecutionPolicy Bypass -File scripts\search.ps1 candidate # pack the table into a submission zip (no upload)
# `run` order: warm-start trial (interlock, starvation) -> per-map search over all training maps in
# 3-hour sessions -> cross-map generalisation study. The PC stays awake while a step runs.
param([Parameter(Position = 0)][ValidateSet("run", "status", "stop", "results", "candidate")][string]$cmd = "status")

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$py = Join-Path $root ".venv\Scripts\python.exe"
$logs = Join-Path $root "runs\search"
function Py { & $py -X utf8 -u -m harness.search @args }
function Log($m) { $line = "$(Get-Date -Format 'MM-dd HH:mm') $m"; Write-Host $line -ForegroundColor Cyan; $line | Out-File -Append -Encoding utf8 (Join-Path $logs "run.log") }
function Running { Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match 'harness.search' } }
function TrainLeft {
    & $py -X utf8 -c "import sys,json;sys.path.insert(0,'.');from harness.search import search as S;sp=json.load(open('runs/search/splits.json'));t=S.table();print(sum(1 for m in sp['maps_train'] if m not in t))" 2>$null | Select-Object -Last 1
}

switch ($cmd) {
    "run" {
        if (Running) { Write-Host "a search step is already running (scripts\search.ps1 status)" -ForegroundColor Yellow; exit 1 }
        Log "warm-start trial"
        Py warm --maps interlock,starvation *>> (Join-Path $logs "warm.log")
        while ([int](TrainLeft) -gt 0) {
            Log "search session ($(TrainLeft) training maps left)"
            Py search --hours 3 *>> (Join-Path $logs "search.log")
        }
        Log "training maps done; cross-map generalisation study"
        Py crossmap run *>> (Join-Path $logs "crossmap.log")
        Log "all done: scripts\search.ps1 results"
    }
    "status" {
        $r = Running
        if ($r) { Write-Host "running: $(($r | Select-Object -First 1).CommandLine -replace '.*harness.search ', '')" -ForegroundColor Green } else { Write-Host "nothing running" -ForegroundColor Yellow }
        Write-Host "training maps left: $(TrainLeft)"
        Py sstatus 2>$null | Select-Object -Last 4
        if (Test-Path (Join-Path $logs "run.log")) { Get-Content (Join-Path $logs "run.log") -Tail 3 }
    }
    "stop" {
        Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'search\.ps1 run|queue_a\.ps1' -and $_.ProcessId -ne $PID } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
        Running | ForEach-Object { taskkill /PID $_.ProcessId /T /F | Out-Null }
        Write-Host "stopped; progress is saved (resume: scripts\search.ps1 run)" -ForegroundColor Yellow
    }
    "results" {
        Py table
        Py warm-report
        if (Test-Path (Join-Path $logs "crossmap.json")) { Py crossmap report }
    }
    "candidate" {
        Py export
        $bot = "bots\sub-plan"
        & $py -X utf8 -c "import sys;sys.path.insert(0,'.');from harness.search import common as C;import shutil;d=C.ROOT/'bots'/'sub-plan';C._copy_source(C.slot_dir(0),d);shutil.copy2(C.ROOT/'bots'/'main'/'src'/'pack'/'pack.bin',d/'src'/'pack'/'pack.bin');C.runner.build(str(d),release=True);print('built',d)"
        New-Item -ItemType Directory -Force (Join-Path $root "submission") | Out-Null
        & (Join-Path $root ".venv\Scripts\cambc.exe") bundle $bot --output (Join-Path $root "submission\sub-plan.zip")
        Write-Host "submission\sub-plan.zip is ready (not uploaded; test it before you submit)" -ForegroundColor Green
    }
}
