# The offline plan search (harness/search), end to end. Every step saves its progress and
# resumes where it stopped, so stopping or shutting down loses at most a few minutes.
#   powershell -ExecutionPolicy Bypass -File scripts\search.ps1 prepare   # once per engine / bot change: snapshot, baseline, splits, audit
#   powershell -ExecutionPolicy Bypass -File scripts\search.ps1 run       # everything, in order (resumable)
#   powershell -ExecutionPolicy Bypass -File scripts\search.ps1 status    # what is done, what runs now
#   powershell -ExecutionPolicy Bypass -File scripts\search.ps1 stop      # stop safely (resume with run)
#   powershell -ExecutionPolicy Bypass -File scripts\search.ps1 results   # per-map results, warm trial, study
#   powershell -ExecutionPolicy Bypass -File scripts\search.ps1 candidate # pack the table into a submission zip (no upload)
# `run` order: per-map search over all training maps in 3-hour sessions -> cross-map
# generalisation study -> MVP check (A + rules vs our rule bot, the fresh top 10 and the scoring
# opponents) -> B on top of A (tribute timing: labels with A active, net, A+B vs A+rules) ->
# search the 10 held-out maps too (a complete 40-map table).
# The PC stays awake while a step runs.
param([Parameter(Position = 0)][ValidateSet("prepare", "run", "status", "stop", "results", "candidate")][string]$cmd = "status")

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
# Self-contained checkout: opponents, rules bot and slots all come from this worktree (the default
# would be the league checkout, whose bots are built on the old engine).
$env:BC_MAIN = $root
$py = Join-Path $root ".venv\Scripts\python.exe"
$logs = Join-Path $root "runs\search"
# One BLAS thread: numpy is used lightly, and 16 BLAS threads failed to allocate on a low-memory PC.
$env:OPENBLAS_NUM_THREADS = "1"
function Py { & $py -X utf8 -u -m harness.search @args }
function Log($m) { $line = "$(Get-Date -Format 'MM-dd HH:mm') $m"; Write-Host $line -ForegroundColor Cyan; $line | Out-File -Append -Encoding utf8 (Join-Path $logs "run.log") }
function Running { Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match 'harness.search' } }
function HeldLeft {
    & $py -X utf8 -c "import sys,json;sys.path.insert(0,'.');from harness.search import search as S;sp=json.load(open('runs/search/splits.json'));t=S.table();print(sum(1 for m in sp['maps_heldout'] if m not in t))" 2>$null | Select-Object -Last 1
}
function TrainLeft {
    & $py -X utf8 -c "import sys,json;sys.path.insert(0,'.');from harness.search import search as S;sp=json.load(open('runs/search/splits.json'));t=S.table();print(sum(1 for m in sp['maps_train'] if m not in t))" 2>$null | Select-Object -Last 1
}

switch ($cmd) {
    "prepare" {
        # A new snapshot (engine 2.0.0 / bot changes): rebuild the slot pool on the reference pack,
        # check patching, measure the base bot, add new pool maps to the splits, re-audit the knobs.
        # Per-map states of the old snapshot are archived by the search itself.
        if (Running) { Write-Host "a search step is already running (scripts\search.ps1 status)" -ForegroundColor Yellow; exit 1 }
        Start-Process powershell -WindowStyle Hidden -ArgumentList "-ExecutionPolicy Bypass -File `"$PSScriptRoot\keepawake.ps1`""
        Log "prepare: setup (slot pool on data\search\live-pack.bin)"
        Py setup *>> (Join-Path $logs "prepare.log")
        Py smoke *>> (Join-Path $logs "prepare.log")
        Log "prepare: baseline (base bot vs every opponent, all maps)"
        Py baseline *>> (Join-Path $logs "prepare.log")
        Py splits --extend *>> (Join-Path $logs "prepare.log")
        Log "prepare: audit (which knobs matter per map)"
        Py audit *>> (Join-Path $logs "prepare.log")
        Log "prepare done: scripts\search.ps1 run"
    }
    "run" {
        if (Running) { Write-Host "a search step is already running (scripts\search.ps1 status)" -ForegroundColor Yellow; exit 1 }
        # Hold the stay-awake lock across the gaps between steps too (Windows slept "System Idle"
        # in the second between two steps).
        Start-Process powershell -WindowStyle Hidden -ArgumentList "-ExecutionPolicy Bypass -File `"$PSScriptRoot\keepawake.ps1`""
        while ([int](TrainLeft) -gt 0) {
            Log "search session ($(TrainLeft) training maps left)"
            Py search --hours 3 *>> (Join-Path $logs "search.log")
        }
        Log "training maps done; searching the held-out maps (complete table)"
        $held = (& $py -X utf8 -c "import json;print(','.join(json.load(open('runs/search/splits.json'))['maps_heldout']))" | Select-Object -Last 1)
        while ([int](HeldLeft) -gt 0) {
            Log "held-out search session ($(HeldLeft) maps left)"
            Py search --maps $held --hours 3 *>> (Join-Path $logs "search.log")
        }
        Log "all maps searched; MVP check (A + rules vs the rules bot, the fresh top 10)"
        Py mvp *>> (Join-Path $logs "mvp.log")
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
        # Wrappers first (go.ps1, search.ps1 prepare|run, queues), so nothing starts the next step;
        # then the Python workers with their game processes. Progress is saved after every batch.
        Get-CimInstance Win32_Process | Where-Object { $_.ProcessId -ne $PID -and $_.CommandLine -match 'go\.ps1|search\.ps1 (run|prepare)|queue_[a-z]+\.ps1|ab_checks\.ps1' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
        Running | ForEach-Object { taskkill /PID $_.ProcessId /T /F | Out-Null }
        Get-Process cambc -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
        Write-Host "stopped; progress is saved (resume: scripts\search.ps1 run, or scripts\go.ps1 to include prepare)" -ForegroundColor Yellow
    }
    "results" {
        Py table
        if (Test-Path (Join-Path $logs "crossmap.json")) { Py crossmap report }
        if (Test-Path (Join-Path $logs "mvp.log")) { Get-Content (Join-Path $logs "mvp.log") -Tail 3 }
        $b = Join-Path $logs "b\eval_tribute.json"
        if (Test-Path $b) { Write-Host "A+B vs A+rules:"; Get-Content $b }
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
