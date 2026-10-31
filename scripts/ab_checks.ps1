# The five paired A/B checks of the engine-2 rule defaults (docs/plan-search.md, "Run on the PC",
# step 1), one after the other. Results: runs\ab\<name>.json (dwin = new default - old behaviour).
#   powershell -ExecutionPolicy Bypass -File scripts\ab_checks.ps1
# Resumable: a check whose runs\ab\<name>.json exists is skipped.
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:BC_MAIN = $root
$env:OPENBLAS_NUM_THREADS = "1"
$env:BCTOOLS_JOBS = [string]([Environment]::ProcessorCount - 2)
$py = Join-Path $root ".venv\Scripts\python.exe"
$logs = Join-Path $root "runs\ab"
New-Item -ItemType Directory -Force $logs | Out-Null
Start-Process powershell -WindowStyle Hidden -ArgumentList "-ExecutionPolicy Bypass -File `"$PSScriptRoot\keepawake.ps1`""
$checks = @(
    @("regroup", "--b-knob", "FIGHT_V6=0", "--no-table"),
    @("panic", "--b-knob", "K2_TRIBUTE_DETOUR=0", "--no-table"),
    @("onecap", "--b-knob", "CAMP_FEASIBLE=24", "--no-table"),
    @("draft", "--b-knob", "TAVI_META=1", "--maps", "corners,gemfight,yellow-brick-road"),
    @("tavibld", "--b-knob", "TAVI_META=3")
)
foreach ($c in $checks) {
    $name = $c[0]
    if (Test-Path (Join-Path $logs "$name.json")) { "skip $name (done)"; continue }
    "$(Get-Date -Format 'MM-dd HH:mm') A/B $name" | Out-File -Append -Encoding utf8 (Join-Path $logs "ab.log")
    & $py -X utf8 -u -m harness.ab @c *>> (Join-Path $logs "$name.log")
}
"$(Get-Date -Format 'MM-dd HH:mm') A/B checks done" | Out-File -Append -Encoding utf8 (Join-Path $logs "ab.log")
