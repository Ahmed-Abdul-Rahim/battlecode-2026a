# prepare then run, one job (resumable: every step skips what is already done)
Set-Location (Split-Path -Parent $PSScriptRoot)
powershell -ExecutionPolicy Bypass -File scripts\search.ps1 prepare
powershell -ExecutionPolicy Bypass -File scripts\search.ps1 run
