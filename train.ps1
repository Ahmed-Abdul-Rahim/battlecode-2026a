# Root wrapper for scripts\train.ps1
# Allows running: .\train start, .\train status, .\train benchmark, etc.
param(
    [Parameter(Position=0)]
    [ValidateSet("start", "stop", "status", "benchmark", "push", "logs", "run-once")]
    [string]$cmd = "status"
)

& "$PSScriptRoot\scripts\train.ps1" $cmd
