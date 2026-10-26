# Keep Windows awake while any plan-search job or queue runs, including the gaps between steps
# (a Python step holds its own lock, but Windows slept "System Idle" in the second between two
# steps, 2026-10-26 10:17). Exits after 5 minutes with no search process. The display may still
# turn off.
Add-Type -Namespace Win -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
$ES_CONTINUOUS = [uint32]"0x80000000"; $ES_SYSTEM_REQUIRED = [uint32]1
[Win.Power]::SetThreadExecutionState($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED) | Out-Null
$idle = 0
while ($idle -lt 10) {
    Start-Sleep 30
    $busy = Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'harness\.search|search\.ps1 run|queue_side\.ps1|queue_a\.ps1' -and $_.CommandLine -notmatch 'keepawake' }
    if ($busy) { $idle = 0; [Win.Power]::SetThreadExecutionState($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED) | Out-Null } else { $idle++ }
}
[Win.Power]::SetThreadExecutionState($ES_CONTINUOUS) | Out-Null
