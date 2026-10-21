#!/usr/bin/env bash
# The league daemon (harness/league.py) on Linux / macOS / Git Bash. Extra arguments pass through.
#   scripts/train.sh start                          # worker: train and push to branch 'league'
#   scripts/train.sh start --lead                   # the one PC that reads the ladder + writes reviews
#   scripts/train.sh start --lead --auto-submit     # ... and uploads (team-agreed rules, one PC only)
#   scripts/train.sh stop | status | logs | review | sync | maps | bench | submit
#   scripts/train.sh once [--no-push]               # one cycle in the foreground
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
[ -x "$PY" ] || PY=.venv/Scripts/python.exe
cmd="${1:-status}"
shift || true

stop_league() {
  pkill -f "harness/league.py run" 2>/dev/null || true
  rm -f runs/league/lock
}

case "$cmd" in
  start)
    stop_league
    mkdir -p runs/league
    "$PY" harness/params.py --check
    nohup "$PY" -u harness/league.py run --forever "$@" >> runs/league/league.log 2>&1 &
    echo "league started ($*). Log: runs/league/league.log; status: scripts/train.sh status" ;;
  stop)
    stop_league
    echo "league stopped." ;;
  logs)
    tail -F runs/league/league.log ;;
  once)
    "$PY" -u harness/league.py run --cycles 1 "$@" ;;
  experiments)
    wt="$(cd .. && pwd)/battlecode-autopilot"
    [ -d "$wt" ] || { git fetch -q origin && git worktree add -B autopilot "$wt" origin/league; }
    mkdir -p "$wt/runs/league"
    py="$(pwd)/$PY"
    (cd "$wt" && nohup "$py" -u harness/experiments.py run --forever >> runs/league/experiments.log 2>&1 &)
    echo "experiment autopilot started in $wt (log: runs/league/experiments.log)" ;;
  status|review|sync|maps|bench|submit|replay)
    "$PY" harness/league.py "$cmd" "$@" ;;
  *)
    echo "usage: $0 start|stop|status|logs|review|sync|maps|bench|submit|replay|once [args]"; exit 2 ;;
esac
