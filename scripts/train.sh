#!/usr/bin/env bash
# The league daemon (harness/league.py) on Linux / macOS / Git Bash. Windows: scripts\train.ps1 (same commands).
#   scripts/train.sh train [--lead --auto-submit]   # start; the flags are remembered for the next `train`
#   scripts/train.sh stop | status | logs
#   scripts/train.sh experiments [list]             # the queue; queued experiments run inside the daemon
#   scripts/train.sh experiments add --title "..." --area combat --ref origin/feat/x --knobs K=1
#   scripts/train.sh exp-status                     # what was tried, merged, rejected, next
# Without flags `train` starts a worker (trains θ, tests experiments, pushes). Exactly one PC runs
# --lead (reads the ladder, builds mimics, writes reviews); only the lead may add --auto-submit.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
[ -x "$PY" ] || PY=.venv/Scripts/python.exe
SCRATCH=runs/league
SAVED=$SCRATCH/train.args
cmd="${1:-status}"
shift || true

stop_league() {
  if [ -f "$SCRATCH/lock" ]; then kill "$(cat "$SCRATCH/lock")" 2>/dev/null || true; fi
  pkill -f "harness/(league|experiments).py run" 2>/dev/null || true
  rm -f "$SCRATCH/lock"
}

case "$cmd" in
  train)
    mkdir -p "$SCRATCH"
    if [ $# -gt 0 ]; then
      echo "$*" > "$SAVED"
    elif [ -f "$SAVED" ]; then
      read -r -a saved_args < "$SAVED" || true
      set -- "${saved_args[@]}"
    fi
    stop_league
    "$PY" harness/params.py --check
    for f in league.log league.err; do
      [ -f "$SCRATCH/$f" ] && mv -f "$SCRATCH/$f" "$SCRATCH/${f%.*}.prev.${f##*.}"
    done
    nohup "$PY" -u harness/league.py run --forever "$@" > "$SCRATCH/league.log" 2> "$SCRATCH/league.err" < /dev/null &
    role=worker
    for x in "$@"; do [ "$x" = "--lead" ] && role=lead; done
    echo "league started as $role ($*). Log: scripts/train.sh logs; status: scripts/train.sh status" ;;
  stop)
    stop_league
    echo "league stopped." ;;
  status)
    "$PY" harness/league.py status "$@" ;;
  logs)
    if [ -f "$SCRATCH/league.log" ]; then tail -n 40 -F "$SCRATCH/league.log"; else echo "no log yet: scripts/train.sh train first"; fi ;;
  experiments)
    [ $# -gt 0 ] || set -- list
    "$PY" harness/experiments.py "$@" ;;
  exp-status)
    "$PY" harness/experiments.py status ;;
  *)
    echo "usage: $0 train|stop|status|logs|experiments|exp-status [args]"; exit 2 ;;
esac
