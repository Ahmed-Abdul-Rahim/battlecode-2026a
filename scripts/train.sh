#!/usr/bin/env bash
# Start / stop / inspect the self-improvement daemons (see docs/training.md).
#   scripts/train.sh start | stop | status | logs
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
[ -x "$PY" ] || PY=.venv/bin/python

stop_trainers() {
  pkill -f "harness/league_train.py" 2>/dev/null || true
  pkill -f "harness/coevo.py" 2>/dev/null || true
  rm -f runs/league/lock runs/coevo/lock
}

case "${1:-status}" in
  start)
    stop_trainers
    mkdir -p runs/league runs/coevo
    "$PY" harness/params.py --check
    nohup "$PY" harness/league_train.py --forever >> runs/league/train.log 2>&1 &
    nohup "$PY" harness/coevo.py --forever >> runs/coevo/train.log 2>&1 &
    echo "started; logs in runs/league/train.log and runs/coevo/train.log" ;;
  stop) stop_trainers; echo stopped ;;
  status) "$PY" harness/status.py ;;
  logs) tail -F runs/coevo/train.log runs/league/train.log ;;
  *) echo "usage: $0 start|stop|status|logs"; exit 2 ;;
esac
