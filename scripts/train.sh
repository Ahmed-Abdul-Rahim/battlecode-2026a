#!/usr/bin/env bash
# Start / stop / inspect the autonomous self-improvement harness.
#   scripts/train.sh start | stop | status | benchmark | push | logs | run-once
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
[ -x "$PY" ] || PY=.venv/bin/python

stop_trainers() {
  pkill -f "harness/auto_train.py" 2>/dev/null || true
  pkill -f "harness/league_train.py" 2>/dev/null || true
  pkill -f "harness/coevo.py" 2>/dev/null || true
  rm -f runs/auto_train/lock runs/league/lock runs/coevo/lock
}

case "${1:-status}" in
  start)
    stop_trainers
    mkdir -p runs/auto_train
    "$PY" harness/params.py --check
    nohup "$PY" harness/auto_train.py --forever --auto-submit >> runs/auto_train/train.log 2>&1 &
    echo "Autonomous harness started in background (auto-submit on)."
    echo "Logs: runs/auto_train/train.log" ;;
  stop)
    stop_trainers
    echo "Trainers stopped and lock files cleaned." ;;
  status)
    "$PY" harness/auto_train.py --status ;;
  benchmark)
    "$PY" harness/auto_train.py --benchmark ;;
  push)
    "$PY" harness/auto_train.py --push ;;
  run-once)
    "$PY" harness/auto_train.py --cycles 1 --auto-submit ;;
  logs)
    tail -F runs/auto_train/train.log ;;
  *)
    echo "usage: $0 start|stop|status|benchmark|push|logs|run-once"; exit 2 ;;
esac
