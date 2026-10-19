"""One-screen status of the self-improvement system (read-only).

Shows each co-evolution lineage's champion, recent per-map acceptances and Elo, the league
trainer's champion and track record, the mimic teams, the latest ladder results seen by the
watcher, and which zip in submission/ to upload.

    python harness/status.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(p: Path) -> dict:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def running(lock: Path) -> str:
    if not lock.exists():
        return "stopped"
    age = time.time() - lock.stat().st_mtime
    return f"running (pid {lock.read_text().strip()}, last heartbeat {int(age // 60)} min ago)"


def tail(p: Path, n: int) -> list[str]:
    try:
        return p.read_text(encoding="utf-8", errors="replace").splitlines()[-n:]
    except OSError:
        return []


def main() -> int:
    co = load(ROOT / "runs" / "coevo" / "state.json")
    lg = load(ROOT / "runs" / "league" / "state.json")
    print("== Co-evolution (harness/coevo.py):", running(ROOT / "runs" / "coevo" / "lock"))
    for key, lin in (co.get("lineages") or {}).items():
        maps = sorted(lin.get("pack", {}))
        print(f"  lineage {key}: champion {lin.get('champion')} (v{lin.get('version')}), src bots/{lin.get('src')}, "
              f"evolved maps {len(maps)}: {', '.join(maps)}")
    elo = co.get("elo") or {}
    if elo:
        top = sorted(elo.items(), key=lambda kv: -kv[1])[:8]
        print("  Elo:", ", ".join(f"{k} {v:.0f}" for k, v in top))
    acc = [line for line in tail(ROOT / "runs" / "coevo" / "train.log", 400) if "accepted" in line][-5:]
    for line in acc:
        print("   ", line[:160])
    print("\n== League trainer (harness/league_train.py):", running(ROOT / "runs" / "league" / "lock"))
    if lg:
        print(f"  champion {lg.get('champion')} (v{lg.get('version')}), tracks "
              + ", ".join(f"{k} {v['promos']}/{v['tries']}" for k, v in lg.get("tracks", {}).items()))
        res = (lg.get("ladder") or {}).get("results") or []
        if res:
            last = res[-20:]
            wr = sum(r["win"] for r in last) / len(last)
            d = [r["deaths"] for r in last if r.get("deaths") is not None]
            print(f"  ladder (last {len(last)}): win {wr:.2f}, deaths/game {sum(d) / max(len(d), 1):.1f}, "
                  f"death weight {lg.get('weights', {}).get('death')}")
    print("\n== Mimics (harness/mimic.py, runs/mimic):")
    for f in sorted((ROOT / "runs" / "mimic").glob("*.json")):
        if f.name.endswith(".spec.json"):
            continue
        m = load(f)
        print(f"  {m.get('team', f.stem):28s} games {m.get('games')}, win {m.get('win_rate')}, draft5 {m.get('draft', {}).get('5')}, "
              f"cadence {m.get('cadence')}")
    zips = sorted((ROOT / "submission").glob("*.zip"), key=lambda p: p.stat().st_mtime)
    print("\n== Newest zips in submission/:")
    for z in zips[-5:]:
        print(f"  {z.name}  ({time.strftime('%m-%d %H:%M', time.localtime(z.stat().st_mtime))})")
    a = (co.get("lineages") or {}).get("A", {}).get("champion")
    if a:
        print(f"\nUpload candidate: submission/{a}.zip (lineage A champion; check the cross-check Elo in "
              f"runs/coevo/report.md first). Never upload without the team agreeing (docs/cambc-cli.md).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
