"""League gate: a candidate against every bot in the league, all maps, both sides.

The plain gate (bctools gate) checks one opponent. This one plays the whole
league (baseline, starter, Qwen and hand-written exploiters, main-v* snapshots)
and reports per-opponent and held-out win rates, plus fuel-suspended turns.

    python harness/league_gate.py main
    python harness/league_gate.py main-learned --release --pool main

Local only: nothing here touches the Battlecode platform.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tune_knobs import default_pool

from bctools import ROOT, gate, replay, runner


def suspended(results: list[runner.MatchResult], bot: str) -> tuple[int, int]:
    """(suspended turns, total turns) for `bot` across the matches."""
    bad = total = 0
    for r in results:
        try:
            log = replay.load(r.replay).team("a" if r.bot_a == bot else "b")
        except (OSError, KeyError, ValueError):
            continue
        bad += replay.exhausted_turns(log)
        total += len(log.fuel)
    return bad, total


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("candidate")
    p.add_argument("--pool", nargs="*", help="opponents (default: the league)")
    p.add_argument("--release", action="store_true", help="platform build (no debug telemetry)")
    a = p.parse_args(argv)
    pool = [o for o in (a.pool or default_pool()) if o != a.candidate]
    out = ROOT / "runs" / "league-gate" / f"{a.candidate}-{time.strftime('%Y%m%d-%H%M%S')}"
    held = gate.heldout_names()
    rows, all_results = {}, []
    for opp in pool:
        results = runner.run_matches(a.candidate, opp, out_dir=out / opp, swap=True, release=a.release)
        all_results += results
        s = runner.summarize(results, a.candidate)
        h = [r.score_for(a.candidate) for r in results if r.map in held]
        rows[opp] = {
            "win_rate": s["win_rate"],
            "heldout": round(sum(h) / len(h), 3) if h else None,
            "losses": [f"{x['map']} ({x['a']} vs {x['b']}) {x['gems']}" for x in s["losses"]],
        }
        print(f"{opp:28s} win {s['win_rate']:.3f}  heldout {rows[opp]['heldout']}", flush=True)
    total = runner.summarize(all_results, a.candidate)
    held_all = [r.score_for(a.candidate) for r in all_results if r.map in held]
    bad, turns = suspended(all_results, a.candidate)
    summary = {
        "candidate": a.candidate,
        "release": a.release,
        "matches": total["matches"],
        "win_rate": total["win_rate"],
        "wilson_95": total["wilson_95"],
        "heldout_win_rate": round(sum(held_all) / len(held_all), 3) if held_all else None,
        "unfinished": total["unfinished"],
        "suspended_turns": [bad, turns],
        "per_opponent": rows,
        "out": str(out),
    }
    (out / "league.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "per_opponent"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
