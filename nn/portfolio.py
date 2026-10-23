"""Strategy per opponent: is there a gain in choosing a different bot style per opponent?

    python nn/portfolio.py run       # plays the matrix (runs/nn/portfolio/)
    python nn/portfolio.py report

Every portfolio style plays every opponent on the same maps and sides. Reported:
- single best style overall (what we ship now is `main`);
- oracle: the best style per opponent (upper bound of a perfect opponent detector);
- honest: the style picked on half the maps, scored on the other half (and swapped), so the
  gain isn't just picking whichever style got lucky on these games.
If the honest gain clears the noise, an in-game selector (draft picks / early behaviour ->
style) is worth building.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from bctools import runner  # noqa: E402

from nn import run as R  # noqa: E402

OUT = R.OUT / "portfolio"
# v1 (2026-10-23): the old style bots won 0-10% vs main's 83%: main was best against every
# opponent, so no gain. v2: styles of comparable strength, variants of main by knob.
STYLES = ["main", "_pf_aggro", "_pf_careful", "_pf_hunt", "_pf_teamfight", "_pf_search", "_pf_bank"]
VARIANTS = {
    "_pf_aggro": {"AGGRESSION": 210, "FIGHT_MARGIN_PCT": 5},
    "_pf_careful": {"AGGRESSION": 60, "FIGHT_MARGIN_PCT": 60, "RETREAT_HP_PCT": 50},
    "_pf_hunt": {"HUNT_MODE": 2},
    "_pf_teamfight": {"TEAM_FIGHT": 1},
    "_pf_search": {"COMBAT_SEARCH": 1},
    "_pf_bank": {"TRIBUTE_MAX_GEMS": 40},
}
N_OPP = 14
N_MAPS = 6


def setup() -> tuple[list[str], list[Path]]:
    rng = random.Random(11)
    opps = rng.sample(R.mimics(), N_OPP - 6) + R.held_mimics()[:6]
    train, held = R.maps()
    maps = rng.sample(train + held, N_MAPS)
    return opps, maps


def cmd_run(_: argparse.Namespace) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    from harness import experiments as X
    for name, knobs in VARIANTS.items():
        X.knob_variant(name, knobs)
    opps, maps = setup()
    games = [(s, o, m) for s in STYLES for o in opps for m in maps] + \
            [(o, s, m) for s in STYLES for o in opps for m in maps]
    res = runner.run_games(games, OUT / "games", log_limit=0, timeout=R.GAME_TIMEOUT)
    rows = [{"style": r.bot_a if r.bot_a in STYLES else r.bot_b,
             "opp": r.bot_b if r.bot_a in STYLES else r.bot_a,
             "map": r.map, "side": "a" if r.bot_a in STYLES else "b",
             "score": r.score_for(r.bot_a if r.bot_a in STYLES else r.bot_b),
             "margin": (r.gems_a - r.gems_b) * (1 if r.bot_a in STYLES else -1),
             "done": r.winner is not None or r.rounds > 0} for r in res]
    (OUT / "matrix.json").write_text(json.dumps(rows, indent=0))
    return cmd_report(_)


def cmd_report(_: argparse.Namespace) -> int:
    rows = [r for r in json.loads((OUT / "matrix.json").read_text()) if r["done"]]
    sc = defaultdict(list)  # (style, opp) -> scores
    by_map = defaultdict(list)  # (style, opp, map) -> scores
    for r in rows:
        sc[(r["style"], r["opp"])].append(r["score"])
        by_map[(r["style"], r["opp"], r["map"])].append(r["score"])
    opps = sorted({r["opp"] for r in rows})
    maps = sorted({r["map"] for r in rows})
    mean = lambda v: sum(v) / len(v) if v else 0.0  # noqa: E731
    overall = {s: mean([x for o in opps for x in sc[(s, o)]]) for s in STYLES}
    print("overall win rate per style:")
    for s, v in sorted(overall.items(), key=lambda kv: -kv[1]):
        print(f"  {s:28s} {v:.3f}")
    print("\nper opponent (best style, its win rate, main's):")
    oracle = []
    for o in opps:
        best = max(STYLES, key=lambda s: mean(sc[(s, o)]))
        oracle.append(mean(sc[(best, o)]))
        print(f"  {o:42s} {best:28s} {mean(sc[(best, o)]):.2f}  main {mean(sc[('main', o)]):.2f}")
    # honest: choose on half the maps, score on the other half, both ways
    half = [maps[::2], maps[1::2]]
    honest = []
    for pick_maps, test_maps in (half, half[::-1]):
        for o in opps:
            best = max(STYLES, key=lambda s: mean([x for m in pick_maps for x in by_map[(s, o, m)]]))
            honest += [x for m in test_maps for x in by_map[(best, o, m)]]
    main_all = [x for o in opps for x in sc[("main", o)]]
    print(f"\nmain {mean(main_all):.3f} | single best style {max(overall.values()):.3f} | "
          f"oracle per opponent {mean(oracle):.3f} | honest per opponent {mean(honest):.3f} "
          f"({len(honest)} games)")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("run").set_defaults(fn=cmd_run)
    sub.add_parser("report").set_defaults(fn=cmd_report)
    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
