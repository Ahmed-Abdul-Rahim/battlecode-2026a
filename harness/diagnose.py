"""Behaviour diagnosis from debug replays: the general failure patterns behind a map's losses.

    python -m harness.diagnose runs/<dir> <our bot folder name> [--map NAME]

Run debug games first (bctools run, or runner.run_games with release=False and a log limit),
so the replays carry our telemetry. Per map it reports:

- loop: share of alive turns spent oscillating between two tiles (A B A B);
- arbiter: utility-arbiter overrides per game and how many came while calm (hp >= 70%, no
  gems, no enemy in view), by option (stale learned weights show up here);
- deaths: per game, the share while retreating and with no ally within 5 tiles;
- bag: gems lost on deaths per game and how many of those deaths were within 4 tiles of an
  altar (tribute was possible);
- capture: share of capture turns with a second champion on the same group;
- intents: where the turns go.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from bctools import replay  # noqa: E402

ARBITER = {0: "keep", 1: "step_back", 2: "support", 3: "fight", 4: "walk_home", 5: "recall"}
#: A gap of this many rounds between a champion's turn lines is a death (respawn takes 60).
DEATH_GAP = 40


def _layer(hexs: str, w: int, h: int) -> set[tuple[int, int]]:
    b = bytes.fromhex(hexs)
    return {(x, y) for y in range(h) for x in range(w) if (b[(y * w + x) >> 3] >> ((y * w + x) & 7)) & 1}


def _cheb(a, b) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _map_name(stem: str) -> str:
    m = stem.split("__", 1)[0]
    return m[2:] if m[:2].isdigit() else m


def diagnose_game(path: Path, ours: str) -> tuple[str, dict]:
    stem = path.stem
    a, b = stem.split("__", 1)[1].split("-vs-", 1)
    side = "a" if a.lstrip("_") == ours.lstrip("_") else "b"
    t = replay.load(path).team(side)
    out: collections.Counter = collections.Counter()
    by_c: dict[str, list[dict]] = collections.defaultdict(list)
    by_r: dict[int, dict[str, dict]] = collections.defaultdict(dict)
    for r in t.turns:
        by_c[r["c"]].append(r)
        by_r[r["r"]][r["c"]] = r
        out["turns"] += 1
        out["intent:" + r.get("i", "?").split("(")[0]] += 1
    altars = _layer(t.map["altars"], t.map["w"], t.map["h"]) if t.map else set()
    for rows in by_c.values():
        for i in range(3, len(rows)):
            w = rows[i - 3 : i + 1]
            ps = [tuple(x["p"]) for x in w]
            if ps[0] == ps[2] and ps[1] == ps[3] and ps[0] != ps[1] and all(x["hp"] > 0 for x in w):
                out["loop"] += 1
        for x, y in zip(rows, rows[1:]):
            if y["r"] - x["r"] >= DEATH_GAP:
                out["deaths"] += 1
                if x.get("i", "").startswith(("Retreat", "WalkHome")):
                    out["deaths_retreating"] += 1
                near = sum(1 for c, o in by_r[x["r"]].items() if c != x["c"] and _cheb(o["p"], x["p"]) <= 5)
                if near == 0:
                    out["deaths_alone"] += 1
            drop = x["gm"] - y["gm"]
            if drop > 0 and y["at"] - x["at"] < drop * 0.8:
                out["bag_lost"] += drop
                if altars and min(_cheb(x["p"], al) for al in altars) <= 4:
                    out["bag_lost_by_altar"] += drop
    caps: dict[tuple[int, str], int] = collections.Counter()
    for r in t.turns:
        if r.get("i", "").startswith("Capture"):
            caps[(r["r"], r["i"])] += 1
    out["capture_turns"] = sum(caps.values())
    out["capture_extra"] = sum(n - 1 for n in caps.values() if n > 1)
    rows_at = {(r["r"], r["c"]): r for r in t.turns}
    for e in t.events:
        if e.get("k") != "utility":
            continue
        out["arbiter"] += 1
        r = rows_at.get((e["r"], e["c"]))
        if r and r["hp"] >= 0.7 and r["gm"] == 0 and not r.get("en"):
            out["arbiter_calm"] += 1
            out["arbiter_calm:" + ARBITER.get(e["v"], str(e["v"]))] += 1
    return _map_name(stem), dict(out)


def report(games: dict[str, list[dict]]) -> None:
    for m, gs in sorted(games.items()):
        n = len(gs)
        s: collections.Counter = collections.Counter()
        for g in gs:
            s.update(g)
        turns = max(s["turns"], 1)
        deaths = max(s["deaths"], 1)
        calm = {k.split(":", 1)[1]: v // n for k, v in s.items() if k.startswith("arbiter_calm:")}
        intents = {k.split(":", 1)[1]: round(v / turns, 2) for k, v in s.most_common() if k.startswith("intent:")}
        print(f"{m} ({n} games)")
        print(f"  loop {s['loop'] / turns:.1%} of turns | arbiter {s['arbiter'] // n}/game, calm {s['arbiter_calm'] // n} {calm}")
        print(f"  deaths {s['deaths'] / n:.1f}/game: retreating {s['deaths_retreating'] / deaths:.0%}, alone {s['deaths_alone'] / deaths:.0%}"
              f" | bag lost {s['bag_lost'] / n:.0f}/game ({s['bag_lost_by_altar'] / n:.0f} within 4 of an altar)"
              f" | extra capturers {s['capture_extra'] / max(s['capture_turns'], 1):.0%}")
        print(f"  intents {dict(list(intents.items())[:7])}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dir", type=Path)
    p.add_argument("ours", help="our bot's folder name as in the replay file names (e.g. _dbg)")
    p.add_argument("--map")
    p.add_argument("--json", action="store_true")
    a = p.parse_args(argv)
    games: dict[str, list[dict]] = collections.defaultdict(list)
    for f in sorted(a.dir.glob("*.repl26a")):
        if a.ours not in f.stem:
            continue
        m, d = diagnose_game(f, a.ours)
        if a.map and m != a.map:
            continue
        games[m].append(d)
    if a.json:
        print(json.dumps(games, indent=1))
    else:
        report(games)
    return 0


if __name__ == "__main__":
    sys.exit(main())
