"""Speedrun: how fast a bot farms to the winning gem count against an idle opponent.

Every earlier test compared our bot with copies of itself, so an economy that is
uniformly slow was invisible. Against `bots/idle` (champions never act) nothing
contests us, and the result is an absolute farming speed that doesn't depend on
any opponent population. Ladder reference: the best bot reaches 1000 gems by
round 350-500 while fighting us (best_bot_dump/).

    python harness/speedrun.py --bots main main-v5 main-v6
    python harness/speedrun.py --variants "power:BUILD_PROFILE=13" "best:BUILD_PROFILE=13,DRAFT_BEST=1"

Score per game: rounds to win, or 2000 + (1000 - gems) if unfinished (lower is
better). Both sides of every map are played (spawns differ). Debug builds by
default so turn shares come from our telemetry; --release for the platform build.
Local only: nothing here touches the Battlecode platform.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import v6_trials

from bctools import ROOT, replay, runner

IDLE = "idle"
WIN_GEMS = 1000
MAX_ROUNDS = 2000


def score(r: runner.MatchResult, bot: str) -> int:
    gems = r.gems_a if r.bot_a == bot else r.gems_b
    if r.winner == bot:
        return r.rounds
    return MAX_ROUNDS + max(0, WIN_GEMS - gems)


def stats_metrics(r: runner.MatchResult, bot: str) -> dict:
    """Economy numbers from the engine statistics JSON (same format as ladder summaries)."""
    try:
        d = json.loads(Path(r.stats).read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return {}
    us = "teamA" if r.bot_a == bot else "teamB"
    rounds = max(r.rounds, 1)
    champs = [c for c in d.get("champions", []) if c["team"] == us]
    n = max(len(champs), 1)
    clears = emblems400 = 0
    first_emblem = None
    for e in d.get("events", []):
        if (e.get("owner") or {}).get("team") != us:
            continue
        for en in e["entries"]:
            rnd = en["timestamp"]["round"]
            if e["kind"] == "campClear":
                clears += 1
            elif e["kind"] == "itemCombination":
                emblems400 += rnd <= 400
                first_emblem = rnd if first_emblem is None else min(first_emblem, rnd)
    gold300 = 0
    for tl in d.get("championTimelines", []):
        if tl["metric"] != "earnedGold" or tl["owner"].get("team") != us:
            continue
        t = value = 0
        for run in tl["runs"]:
            if t > 300:
                break
            value = int(float(run["value"]))
            t += run["duration"]
        gold300 += value
    monster = sum(c["metrics"]["monsterDamageDone"] for c in champs)
    return {
        "clears_per_100": 100 * clears / rounds,
        "gold300": gold300,
        "emblems400_per_champ": emblems400 / n,
        "first_emblem": first_emblem or MAX_ROUNDS,
        "monster_per_champ_round": monster / (n * rounds),
    }


def turn_shares(r: runner.MatchResult, bot: str) -> Counter:
    """Champion-turns by intent (and dead turns) from our telemetry; empty for release builds."""
    try:
        log = replay.load(r.replay).team("a" if r.bot_a == bot else "b")
    except (OSError, KeyError, ValueError):
        return Counter()
    c = Counter(t["i"].split("(")[0] for t in log.turns)
    c["DEAD"] = sum(1 for e in log.events if e["k"] == "dead")
    return c


def evaluate(bot: str, out: Path, release: bool) -> dict:
    t0 = time.time()
    results = runner.run_matches(bot, IDLE, out_dir=out / bot, swap=True, release=release)
    scores = [score(r, bot) for r in results]
    metrics = [stats_metrics(r, bot) for r in results]
    keys = [k for k in metrics[0]] if metrics and metrics[0] else []
    shares = Counter()
    for r in results:
        shares += turn_shares(r, bot)
    total = sum(shares.values()) or 1
    per_map: dict[str, list[int]] = {}
    for r, s in zip(results, scores):
        per_map.setdefault(r.map, []).append(s)
    return {
        "bot": bot,
        "mean": round(statistics.mean(scores), 1),
        "median": statistics.median(scores),
        "unfinished": sum(1 for r in results if r.winner != bot),
        **{k: round(statistics.mean(m[k] for m in metrics if m), 2) for k in keys},
        "turns": {k: round(v / total, 3) for k, v in shares.most_common()},
        "per_map": {m: round(statistics.mean(v)) for m, v in sorted(per_map.items())},
        "secs": round(time.time() - t0),
    }


def line(name: str, row: dict) -> str:
    turns = " ".join(f"{k}:{v:.0%}" for k, v in list(row["turns"].items())[:7])
    return (
        f"{name:22s} mean {row['mean']:7.1f} median {row['median']:6.0f} unfinished {row['unfinished']:2d} | "
        f"clears/100r {row.get('clears_per_100', 0):5.2f} gold@300 {row.get('gold300', 0):6.0f} "
        f"emblems@400/ch {row.get('emblems400_per_champ', 0):4.2f} 1st emblem {row.get('first_emblem', 0):5.0f} "
        f"mon/ch/r {row.get('monster_per_champ_round', 0):5.1f} | {turns}"
    )


def parse_variant(spec: str) -> tuple[str, dict[str, int]]:
    name, _, knobs = spec.partition(":")
    return name, {k: int(v) for k, v in (kv.split("=") for kv in knobs.split(",") if kv)}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bots", nargs="*", default=[], help="existing bots, played as they are")
    p.add_argument("--variants", nargs="*", default=[], help='"name:KNOB=V,KNOB=V" built from bots/main')
    p.add_argument("--release", action="store_true")
    a = p.parse_args(argv)
    out = ROOT / "runs" / "speedrun" / time.strftime("%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    table: dict[str, dict] = {}
    jobs = [(b, b) for b in a.bots]
    for spec in a.variants:
        name, knobs = parse_variant(spec)
        jobs.append((name, v6_trials.make("main", f"_sr-{name}", knobs)))
    for name, bot in jobs:
        table[name] = evaluate(bot, out, a.release)
        print(line(name, table[name]), flush=True)
        (out / "table.json").write_text(json.dumps(table, indent=1), encoding="utf-8")
    print(f"table: {out / 'table.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
