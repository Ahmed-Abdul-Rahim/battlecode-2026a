"""Per-map strategy selection from variant tournaments (harness/variants.py output).

For every map and variant: mean score (win + 0.25 × gem margin) over every match on
that map (all pool opponents, both sides). A map gets a non-default variant only if
it beats the general choice on that map by >= MIN_GAIN *and* never scores worse than
it against any single opponent there (the generalisation guard; each map has only a
handful of matches per variant).

    python harness/map_plans.py runs/variants/macro-* runs/variants/squads-* ...
    python harness/map_plans.py --general squads-1.6 runs/variants/*-0*
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

MARGIN_WEIGHT = 0.25
MIN_GAIN = 0.1

# Variant -> (knob overrides, draft profile) in bots/main-maps (bc_core::memory::knob,
# map_plans::MapPlan). Variants not listed can't be switched per map and are skipped.
KNOB = {"BUILD_PROFILE": 8, "SQUAD_JOIN_X10": 9, "VALUE_PROFILE": 10, "REMOTE_FARMER": 11}
SWITCHABLE: dict[str, tuple[dict[str, int], int]] = {
    "base": ({}, 0),
    "monster-first": ({"BUILD_PROFILE": 1}, 0),
    "all-primary": ({"BUILD_PROFILE": 2}, 0),
    "outlast-like": ({"BUILD_PROFILE": 3}, 0),
    "cache-like": ({"BUILD_PROFILE": 4}, 0),
    "power-first": ({"BUILD_PROFILE": 5}, 0),
    "treasure-first": ({"BUILD_PROFILE": 6}, 0),
    "durable-first": ({"BUILD_PROFILE": 7}, 0),
    "home-first": ({"BUILD_PROFILE": 8}, 0),
    "bruiser-first": ({"BUILD_PROFILE": 9}, 0),
    "squads-1.6": ({"SQUAD_JOIN_X10": 16}, 0),
    "squads-2.5": ({"SQUAD_JOIN_X10": 25}, 0),
    "all-gems": ({"VALUE_PROFILE": 1}, 0),
    "split-2-gold": ({"VALUE_PROFILE": 2}, 0),
    "r3-remote": ({"REMOTE_FARMER": 1}, 0),
    "all-ranged": ({}, 1),
    "all-melee": ({}, 2),
    "double-body": ({}, 3),
    "counter": ({}, 4),
}


def rust_table(plan: dict, fingerprints: dict[str, int], general_knobs: dict[str, int]) -> str:
    """map_plans.rs PLANS entries for maps whose chosen variant differs from the general one."""
    rows = []
    for m, choice in sorted(plan.items()):
        v = choice["variant"]
        if v == plan_general(plan) or v not in SWITCHABLE or m not in fingerprints:
            continue
        knobs, draft = SWITCHABLE[v]
        merged = {**general_knobs, **knobs}
        pairs = ", ".join(f"({KNOB[k]}, {val})" for k, val in sorted(merged.items()))
        rows.append(
            f'    MapPlan {{ fingerprint: {fingerprints[m]}, name: "{m}", knobs: &[{pairs}], draft: {draft} }},'
        )
    nl = chr(10)
    body = nl.join(rows) + (nl if rows else "")
    return "pub const PLANS: &[MapPlan] = &[" + nl + body + "];"


def plan_general(plan: dict) -> str:
    return plan.get("__general__", {}).get("variant", "base")


def match_score(r: dict, bot: str) -> float:
    win = 0.5 if r["winner"] is None else 1.0 if r["winner"] == bot else 0.0
    ours, theirs = (r["gems_a"], r["gems_b"]) if r["bot_a"] == bot else (r["gems_b"], r["gems_a"])
    return win + MARGIN_WEIGHT * (ours - theirs) / max(ours + theirs, 1)


def load(dirs: list[Path]) -> dict:
    """{variant: {map: {opponent: [scores]}}} — the latest run of each variant wins."""
    table: dict = {}
    for d in sorted(dirs):
        for vdir in sorted(p for p in d.iterdir() if p.is_dir()):
            variant = vdir.name
            per = defaultdict(lambda: defaultdict(list))
            for res in vdir.glob("*/results.json"):
                opp = res.parent.name
                for r in json.loads(res.read_text(encoding="utf-8")):
                    bot = f"_var-{variant}"
                    if bot not in (r["bot_a"], r["bot_b"]):
                        continue
                    per[r["map"]][opp].append(match_score(r, bot))
            if per:
                table[variant] = per
    return table


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def choose(table: dict, general: str) -> dict:
    maps = sorted({m for v in table.values() for m in v})
    plan = {}
    for m in maps:
        base = table[general].get(m, {})
        base_all = mean([s for xs in base.values() for s in xs])
        best, best_gain = general, 0.0
        for variant, per in table.items():
            cand = per.get(m, {})
            if not cand or variant == general:
                continue
            gain = mean([s for xs in cand.values() for s in xs]) - base_all
            never_worse = all(mean(cand.get(o, [])) >= mean(xs) - 1e-9 for o, xs in base.items())
            if gain >= MIN_GAIN and never_worse and gain > best_gain:
                best, best_gain = variant, gain
        plan[m] = {"variant": best, "gain": round(best_gain, 3), "general_score": round(base_all, 3)}
    return plan


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dirs", nargs="+", type=Path)
    p.add_argument("--general", default="base", help="variant used where no map override is justified")
    p.add_argument("--out", type=Path, default=Path("runs/variants/map_plans.json"))
    p.add_argument("--fingerprints", type=Path, default=Path("runs/variants/fingerprints.json"))
    p.add_argument("--general-knobs", default="", help="e.g. SQUAD_JOIN_X10=16,BUILD_PROFILE=5")
    p.add_argument("--rust", type=Path, help="write the PLANS table into this map_plans.rs")
    a = p.parse_args(argv)
    table = load([d for d in a.dirs if d.is_dir()])
    if a.general not in table:
        print(f"general variant {a.general!r} not found; have {sorted(table)}")
        return 1
    table = {v: per for v, per in table.items() if v in SWITCHABLE}
    plan = choose(table, a.general)
    # Overall per-variant means, for context.
    overall = {
        v: round(mean([s for per in t.values() for xs in per.values() for s in xs]), 4)
        for v, t in table.items()
    }
    out = {
        "general": a.general,
        "overall": dict(sorted(overall.items(), key=lambda kv: -kv[1])),
        "maps": plan,
    }
    a.out.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))
    if a.rust:
        fps = json.loads(a.fingerprints.read_text(encoding="utf-8"))
        general_knobs = dict(
            (kv.split("=")[0], int(kv.split("=")[1])) for kv in a.general_knobs.split(",") if kv
        )
        plan["__general__"] = {"variant": a.general}
        table_rs = rust_table(plan, fps, general_knobs)
        src = a.rust.read_text(encoding="utf-8")
        start = src.index("pub const PLANS: &[MapPlan] = &[")
        end = src.index("];", start) + 2
        a.rust.write_text(src[:start] + table_rs + src[end:], encoding="utf-8")
        print(f"wrote {table_rs.count('MapPlan {')} map overrides to {a.rust}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
