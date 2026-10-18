"""v6 feature trials: one knob per feature, measured against the deployed v5.

Matches are deterministic (fuel, not time; no RNG), so one pairing is exactly 34
games (17 maps x 2 sides). To get more than one sample of "v5" we also play
jittered v5 clones (tiny knob deltas the draft doesn't override): the v5 family.

    python harness/v6_trials.py --only base camp-escape        # named trials
    python harness/v6_trials.py --tier A                       # a whole tier
    python harness/v6_trials.py --knobs CAMP_SAFE=3 BUILD_PROFILE=11 --name stack1

Pools:
  family   main-v5 (bots/main-v5 = the deployed zip) + main-v5-j1..j4 (jitter)
  exploit  main-v4, claude-*, exploit-qwen-pillager, baseline
  mimic    our bot forced to the drafts/builds of the ladder teams that beat v5

Accept (vs the `base` control run in the same table):
  family win >= base + 0.04, gem score > base, exploit wins >= base - 1 game,
  held-out family win >= 0.5, suspended turns <= 2x base (+20).
Local only: nothing here touches the Battlecode platform.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from league_gate import suspended

from bctools import ROOT, gate, runner
from bctools.stats import wilson

MEMORY = "src/bc_core/memory.rs"
DRAFT = "src/draft.rs"
MARGIN_WEIGHT = 0.25

REFERENCE = "main-v5"
JITTER = {
    "main-v5-j1": {"AGGRESSION": 120},
    "main-v5-j2": {"AGGRESSION": 136},
    "main-v5-j3": {"FIGHT_MARGIN_PCT": 20},
    "main-v5-j4": {"FIGHT_MARGIN_PCT": 30},
}
FAMILY = [REFERENCE, *JITTER, "main-v6"]
EXPLOIT = ["main-v4", "claude-focus", "claude-pillager", "claude-ambusher", "exploit-qwen-pillager", "baseline"]
# Ladder teams that beat v5 (runs/ranked_v5): (draft profile in draft.rs, build profile).
MIMICS = {
    "mimic-unnamed": (6, 11),  # Stellar Tavi Varg Prism Vendetta; passives on primary stat
    "mimic-on2": (7, 3),  # Vendetta Varg Ryth Willow Cassi&Issac; Power/Bruiser
    "mimic-reddragon": (5, 10),  # Varg Ryth Tavi Ren Willow; Monster then Power, Strength
}
# Maps the ladder plays with 3 picks (where v5 went 2-9).
SMALL_MAPS = {"expanse", "collision", "packs", "fragment"}

TRIALS: dict[str, dict[str, dict[str, int]]] = {
    "A": {
        "power-first": {"BUILD_PROFILE": 5},
        "deny": {"DENY_TRIBUTE": 1},
        "endgame": {"ENDGAME_BANK": 1},
        "camp-escape": {"CAMP_SAFE": 1},
        "camp-gather": {"CAMP_SAFE": 2},
        "camp-both": {"CAMP_SAFE": 3},
        "build-passives": {"BUILD_PROFILE": 11},
    },
    "B": {
        "deny-ground": {"DENY_TRIBUTE": 2},
        "deny-team": {"DENY_TRIBUTE": 3},
        "altar-lock": {"ALTAR_LOCK": 1},
    },
    # v7 ladder fixes (sanity only: the ladder population isn't reproducible locally).
    "L": {
        "tavi-meta": {"TAVI_META": 1},
        "recall-safe": {"RECALL_SAFE": 1},
        "fight-v6": {"FIGHT_V6": 1},
        "stun-threat": {"STUN_THREAT": 1},
        "camp-feasible": {"CAMP_FEASIBLE": 1},
        "v7-all": {"TAVI_META": 1, "RECALL_SAFE": 1, "FIGHT_V6": 1, "STUN_THREAT": 1, "CAMP_FEASIBLE": 1},
    },
}


def set_knobs(text: str, knobs: dict[str, int]) -> str:
    for k, v in knobs.items():
        text, n = re.subn(rf"\(({k}), \d+\),", rf"(\1, {v}),", text)
        if n != 1:
            raise ValueError(f"knob {k} not found exactly once in DEFAULTS")
    return text


def copy_bot(src: str, dst: str) -> Path:
    d = ROOT / "bots" / dst
    if d.exists():
        shutil.rmtree(d / "src", ignore_errors=True)
        shutil.copytree(ROOT / "bots" / src / "src", d / "src")
    else:
        shutil.copytree(ROOT / "bots" / src, d, ignore=shutil.ignore_patterns(".cambc", "target"))
    return d


def make(src: str, dst: str, knobs: dict[str, int], draft_profile: int | None = None) -> str:
    d = copy_bot(src, dst)
    mem = d / MEMORY
    mem.write_text(set_knobs(mem.read_text(encoding="utf-8"), knobs), encoding="utf-8", newline="\n")
    if draft_profile is not None:
        f = d / DRAFT
        old = "draft_prefs(c.plan.map_or(0, |p| p.draft), &state)"
        text = f.read_text(encoding="utf-8")
        if text.count(old) != 1:
            raise ValueError("draft_prefs call site not found")
        f.write_text(text.replace(old, f"draft_prefs({draft_profile}, &state)"), encoding="utf-8", newline="\n")
    return dst


def ensure_pools(rebuild: bool) -> None:
    for name, knobs in JITTER.items():
        if rebuild or not (ROOT / "bots" / name).exists():
            make(REFERENCE, name, knobs)
    for name, (draft, build) in MIMICS.items():
        if rebuild or not (ROOT / "bots" / name).exists():
            make("main", name, {"BUILD_PROFILE": build}, draft_profile=draft)


def margin(r: runner.MatchResult, bot: str) -> float:
    ours, theirs = (r.gems_a, r.gems_b) if r.bot_a == bot else (r.gems_b, r.gems_a)
    return r.score_for(bot) + MARGIN_WEIGHT * (ours - theirs) / max(ours + theirs, 1)


def rate(results: list[runner.MatchResult], bot: str) -> dict:
    n = len(results)
    s = sum(r.score_for(bot) for r in results)
    lo, _ = wilson(s, n) if n else (0.0, 0.0)
    return {"n": n, "win": round(s / n, 3) if n else None, "lo95": round(lo, 3), "wins": s}


def evaluate(bot: str, out: Path, pools: list[str]) -> dict:
    held = gate.heldout_names()
    by_pool: dict[str, list[runner.MatchResult]] = {"family": [], "exploit": [], "mimic": []}
    per_opp = {}
    for pool in pools:
        opps = {"family": FAMILY, "exploit": EXPLOIT, "mimic": list(MIMICS)}[pool]
        for opp in opps:
            res = runner.run_matches(bot, opp, out_dir=out / opp, swap=True, release=True)
            by_pool[pool] += res
            per_opp[opp] = rate(res, bot)["win"]
    allres = [r for v in by_pool.values() for r in v]
    fam = by_pool["family"]
    row = {
        "bot": bot,
        **{p: rate(v, bot) for p, v in by_pool.items() if v},
        "family_heldout": rate([r for r in fam if r.map in held], bot)["win"],
        "small_maps": rate([r for r in allres if r.map in SMALL_MAPS], bot)["win"],
        "score": round(sum(margin(r, bot) for r in fam) / max(len(fam), 1), 4),
        "suspended": suspended(allres, bot)[0],
        "per_opp": per_opp,
        "per_map_family": {
            m: round(sum(r.score_for(bot) for r in fam if r.map == m) / max(1, sum(1 for r in fam if r.map == m)), 2)
            for m in sorted({r.map for r in fam})
        },
    }
    return row


def verdict(row: dict, base: dict) -> str:
    if row is base:
        return "control"
    why = []
    if row["family"]["win"] < base["family"]["win"] + 0.04:
        why.append("family")
    if row["score"] <= base["score"]:
        why.append("score")
    if "exploit" in row and row["exploit"]["wins"] < base["exploit"]["wins"] - 1:
        why.append("exploit")
    if (row["family_heldout"] or 0) < 0.5:
        why.append("heldout")
    if row["suspended"] > 2 * base["suspended"] + 20:
        why.append("fuel")
    return "ACCEPT" if not why else "reject(" + ",".join(why) + ")"


def line(name: str, row: dict, v: str) -> str:
    f = row["family"]
    parts = [f"{name:18s} fam {f['win']:.3f} (lo {f['lo95']:.2f})", f"held {row['family_heldout']}"]
    for p in ("exploit", "mimic"):
        if p in row:
            parts.append(f"{p} {row[p]['win']:.3f}")
    parts += [f"small {row['small_maps']}", f"score {row['score']:.4f}", f"susp {row['suspended']}", v]
    return "  ".join(parts)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tier", choices=list(TRIALS))
    p.add_argument("--only", nargs="*", help="trial names (from any tier); 'base' is always run")
    p.add_argument("--knobs", nargs="*", help="ad-hoc trial: KNOB=V ...")
    p.add_argument("--name", default="adhoc")
    p.add_argument("--pools", nargs="*", default=["family", "exploit", "mimic"])
    p.add_argument("--rebuild-pools", action="store_true")
    p.add_argument("--no-base", action="store_true", help="reuse base row from --base-from table")
    p.add_argument("--base-from", type=Path)
    a = p.parse_args(argv)

    trials: dict[str, dict[str, int]] = {}
    every = {k: v for t in TRIALS.values() for k, v in t.items()}
    if a.tier:
        trials.update(TRIALS[a.tier])
    for n in a.only or []:
        if n != "base":
            trials[n] = every[n]
    if a.knobs:
        trials[a.name] = {k: int(v) for k, v in (kv.split("=") for kv in a.knobs)}

    ensure_pools(a.rebuild_pools)
    out = ROOT / "runs" / "v6" / time.strftime("%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    table: dict[str, dict] = {}
    if a.no_base and a.base_from:
        table["base"] = json.loads(a.base_from.read_text(encoding="utf-8"))["base"]
    else:
        t0 = time.time()
        table["base"] = evaluate(make("main", "_v6-base", {}), out / "base", a.pools)
        table["base"]["knobs"], table["base"]["secs"] = {}, round(time.time() - t0)
        print(line("base", table["base"], "control"), flush=True)
    base = table["base"]
    for name, knobs in trials.items():
        t0 = time.time()
        row = evaluate(make("main", f"_v6-{name}", knobs), out / name, a.pools)
        row["knobs"], row["secs"] = knobs, round(time.time() - t0)
        row["verdict"] = verdict(row, base)
        table[name] = row
        print(line(name, row, row["verdict"]), flush=True)
        (out / "table.json").write_text(json.dumps(table, indent=1), encoding="utf-8")
    (out / "table.json").write_text(json.dumps(table, indent=1), encoding="utf-8")
    print(f"table: {out / 'table.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
