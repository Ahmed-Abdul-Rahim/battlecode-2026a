"""Tune main's θ_match knob defaults against a pool of sparring bots.

Each candidate is a copy of bots/main in bots/_tune-0/ (it has to sit under bots/
so the `../../sdk/rust` path dependency resolves), with the knob::DEFAULTS array
patched in its vendored bc_core/memory.rs. Search is coordinate descent: try
± step on each knob, keep the best improvement, halve the steps after each pass.

Objective, per match: 1 win / 0 loss (0.5 unfinished) + 0.25 × normalised gem
margin. The margin term gives a signal when we already win everything. Tuning
uses only the non-held-out maps; the held-out ones are for the gate.

    python harness/tune_knobs.py                          # pool: baseline, starter-python, exploit-qwen-*
    python harness/tune_knobs.py --evals 2                # smoke test
    python harness/tune_knobs.py --apply runs/tune/<ts>/best.json   # write winner into crates/ + sync

Local only: nothing here touches the Battlecode platform.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from bctools import ROOT, gate, runner

# name: (lo, hi, initial step). TRIBUTE_MIN_GEMS, TRIBUTE_DISTANCE_SCALE and
# RETREAT_HP_PCT are left out: knobs_for_map (bc-draft theta.rs) overwrites them.
KNOBS = {
    "AGGRESSION": (0, 255, 48),
    "TRIBUTE_MAX_GEMS": (30, 200, 30),
    "MONOLITH_MIN_POWER": (10, 120, 20),
    "RECALL_GOLD": (5, 50, 6),
    "FIGHT_MARGIN_PCT": (0, 100, 15),
}
CANONICAL = ROOT / "crates" / "bc-core" / "src" / "memory.rs"
CANDIDATE = "_tune-0"
MARGIN_WEIGHT = 0.25


def read_defaults(path: Path) -> dict[str, int]:
    text = path.read_text(encoding="utf-8")
    return {k: int(v) for k, v in re.findall(r"\((\w+), (\d+)\)", text.split("DEFAULTS")[1].split("];")[0])}


def patch_defaults(path: Path, knobs: dict[str, int]) -> None:
    text = path.read_text(encoding="utf-8")
    head, tail = text.split("pub const DEFAULTS", 1)
    block, rest = tail.split("];", 1)
    for name, value in knobs.items():
        block, n = re.subn(rf"\({name}, \d+\)", f"({name}, {value})", block)
        if n != 1:
            raise ValueError(f"knob {name} not found in DEFAULTS of {path}")
    path.write_text(head + "pub const DEFAULTS" + block + "];" + rest, encoding="utf-8")


def make_candidate(knobs: dict[str, int]) -> str:
    d = ROOT / "bots" / CANDIDATE
    if not d.exists():
        shutil.copytree(ROOT / "bots" / "main", d, ignore=shutil.ignore_patterns(".cambc", "target"))
    else:  # refresh sources, keep the build cache
        shutil.rmtree(d / "src")
        shutil.copytree(ROOT / "bots" / "main" / "src", d / "src")
    patch_defaults(d / "src" / "bc_core" / "memory.rs", knobs)
    return CANDIDATE


def default_pool() -> list[str]:
    """League (P-006): fixed bots, Qwen and hand-written exploiters, and frozen snapshots of main."""
    bots = ROOT / "bots"
    python = sorted(
        p.name
        for pat in ("exploit-qwen-*", "claude-*")
        for p in bots.glob(pat)
        if (p / "src" / "bot.py").exists()
    )
    snapshots = sorted(p.name for p in bots.glob("main-v*") if (p / "Cargo.toml").exists())
    return ["baseline", "starter-python", *python, *snapshots]


def pfsp_weights(per_opp: dict, power: float = 2.0, floor: float = 0.1) -> dict[str, float]:
    """Prioritised fictitious self-play (P-006): weight hard opponents up.

    p = how well we do against the opponent on a 0..1 scale (score / its maximum);
    weight = (1 - p)^power + floor.
    """
    top = 1.0 + MARGIN_WEIGHT
    return {
        o: round((1 - min(max(v["score"] / top, 0.0), 1.0)) ** power + floor, 4) for o, v in per_opp.items()
    }


def weighted(per_opp: dict, weights: dict[str, float] | None) -> float:
    if not weights:
        n = sum(v["n"] for v in per_opp.values())
        return round(sum(v["score"] * v["n"] for v in per_opp.values()) / max(n, 1), 4)
    total = sum(weights[o] for o in per_opp)
    return round(sum(per_opp[o]["score"] * weights[o] for o in per_opp) / total, 4)


def tuning_maps() -> list[Path]:
    held = gate.heldout_names()
    return [m for m in runner.list_maps() if m.stem not in held]


def evaluate(bot: str, pool: list[str], maps: list[Path], out: Path) -> dict:
    per_opp = {}
    for opp in pool:
        results = runner.run_matches(bot, opp, maps, out / opp, swap=True)
        s = 0.0
        for r in results:
            ours, theirs = (r.gems_a, r.gems_b) if r.bot_a == bot else (r.gems_b, r.gems_a)
            margin = (ours - theirs) / max(ours + theirs, 1)
            s += r.score_for(bot) + MARGIN_WEIGHT * margin
        per_opp[opp] = {
            "score": round(s / len(results), 4),
            "wins": sum(r.score_for(bot) for r in results),
            "n": len(results),
        }
    return {"per_opponent": per_opp}


def search(pool: list[str], max_evals: int, passes: int, pfsp: bool = True, max_opponents: int = 0) -> dict:
    out = ROOT / "runs" / "tune" / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    maps = tuning_maps()
    trace = out / "trace.jsonl"
    evals = 0
    weights: dict[str, float] | None = None
    full_pool = list(pool)

    def run(knobs: dict[str, int]) -> float:
        nonlocal evals
        evals += 1
        t0 = time.time()
        res = evaluate(make_candidate(knobs), pool, maps, out / f"e{evals:03d}")
        res["objective"] = weighted(res["per_opponent"], weights)
        row = {"eval": evals, "knobs": knobs, **res, "weights": weights, "secs": round(time.time() - t0)}
        with trace.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        print(f"eval {evals}: {res['objective']:.4f} {knobs}", flush=True)
        return res["objective"]

    defaults = read_defaults(CANONICAL)
    best = {k: defaults[k] for k in KNOBS}
    f_best = run(best)
    if pfsp:
        first = json.loads(trace.read_text(encoding="utf-8").splitlines()[0])["per_opponent"]
        weights = pfsp_weights(first)
        if max_opponents:  # keep only the hardest opponents for the rest of the search
            pool = sorted(pool, key=lambda o: -weights[o])[:max_opponents]
            first = {o: first[o] for o in pool}
        f_best = weighted(first, weights)
        print(f"PFSP weights: {weights}\npool: {pool}", flush=True)
    f_start = f_best
    steps = {k: v[2] for k, v in KNOBS.items()}
    for _ in range(passes):
        for name, (lo, hi, _) in KNOBS.items():
            options = []
            for d in (steps[name], -steps[name]):
                v = min(hi, max(lo, best[name] + d))
                if v == best[name] or evals >= max_evals:
                    continue
                trial = {**best, name: v}
                options.append((run(trial), trial))
            if options:
                f, trial = max(options, key=lambda o: o[0])
                if f > f_best + 1e-4:
                    f_best, best = f, trial
        steps = {k: max(1, s // 2) for k, s in steps.items()}
        if evals >= max_evals:
            break
    summary = {
        "pool": pool,
        "full_pool": full_pool,
        "weights": weights,
        "maps": [m.stem for m in maps],
        "start": f_start,
        "best": f_best,
        "knobs": best,
        "defaults": {k: defaults[k] for k in KNOBS},
        "evals": evals,
    }
    (out / "best.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return summary


def apply(best_json: Path) -> int:
    knobs = json.loads(best_json.read_text(encoding="utf-8"))["knobs"]
    patch_defaults(CANONICAL, knobs)
    print(f"wrote {knobs} to {CANONICAL.relative_to(ROOT)}")
    return subprocess.run([sys.executable, "-m", "bctools.cli", "sync"], cwd=ROOT).returncode


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--pool", nargs="*", help="opponent bots (default: baseline, starter-python, exploit-qwen-*)"
    )
    p.add_argument("--evals", type=int, default=30)
    p.add_argument("--passes", type=int, default=2)
    p.add_argument("--no-pfsp", action="store_true", help="plain mean over the pool")
    p.add_argument("--max-opponents", type=int, default=0, help="after eval 1 keep the K hardest (PFSP)")
    p.add_argument("--apply", type=Path, help="best.json to write into crates/bc-core/src/memory.rs")
    a = p.parse_args(argv)
    if a.apply:
        return apply(a.apply)
    search(a.pool or default_pool(), a.evals, a.passes, pfsp=not a.no_pfsp, max_opponents=a.max_opponents)
    return 0


if __name__ == "__main__":
    sys.exit(main())
