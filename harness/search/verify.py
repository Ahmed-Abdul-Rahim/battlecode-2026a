"""Final check of the submission build (scripts/search.ps1 candidate): the real bot, plans read
from its pack (no patching), against a reference build of the same source without the table.
Opponents: our rule bots and the fresh ladder top 10 (never used by the search), all 40 maps,
both sides, paired.

    python -m harness.search verify
"""

from __future__ import annotations

import shutil

from harness.search import common as C


def build_reference() -> str:
    d = C.ROOT / "bots" / "sub-ref"
    C._copy_source(C.slot_dir(0), d)
    shutil.copy2(C.MAIN / "bots" / "main" / "src" / "pack" / "pack.bin", d / "src" / "pack" / "pack.bin")
    C.runner.build(str(d), release=True)
    return str(d)


def build_submission() -> str:
    """The submission build: this worktree's bots/main as it is (its pack.bin holds the table)."""
    d = C.ROOT / "bots" / "sub-plan"
    C._copy_source(C.ROOT / "bots" / "main", d)
    C.runner.build(str(d), release=True)
    return str(d)


def run(maps_only: list[str] | None = None) -> dict:
    splits = C.load_json(C.RUNS / "splits.json", None)
    cand, ref = build_submission(), build_reference()
    opps = C.opponents()
    test = [o for o in splits["head_to_head"]] + sorted(C.fresh_top())
    maps = {m: p for m, p in C.maps().items() if not maps_only or m in maps_only}
    games, meta = [], []
    for m, path in maps.items():
        for o in test:
            for s in "ab":
                for name, bot in (("plan", cand), ("ref", ref)):
                    games.append((bot, opps[o], path) if s == "a" else (opps[o], bot, path))
                    meta.append((name, m, o, s, bot))
    out_dir = C.RUNS / "tmp" / "verify"
    res = C.runner.run_games(games, out_dir, release=True, log_limit=0)
    rows = {}
    for r, (name, m, o, s, bot) in zip(res, meta):
        rows[(name, m, o, s)] = C._row(r, C.Game(name, m, o, s), "verify", bot)
    shutil.rmtree(out_dir, ignore_errors=True)
    rep = {"per_map": {}}
    allw, allf = [], []
    for m in maps:
        dw = [rows[("plan", m, o, s)]["res"] - rows[("ref", m, o, s)]["res"] for o in test for s in "ab"]
        df = [C.fitness(rows[("plan", m, o, s)]) - C.fitness(rows[("ref", m, o, s)]) for o in test for s in "ab"]
        st = sum(C.stalls(rows[("plan", m, o, s)]) - C.stalls(rows[("ref", m, o, s)]) for o in test for s in "ab")
        rep["per_map"][m] = {**C._mean_se(dw, "win"), **C._mean_se(df, "fit"), "extra_stalls": st}
        allw += dw
        allf += df
    # Absolute results per map and per opponent: where the submission still loses.
    rep["abs_map"] = {}
    for m in maps:
        for name in ("plan", "ref"):
            v = [rows[(name, m, o, s)]["res"] for o in test for s in "ab"]
            rep["abs_map"].setdefault(m, {})[name] = round(sum(v) / len(v), 3)
    rep["abs_opp"] = {}
    for o in test:
        for name in ("plan", "ref"):
            v = [rows[(name, m, o, s)]["res"] for m in maps for s in "ab"]
            rep["abs_opp"].setdefault(o, {})[name] = round(sum(v) / len(v), 3)
    rep["all"] = {"n": len(allw), **C._mean_se(allw, "win"), **C._mean_se(allf, "fit")}
    for name in ("plan", "ref"):
        for o in splits["head_to_head"]:
            v = [rows[(name, m, o, s)]["res"] for m in maps for s in "ab"]
            rep[f"{name}_vs_{o}"] = round(sum(v) / len(v), 3)
    C.save_json(C.RUNS / ("verify.json" if not maps_only else "verify-subset.json"), rep)
    a = rep["all"]
    print(f"submission build vs reference (no table), {a['n']} paired games: Δwin {a['win']:+.3f} ± {a['win_se']:.3f}, "
          f"Δfit {a['fit']:+.3f} ± {a['fit_se']:.3f}")
    for o in splits["head_to_head"]:
        print(f"  win rate vs {o}: submission {rep[f'plan_vs_{o}']:.3f}, reference {rep[f'ref_vs_{o}']:.3f}")
    worst = sorted(rep["per_map"].items(), key=lambda kv: kv[1]["win"])[:4]
    print("  weakest maps:", ", ".join(f"{m} {v['win']:+.2f}" for m, v in worst))
    stalls = sum(v["extra_stalls"] for v in rep["per_map"].values())
    print(f"  extra out-of-fuel turns (all maps): {stalls:+d}")
    print("  lowest absolute win rate (submission / no table):")
    for m, v in sorted(rep["abs_map"].items(), key=lambda kv: kv[1]["plan"])[:10]:
        print(f"    {m:22s} {v['plan']:.3f} / {v['ref']:.3f}")
    print("  per opponent (submission / no table):")
    for o, v in sorted(rep["abs_opp"].items(), key=lambda kv: kv[1]["plan"]):
        print(f"    {o:40s} {v['plan']:.3f} / {v['ref']:.3f}")
    return rep
