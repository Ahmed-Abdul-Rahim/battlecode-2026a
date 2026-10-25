"""Offline per-map plan search (plan: learning goes into the draft plan).

    python -m harness.search setup               # build the slot pool + the rules opponent
    python -m harness.search smoke               # patch path: null = identical, settings take effect
    python -m harness.search baseline [--hours H] # snapshot vs every opponent, all maps, both sides
    python -m harness.search splits              # opponent A/B and map train/held-out splits
    python -m harness.search audit [--hours H]   # probe every plan-settable decision
    python -m harness.search search [--maps a,b] [--hours H]  # per-map search (train maps)
    python -m harness.search table | sstatus     # accepted settings per map / search progress
    python -m harness.search mvp [--hours H]     # honest (cross-fitted) gain + h2h vs our rules
    python -m harness.search export              # league/plan_table.json + this worktree's pack.bin
    python -m harness.search status

Every job resumes where it stopped (rows in runs/search/<job>/games.jsonl).
"""

from __future__ import annotations

import argparse
import os
import statistics
import time

from harness.search import common as C
from harness.search import split


def cmd_setup(a) -> None:
    t0 = time.time()
    C.build_rules()
    snap = C.build_slots(a.slots)
    print(f"snapshot {snap['wasm'][:12]} ({snap['slots']} slots, commit {snap['commit'][:8]}"
          f"{' dirty' if snap['dirty'] else ''}) in {time.time() - t0:.0f}s")


SMOKE_MAPS = ["ambush", "lanes"]
SMOKE_OPPS = ["mimic-procless", "mimic-3", "baseline"]


def cmd_smoke(a) -> None:
    snap = C.check_snapshot()
    store = C.Store("smoke", snap["wasm"])
    ks = C.knob_slots()
    recs = {
        "base": C.record(),
        "null2": C.record(),
        "best": C.record(draft=8),
        "aggro": C.record(overrides={ks["AGGRESSION"]: 230, ks["FIGHT_MARGIN_PCT"]: 0}),
    }
    games = [C.Game(c, m, o, s) for c in recs for m in SMOKE_MAPS for o in SMOKE_OPPS for s in "ab"]
    C.play(games, recs, store, C.opponents())
    fields = ("res", "ours", "theirs", "deaths", "rounds", "picks", "fmean", "oof")
    same = lambda x, y: all(x.get(f) == y.get(f) for f in fields)  # noqa: E731
    for c in ("null2", "best", "aggro"):
        n = d = 0
        for g in games:
            if g.cand != c:
                continue
            r, b = store.get(g), store.get(C.Game("base", g.map, g.opp, g.side))
            n += 1
            d += not same(r, b)
        print(f"{c}: {d}/{n} games differ from base")
    picks = {tuple(r["picks"]) for r in store.select("best")}
    print("best picks:", picks, "base picks:", {tuple(r["picks"]) for r in store.select("base")})
    bad = sum(1 for g in games if g.cand == "null2" and not same(store.get(g), store.get(C.Game("base", g.map, g.opp, g.side))))
    print("PASS" if bad == 0 and any(p and p[0] == "Stellar" for p in picks) else "FAIL")


def cmd_baseline(a) -> None:
    snap = C.check_snapshot()
    store = C.Store("baseline", snap["wasm"])
    opps, maps = C.opponents(), C.maps()
    games = [C.Game("base", m, o, s) for m in maps for o in opps for s in "ab"]
    # The null-patch check on another slot: identical games are the proof that patching is clean.
    games += [C.Game("null2", m, o, s) for m in list(maps)[:4] for o in list(opps)[:8] for s in "ab"]
    ok = C.play(games, {"base": C.record(), "null2": C.record()}, store, opps,
                C.deadline_from(a.hours))
    report_baseline(store, opps, maps)
    if not ok:
        print("stopped at the deadline: run baseline again to continue")


def report_baseline(store: C.Store, opps: dict, maps: dict) -> None:
    rows = store.select("base")
    print(f"{len(rows)} base games")
    if not rows:
        return
    diff = 0
    for r in store.select("null2"):
        b = store.get(C.Game("base", r["map"], r["opp"], r["side"]))
        diff += b is not None and (b["res"], b["ours"], b["theirs"], b.get("fmean")) != (
            r["res"], r["ours"], r["theirs"], r.get("fmean"))
    print(f"null patch: {diff} of {len(store.select('null2'))} games differ (must be 0)")
    out = {"per_map": {}, "per_opp": {}}
    for key, groups in (("per_map", maps), ("per_opp", opps)):
        for name in groups:
            rs = [r for r in rows if r["map" if key == "per_map" else "opp"] == name]
            if not rs:
                continue
            out[key][name] = {
                "n": len(rs),
                "win": round(sum(r["res"] for r in rs) / len(rs), 3),
                "fit": round(sum(C.fitness(r) for r in rs) / len(rs), 3),
                "deaths": round(sum(r["deaths"] for r in rs) / len(rs), 1),
                "oof": sum(max(r.get("oof", 0), 0) for r in rs),
                "fmean": round(statistics.mean(r["fmean"] for r in rs if "fmean" in r)) if any("fmean" in r for r in rs) else None,
                "fp95": max((r.get("fp95", 0) for r in rs), default=0),
                "draft_fuel_max": [max((r["draft_fuel"][i] for r in rs if len(r.get("draft_fuel", [])) > i), default=0) for i in range(5)],
            }
    n = len(rows)
    out["all"] = {
        "n": n,
        "win": round(sum(r["res"] for r in rows) / n, 3),
        "fit": round(sum(C.fitness(r) for r in rows) / n, 3),
        "oof": sum(max(r.get("oof", 0), 0) for r in rows),
    }
    C.save_json(store.dir / "baseline.json", out)
    print(f"win {out['all']['win']}, fitness {out['all']['fit']}, OOF turns {out['all']['oof']}")
    worst = sorted(out["per_map"].items(), key=lambda kv: kv[1]["win"])[:5]
    print("weakest maps:", ", ".join(f"{m} {v['win']}" for m, v in worst))
    fuel = sorted(out["per_map"].items(), key=lambda kv: -kv[1]["oof"])[:5]
    print("most OOF:", ", ".join(f"{m} {v['oof']} (p95 {v['fp95']})" for m, v in fuel))
    for name in ("rules", "lg-v23-fuel"):
        if name in out["per_opp"]:
            print(f"vs {name}: win {out['per_opp'][name]['win']} fit {out['per_opp'][name]['fit']}")


def cmd_splits(a) -> None:
    snap = C.check_snapshot()
    store = C.Store("baseline", snap["wasm"])
    s = split.make(store)
    C.save_json(C.RUNS / "splits.json", s)
    print({k: len(v) for k, v in s.items()})
    print("held-out maps:", s["maps_heldout"])


def cmd_audit(a) -> None:
    from harness.search import audit

    if not audit.run(a.hours):
        print("stopped at the deadline: run audit again to continue")


def cmd_search(a) -> None:
    from harness.search import search

    search.run(a.maps.split(",") if a.maps else None, a.hours)


def cmd_table(a) -> None:
    from harness.search import search

    search.table()


def cmd_sstatus(a) -> None:
    from harness.search import search

    search.status()


def cmd_mvp(a) -> None:
    from harness.search import mvp

    mvp.evaluate(a.hours)


def cmd_export(a) -> None:
    from harness.search import mvp

    mvp.export()


def cmd_generalize(a) -> None:
    from harness.search import generalize

    generalize.run(a.hours)


def cmd_branch(a) -> None:
    from harness.search import branch

    {"setup": lambda: branch.setup(), "labels": lambda: branch.labels(a.hours, a.decision),
     "train": lambda: branch.train(a.decision), "eval": lambda: branch.evaluate(a.hours, a.decision)}[a.action]()


def cmd_status(a) -> None:
    snap = C.load_json(C.RUNS / "snapshot.json", {})
    print("snapshot:", snap.get("wasm", "-")[:12], snap.get("time", ""))
    for d in sorted(C.RUNS.glob("*/games.jsonl")):
        n = sum(1 for _ in open(d, encoding="utf-8"))
        print(f"  {d.parent.name}: {n} game rows")


def stay_awake() -> None:
    """Keep Windows from sleeping while a job runs (it slept after 60 idle minutes mid-search,
    freezing games). The display may still turn off; normal sleep returns when we exit."""
    if os.name == "nt":
        import ctypes

        es_continuous, es_system_required = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(es_continuous | es_system_required)


def main() -> None:
    stay_awake()
    ap = argparse.ArgumentParser(prog="harness.search")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("setup")
    p.add_argument("--slots", type=int, default=C.N_SLOTS)
    sub.add_parser("smoke")
    p = sub.add_parser("baseline")
    p.add_argument("--hours", type=float)
    sub.add_parser("splits")
    p = sub.add_parser("audit")
    p.add_argument("--hours", type=float)
    p = sub.add_parser("search")
    p.add_argument("--maps")
    p.add_argument("--hours", type=float)
    sub.add_parser("table")
    sub.add_parser("sstatus")
    p = sub.add_parser("mvp")
    p.add_argument("--hours", type=float)
    sub.add_parser("export")
    p = sub.add_parser("branch")
    p.add_argument("action", choices=["setup", "labels", "train", "eval"])
    p.add_argument("--decision", choices=["contact", "tribute"], default="contact")
    p.add_argument("--hours", type=float)
    p = sub.add_parser("generalize")
    p.add_argument("--hours", type=float)
    sub.add_parser("status")
    a = ap.parse_args()
    {"setup": cmd_setup, "smoke": cmd_smoke, "baseline": cmd_baseline, "splits": cmd_splits,
     "audit": cmd_audit, "search": cmd_search, "table": cmd_table, "sstatus": cmd_sstatus,
     "mvp": cmd_mvp, "export": cmd_export, "generalize": cmd_generalize,
     "branch": cmd_branch,
     "status": cmd_status}[a.cmd](a)


if __name__ == "__main__":
    main()
