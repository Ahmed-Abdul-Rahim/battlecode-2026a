"""Generalisation study (plan steps 4-5): which way of choosing a plan for an unseen map works?

1. The cross-map matrix: every proven plan (each searched map's accepted plan and runner-up)
   played on every searched map and every held-out map, against the scoring (B) opponents,
   both sides, paired with the defaults: G[plan][map] = Δwin, Δfit.
2. Methods choose a plan for a map without using that map's own search (leave-one-map-out on
   the searched maps; on held-out maps they use all searched maps), scored from the matrix:
     defaults, 1-NN copy, gated 1-NN, weighted kNN on measured gains, best-on-average plan,
     mix-and-match (style / roster vote + weighted-median knobs of the two nearest plans:
     a new plan, played separately).

    python -m harness.search crossmap run [--hours H]
    python -m harness.search crossmap report
"""

from __future__ import annotations

import math
import pickle

from harness.search import common as C
from harness.search import generalize as G
from harness.search import search as S

PLANS_PER_MAP = 1
#: Scoring opponents per (plan, map) cell: 30 plans x 40 maps x 35 opponents was ~150k games.
CELL_OPPS = 12


def proven() -> dict[str, dict]:
    """plan id -> {from, rank, rec (hex), settings}: the top PLANS_PER_MAP of each searched map."""
    snap = C.check_snapshot()["wasm"]
    out = {}
    for d in sorted((C.RUNS / "search").glob("*/state.pkl")):
        with open(d, "rb") as f:
            st = pickle.load(f)
        r = st.get("result")
        if not r or not r.get("accepted"):
            continue
        m = d.parent.name
        store = C.Store(f"search/{m}", snap)
        lead = [r["accepted"]] + [c for c in r["b"] if c != r["accepted"]]
        for rank, c in enumerate(lead[:PLANS_PER_MAP]):
            out[f"{m}#{rank}"] = {"from": m, "rank": rank, "rec": store.cands[c], "settings": st["meta"].get(c)}
    return out


def _std(names: list[str]) -> dict[str, list[float]]:
    f = {m: G.features(m) for m in names}
    k = len(next(iter(f.values())))
    mu = [sum(f[m][i] for m in names) / len(names) for i in range(k)]
    sd = [max((sum((f[m][i] - mu[i]) ** 2 for m in names) / len(names)) ** 0.5, 1e-9) for i in range(k)]
    return {m: [(f[m][i] - mu[i]) / sd[i] for i in range(k)] for m in names}


def dist(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def mix_plan(target: str, pool: list[str], z: dict, plans: dict) -> bytes:
    """Mix and match the accepted plans of the two nearest maps: draft profile and style knobs
    by the nearer one where they disagree, every other knob by inverse-distance weighted median."""
    near = sorted(pool, key=lambda m: dist(z[target], z[m]))[:2]
    sets = [plans[f"{m}#0"]["settings"] for m in near]
    w = [1 / max(dist(z[target], z[m]), 1e-3) for m in near]
    roster = sets[0]["roster"]
    keys = set(sets[0]["overrides"]) | set(sets[1]["overrides"])
    dflt = C.defaults()
    ov = {}
    for k in keys:
        vals = sorted(((s["overrides"].get(k, dflt[int(k)]), wi) for s, wi in zip(sets, w)), key=lambda t: t[0])
        half, acc = sum(wi for _, wi in vals) / 2, 0.0
        for v, wi in vals:
            acc += wi
            if acc >= half:
                ov[int(k)] = v
                break
    return C.record(draft=roster, overrides=ov)


def cell_opps(splits: dict) -> list[str]:
    import random

    return sorted(random.Random(31).sample(splits["opps_B"], CELL_OPPS))


def run(hours: float | None = None) -> None:
    snap = C.check_snapshot()
    splits = C.load_json(C.RUNS / "splits.json", None)
    plans = proven()
    searched = sorted({p["from"] for p in plans.values()})
    held = splits["maps_heldout"]
    z = _std(searched + held)
    store = C.Store("crossmap", snap["wasm"])
    recs = {"base": C.record(), **{pid: bytes.fromhex(p["rec"]) for pid, p in plans.items()}}
    games = []
    for m in searched + held:
        for pid in ["base", *plans]:
            games += [C.Game(pid, m, o, s) for o in cell_opps(splits) for s in "ab"]
    # Mix-and-match plans: one per map, built without that map (leave-one-map-out).
    for m in searched + held:
        pool = [s for s in searched if s != m]
        if len(pool) >= 2:
            pid = f"mix:{m}"
            recs[pid] = mix_plan(m, pool, z, plans)
            games += [C.Game(pid, m, o, s) for o in cell_opps(splits) for s in "ab"]
    print(f"{len(plans)} proven plans, {len(searched)} searched + {len(held)} held-out maps, {len(games)} games")
    if not C.play(games, recs, store, C.opponents(), C.deadline_from(hours)):
        print("stopped at the deadline: run crossmap again to continue")
        return
    report()


def report() -> dict:
    snap = C.check_snapshot()
    splits = C.load_json(C.RUNS / "splits.json", None)
    plans = proven()
    searched = sorted({p["from"] for p in plans.values()})
    held = splits["maps_heldout"]
    z = _std(searched + held)
    store = C.Store("crossmap", snap["wasm"])
    g = {(pid, m): C.paired(store, pid, "base", map=m) for pid in plans for m in searched + held}
    gm = {m: C.paired(store, f"mix:{m}", "base", map=m) for m in searched + held}

    def val(pid, m):
        return g[(pid, m)]["win"] if g[(pid, m)]["n"] else None

    def choose(method: str, target: str, pool: list[str], tau: float = 0.0, h: float = 1.5):
        cands = [p for p in plans if plans[p]["from"] in pool]
        if method == "defaults":
            return None
        if method in ("1nn", "gated"):
            near = min(pool, key=lambda m: dist(z[target], z[m]))
            if method == "gated" and dist(z[target], z[near]) > tau:
                return None
            return f"{near}#0"
        if method == "robust":
            mean = {p: sum(val(p, m) or 0 for m in pool) / len(pool) for p in cands}
            best = max(mean, key=mean.get)
            return best if mean[best] > 0 else None
        if method == "wknn":
            w = {m: math.exp(-(dist(z[target], z[m]) / h) ** 2) for m in pool}
            pred = {p: sum(w[m] * (val(p, m) or 0) for m in pool) / max(sum(w.values()), 1e-9) for p in cands}
            best = max(pred, key=pred.get)
            return best if pred[best] > tau else None
        if method == "mix":
            return "mix"
        raise ValueError(method)

    def score(method: str, maps: list[str], loo: bool, **kw) -> dict:
        res = []
        for m in maps:
            pool = [s for s in searched if s != m] if loo else searched
            p = choose(method, m, pool, **kw)
            if p is None:
                res.append(0.0)
            elif p == "mix":
                res.append(gm[m]["win"] if gm[m]["n"] else 0.0)
            else:
                res.append(val(p, m) or 0.0)
        n = len(res)
        mu = sum(res) / n
        se = (sum((x - mu) ** 2 for x in res) / max(n - 1, 1) / n) ** 0.5
        return {"dwin": round(mu, 4), "se": round(se, 4), "worst": round(min(res), 3)}

    # Thresholds tuned leave-one-map-out on the searched maps, then fixed for the held-out maps.
    best_tau = max([0.5, 1.0, 1.5, 2.0, 3.0, 99.0], key=lambda t: score("gated", searched, True, tau=t)["dwin"])
    best_h = max([0.5, 1.0, 1.5, 2.5], key=lambda h: score("wknn", searched, True, h=h)["dwin"])
    oracle = {m: max([0.0] + [val(p, m) or 0 for p in plans]) for m in held}
    rows = {}
    for name, kw in (("defaults", {}), ("1nn", {}), ("gated", {"tau": best_tau}), ("wknn", {"h": best_h}),
                     ("robust", {}), ("mix", {})):
        rows[name] = {"loo_searched": score(name, searched, True, **kw), "heldout": score(name, held, False, **kw)}
    out = {"searched": searched, "plans": len(plans), "gated_tau": best_tau, "wknn_h": best_h,
           "heldout_oracle_dwin": round(sum(oracle.values()) / len(oracle), 4), "methods": rows}
    C.save_json(C.RUNS / "crossmap.json", out)
    print(f"{len(plans)} plans from {len(searched)} maps; held-out oracle (best proven plan per map): "
          f"Δwin {out['heldout_oracle_dwin']:+.3f}")
    for name, r in rows.items():
        a, b = r["loo_searched"], r["heldout"]
        print(f"  {name:9s} leave-one-out {a['dwin']:+.3f} ± {a['se']:.3f} (worst {a['worst']:+.2f})"
              f" | held-out {b['dwin']:+.3f} ± {b['se']:.3f} (worst {b['worst']:+.2f})")
    return out
