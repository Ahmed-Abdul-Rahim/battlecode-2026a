"""Warm-start trial: copy the plan of the nearest proven map and finetune it (plan step 3).

Same staged search as search.py, but stage A is skipped (two arms only: the donor plan's
style + roster, and the default rules), stage B starts from the donor's knob values (and the
defaults), and the budget is halved. Results live in runs/search/warm/<map>/, apart from the
full search, which later searches the same maps for the comparison.

    python -m harness.search warm --maps a,b [--hours H]
    python -m harness.search warm-report
"""

from __future__ import annotations

import numpy as np
from cmaes import CatCMAwM

from harness.search import audit
from harness.search import common as C
from harness.search import generalize as G
from harness.search import search as S

WARM_BUDGET = 3_000


class WarmSpace(S.Space):
    donor: dict[int, int] = {}

    def optimiser(self, seed: int, start: str) -> CatCMAwM:
        if start != "neighbour":
            return super().optimiser(seed, start)
        zs = [list(range(len(g))) for _, g in self.z] or None
        cs = [len(v) if kind == "cat" else 2 for _, kind, v in self.c] or None
        mean = None
        if self.z:
            vals = []
            for k, grid in self.z:
                v = self.donor.get(self.ks[k], self.d[self.ks[k]])
                vals.append(min(range(len(grid)), key=lambda i: abs((grid[i] if grid[i] is not None else v) - v)))
            mean = np.array(vals, dtype=float)
        cat = None
        if self.c:
            cat = np.zeros((len(cs), max(cs)))
            for i, n in enumerate(cs):
                k, kind, vals = self.c[i]
                cur = self.donor.get(self.ks[k], self.d[self.ks[k]])
                j = (vals.index(cur) if cur in vals else 0) if kind == "cat" else int(bool(cur & vals[0]))
                cat[i, :n] = 0.4 / max(n - 1, 1)
                cat[i, j] = 0.6
        cov = np.diag([(len(g) / 6.0) ** 2 for _, g in self.z]) if self.z else None
        return CatCMAwM(z_space=zs, c_space=cs, population_size=S.POP, mean=mean,
                        sigma=1.0 if self.z else None, cov=cov, cat_param=cat, seed=seed)


def donor_for(name: str, table: dict) -> str:
    z = {m: G.features(m) for m in [name, *table]}
    k = len(z[name])
    names = list(z)
    mu = [sum(z[m][i] for m in names) / len(names) for i in range(k)]
    sd = [max((sum((z[m][i] - mu[i]) ** 2 for m in names) / len(names)) ** 0.5, 1e-9) for i in range(k)]
    std = {m: [(z[m][i] - mu[i]) / sd[i] for i in range(k)] for m in names}
    return min(table, key=lambda t: sum((a - b) ** 2 for a, b in zip(std[name], std[t])))


class WarmSearch(S.MapSearch):
    def __init__(self, name: str, splits: dict, snap: str, table: dict):
        super().__init__(name, splits, snap)
        self.job = f"warm/{name}"
        self.store = C.Store(self.job, snap)
        self.path = self.store.dir / "state.pkl"
        self.state = self._load() or {"stage": "A", "round": 0, "alive": None, "cma": {}, "log": [],
                                      "meta": {}, "accepted": None}
        src = donor_for(name, table)
        self.donor_map, self.donor = src, table[src]["settings"]

    def stage_a(self, deadline) -> bool:
        st = self.state
        d = (self.donor["style"], self.donor["roster"])
        st["top_arms"] = [list(d), ["base", 0]] if d != ("base", 0) else [["base", 0]]
        key = f"{d[0]}/{d[1]}"
        st["cma"][key] = {"starts": ["neighbour", "default"], "opt": None, "gen": 0, "best": -9.0,
                          "stall": 0, "pending": None, "done": False}
        st["cma"]["base/0"] = {"starts": ["default"], "opt": None, "gen": 0, "best": -9.0,
                               "stall": 0, "pending": None, "done": False}
        st["stage"] = "B"
        self.note(f"warm: donor {self.donor_map} ({d[0]}/{d[1]})")
        return True

    def run(self, deadline) -> bool:
        S.Space, S.BUDGET = WarmSpace, WARM_BUDGET
        WarmSpace.donor = {int(k): v for k, v in self.donor["overrides"].items()}
        return super().run(deadline)


def run(maps: list[str], hours: float | None) -> None:
    snap = C.check_snapshot()
    splits = C.load_json(C.RUNS / "splits.json", None)
    table = {m: r for m, r in S.table().items() if r["accepted"]}
    deadline = C.deadline_from(hours)
    for m in maps:
        if m in table:
            print(f"{m} is already searched; pick an unsearched map")
            continue
        if not WarmSearch(m, splits, snap["wasm"], table).run(deadline):
            print(f"stopped at the deadline in {m}: run warm again to continue")
            return
    report()


def report() -> None:
    import pickle

    full = {m: r for m, r in S.table().items()}
    for d in sorted((C.RUNS / "warm").glob("*/state.pkl")):
        with open(d, "rb") as f:
            st = pickle.load(f)
        m, r = d.parent.name, st.get("result")
        if not r:
            print(f"{m}: warm not finished")
            continue
        wb = r["b"].get(r["accepted"]) if r["accepted"] else None
        fr = full.get(m)
        fb = fr["b"].get(fr["accepted"]) if fr and fr["accepted"] else None
        print(f"{m}: warm {r['games']} games -> {'Δwin %+.3f fit %+.3f' % (wb['win'], wb['fit']) if wb else 'keep defaults'}"
              f" | full {fr['games'] if fr else '-'} games -> "
              f"{'Δwin %+.3f fit %+.3f' % (fb['win'], fb['fit']) if fb else ('keep defaults' if fr else 'not searched yet')}")
