"""Per-map plan search (plan step 4).

Stage A  successive halving over style x roster arms on growing subsets of the A opponents.
Stage B  CatCMAwM (cmaes) over the audit's plan knobs not set by the arm's style, one optimiser
         per top arm, restarted from a random and from an extreme start when it stalls.
Stage C  the best 4 candidates play every A game, the A leader and runner-up play every B game
         (scoring opponents never used to choose); accept only if B agrees (see `accept`).

Everything is scored paired against the base settings on the same (map, opponent, side) games.
State per map lives in runs/search/search/<map>/state.pkl (atomic), games in its games.jsonl.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import random
import time

import numpy as np
from cmaes import CatCMAwM

from harness.search import audit
from harness.search import common as C

#: Ordinal knobs: value grid (None = no override: keep the compiled / map-derived value).
GRIDS: dict[str, list[int | None]] = {
    "AGGRESSION": [0, 32, 64, 96, 128, 160, 192, 224, 255],
    "RETREAT_HP_PCT": [5, 15, 25, None, 45, 55, 70],
    "TRIBUTE_MIN_GEMS": [5, 10, 15, None, 35, 50, 70, 100],
    "TRIBUTE_MAX_GEMS": [30, 50, 70, 90, 120, 150, 180, 220, 255],
    "TRIBUTE_DISTANCE_SCALE": [0, 3, 6, None, 14, 20, 30, 40],
    "RECALL_GOLD": [3, 6, 10, 15, 20, 30, 45, 60],
    "FIGHT_MARGIN_PCT": [0, 5, 10, 25, 40, 60, 80, 100],
    "SQUAD_JOIN_X10": [0, 10, 15, 20],
    "K2_TEAM_UP": [0, 3, 6, 9, 12, 16],
    "K2_WALK_HOME": [0, 3, 6, 9, 12],
    "K2_PUNISH_RECALL": [0, 3, 6, 9],
    "K2_UPGRADE_FIRST": [0, 5, 10, 20],
    "K2_SOLO_CAP": [0, 2, 3, 4, 6],
    "K2_CLUSTER": [0, 5, 10, 15, 20],
}
#: Categorical knobs: the values to choose from.
CATS: dict[str, list[int]] = {
    "BUILD_PROFILE": list(range(18)),
    "VALUE_PROFILE": [0, 1, 2, 4, 5, 6],
    "RECALL_SAFE": [0, 1, 2],
    "HUNT_MODE": [0, 1, 2],
    "EN_ROUTE": [0, 1, 2],
    "K2_GOLD_ORDER": [0, 1, 2],
    "K2_JUMP_HOLES": [0, 1],
}
#: Bit-field knobs: each bit is its own on/off choice.
BITS: dict[str, list[int]] = {
    "CAMP_SAFE": [1, 2],
    "FIGHT_V6": [1, 2, 4, 8, 16],
    "CAMP_FEASIBLE": [1, 2, 8, 16, 32],
    "COMBAT_FSM": [1, 2, 64, 128],
    "K2_DRAFT_PLAN": [1, 2, 4, 8, 16],
    "K2_TRIBUTE_DETOUR": [1, 2, 4],
}
ROSTERS = [0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13, 14]
POP = 12
#: Opponents per stage (both sides each): halving rounds, then CMA candidates.
HALVING = [8, 20, None]  # None = every A opponent
CMA_OPPS = 12
MAX_GENS = 10
STALL_GENS = 4
BUDGET = 6_000  # games per map (trimmed 2026-10-25: ~45-50 min a map)
TOP_C = 4


class Space:
    """The knob dimensions searched for one arm (knobs its style sets stay as the style sets)."""

    def __init__(self, items: list[str], fixed: dict[str, int]):
        ks, d = C.knob_slots(), C.defaults()
        self.z: list[tuple[str, list[int | None]]] = []
        self.c: list[tuple[str, str, list]] = []  # (knob, "cat"|"bit", values or bit)
        for it in items:
            if it in fixed:
                continue
            if it in GRIDS:
                self.z.append((it, GRIDS[it]))
            elif it in CATS:
                self.c.append((it, "cat", CATS[it]))
            elif it in BITS:
                for b in BITS[it]:
                    self.c.append((it, "bit", [b]))
            elif it in ks:  # a plain on/off knob probed at one value: default vs that value
                vals = sorted({d[ks[it]], *_probe_values(it)})
                if len(vals) > 1:
                    self.c.append((it, "cat", vals))
        self.ks, self.d = ks, d

    def empty(self) -> bool:
        return not self.z and not self.c

    def default_mean(self) -> np.ndarray:
        out = []
        for k, grid in self.z:
            cur = self.d[self.ks[k]]
            out.append(min(range(len(grid)), key=lambda i: abs((grid[i] if grid[i] is not None else cur) - cur)))
        return np.array(out, dtype=float)

    def overrides(self, z: np.ndarray, c: np.ndarray) -> dict[int, int]:
        out: dict[int, int] = {}
        for (k, grid), zi in zip(self.z, [] if z is None else z):
            v = grid[int(round(zi))]
            if v is not None:
                out[self.ks[k]] = v
        bits: dict[str, int] = {}
        for (k, kind, vals), ci in zip(self.c, [] if c is None else c):
            j = int(np.argmax(ci))
            if kind == "cat":
                out[self.ks[k]] = vals[j]
            else:
                bits.setdefault(k, 0)
                bits[k] |= vals[0] if j == 1 else 0
        for k, v in bits.items():
            out[self.ks[k]] = v
        return out

    def optimiser(self, seed: int, start: str) -> CatCMAwM:
        rng = np.random.default_rng(seed)
        zs = [list(range(len(g))) for _, g in self.z] or None
        cs = [len(v) if kind == "cat" else 2 for _, kind, v in self.c] or None
        mean = None
        if self.z:
            if start == "default":
                mean = self.default_mean()
            elif start == "random":
                mean = np.array([rng.uniform(0, len(g) - 1) for _, g in self.z])
            else:  # an extreme corner
                mean = np.array([float(rng.choice([0, len(g) - 1])) for _, g in self.z])
        cat = None
        if self.c:
            width = max(cs)
            cat = np.zeros((len(cs), width))
            for i, n in enumerate(cs):
                if start == "default":
                    # Lean towards the current value, keep the others possible.
                    k, kind, vals = self.c[i]
                    cur = self.d[self.ks[k]]
                    j = (vals.index(cur) if cur in vals else 0) if kind == "cat" else int(bool(cur & vals[0]))
                    cat[i, :n] = 0.5 / max(n - 1, 1)
                    cat[i, j] = 0.5
                    if n == 1:
                        cat[i, 0] = 1.0
                else:
                    cat[i, :n] = 1.0 / n
        cov = np.diag([(len(g) / 5.0) ** 2 for _, g in self.z]) if self.z else None
        return CatCMAwM(z_space=zs, c_space=cs, population_size=POP, mean=mean,
                        sigma=1.0 if self.z else None, cov=cov, cat_param=cat, seed=seed)


def _probe_values(item: str) -> list[int]:
    for it, _, vals, _, _ in audit.PROBES:
        if it == item:
            return vals
    return [1]


def plan_items() -> list[str]:
    a = C.load_json(C.RUNS / "audit.json", None)
    if not a:
        raise SystemExit("no audit: run audit first")
    return sorted({r["item"] for r in a["rows"] if r["verdict"] == "plan" and r["item"] not in ("draft", "style")})


def arms() -> list[tuple[str, int]]:
    """Style x roster, without styles and rosters the audit found clearly worse everywhere."""
    a = C.load_json(C.RUNS / "audit.json", {"rows": []})
    bad = {(r["item"], str(r["value"])) for r in a["rows"]
           if r["item"] in ("draft", "style") and r["dfit"] < -0.15 and r["hetero"] < 2}
    styles = [s for s in audit.styles() if ("style", s) not in bad]
    rosters = [r for r in ROSTERS if ("draft", str(r)) not in bad]
    return [(s, r) for s in styles for r in rosters]


def cand_id(rec: bytes) -> str:
    return "c" + hashlib.sha1(rec).hexdigest()[:10] if rec else "base"


class MapSearch:
    def __init__(self, name: str, splits: dict, snap: str):
        self.name = name
        self.job = f"search/{name}"
        self.store = C.Store(self.job, snap)
        self.path = self.store.dir / "state.pkl"
        rng = random.Random(f"{name}-opps")
        a = list(splits["opps_A"])
        rng.shuffle(a)
        self.opps_a, self.opps_b = a, list(splits["opps_B"])
        self.state = self._load() or {"stage": "A", "round": 0, "alive": None, "cma": {}, "log": [],
                                      "meta": {}, "accepted": None}

    # ---- persistence
    def _load(self):
        try:
            with open(self.path, "rb") as f:
                return pickle.load(f)
        except (OSError, pickle.PickleError, EOFError):
            return None

    def save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(self.state, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.path)

    # ---- games
    def games(self, cand: str, opps: list[str]) -> list[C.Game]:
        return [C.Game(cand, self.name, o, s) for o in opps for s in "ab"]

    def score(self, cand: str, opps: list[str] | None = None) -> dict:
        p = C.paired(self.store, cand, "base", map=self.name)
        if opps is not None:
            keep = set(opps)
            base = {(r["opp"], r["side"]): r for r in self.store.select("base", map=self.name) if r["opp"] in keep}
            d = [C.fitness(r) - C.fitness(base[(r["opp"], r["side"])])
                 for r in self.store.select(cand, map=self.name) if (r["opp"], r["side"]) in base]
            w = [r["res"] - base[(r["opp"], r["side"])]["res"]
                 for r in self.store.select(cand, map=self.name) if (r["opp"], r["side"]) in base]
            p = {"n": len(d), **C._mean_se(d, "fit"), **C._mean_se(w, "win")}
        return p

    def used(self) -> int:
        return len(self.store.select(map=self.name))

    def play(self, recs: dict[str, bytes], opps: list[str], deadline) -> bool:
        games = [g for c in ["base", *recs] for g in self.games(c, opps)]
        return C.play(games, {"base": C.record(), **recs}, self.store, C.opponents(), deadline,
                      log=lambda m: None)

    def note(self, msg: str) -> None:
        self.state["log"].append(f"{time.strftime('%H:%M:%S')} {msg}")
        print(f"[{self.name}] {msg}", flush=True)

    # ---- stages
    def run(self, deadline) -> bool:
        """Advance until done or the deadline. True when the map is finished."""
        while True:
            st = self.state["stage"]
            if st == "done":
                return True
            if deadline and time.time() > deadline:
                return False
            ok = {"A": self.stage_a, "B": self.stage_b, "C": self.stage_c}[st](deadline)
            self.save()
            if not ok:
                return False

    def stage_a(self, deadline) -> bool:
        st = self.state
        if st["alive"] is None:
            st["alive"] = [list(a) for a in arms()]
        recs = {}
        for s, r in st["alive"]:
            rec = C.record(draft=r, overrides=audit.style_overrides(s))
            cid = cand_id(rec)
            st["meta"][cid] = {"style": s, "roster": r, "overrides": {}}
            recs[cid] = rec
        k = HALVING[st["round"]]
        opps = self.opps_a[:k] if k else self.opps_a
        if not self.play(recs, opps, deadline):
            return False
        ranked = sorted(st["alive"], key=lambda a: -self.score(
            cand_id(C.record(draft=a[1], overrides=audit.style_overrides(a[0]))), opps)["fit"])
        st["round"] += 1
        self.note(f"A round {st['round']}: {len(st['alive'])} arms on {len(opps)} opps, best {ranked[0]}")
        if st["round"] >= len(HALVING):
            st["top_arms"] = ranked[:2]
            st["stage"] = "B"
        else:
            st["alive"] = ranked[: max(2, len(ranked) // 3)]
        return True

    def stage_b(self, deadline) -> bool:
        st = self.state
        items = plan_items()
        opps = self.opps_a[:CMA_OPPS]
        live = 0
        for ai, (style, roster) in enumerate(st["top_arms"]):
            key = f"{style}/{roster}"
            space = Space(items, audit.styles()[style])
            if space.empty():
                continue
            c = st["cma"].setdefault(key, {"starts": ["default", "random"], "opt": None,
                                           "gen": 0, "best": -9.0, "stall": 0, "pending": None, "done": False})
            if c["done"]:
                continue
            if self.used() > BUDGET:
                c["done"] = True
                continue
            if c["opt"] is None:
                if not c["starts"]:
                    c["done"] = True
                    continue
                start = c["starts"].pop(0)
                seed = int(hashlib.sha1(f"{self.name}{key}{start}".encode()).hexdigest()[:8], 16)
                c.update(opt=space.optimiser(seed, start), gen=0, stall=0, best=-9.0, start=start)
            live += 1
            if c["pending"] is None:
                c["pending"] = [c["opt"].ask() for _ in range(c["opt"].population_size)]
                self.save()
            recs, ids = {}, []
            base_ov = audit.style_overrides(style)
            for sol in c["pending"]:
                ov = {**base_ov, **space.overrides(sol.z, sol.c)}
                rec = C.record(draft=roster, overrides=ov)
                cid = cand_id(rec)
                st["meta"][cid] = {"style": style, "roster": roster, "overrides": ov}
                recs[cid] = rec
                ids.append(cid)
            if not self.play(recs, opps, deadline):
                return False
            vals = [self.score(cid, opps)["fit"] for cid in ids]
            c["opt"].tell([(sol, -v) for sol, v in zip(c["pending"], vals)])
            c["pending"] = None
            c["gen"] += 1
            if max(vals) > c["best"] + 0.01:
                c["best"], c["stall"] = max(vals), 0
            else:
                c["stall"] += 1
            self.note(f"B {key} {c['start']} gen {c['gen']}: best {max(vals):+.3f} (run best {c['best']:+.3f})")
            if c["stall"] >= STALL_GENS or c["gen"] >= MAX_GENS or c["opt"].should_stop():
                c["opt"] = None
        if live == 0:
            st["stage"] = "C"
        return True

    def stage_c(self, deadline) -> bool:
        st = self.state
        if "top" not in st:
            # Chosen once (a resumed run must not re-rank with the B games it already played).
            cands = [c for c in st["meta"] if c != "base"]
            scored = [(self.score(c, self.opps_a)["fit"], c) for c in cands
                      if self.score(c, self.opps_a)["n"] >= 2 * CMA_OPPS]
            st["top"] = [c for _, c in sorted(scored, key=lambda t: -t[0])[:TOP_C]]
            self.save()
        top = st["top"]
        recs = {c: self.store_rec(c) for c in top}
        if not self.play(recs, self.opps_a, deadline):
            return False
        a_scores = {c: self.score(c, self.opps_a) for c in top}
        lead = sorted(top, key=lambda c: -a_scores[c]["fit"])[:2]
        if not self.play({c: recs[c] for c in lead}, self.opps_b, deadline):
            return False
        b_scores = {c: self.score(c, self.opps_b) for c in lead}
        # The A leader that B confirms (B only checks; it never picks between candidates).
        best = next((c for c in lead if accept(a_scores[c], b_scores[c], self.oof(c))), None)
        st["result"] = {
            "a": {c: a_scores[c] for c in top},
            "b": b_scores,
            "accepted": best,
            "settings": st["meta"].get(best) if best else None,
            "candidates": {c: st["meta"].get(c) for c in top},
            "games": self.used(),
        }
        st["accepted"] = best
        st["stage"] = "done"
        self.note(f"C: A {[(c, a_scores[c]['fit']) for c in lead]}, B {[(c, b_scores[c]['fit']) for c in lead]}"
                  f" -> {'accept ' + best if best else 'keep defaults'}")
        return True

    def store_rec(self, cid: str) -> bytes:
        return bytes.fromhex(self.store.cands[cid])

    def oof(self, cid: str) -> int:
        base = {(r["opp"], r["side"]): r for r in self.store.select("base", map=self.name)}
        return sum(C.stalls(r) - C.stalls(base[(r["opp"], r["side"])])
                   for r in self.store.select(cid, map=self.name) if (r["opp"], r["side"]) in base)


def accept(a: dict, b: dict, d_oof: int) -> bool:
    """B (never used to choose) must show the gain: Δfit_B >= 1 SE, Δwin_B >= 0, no extra
    out-of-fuel turns beyond 2, and B keeps at least a third of the A gain."""
    return (b["fit"] >= max(b["fit_se"], 1e-9) and b["win"] >= 0 and d_oof <= 2
            and b["fit"] >= a["fit"] / 3)


def run(maps: list[str] | None, hours: float | None) -> None:
    snap = C.check_snapshot()
    splits = C.load_json(C.RUNS / "splits.json", None)
    if not splits:
        raise SystemExit("no splits: run splits first")
    deadline = C.deadline_from(hours)
    todo = maps or splits["maps_train"]
    for m in todo:
        s = MapSearch(m, splits, snap["wasm"])
        if not s.run(deadline):
            print(f"stopped at the deadline in {m}: run search again to continue")
            return
    table()


def table() -> dict:
    """runs/search/table.json: the accepted settings per map (defaults where none passed)."""
    out = {}
    for d in sorted((C.RUNS / "search").glob("*/state.pkl")):
        with open(d, "rb") as f:
            st = pickle.load(f)
        r = st.get("result")
        if r:
            out[d.parent.name] = r
    C.save_json(C.RUNS / "table.json", out)
    acc = [m for m, r in out.items() if r["accepted"]]
    print(f"table: {len(out)} maps searched, {len(acc)} accepted: {', '.join(acc)}")
    return out


def status() -> None:
    for d in sorted((C.RUNS / "search").glob("*/state.pkl")):
        with open(d, "rb") as f:
            st = pickle.load(f)
        print(f"{d.parent.name}: stage {st['stage']}, {st['log'][-1] if st['log'] else ''}")


if __name__ == "__main__":
    print(json.dumps(table(), indent=1)[:2000])
