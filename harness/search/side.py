"""Side-specific plans: maps where the accepted plan helps one side far more than the other.

The side check (accepted plan's Δwin as team A vs team B on the scoring opponents) found one
map, broken-curve (+0.03 as team A, +0.71 as team B). This searches a plan for the weak side
only (games as that side), from scratch, in runs/search/side/<map>@<side>/. The other side
keeps the map's accepted plan; the bot picks by its spawn (settings.rs record version 2).

    python -m harness.search side --map broken-curve --side a [--hours H]
"""

from __future__ import annotations

from harness.search import common as C
from harness.search import search as S


class SideSearch(S.MapSearch):
    def __init__(self, name: str, side: str, splits: dict, snap: str):
        super().__init__(name, splits, snap)
        self.side = side
        self.job = f"side/{name}@{side}"
        self.store = C.Store(self.job, snap)
        self.path = self.store.dir / "state.pkl"
        self.state = self._load() or {"stage": "A", "round": 0, "alive": None, "cma": {}, "log": [],
                                      "meta": {}, "accepted": None}

    def games(self, cand: str, opps: list[str]) -> list[C.Game]:
        return [C.Game(cand, self.name, o, self.side) for o in opps]

    def note(self, msg: str) -> None:
        super().note(f"(side {self.side}) {msg}")


def run(name: str, side: str, hours: float | None) -> dict | None:
    snap = C.check_snapshot()
    splits = C.load_json(C.RUNS / "splits.json", None)
    s = SideSearch(name, side, splits, snap["wasm"])
    if not s.run(C.deadline_from(hours)):
        print("stopped at the deadline: run side again to continue")
        return None
    r = s.state["result"]
    acc = r["accepted"]
    print(f"{name} as team {side.upper()}: " + (f"accept {acc}: Δwin {r['b'][acc]['win']:+.3f} ± {r['b'][acc]['win_se']:.3f}"
                                                if acc else "keep the map plan (no candidate passed)"))
    return r


def accepted(name: str, side: str) -> str | None:
    """The side plan's settings record (hex), or None."""
    import pickle

    p = C.RUNS / "side" / f"{name}@{side}" / "state.pkl"
    if not p.exists():
        return None
    with open(p, "rb") as f:
        st = pickle.load(f)
    acc = (st.get("result") or {}).get("accepted")
    if not acc:
        return None
    return C.Store(f"side/{name}@{side}", C.check_snapshot()["wasm"]).cands[acc]
