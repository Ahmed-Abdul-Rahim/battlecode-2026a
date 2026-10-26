"""Honest splits (port of archive nn/portfolio.py's map halves, extended to opponents).

- Opponents A (search) / B (scoring only), stratified by the baseline win rate, with the
  ladder-faithful mimics (league/ledger.json pool) split separately so both halves get them.
  Our own rule bots ("rules", "lg-v23-fuel") are in neither: they are the head-to-head test.
- Maps: 30 train / 10 held-out for the draft net, one held-out map per group-count decile;
  maps/heldout (contest, monolith) are always held out. Fixed once, never changed: new pool
  maps are added by `extend` (same rule over the new maps only), old assignments are kept.
"""

from __future__ import annotations

import json
import random

from harness.search import common as C

HEAD_TO_HEAD = ("rules", "lg-v23-fuel")
SEED = 7


def faithful() -> set[str]:
    led = C.load_json(C.MAIN / "league" / "ledger.json", {})
    return {f"mimic-{p['slug']}" for p in led.get("pool", [])}


def _halves(names: list[str], win: dict[str, float], rng: random.Random) -> tuple[list, list]:
    names = sorted(names, key=lambda n: (win.get(n, 0.5), n))
    a, b = [], []
    for i in range(0, len(names), 2):
        pair = names[i : i + 2]
        rng.shuffle(pair)
        a.append(pair[0])
        if len(pair) > 1:
            b.append(pair[1])
    return a, b


def make(store: C.Store) -> dict:
    rng = random.Random(SEED)
    rows = store.select("base")
    win: dict[str, float] = {}
    for o in {r["opp"] for r in rows}:
        rs = [r["res"] for r in rows if r["opp"] == o]
        win[o] = sum(rs) / len(rs)
    opps = [o for o in C.opponents() if o not in HEAD_TO_HEAD]
    fa = faithful()
    a1, b1 = _halves([o for o in opps if o in fa], win, rng)
    a2, b2 = _halves([o for o in opps if o not in fa], win, rng)

    geo = json.loads((C.ROOT / "data" / "map_geometry.json").read_text(encoding="utf-8"))
    groups = {m: len(geo[m]["centres"]) for m in C.maps()}
    forced = {C.map_key(p.stem) for p in (C.MAPS_DIR / "heldout").glob("*.map26a")}
    order = sorted(groups, key=lambda m: (groups[m], m))
    held = []
    for i in range(0, len(order), 4):
        b = order[i : i + 4]
        f = [m for m in b if m in forced]
        held.append(f[0] if f else rng.choice(b))
    held += [m for m in forced if m not in held]
    return {
        "opps_A": a1 + a2,
        "opps_B": b1 + b2,
        "head_to_head": [o for o in HEAD_TO_HEAD if o in C.opponents()],
        "maps_train": [m for m in order if m not in held],
        "maps_heldout": sorted(held),
    }


def extend(splits: dict) -> tuple[dict, list[str]]:
    """Add maps that aren't in the splits yet (a new platform pool): one held-out map per four
    new maps by group count, the rest train. Existing assignments are never changed (results
    and the held-out test stay comparable). Returns (splits, the maps added)."""
    known = set(splits["maps_train"]) | set(splits["maps_heldout"])
    geo = json.loads((C.ROOT / "data" / "map_geometry.json").read_text(encoding="utf-8"))
    new = [m for m in C.maps() if m not in known and m in geo]
    if not new:
        return splits, []
    rng = random.Random(f"{SEED}-{len(known)}")
    order = sorted(new, key=lambda m: (len(geo[m]["centres"]), m))
    held = [rng.choice(order[i : i + 4]) for i in range(0, len(order), 4)]
    out = dict(splits)
    out["maps_train"] = list(splits["maps_train"]) + [m for m in order if m not in held]
    out["maps_heldout"] = sorted(set(splits["maps_heldout"]) | set(held))
    return out, new
