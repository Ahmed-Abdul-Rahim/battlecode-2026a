"""Does A carry over to unseen maps? Nearest-neighbour transfer of the searched table.

Each held-out map (splits.json) takes the accepted settings of its nearest searched map in a
standardised map-feature space, and plays the B opponents (both sides) paired against the
base settings. This is the kNN (k = 1) entry of the draft-net shortlist.
"""

from __future__ import annotations

import json
import math

from harness.search import common as C
from harness.search import search as S

MAPS_CACHE = C.MAIN / "runs" / "gpusim" / "maps"
WOLF, ELK, GEM, GOLD, MONOLITH, GOLEM = range(6)


def _bits(hexs: str) -> int:
    return bin(int(hexs, 16)).count("1") if hexs else 0


def features(name: str) -> list[float]:
    f = next(MAPS_CACHE.glob(f"??{name}.json"))
    m = json.loads(json.loads(f.read_text(encoding="utf-8"))["M"][2:])
    area = m["w"] * m["h"]
    gs = m["groups"]
    camps = [c[0] for g in gs for c in g["camps"]]
    n = max(len(camps), 1)
    gems = sum(g["gems"] for g in gs)
    gold = sum(g["gold"] for g in gs)
    geo = json.loads((C.ROOT / "data" / "map_geometry.json").read_text(encoding="utf-8"))[name]
    sp, ep = geo["spawn_a"], geo["spawn_b"]
    d_own = [max(abs(g["c"][0] - sp[0]), abs(g["c"][1] - sp[1])) for g in gs] or [0]
    return [
        math.log(len(gs) + 1),
        math.log(gems + 1),
        math.log(gold + 1),
        gems / max(gems + gold, 1),
        sum(c == ELK for c in camps) / n,
        sum(c in (GOLEM, MONOLITH) for c in camps) / n,
        math.log(area),
        _bits(m["walls"]) / area,
        _bits(m["bushes"]) / area,
        sum(d_own) / len(d_own) / max(m["w"], m["h"]),
        max(abs(sp[0] - ep[0]), abs(sp[1] - ep[1])) / max(m["w"], m["h"]),
    ]


def run(hours: float | None = None) -> dict:
    snap = C.check_snapshot()
    splits = C.load_json(C.RUNS / "splits.json", None)
    table = {m: r for m, r in S.table().items() if r["accepted"]}
    held = splits["maps_heldout"]
    names = sorted(set(table) | set(held))
    feats = {m: features(m) for m in names}
    k = len(next(iter(feats.values())))
    mu = [sum(feats[m][i] for m in names) / len(names) for i in range(k)]
    sd = [max((sum((feats[m][i] - mu[i]) ** 2 for m in names) / len(names)) ** 0.5, 1e-9) for i in range(k)]
    z = {m: [(feats[m][i] - mu[i]) / sd[i] for i in range(k)] for m in names}
    store = C.Store("generalize", snap["wasm"])
    recs, games, pick = {"base": C.record()}, [], {}
    for h in held:
        src = min(table, key=lambda t: sum((a - b) ** 2 for a, b in zip(z[h], z[t])))
        cid = table[src]["accepted"]
        rec = bytes.fromhex(C.Store(f"search/{src}", snap["wasm"]).cands[cid])
        tag = f"knn:{src}"
        recs[tag] = rec
        pick[h] = src
        games += [C.Game(c, h, o, s) for c in ("base", tag) for o in splits["opps_B"] for s in "ab"]
    ok = C.play(games, recs, store, C.opponents(), C.deadline_from(hours))
    out, dws = {}, []
    for h in held:
        p = C.paired(store, f"knn:{pick[h]}", "base", map=h)
        out[h] = {"from": pick[h], **p}
        dws.append(p["win"])
    n = len(dws)
    mean = sum(dws) / n
    se = (sum((x - mean) ** 2 for x in dws) / max(n - 1, 1) / n) ** 0.5
    res = {"maps": out, "dwin": round(mean, 4), "dwin_se": round(se, 4), "complete": ok}
    C.save_json(C.RUNS / "generalize.json", res)
    for h, r in out.items():
        print(f"  {h:20s} <- {r['from']:20s} Δwin {r['win']:+.3f} ± {r['win_se']:.3f}  Δfit {r['fit']:+.3f}")
    print(f"held-out maps: Δwin {mean:+.3f} ± {se:.3f} (kNN transfer from {len(table)} searched maps)")
    return res
