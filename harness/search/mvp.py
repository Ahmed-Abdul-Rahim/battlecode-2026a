"""MVP gate for the searched table (plan §5) and its export into the pack.

- Honest estimate: B decided acceptance, so reporting B's gain for accepted maps is biased
  upwards. Cross-fit instead: split B in two halves, accept on one half and score on the other,
  swap, average (maps that don't accept count 0: they keep the defaults).
- Head-to-head: the table vs our rule bot (league main, "rules") and the live ladder version,
  on every searched map, both sides, paired against the base settings. These opponents were
  never used by the search.
- Export: league/plan_table.json (committed; league.pack_bytes_for packs it as tag 6) and this
  worktree's bots/main/src/pack/pack.bin rebuilt from the live champion.json + the table.
"""

from __future__ import annotations

import importlib.util
import sys

from harness.search import common as C
from harness.search import search as S

GATE_WIN, WORST_MAP = 0.03, -0.10


def _halves(opps: list[str]) -> tuple[list[str], list[str]]:
    return opps[0::2], opps[1::2]


def evaluate(hours: float | None = None) -> dict:
    snap = C.check_snapshot()
    splits = C.load_json(C.RUNS / "splits.json", None)
    table = S.table()
    deadline = C.deadline_from(hours)
    # Final test: our rule bots + the refreshed ladder top 10 (no search ever chose against them).
    h2h = [o for o in splits["head_to_head"]] + sorted(C.fresh_top())
    per_map, est, est_win = {}, [], []
    for m, r in sorted(table.items()):
        ms = S.MapSearch(m, splits, snap["wasm"])
        lead = list(r["b"])
        b1, b2 = _halves(ms.opps_b)
        fits, wins = [], []
        for choose, score in ((b1, b2), (b2, b1)):
            pick = next((c for c in lead if S.accept(r["a"][c], ms.score(c, choose), 0)), None)
            sc = ms.score(pick, score) if pick else {"fit": 0.0, "win": 0.0}
            fits.append(sc["fit"])
            wins.append(sc["win"])
        acc = r["accepted"]
        row = {"accepted": acc, "honest_fit": sum(fits) / 2, "honest_win": sum(wins) / 2,
               "b": r["b"].get(acc) if acc else None, "games": r["games"]}
        if acc:
            if not ms.play({acc: ms.store_rec(acc)}, h2h, deadline):
                print("stopped at the deadline: run mvp again")
                return {}
            row["h2h"] = ms.score(acc, h2h)
            base = ms.store.select("base", map=m)
            row["h2h_base_win"] = sum(x["res"] for x in base if x["opp"] in h2h) / max(
                sum(1 for x in base if x["opp"] in h2h), 1)
        else:
            if not ms.play({}, h2h, deadline):
                return {}
        per_map[m] = row
        est.append(row["honest_fit"])
        est_win.append(row["honest_win"])
    n = max(len(est), 1)
    mean_w = sum(est_win) / n
    se_w = (sum((x - mean_w) ** 2 for x in est_win) / max(n - 1, 1) / n) ** 0.5
    worst = min((v["b"]["win"] for v in per_map.values() if v["b"]), default=0.0)
    h2h_rows = [v["h2h"] for v in per_map.values() if v.get("h2h")]
    out = {
        "maps": per_map,
        "honest_fit": round(sum(est) / n, 4),
        "honest_win": round(mean_w, 4),
        "honest_win_se": round(se_w, 4),
        "accepted": sum(1 for v in per_map.values() if v["accepted"]),
        "h2h_dwin": round(sum(h["win"] * h["n"] for h in h2h_rows) / max(sum(h["n"] for h in h2h_rows), 1), 4),
        "worst_accepted_b_win": worst,
    }
    out["gate"] = bool(mean_w >= GATE_WIN and mean_w - 1.96 * se_w > 0 and worst >= WORST_MAP)
    ceiling = out["accepted"] < 0.25 * n and mean_w < 0.02
    out["ceiling_report"] = ceiling
    C.save_json(C.RUNS / "mvp.json", out)
    print(f"{out['accepted']}/{len(per_map)} maps accepted; honest Δwin {mean_w:+.3f} ± {se_w:.3f}, "
          f"Δfit {out['honest_fit']:+.3f}; vs our rules + fresh top 10 Δwin {out['h2h_dwin']:+.3f}; gate "
          f"{'PASS' if out['gate'] else 'FAIL'}{' (ceiling: the rules expressiveness is the limit)' if ceiling else ''}")
    return out


def _league():
    sys.path.insert(0, str(C.ROOT / "harness"))
    spec = importlib.util.spec_from_file_location("league_mod", C.ROOT / "harness" / "league.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def export() -> None:
    table = S.table()
    recs = {}
    for m, r in table.items():
        if r["accepted"]:
            s = C.Store(f"search/{m}", C.check_snapshot()["wasm"])
            recs[m] = s.cands[r["accepted"]]
    # Side-specific maps (harness/search/side.py): v2 record = team A's spawn tile, then team A's
    # and team B's records; a side without its own plan keeps the map plan (or the defaults).
    from harness.search import side as SD

    geo = C.load_json(C.ROOT / "data" / "map_geometry.json", {})
    for m in table:
        sa, sb = SD.accepted(m, "a"), SD.accepted(m, "b")
        if not (sa or sb):
            continue
        base = recs.get(m, "")
        ra, rb = bytes.fromhex(sa or base), bytes.fromhex(sb or base)
        ax, ay = geo[m]["spawn_a"]
        recs[m] = (bytes([2, ax, ay, len(ra)]) + ra + rb).hex()
        print(f"{m}: side-specific plan (team A {'own' if sa else 'map'} plan, team B {'own' if sb else 'map'} plan)")
    C.save_json(C.ROOT / "league" / "plan_table.json", {
        "_doc": "harness/search per-map plan settings (settings.rs records, hex); pack tag 6",
        "snapshot": C.check_snapshot()["wasm"],
        "maps": recs,
    })
    lg = _league()
    ch = C.load_json(C.MAIN / "league" / "champion.json", {})
    data = lg.pack_bytes_for(ch["maps"])
    out = C.ROOT / "bots" / "main" / "src" / "pack" / "pack.bin"
    out.write_bytes(data)
    print(f"exported {len(recs)} maps; pack.bin {len(data)} bytes")
