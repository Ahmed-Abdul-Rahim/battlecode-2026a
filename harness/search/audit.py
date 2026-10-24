"""Rule audit (plan step 2): does switching each plan-settable decision change games, and does
its best value differ by map? Port of the probe in archive nn/rules.py.

Every item is probed at two values (an extreme pair, or off/on) on 12 train maps x 8 A
opponents x 2 sides, paired against the base settings. Output: runs/search/audit.json and
docs/rule-audit.md.
"""

from __future__ import annotations

import json
import random

from harness.search import common as C

#: (item, kind, probe values, scope from the code, note). kind: knob name, "draft" or "style".
#: Scope: "map" = reads only static map facts / roster, "turn" = a per-turn reaction whose
#: threshold can still be set per map. Dead knobs are listed without probes.
PROBES: list[tuple[str, str, list[int], str, str]] = [
    ("AGGRESSION", "knob", [40, 220], "turn", "fight margin scale, team params"),
    ("RETREAT_HP_PCT", "knob", [15, 60], "turn", "low_hp threshold"),
    ("TRIBUTE_MIN_GEMS", "knob", [10, 60], "map", "draft writes 25+0.4 walk; this overrides it"),
    ("TRIBUTE_MAX_GEMS", "knob", [50, 220], "map", "bag cap"),
    ("TRIBUTE_DISTANCE_SCALE", "knob", [3, 25], "map", "gems per altar tile x0.1"),
    ("RECALL_GOLD", "knob", [6, 40], "map", "shop trip gold x10"),
    ("FIGHT_MARGIN_PCT", "knob", [5, 60], "turn", "router fight threshold"),
    ("BUILD_PROFILE", "knob", [0, 4], "map", "emblem order (shop.rs profiles)"),
    ("SQUAD_JOIN_X10", "knob", [15], "map", "pairs join each other's group"),
    ("VALUE_PROFILE", "knob", [0, 5], "map", "camp value formula"),
    ("REMOTE_FARMER", "knob", [1], "map", "last slot farms far groups"),
    ("HUNT_MODE", "knob", [1, 2], "turn", "whole-team hunt"),
    ("DENY_TRIBUTE", "knob", [0], "turn", "stun/lock enemy tributes"),
    ("ENDGAME_BANK", "knob", [0], "turn", "bank all from round 1850"),
    ("TEAM_FIGHT", "knob", [1], "turn", "Lanchester skirmish rule"),
    ("COMBAT_SEARCH", "knob", [1], "turn", "skirmish portfolio search (fuel)"),
    ("CAMP_SAFE", "knob", [1, 3], "turn", "leave awake camps / don't wake unsurvivable"),
    ("ALTAR_LOCK", "knob", [0], "turn", "leave a lock on our altars"),
    ("TAVI_META", "knob", [0], "map", "Tavi field shop / build / hunt (draft list is compiled)"),
    ("RECALL_SAFE", "knob", [0, 1], "turn", "recall cancel / start rule"),
    ("FIGHT_V6", "knob", [1, 7], "turn", "focus / join / finish bits"),
    ("STUN_THREAT", "knob", [1], "turn", "stun kits on a 6-round horizon"),
    ("CAMP_FEASIBLE", "knob", [0, 25], "map", "infeasible-group discount, early ramp"),
    ("PASSIVE_ROLES", "knob", [1], "map", "Vesper / Prism passive roles"),
    ("SAFE_ALTAR", "knob", [0], "map", "avoid altars by the enemy spawn"),
    ("MONOLITH_LIVE", "knob", [0], "map", "monolith gate on live damage"),
    ("SHOP_FIRST", "knob", [1], "map", "recall to shop while carrying"),
    ("EN_ROUTE", "knob", [0, 2], "map", "small camps on the way"),
    ("TEAM_RESPAWN", "knob", [0], "turn", "respawn broadcast in the channel"),
    ("COMBAT_FSM", "knob", [0, 193], "turn", "fight hysteresis / help bits"),
    ("K2_PASSIVE_AWARE", "knob", [1], "map", "generic passive rules"),
    ("K2_FOCUS", "knob", [0], "turn", "team focus target"),
    ("K2_ENDGAME", "knob", [0], "turn", "endgame mode"),
    ("K2_LOS_RETREAT", "knob", [1], "turn", "retreat out of line of sight"),
    ("K2_STUN_LOCK", "knob", [1], "turn", "chain stuns"),
    ("K2_UPGRADE_FIRST", "knob", [10], "turn", "recall before hard fights"),
    ("K2_TEAM_UP", "knob", [0, 12], "turn", "join ally fights within N tiles"),
    ("K2_PUNISH_RECALL", "knob", [6], "turn", "attack lone recallers within N"),
    ("K2_UNSTICK", "knob", [1], "turn", "recall when oscillating"),
    ("K2_WALK_HOME", "knob", [0, 12], "turn", "walk instead of recall near spawn"),
    ("K2_SOLO_CAP", "knob", [3], "map", "don't solo big groups"),
    ("K2_EASE_FIRST", "knob", [1], "map", "easier groups first alone"),
    ("K2_WEAK_CAMP_FIRST", "knob", [1], "turn", "weakest monster first"),
    ("K2_EARLY_GOLD", "knob", [1], "map", "early gold bonus"),
    ("K2_TP_CRYSTAL", "knob", [1], "map", "teleport crystal"),
    ("K2_GOLD_ORDER", "knob", [1, 2], "turn", "who takes gold rewards"),
    ("K2_CLUSTER", "knob", [10], "map", "cluster value (per-turn compute: fuel)"),
    ("K2_KEEP_BUYING", "knob", [0], "map", "keep buying the last emblem"),
    ("K2_DRAFT_PLAN", "knob", [1, 4, 8, 16, 29], "map", "v3 plan readers (bits)"),
    ("draft", "draft", [1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13, 14], "map", "roster (draft_prefs profile)"),
    ("style", "style", [], "map", "data/styles.json bundles"),
]
DEAD = ["K2_SPACING", "K2_CAMP_DIVERT", "K2_PREDICT", "K2_WARDS", "DRAFT_BEST", "MONOLITH_MIN_POWER"]
N_MAPS, N_OPPS = 12, 8


def styles() -> dict[str, dict[str, int]]:
    s = json.loads((C.ROOT / "data" / "styles.json").read_text(encoding="utf-8"))
    return {k: v for k, v in s.items() if not k.startswith("_")}


def style_overrides(name: str) -> dict[int, int]:
    ks = C.knob_slots()
    return {ks[k]: v for k, v in styles()[name].items()}


def candidates() -> dict[str, tuple[str, int | str, bytes]]:
    """Candidate id -> (item, value, record)."""
    ks = C.knob_slots()
    out = {}
    for item, kind, values, _, _ in PROBES:
        if kind == "knob":
            for v in values:
                out[f"{item}={v}"] = (item, v, C.record(overrides={ks[item]: v}))
        elif kind == "draft":
            for v in values:
                out[f"draft={v}"] = (item, v, C.record(draft=v))
        else:
            for name in styles():
                if name != "base":
                    out[f"style={name}"] = (item, name, C.record(overrides=style_overrides(name)))
    return out


def setups(splits: dict) -> tuple[list[str], list[str]]:
    """12 train maps spread over group counts, 8 A opponents spread over strength (ladder
    mimics first)."""
    rng = random.Random(11)
    maps = splits["maps_train"]  # sorted by group count
    step = len(maps) / N_MAPS
    ms = [maps[int(i * step)] for i in range(N_MAPS)]
    a = splits["opps_A"]
    fa = [o for o in a if o in _faithful()]
    rest = [o for o in a if o not in fa]
    opps = rng.sample(fa, min(4, len(fa))) + rng.sample(rest, N_OPPS - min(4, len(fa)))
    return ms, opps


def _faithful() -> set[str]:
    from harness.search.split import faithful

    return faithful()


def run(hours: float | None) -> bool:
    snap = C.check_snapshot()
    splits = C.load_json(C.RUNS / "splits.json", None)
    if not splits:
        raise SystemExit("no splits: run baseline and splits first")
    store = C.Store("audit", snap["wasm"])
    ms, opps = setups(splits)
    cands = candidates()
    recs = {"base": C.record(), **{c: r for c, (_, _, r) in cands.items()}}
    games = [C.Game(c, m, o, s) for c in recs for m in ms for o in opps for s in "ab"]
    ok = C.play(games, recs, store, C.opponents(), C.deadline_from(hours))
    report(store, cands, ms)
    return ok


def _changed(r: dict, b: dict) -> bool:
    return any(r.get(f) != b.get(f) for f in ("res", "ours", "theirs", "deaths", "rounds", "fmean"))


def report(store: C.Store, cands: dict, ms: list[str]) -> None:
    base = {(r["map"], r["opp"], r["side"]): r for r in store.select("base")}
    rows = []
    for cid, (item, value, _) in cands.items():
        rs = [r for r in store.select(cid) if (r["map"], r["opp"], r["side"]) in base]
        if not rs:
            continue
        p = C.paired(store, cid, "base")
        changed = sum(_changed(r, base[(r["map"], r["opp"], r["side"])]) for r in rs) / len(rs)
        # Map heterogeneity: Cochran's Q of the per-map mean differences over its df.
        per_map, q = {}, 0.0
        for m in ms:
            d = [C.fitness(r) - C.fitness(base[(r["map"], r["opp"], r["side"])]) for r in rs if r["map"] == m]
            if len(d) > 1:
                mu = sum(d) / len(d)
                var = sum((x - mu) ** 2 for x in d) / (len(d) - 1)
                per_map[m] = (round(mu, 3), var / len(d))
        if per_map:
            w = {m: 1 / max(v, 1e-4) for m, (_, v) in per_map.items()}
            mean_w = sum(w[m] * mu for m, (mu, _) in per_map.items()) / sum(w.values())
            q = sum(w[m] * (mu - mean_w) ** 2 for m, (mu, _) in per_map.items())
        df = max(len(per_map) - 1, 1)
        best = max(per_map.items(), key=lambda kv: kv[1][0])[0] if per_map else None
        dfuel = [r.get("fmean", 0) - base[(r["map"], r["opp"], r["side"])].get("fmean", 0) for r in rs]
        oof = sum(max(r.get("oof", 0), 0) for r in rs) - sum(
            max(base[(r["map"], r["opp"], r["side"])].get("oof", 0), 0) for r in rs)
        hetero = q / df
        if changed < 0.05:
            verdict = "leave (no effect)"
        elif hetero > 2.0 or abs(p["fit"]) > 2 * p["fit_se"]:
            verdict = "plan"
        else:
            verdict = "leave"
        rows.append({
            "cand": cid, "item": item, "value": value, "n": p["n"], "changed": round(changed, 3),
            "dfit": p["fit"], "dfit_se": p["fit_se"], "dwin": p["win"], "dwin_se": p["win_se"],
            "hetero": round(hetero, 2), "best_map": best,
            "best_map_dfit": per_map[best][0] if best else None,
            "dfuel_mean": round(sum(dfuel) / len(dfuel)), "doof": oof, "verdict": verdict,
        })
    C.save_json(C.RUNS / "audit.json", {"maps": ms, "rows": rows, "dead": DEAD})
    write_md(rows)
    plan_items = sorted({r["item"] for r in rows if r["verdict"] == "plan"})
    print(f"{len(rows)} probes, {len(plan_items)} plan items: {', '.join(plan_items)}")


def write_md(rows: list[dict]) -> None:
    meta = {p[0]: p for p in PROBES}
    ks = C.knob_slots()
    lines = [
        "# Rule audit (harness/search/audit.py)",
        "",
        "Each plan-settable decision probed at two values on 12 train maps × 8 search (A) opponents × 2 sides,",
        "paired against the base settings (same snapshot wasm, byte-patched: games differ only through the setting).",
        "**changed** = share of games with any different outcome or fuel; **hetero** = Cochran's Q / df of the per-map",
        "fitness differences (> 2: the best value differs by map); **verdict** plan = changes games and is map-heterogeneous",
        "or clearly better/worse everywhere.",
        "",
        "| item | slot | scope (code) | value | changed | Δfit ± SE | Δwin ± SE | hetero | best map (Δfit) | Δfuel mean | ΔOOF | verdict | note |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(rows, key=lambda r: (r["item"], str(r["value"]))):
        m = meta.get(r["item"], ("", "", [], "", ""))
        slot = ks.get(r["item"], "draft" if r["item"] == "draft" else "style")
        lines.append(
            f"| {r['item']} | {slot} | {m[3]} | {r['value']} | {r['changed']:.0%} | {r['dfit']:+.3f} ± {r['dfit_se']:.3f} "
            f"| {r['dwin']:+.3f} ± {r['dwin_se']:.3f} | {r['hetero']} | {r['best_map']} ({r['best_map_dfit']:+.2f}) "
            f"| {r['dfuel_mean']:+d} | {r['doof']:+d} | {r['verdict']} | {m[4]} |"
        )
    lines += ["", "Dead (never read in play, slots free): " + ", ".join(DEAD), ""]
    (C.ROOT / "docs" / "rule-audit.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
