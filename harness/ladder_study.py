"""What do strong ladder teams have in common? Tempo, comps, builds and roles per team.

Reads match summaries (`cambc matches summary --json`) plus the match lists that name
the teams, and prints one row per team: win rate, economic tempo (first emblem, clears
by round 150, clears per champion per 100 rounds, gold per champion-round, camp groups
in play), bag size, shop trips, kills/deaths, comps and build families; then per
champion kind (who farms, who fights). Read-only; nothing touches the platform.

    python harness/ladder_study.py runs/ladder_v8 runs/ladder_top runs/ranked_v5
"""

from __future__ import annotations

import json
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

NAMES = {0: "Ryth", 1: "Vesper", 2: "Vendetta", 3: "C&I", 4: "Stellar", 5: "Prism", 6: "Marcella", 7: "Willow",
         8: "Ren", 9: "Varg", 10: "Tavi"}
F = lambda x: int(float(x))  # noqa: E731


def load(dirs: list[Path]) -> list[tuple[dict, dict]]:
    """(list entry, summary) pairs; the list entry names the teams and sides."""
    meta: dict[str, dict] = {}
    for d in dirs:
        for f in list(d.glob("*.json")):
            try:
                j = json.loads(f.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            ms = j if isinstance(j, list) else j.get("matches") if isinstance(j, dict) else None
            for m in ms or []:
                if isinstance(m, dict) and "matchId" in m:
                    meta.setdefault(m["matchId"], m)
    out, seen = [], set()
    for d in dirs:
        for f in d.glob("*-*-*-*-*.json"):
            mid = f.stem.replace("match-", "")
            if mid in seen or mid not in meta:
                continue
            seen.add(mid)
            out.append((meta[mid], json.loads(f.read_text(encoding="utf-8"))))
    return out


def team_stats(d: dict, team: str) -> dict:
    rounds = max(d["match"]["roundsCompleted"], 1)
    champs = [c for c in d["champions"] if c["team"] == team]
    n = max(len(champs), 1)
    ev = defaultdict(list)
    for e in d["events"]:
        if (e.get("owner") or {}).get("team") != team:
            continue
        for en in e["entries"]:
            ev[e["kind"]].append((en["timestamp"]["round"], e["owner"].get("id"), en.get("value"), en.get("x"), en.get("y"),
                                  (en.get("target") or {}) if isinstance(en.get("target"), dict) else {}))
    emblems = sorted(ev["itemCombination"], key=lambda x: x[0])
    clears = sorted(r for r, *_ in ev["campClear"])
    subs = [F(v) for _, _, v, *_ in ev["gemSubmission"] if v is not None]
    windows = defaultdict(set)
    for r, _, _, x, y, _ in ev["campClear"]:
        windows[r // 50].add((x // 4, y // 4))
    fams = [t.get("name", "").split(" Emblem")[0] for *_, t in emblems[: 3 * n]]
    stats = Counter()
    for c in champs:
        for it in c["items"]:
            stats[it["item"].split(" of ")[-1]] += F(it["quantity"])
    return {
        "first_emblem": emblems[0][0] if emblems else rounds,
        "clears@150": sum(1 for r in clears if r <= 150),
        "clears/100r/ch": 100 * len(clears) / rounds / n,
        "gold/ch/r": sum(F(c["metrics"]["earnedGold"]) for c in champs) / n / rounds,
        "mon/ch/r": sum(c["metrics"]["monsterDamageDone"] for c in champs) / n / rounds,
        "groups/50r": st.mean(len(v) for v in windows.values()) if windows else 0.0,
        "bag": st.mean(subs) if subs else 0.0,
        "shops/ch/100r": 100 * len({(cid, r) for r, cid, *_ in ev["buyItem"]}) / n / rounds,
        "kills": sum(F(c["metrics"]["kills"]) for c in champs),
        "deaths": sum(F(c["metrics"]["deaths"]) for c in champs),
        "rounds": rounds,
        "early_families": Counter(fams),
        "stat_mix": stats,
        "champs": champs,
    }


KEYS = ["first_emblem", "clears@150", "clears/100r/ch", "gold/ch/r", "mon/ch/r", "groups/50r", "bag",
        "shops/ch/100r", "kills", "deaths", "rounds"]


def main(argv: list[str]) -> int:
    dirs = [Path(a) for a in argv] or [Path("runs/ladder_v8"), Path("runs/ladder_top")]
    games = load(dirs)
    rows: dict[str, dict] = defaultdict(lambda: {"res": [], "vals": defaultdict(list), "comps": Counter(),
                                                 "fam": Counter(), "stat": Counter(), "kind": defaultdict(list)})
    for m, d in games:
        for t in m["teams"]:
            name = (t["name"] or "?").replace("​", "") or "blank-name"
            team = "team" + t["side"].upper()
            s = team_stats(d, team)
            r = rows[name]
            r["res"].append(t["result"] == "win")
            for k in KEYS:
                r["vals"][k].append(s[k])
            r["comps"][" ".join(NAMES[p] for p in t.get("draftPicks") or [])] += 1
            r["fam"] += s["early_families"]
            r["stat"] += s["stat_mix"]
            for c in s["champs"]:
                mt = c["metrics"]
                r["kind"][c["name"]].append((mt["monsterDamageDone"] / s["rounds"], F(mt["kills"]), F(mt["deaths"]),
                                             F(mt["submittedGems"]), F(mt["earnedGold"]) / s["rounds"]))
    order = sorted(rows.items(), key=lambda kv: (-len(kv[1]["res"]) >= -3, -sum(kv[1]["res"]) / len(kv[1]["res"])))
    print(f"{'team':22s} {'n':>3s} {'win':>5s} " + " ".join(f"{k[:12]:>12s}" for k in KEYS))
    for name, r in order:
        if len(r["res"]) < 3:
            continue
        print(f"{name[:22]:22s} {len(r['res']):3d} {sum(r['res'])/len(r['res']):5.2f} "
              + " ".join(f"{st.mean(r['vals'][k]):12.2f}" for k in KEYS))
    print("\nComps, early emblem families (first 3 per champion), stat mix:")
    for name, r in order:
        if len(r["res"]) < 3:
            continue
        tot = sum(r["stat"].values()) or 1
        print(f"  {name[:22]:22s} comps {dict(r['comps'].most_common(3))}")
        print(f"  {'':22s} early {dict(r['fam'].most_common(4))}  stats "
              f"{ {k: round(v / tot, 2) for k, v in r['stat'].most_common()} }")
    print("\nPer champion kind (top teams vs us): monster dmg/round, kills, deaths, gems tributed, gold/round")
    for name, r in order:
        if len(r["res"]) < 3:
            continue
        parts = []
        for kind, xs in sorted(r["kind"].items(), key=lambda kv: -len(kv[1])):
            a = [st.mean(x[i] for x in xs) for i in range(5)]
            parts.append(f"{kind[:7]} {a[0]:.0f}/{a[1]:.1f}k/{a[2]:.1f}d/{a[3]:.0f}g/{a[4]:.2f}")
        print(f"  {name[:22]:22s} " + " | ".join(parts[:6]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
