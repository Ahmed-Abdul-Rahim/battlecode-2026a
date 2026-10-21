"""Clone a ladder team's macro policy from match summaries into a policy-pack spec.

From every summary of a team's games (any opponent) we extract:
  draft    most common pick order per team size (3/4/5), DraftChoice ints
  build    per champion kind: emblem order (family, stat) by majority vote per position
  opening  per map and spawn side: each champion kind's first 4 camp groups cleared
  cadence  mean tributed bag, shop trips per champion per 100 rounds, first emblem

Camp clears are mapped to our group indices through each map's group centres (taken
from our debug map dumps). The output pack spec (harness/pack.py write) holds, per map:
draft, build, opening and a theta whose tribute/shop multipliers follow the cadence.
Their fight micro can't be cloned (no per-turn data); our engine plays it.

    python harness/mimic.py --team "Nyx" --out runs/mimic/nyx.json
    python harness/mimic.py --all-top        # every team in TOP
Read-only on the platform (uses already downloaded summaries). harness/league.py calls
extract() and pack_spec() for the live top 10 and writes data/mimic/.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "harness"))

from bctools import replay  # noqa: E402

# Engine 1.0.0 games only (ladder_v8/v10 hold old-engine games: other stats, maps and meta).
SUMMARY_DIRS = ["runs/ladder_top", "runs/ladder_live"]
MIN_ENGINE_MAJOR = 1
MAP_DUMPS = "runs/speedrun/*/main-v7/*__main-v7-vs-idle.repl26a"
#: Committed cache of map_geometry(): the debug dumps it is built from are not in git.
GEOMETRY = ROOT / "data" / "map_geometry.json"
TOP = ["Intercontinental Fish", "Mite", "Nyx", "Univardisty of Shingtonwash", "Sneaky Bush Campers"]
KIND = {
    "Ryth": 0,
    "Vesper Vale": 1,
    "Vendetta": 2,
    "Cassi": 3,
    "Issac": 4,
    "Stellar": 5,
    "Prism": 6,
    "Marcella": 7,
    "Willow": 8,
    "Ren": 9,
    "Varg": 10,
    "Tavi": 11,
    "Heart of Ruin": 12,
}
FAMILY = {"Monster": 0, "Power": 1, "Durability": 2, "Treasure": 3, "Jungle": 4, "Home": 5, "Bruiser": 6}
STAT = {"Strength": 0, "Focus": 1, "Health": 2}
OPENING_LEN = 4
BUILD_LEN = 12
F = lambda x: int(float(x))  # noqa: E731


def slug(team: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", team.lower()).strip("-")


def map_name(platform: str) -> str:
    """Platform map name -> our map stem: '00contest' -> 'contest', '01big_world' -> 'big-world'."""
    return re.sub(r"^\d+", "", platform).replace("_", "-").lower()


def geometry_from_dump(m: dict) -> dict:
    """Group centres and both spawn tiles from our bot's debug `M ` dump (written as team A)."""
    from bctools.render import _bits

    w, h = m["w"], m["h"]

    def cells(key: str) -> list[tuple[int, int]]:
        return [(i % w, i // w) for i, v in enumerate(_bits(m[key], w, h)) if v]

    return {
        "centres": [g["c"] for g in m["groups"]],
        "spawn_a": cells("ally_spawn")[0],
        "spawn_b": cells("enemy_spawn")[0],
    }


def map_geometry() -> dict[str, dict]:
    """Per map: group centres, team-A and team-B spawn tiles (from our debug dumps; cached in
    data/map_geometry.json, so a fresh clone needs no dumps; harness/league.py adds new maps)."""
    if GEOMETRY.exists():
        return json.loads(GEOMETRY.read_text(encoding="utf-8"))
    geo = {}
    for f in glob.glob(str(ROOT / MAP_DUMPS)):
        name = Path(f).name.split("__")[0]
        if name in geo or "main-v7-vs-idle" not in Path(f).name:
            continue
        m = replay.load(f).team("a").map
        if m:
            geo[name] = geometry_from_dump(m)
    return geo


def load_games(team: str) -> list[tuple[dict, dict]]:
    meta, out, seen = {}, [], set()
    for d in SUMMARY_DIRS:
        for f in glob.glob(str(ROOT / d / "*.json")):
            try:
                j = json.loads(Path(f).read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            for m in (j.get("matches") if isinstance(j, dict) else None) or []:
                meta.setdefault(m["matchId"], m)
    for d in SUMMARY_DIRS:
        for f in glob.glob(str(ROOT / d / "*-*-*-*-*.json")):
            mid = Path(f).stem.replace("match-", "")
            m = meta.get(mid)
            if not m or mid in seen or m.get("status") != "finished":
                continue
            if not any(t["name"] == team for t in m["teams"]):
                continue
            seen.add(mid)
            summary = json.loads(Path(f).read_text(encoding="utf-8"))
            if engine_major(summary) < MIN_ENGINE_MAJOR:
                continue
            out.append((m, summary))
    return out


def engine_major(summary: dict) -> int:
    """Engine major version a summary was played on (0 if unknown: treated as old)."""
    v = str((summary.get("match") or {}).get("engineVersion") or "0")
    try:
        return int(v.split(".")[0])
    except ValueError:
        return 0


def nearest_group(centres: list[list[int]], x: int, y: int) -> int:
    return min(range(len(centres)), key=lambda i: max(abs(centres[i][0] - x), abs(centres[i][1] - y)))


def majority(seqs: list[list], length: int) -> list:
    out = []
    for i in range(length):
        c = Counter(tuple(s[i]) if isinstance(s[i], list) else s[i] for s in seqs if len(s) > i)
        if not c:
            break
        v = c.most_common(1)[0][0]
        out.append(list(v) if isinstance(v, tuple) else v)
    return out


def extract(team: str, geo: dict) -> dict:
    games = load_games(team)
    drafts: dict[int, Counter] = defaultdict(Counter)
    builds: dict[int, list] = defaultdict(list)
    openings: dict[tuple[str, str], dict[int, list]] = defaultdict(lambda: defaultdict(list))
    bags, shops, firsts, wins = [], [], [], []
    for m, d in games:
        t = next(x for x in m["teams"] if x["name"] == team)
        side = t["side"]
        tm = "team" + side.upper()
        wins.append(t["result"] == "win")
        picks = t.get("draftPicks") or []
        drafts[len(picks)][tuple(picks)] += 1
        rounds = max(d["match"]["roundsCompleted"], 1)
        ev = defaultdict(list)
        for e in d["events"]:
            o = e.get("owner") or {}
            if o.get("team") != tm:
                continue
            for en in e["entries"]:
                ev[(e["kind"], o.get("id"), o.get("name"))].append(en)
        champs = [c for c in d["champions"] if c["team"] == tm]
        for c in champs:
            kind = KIND.get(c["name"])
            if kind is None:
                continue
            emb = sorted(ev[("itemCombination", c["id"], c["name"])], key=lambda en: en["timestamp"]["round"])
            seq = []
            for en in emb:
                name = (en.get("target") or {}).get("name", "")
                mm = re.match(r"(\w+) Emblem of (\w+)", name)
                if mm and mm.group(1) in FAMILY and mm.group(2) in STAT:
                    seq.append([FAMILY[mm.group(1)], STAT[mm.group(2)]])
            if seq:
                builds[kind].append(seq)
                firsts.append(emb[0]["timestamp"]["round"])
            mp = map_name(m["map"])
            if mp in geo:
                clears = sorted(
                    ev[("campClear", c["id"], c["name"])], key=lambda en: en["timestamp"]["round"]
                )
                route = []
                for en in clears:
                    g = nearest_group(geo[mp]["centres"], en["x"], en["y"])
                    if g not in route:
                        route.append(g)
                    if len(route) >= OPENING_LEN:
                        break
                if route:
                    openings[(mp, side)][kind].append(route)
            sess = {en["timestamp"]["round"] for en in ev[("buyItem", c["id"], c["name"])]}
            shops.append(100 * len(sess) / rounds)
        bags += [
            F(en["value"])
            for c in champs
            for en in ev[("gemSubmission", c["id"], c["name"])]
            if en.get("value")
        ]
    draft = {n: list(c.most_common(1)[0][0]) for n, c in drafts.items() if n in (3, 4, 5)}
    build = {k: majority(v, BUILD_LEN) for k, v in builds.items()}
    opening = {}
    for (mp, side), kinds in openings.items():
        spawn = geo[mp]["spawn_a" if side == "a" else "spawn_b"]
        opening.setdefault(mp, []).append(
            {"spawn": list(spawn), "kinds": {k: majority(v, OPENING_LEN) for k, v in kinds.items()}}
        )
    return {
        "team": team,
        "games": len(games),
        "win_rate": round(sum(wins) / max(len(wins), 1), 3),
        "draft": draft,
        "build": build,
        "opening": opening,
        "cadence": {
            "bag": round(st.mean(bags), 1) if bags else None,
            "shops_per_100r": round(st.mean(shops), 2) if shops else None,
            "first_emblem": round(st.median(firsts)) if firsts else None,
        },
    }


def theta_from_cadence(base: list[float], cad: dict) -> list[float]:
    """Base θ (pack layout: W multipliers, TARGET, FIGHT, HOTSPOT_W) with tribute/shop set
    to the team's cadence. W indices: RECALL_GOLD 14, TRIBUTE_MIN 15, TRIBUTE_MAX 16."""
    th = list(base)
    if cad.get("bag"):
        th[15] = min(max(cad["bag"] / 60.0, 0.5), 2.5)
        th[16] = min(max(cad["bag"] * 1.4 / 120.0, 0.5), 2.0)
    if cad.get("shops_per_100r"):
        th[14] = min(max(0.6 / cad["shops_per_100r"], 0.4), 2.0)
    return th


def base_theta() -> list[float]:
    text = (ROOT / "bots" / "main" / "src" / "learned.rs").read_text(encoding="utf-8")

    def arr(name: str, n: int, dflt: float) -> list[float]:
        mm = re.search(rf"pub const {name}: \[f32; \w+\] = \[(.*?)\];", text, re.S)
        if not mm or ";" in mm.group(1):
            return [dflt] * n
        v = [float(x) for x in mm.group(1).split(",") if x.strip()]
        return v if len(v) == n else [dflt] * n

    return arr("W", 18, 1.0) + arr("TARGET", 10, 0.0) + arr("FIGHT", 9, 0.0) + [0.0]


def pack_spec(clone: dict, maps: list[str]) -> dict:
    theta = theta_from_cadence(base_theta(), clone["cadence"])
    spec = {"maps": {}}
    for mp in maps:
        e = {"theta": theta, "draft": clone["draft"], "build": clone["build"]}
        if mp in clone["opening"]:
            e["opening"] = clone["opening"][mp]
        spec["maps"][mp] = e
    return spec


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--team", action="append", default=[])
    p.add_argument("--all-top", action="store_true")
    p.add_argument("--outdir", type=Path, default=ROOT / "runs" / "mimic")
    a = p.parse_args(argv)
    teams = a.team + (TOP if a.all_top else [])
    geo = map_geometry()
    a.outdir.mkdir(parents=True, exist_ok=True)
    for team in teams:
        clone = extract(team, geo)
        (a.outdir / f"{slug(team)}.json").write_text(json.dumps(clone, indent=1), encoding="utf-8")
        spec = pack_spec(clone, sorted(geo))
        (a.outdir / f"{slug(team)}.spec.json").write_text(json.dumps(spec), encoding="utf-8")
        print(
            f"{team}: {clone['games']} games, win {clone['win_rate']}, draft {clone['draft']}, "
            f"cadence {clone['cadence']}, builds for {len(clone['build'])} kinds, openings on {len(clone['opening'])} maps"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
