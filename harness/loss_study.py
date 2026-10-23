"""Why do we lose? Per-map loss causes from our ranked games (read-only, small on disk).

For each of our recent ranked games it fetches the match summary (`cambc matches summary`,
~0.7 MB), keeps a ~2 KB extract in runs/ranked_study/<match>.json and deletes the summary.
Then it prints, per map, what our losses have in common:

  deaths      ours per game: to camps (by camp kind, from the map's nearest camp group) and
              to champions; the enemy's deaths
  gems lost   carried gems destroyed by camps / taken by enemy champions
  stolen      camp groups we cleared whose reward the enemy captured (within STEAL_ROUNDS)
  economy     clears, earned gold, first emblem round (ours / theirs)
  score       submitted gems ours / theirs at rounds 500 / 1000 / end; tiebreak games and
              the gems we still carried when the game ended
  fuel        turns that ran out of fuel

    python harness/loss_study.py                          # last 400 ranked games, all maps
    python harness/loss_study.py --maps starvation gold-rush --limit 300
    python harness/loss_study.py --json out.json          # the per-map table as JSON
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from bctools import CAMBC  # noqa: E402

TEAM_ID = "1ec259ba-7596-4a78-9fd6-090c77f543bf"
CACHE = ROOT / "runs" / "ranked_study"
LOCAL_SUMMARIES = ROOT / "runs" / "ladder_live"
STEAL_ROUNDS = 15
CAMP_KINDS = ("wolf", "elk", "gem", "gold", "monolith", "golem")


def cambc_json(args: list[str], tries: int = 4) -> dict:
    for i in range(tries):
        r = subprocess.run([str(CAMBC), *args], capture_output=True, text=True, cwd=ROOT)
        if r.returncode == 0:
            return json.loads(r.stdout)
        time.sleep(5 * (i + 1))  # platform / DNS hiccups
    raise SystemExit(f"cambc {' '.join(args[:2])} failed: {r.stderr.strip()[:200]}")


def ranked_games(limit: int) -> list[dict]:
    games, cursor = [], None
    while len(games) < limit:
        args = ["matches", "list", "--team", TEAM_ID, "--kind", "ranked", "--limit", "100", "--json"]
        d = cambc_json(args + (["--cursor", cursor] if cursor else []))
        games += [m for m in d["matches"] if m["status"] == "finished"]
        cursor = d.get("nextCursor")
        if not cursor:
            break
    return games[:limit]


#: Damaging camps: engine CampType -> (attack range, strength). Gem / gold clusters deal no damage.
CAMP_HIT = {0: (3, 20), 1: (4, 60), 4: (5, 200), 5: (4, 120)}


def _camp_groups(map_name: str) -> list[tuple[int, int, str, int, int]] | None:
    """(x, y, kind, range, strength) of every damaging camp on a local map (gpusim map cache),
    for naming the camp that killed us. The summary's camp ids don't index the map's camps."""
    try:
        from gpusim import maps

        md = maps.load(map_name.replace("_", "-"))
    except Exception:  # noqa: BLE001 - no local copy / dump: deaths stay "camp"
        return None
    return [(c.x, c.y, CAMP_KINDS[c.kind], *CAMP_HIT[c.kind]) for g in md.groups for c in g.camps if c.kind in CAMP_HIT]


def _timeline_at(runs: list[dict], rnd: int) -> int:
    at = 0
    for r in runs:
        at += int(r["duration"])
        if rnd < at:
            return int(float(r["value"]))
    return int(float(runs[-1]["value"])) if runs else 0


def extract(m: dict, s: dict) -> dict:
    """The compact per-game record (our side vs theirs)."""
    us = next(t for t in m["teams"] if t["teamId"] == TEAM_ID)
    them = next(t for t in m["teams"] if t["teamId"] != TEAM_ID)
    side, other = f"team{us['side'].upper()}", f"team{them['side'].upper()}"
    ev = defaultdict(list)
    for e in s["events"]:
        for x in e["entries"]:
            ev[(e["kind"], e["owner"]["team"])].append((x, e["owner"]))
    camps = _camp_groups(m["map"])

    def camp_kind(x: dict) -> str:
        """The strongest damaging camp in range of where we died (88% of camp deaths have one)."""
        hit = [c for c in camps or [] if max(abs(c[0] - x["x"]), abs(c[1] - x["y"])) <= c[3]]
        return max(hit, key=lambda c: c[4])[2] if hit else "camp?"

    deaths = Counter()
    for x, _ in ev[("death", side)]:
        deaths["champion" if x["context"] == "champion" else camp_kind(x)] += 1
    lost = Counter()
    for x, _ in ev[("gemLoss", side)]:
        lost["champion" if x["context"] == "champion" else "camp"] += int(x["value"])
    # Stolen: a group we cleared whose gems the enemy collected soon after.
    cleared = [(x["target"].get("id"), x["timestamp"]["round"]) for x, _ in ev[("campClear", side)]]
    taken = [(x["target"].get("id"), x["timestamp"]["round"], int(x["value"])) for x, _ in ev[("gemCollection", other)]
             if x["context"] == "campGroup"]
    stolen = [(g, r2, v) for g, r2, v in taken if any(g == g1 and 0 <= r2 - r1 <= STEAL_ROUNDS for g1, r1 in cleared)]

    def first_emblem(team: str) -> int | None:
        rs = [x["timestamp"]["round"] for x, _ in ev[("itemCombination", team)] if "Emblem" in x["target"].get("name", "")]
        return min(rs) if rs else None

    tl = {t["owner"]: t["runs"] for t in s["teamTimelines"] if t["metric"] == "submittedGems"}
    carried = {c["team"]: 0 for c in s["champions"]}
    gold = Counter()
    for c in s["champions"]:
        carried[c["team"]] += int(float(c["metrics"]["gems"]))
        gold[c["team"]] += int(float(c["metrics"]["earnedGold"]))
    return {
        "match": m["matchId"],
        "time": m["createdAt"],
        "map": m["map"][2:],
        "opp": them["name"],
        "result": us["result"],
        "end": m["endReason"],
        "rounds": s["match"]["roundsCompleted"],
        "draft": us["draftPicks"],
        "opp_draft": them["draftPicks"],
        "deaths": dict(deaths),
        "enemy_deaths": len(ev[("death", other)]),
        "gems_lost": dict(lost),
        "stolen": len(stolen),
        "stolen_gems": sum(v for *_, v in stolen),
        "clears": len(cleared),
        "enemy_clears": len(ev[("campClear", other)]),
        "gold": gold[side],
        "enemy_gold": gold[other],
        "emblem": first_emblem(side),
        "enemy_emblem": first_emblem(other),
        "submitted": {r: _timeline_at(tl.get(side, []), r) for r in (500, 1000, 2000)},
        "enemy_submitted": {r: _timeline_at(tl.get(other, []), r) for r in (500, 1000, 2000)},
        "carried": carried.get(side, 0),
        "enemy_carried": carried.get(other, 0),
        "fuel_out": len(ev[("fuelExhausted", side)]),
    }


def record(m: dict, offline: bool = False) -> dict | None:
    out = CACHE / f"{m['matchId']}.json"
    if out.exists():
        return json.loads(out.read_text(encoding="utf-8"))
    saved = LOCAL_SUMMARIES / f"{m['matchId']}.json"  # the daemon keeps summaries of its loss queue
    if saved.exists():
        rec = extract(m, json.loads(saved.read_text(encoding="utf-8")))
    elif offline:
        return None
    else:
        rec = None
    if rec is not None:
        CACHE.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rec), encoding="utf-8")
        return rec
    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp) / "s.json"
        subprocess.run([str(CAMBC), "matches", "summary", m["matchId"], "--json", "-o", str(raw)],
                       capture_output=True, cwd=ROOT)
        if not raw.exists():
            return None
        rec = extract(m, json.loads(raw.read_text(encoding="utf-8")))
    CACHE.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec), encoding="utf-8")
    return rec


def table(recs: list[dict]) -> dict[str, dict]:
    by = defaultdict(list)
    for r in recs:
        by[r["map"]].append(r)
    rows = {}
    for mp, rs in by.items():
        n, losses = len(rs), [r for r in rs if r["result"] == "loss"]
        nl = max(len(losses), 1)
        d = Counter()
        for r in losses:
            d.update(r["deaths"])
        avg = lambda f: round(sum(f(r) for r in losses) / nl, 1)  # noqa: E731
        rows[mp] = {
            "games": n,
            "win": round(sum(r["result"] == "win" for r in rs) / n, 2),
            "losses": len(losses),
            "deaths_by": {k: round(v / nl, 1) for k, v in d.most_common()},
            "enemy_deaths": avg(lambda r: r["enemy_deaths"]),
            "gems_lost_camp": avg(lambda r: r["gems_lost"].get("camp", 0)),
            "gems_lost_champ": avg(lambda r: r["gems_lost"].get("champion", 0)),
            "stolen": avg(lambda r: r["stolen"]),
            "stolen_gems": avg(lambda r: r["stolen_gems"]),
            "clears": f"{avg(lambda r: r['clears'])}/{avg(lambda r: r['enemy_clears'])}",
            "gold": f"{avg(lambda r: r['gold'])}/{avg(lambda r: r['enemy_gold'])}",
            "emblem": f"{avg(lambda r: r['emblem'] or 2000)}/{avg(lambda r: r['enemy_emblem'] or 2000)}",
            "at500": f"{avg(lambda r: _at(r, 'submitted', 500))}/{avg(lambda r: _at(r, 'enemy_submitted', 500))}",
            "at1000": f"{avg(lambda r: _at(r, 'submitted', 1000))}/{avg(lambda r: _at(r, 'enemy_submitted', 1000))}",
            "final": f"{avg(lambda r: _at(r, 'submitted', 2000))}/{avg(lambda r: _at(r, 'enemy_submitted', 2000))}",
            "tiebreak": sum(r["end"].startswith("tiebreak") for r in losses),
            "carried_end": avg(lambda r: r["carried"] if r["end"].startswith("tiebreak") else 0),
            "fuel_out": avg(lambda r: r["fuel_out"]),
        }
    return rows


def _at(r: dict, key: str, rnd: int) -> int:
    """Submitted gems at a round (keys are strings once a record went through JSON)."""
    v = r[key]
    return v.get(rnd, v.get(str(rnd), 0))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--limit", type=int, default=400, help="most recent ranked games")
    p.add_argument("--maps", nargs="*", help="only these maps (e.g. starvation gold-rush)")
    p.add_argument("--losses-only", action="store_true", help="fetch summaries of lost games only")
    p.add_argument("--json", type=Path, help="also write the per-map table here")
    p.add_argument("--matches", type=Path, help="offline: a saved `cambc matches list --json` (or its list)")
    a = p.parse_args(argv)
    want = {m.replace("_", "-") for m in a.maps or []}
    if a.matches:
        listed = json.loads(a.matches.read_text(encoding="utf-8"))
        listed = listed["matches"] if isinstance(listed, dict) else listed
        listed = [m for m in listed if m["status"] == "finished" and m["kind"] == "ranked"][: a.limit]
    else:
        listed = ranked_games(a.limit)
    games = [m for m in listed if not want or m["map"][2:].replace("_", "-") in want]
    if a.losses_only:
        games = [m for m in games if next(t for t in m["teams"] if t["teamId"] == TEAM_ID)["result"] == "loss"]
    recs = []
    for i, m in enumerate(games, 1):
        r = record(m, offline=a.matches is not None)
        if r:
            recs.append(r)
        if i % 25 == 0:
            print(f"  {i}/{len(games)} summaries", file=sys.stderr)
    rows = table(recs)
    cols = ["games", "win", "losses", "deaths_by", "enemy_deaths", "gems_lost_camp", "gems_lost_champ",
            "stolen", "stolen_gems", "clears", "gold", "emblem", "at500", "at1000", "final", "tiebreak", "carried_end", "fuel_out"]
    for mp, row in sorted(rows.items(), key=lambda kv: kv[1]["win"]):
        print(f"{mp:22s} " + " ".join(f"{c}={row[c]}" for c in cols))
    if a.json:
        a.json.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
