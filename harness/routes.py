"""Precomputed camp clusters and farming routes per map (pack section TAG_ROUTES).

Camp groups whose walking distance is short are merged into clusters; per spawn side the best
few routes (ordered cluster visits) are kept. Everything is computed here, offline, from the
bot's terrain dump, so the bot only looks things up (knob2 ROUTES, bots/main/src/bot.rs).

    python harness/routes.py dump      # one short debug game per map -> data/map_terrain.json
    python harness/routes.py build     # -> data/routes.json (read by harness/pack.py)

Walking: 8 directions; walls, holes and altars block (both spawns are treated as open for the
cluster distances, and the enemy's spawn as blocked for the spawn-to-cluster distances).
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import zipfile
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TERRAIN = ROOT / "data" / "map_terrain.json"
ROUTES = ROOT / "data" / "routes.json"

#: Groups join one cluster while every pair in it is within this many walking tiles.
CLUSTER_DIAMETER = 12
#: Clusters per route and routes kept per spawn side.
ROUTE_LEN, ROUTES_PER_SIDE = 4, 8
#: Rounds spent per camp tile of a cluster when ranking routes offline (clear + capture).
DWELL_PER_TILE = 4
UNREACHABLE = 255


def bits(hex_str: str, w: int, h: int) -> list[bool]:
    data = bytes.fromhex(hex_str)
    return [bool(data[i >> 3] >> (i & 7) & 1) for i in range(w * h)]


def bfs(w: int, h: int, blocked: list[bool], sources: list[tuple[int, int]]) -> list[int]:
    """Walking distance (8 directions) from the nearest source to every tile; -1 = unreachable."""
    dist = [-1] * (w * h)
    q: deque[tuple[int, int]] = deque()
    for x, y in sources:
        if 0 <= x < w and 0 <= y < h and dist[y * w + x] < 0:
            dist[y * w + x] = 0
            q.append((x, y))
    while q:
        x, y = q.popleft()
        d = dist[y * w + x] + 1
        for dx, dy in itertools.product((-1, 0, 1), repeat=2):
            nx, ny = x + dx, y + dy
            if (dx or dy) and 0 <= nx < w and 0 <= ny < h:
                i = ny * w + nx
                if dist[i] < 0 and not blocked[i]:
                    dist[i] = d
                    q.append((nx, ny))
    return dist


def nearest(dist: list[int], w: int, tiles: list[tuple[int, int]]) -> int:
    ds = [dist[y * w + x] for x, y in tiles if dist[y * w + x] >= 0]
    return min(ds) if ds else UNREACHABLE


def clusters_of(gd: list[list[int]]) -> list[int]:
    """Complete-linkage agglomeration: merge the closest pair of clusters while the merged
    cluster's largest group-to-group distance stays <= CLUSTER_DIAMETER. Deterministic."""
    n = len(gd)
    members = [[i] for i in range(n)]

    def diameter(a: list[int], b: list[int]) -> int:
        return max(gd[i][j] for i in a for j in b)

    while True:
        best = None
        for a, b in itertools.combinations(range(len(members)), 2):
            d = diameter(members[a], members[b])
            if d <= CLUSTER_DIAMETER and (best is None or d < best[0]):
                best = (d, a, b)
        if best is None:
            break
        _, a, b = best
        members[a] = sorted(members[a] + members[b])
        del members[b]
    members.sort(key=lambda m: m[0])
    of = [0] * n
    for c, m in enumerate(members):
        for i in m:
            of[i] = c
    return of


def build_map(m: dict) -> dict:
    w, h = m["w"], m["h"]
    walls, holes, altars = (bits(m[k], w, h) for k in ("walls", "holes", "altars"))
    ally, enemy = bits(m["ally_spawn"], w, h), bits(m["enemy_spawn"], w, h)
    blocked = [a or b or c for a, b, c in zip(walls, holes, altars, strict=True)]
    groups = m["groups"]
    tiles = [[tuple(t) for t in g["tiles"]] or [tuple(g["c"])] for g in groups]
    fields = [bfs(w, h, blocked, t) for t in tiles]
    n = len(groups)
    gd = [[0 if i == j else nearest(fields[i], w, tiles[j]) for j in range(n)] for i in range(n)]
    gd = [[min(gd[i][j], gd[j][i], UNREACHABLE) for j in range(n)] for i in range(n)]
    of = clusters_of(gd)
    k = max(of) + 1 if of else 0
    members = [[i for i in range(n) if of[i] == c] for c in range(k)]
    cd = [[min((gd[i][j] for i in members[a] for j in members[b]), default=UNREACHABLE) for b in range(k)] for a in range(k)]
    value = [sum(groups[i]["gems"] + groups[i]["gold"] / 3 for i in members[c]) for c in range(k)]
    dwell = [DWELL_PER_TILE * sum(len(tiles[i]) for i in members[c]) for c in range(k)]
    centroid = [
        (
            round(sum(groups[i]["c"][0] for i in members[c]) / len(members[c])),
            round(sum(groups[i]["c"][1] for i in members[c]) / len(members[c])),
        )
        for c in range(k)
    ]
    sides = []
    for own, other in ((ally, enemy), (enemy, ally)):
        spawn = [(i % w, i // w) for i, v in enumerate(own) if v]
        side_blocked = [a or b for a, b in zip(blocked, other, strict=True)]
        home_field = bfs(w, h, side_blocked, spawn)
        home = [min((nearest(home_field, w, tiles[i]) for i in members[c]), default=UNREACHABLE) for c in range(k)]
        sides.append({"spawn": list(spawn[0]), "home": [min(d, UNREACHABLE) for d in home], "routes": best_routes(k, cd, home, value, dwell)})
    return {
        "cluster_of": of,
        "centroid": [list(c) for c in centroid],
        "value": [round(v) for v in value],
        "dist": [[min(d, UNREACHABLE) for d in row] for row in cd],
        "sides": sides,
    }


def route_score(route: tuple[int, ...], start: list[int], cd: list[list[int]], value: list[float], dwell: list[int]) -> float:
    t = start[route[0]] + dwell[route[0]]
    for a, b in itertools.pairwise(route):
        t += cd[a][b] + dwell[b]
    return sum(value[c] for c in route) / max(t, 1)


def best_routes(k: int, cd: list[list[int]], home: list[int], value: list[float], dwell: list[int]) -> list[list[int]]:
    """The top routes from spawn by value per round, kept diverse: a route is skipped when it
    shares its first two clusters with one already kept."""
    reach = [c for c in range(k) if home[c] < UNREACHABLE and value[c] > 0]
    length = min(ROUTE_LEN, len(reach))
    if length == 0:
        return []
    scored = []
    for r in itertools.permutations(reach, length):
        if any(cd[a][b] >= UNREACHABLE for a, b in itertools.pairwise(r)):
            continue
        scored.append((route_score(r, home, cd, value, dwell), r))
    scored.sort(key=lambda s: (-s[0], s[1]))
    kept: list[tuple[int, ...]] = []
    for _, r in scored:
        if all(r[:2] != q[:2] for q in kept):
            kept.append(r)
        if len(kept) == ROUTES_PER_SIDE:
            break
    return [list(r) for r in kept]


def routes_bytes(r: dict) -> bytes:
    """TAG_ROUTES layout (bots/main/src/pack.rs Routes):
    G u8, K u8, cluster_of[G], centroid[K] (x, y), value[K] (/4, u8), dist[K*K] u8,
    sides u8, per side: spawn (x, y), home[K], R u8, R x ROUTE_LEN cluster ids (255 = none)."""
    of, k = r["cluster_of"], len(r["value"])
    out = bytearray([len(of), k, *of])
    for x, y in r["centroid"]:
        out += bytes([x & 0xFF, y & 0xFF])
    out += bytes(min(round(v / 4), 255) for v in r["value"])
    out += bytes(d for row in r["dist"] for d in row)
    out.append(len(r["sides"]))
    for s in r["sides"]:
        out += bytes([s["spawn"][0] & 0xFF, s["spawn"][1] & 0xFF, *s["home"], len(s["routes"])])
        for route in s["routes"]:
            out += bytes(route + [UNREACHABLE] * (ROUTE_LEN - len(route)))
    return bytes(out)


def cmd_dump(_: argparse.Namespace) -> int:
    sys.path.insert(0, str(ROOT / "tools"))
    from bctools import runner

    terrain = json.loads(TERRAIN.read_text(encoding="utf-8")) if TERRAIN.exists() else {}
    out = ROOT / "runs" / "routes-dump"
    todo = [p for p in runner.list_maps() if p.stem.lstrip("0123456789") not in terrain]
    if todo:
        res = runner.run_pairs([("main", "baseline")], todo, out, rounds=12)
        for r in res:
            name = Path(r.replay).name.split("__")[0].lstrip("0123456789")
            raw = zipfile.ZipFile(r.replay).read("team-a.stdout").replace(b"\x00", b"").decode("utf-8", "replace")
            m = next((json.loads(ln[2:]) for ln in raw.splitlines() if ln.startswith("M ")), None)
            if m:
                terrain[name] = m
    TERRAIN.write_text(json.dumps(terrain, separators=(",", ":"), sort_keys=True), encoding="utf-8")
    print(f"{len(terrain)} maps in {TERRAIN.relative_to(ROOT)}")
    return 0


def cmd_build(_: argparse.Namespace) -> int:
    terrain = json.loads(TERRAIN.read_text(encoding="utf-8"))
    out = {name: build_map(m) for name, m in sorted(terrain.items())}
    ROUTES.write_text(json.dumps(out, separators=(",", ":"), sort_keys=True), encoding="utf-8")
    for name, r in out.items():
        print(f"{name:24s} groups {len(r['cluster_of']):2d} clusters {len(r['value']):2d} routes {[len(s['routes']) for s in r['sides']]}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("dump").set_defaults(fn=cmd_dump)
    sub.add_parser("build").set_defaults(fn=cmd_build)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
