"""Map layouts for the GPU engine, extracted from the bot's debug map dump (`M` line).

    python -m nn.gpusim.maps        # (re)extract every map in maps/ into runs/nn/gpusim/maps/

Each map: width, height, boolean grids (walls, holes, bushes, altars, spawn_a, spawn_b) and
camp groups (camps with kind and tile, gems, gold, respawn). Team A's spawn is the dump's
"ally_spawn" (the dump is read from team A's log).
"""

from __future__ import annotations

import json
import shutil
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "runs" / "nn" / "gpusim" / "maps"


@dataclass
class Camp:
    kind: int  # engine CampType: 0 wolf, 1 elk, 2 gem cluster, 3 gold cluster, 4 monolith, 5 golem
    x: int
    y: int


@dataclass
class Group:
    camps: list[Camp]
    tiles: list[tuple[int, int]]
    gems: int
    gold: int
    respawn: int


@dataclass
class MapData:
    name: str
    w: int
    h: int
    walls: np.ndarray  # bool [h, w]
    holes: np.ndarray
    bushes: np.ndarray
    altars: np.ndarray
    spawn_a: np.ndarray
    spawn_b: np.ndarray
    groups: list[Group]


def _grid(hexs: str, w: int, h: int) -> np.ndarray:
    bits = np.unpackbits(np.frombuffer(bytes.fromhex(hexs), np.uint8), bitorder="little")[: w * h]
    return bits.reshape(h, w).astype(bool)


def parse_dump(line: str, name: str) -> MapData:
    o = json.loads(line[2:])
    w, h = o["w"], o["h"]
    groups = [
        Group([Camp(k, x, y) for k, x, y in g.get("camps", [])], [tuple(t) for t in g["tiles"]], g["gems"], g["gold"],
              g.get("respawn", 0))
        for g in o["groups"]
    ]
    return MapData(name, w, h, _grid(o["walls"], w, h), _grid(o["holes"], w, h), _grid(o["bushes"], w, h),
                   _grid(o["altars"], w, h), _grid(o["ally_spawn"], w, h), _grid(o["enemy_spawn"], w, h), groups)


def load(name: str) -> MapData:
    p = CACHE / f"{name}.json"
    return parse_dump(json.loads(p.read_text())["M"], name)


def extract_all() -> None:
    sys.path.insert(0, str(ROOT / "tools"))
    from bctools import runner

    CACHE.mkdir(parents=True, exist_ok=True)
    maps = runner.list_maps()
    out = ROOT / "runs" / "nn" / "gpusim" / "extract"
    res = runner.run_games([("main", "baseline", m) for m in maps], out, rounds=2)
    for r in res:
        raw = zipfile.ZipFile(r.replay).read("team-a.stdout").replace(b"\x00", b"").decode("utf-8", "replace")
        line = next((l for l in raw.splitlines() if l.startswith("M ")), None)
        if line is None:
            print(f"{r.map}: no map dump")
            continue
        md = parse_dump(line, r.map)
        (CACHE / f"{r.map}.json").write_text(json.dumps({"M": line}))
        print(f"{r.map}: {md.w}x{md.h}, {len(md.groups)} groups, {sum(len(g.camps) for g in md.groups)} camps")
    shutil.rmtree(out, ignore_errors=True)


if __name__ == "__main__":
    extract_all()
