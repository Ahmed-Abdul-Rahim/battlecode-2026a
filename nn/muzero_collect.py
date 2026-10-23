"""Exploration games for the MuZero-style model (nn/muzero.py).

    python nn/muzero_collect.py [--games 384] [--rate 100] [--hold 20]

The rule bot (TEACHER = 1: rules decide, every decision logged) plays full games with
exploration switched on through the patchable branch block: RATE per mille of decisions take a
random other allowed goal and hold it HOLD rounds. Each game gets its own seed, so a pool of
prebuilt bots covers every game without rebuilds. Saved per shard (runs/nn/muzero/data/):
the usual sample arrays with A[:, 0] = the goal actually executed ("ex" in the log),
B = the rules' own choice (the behaviour prior), and R = the team value change until the
champion's next decision (from the bot's "S" score lines: submitted ours - theirs + half of
carried + earned gold / 10), in gems.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import struct
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from bctools import runner  # noqa: E402

from nn import data  # noqa: E402
from nn import expert as E  # noqa: E402
from nn import run as R  # noqa: E402

OUT = R.OUT / "muzero"
POOL = 32


def patch_explore(bot: str, rate: int, hold: int, seed: int) -> None:
    words = struct.pack("<8I", 0xB7A1C4E5, 0x5EEDF00D, 0xFFFFFFFF, 0, 0, hold, rate, seed)
    n = 0
    for f in (runner.bot_dir(bot) / ".cambc").rglob("*.wasm"):
        b = bytearray(f.read_bytes())
        i = b.find(E.MAGIC)
        if i >= 0:
            b[i : i + 32] = words
            f.write_bytes(b)
            n += 1
    if not n:
        raise RuntimeError(f"{bot}: branch block not found")


def parse(job: tuple[str, str, str, float, float]) -> dict:
    """One game -> samples of our side with executed actions and team-value rewards."""
    replay, stats, side, won, margin = job
    try:
        d = data.game_samples(Path(replay), Path(stats), side, won, margin)
        if not d:
            return {}
        dec, sc = [], E.scores(Path(replay), side)
        for ln in data._lines(Path(replay), side):
            if ln.startswith("N "):
                o = json.loads(ln[2:])
                dec.append((data._norm(o["c"]), o["r"], o["a"][0], o.get("ex", -1)))
        dec.sort(key=lambda x: (x[0], x[1]))  # game_samples' row order
        if len(dec) != len(d["S"]):
            return {}
        n = len(dec)
        A = d["A"].copy()
        B = A[:, 0].copy()
        Rw = np.zeros(n, np.float32)
        last_round = max(r for _, r, _, _ in dec)
        for i, (c, r, a, ex) in enumerate(dec):
            if ex >= 0:
                A[i, 0] = ex
                d["M"][i, 0] |= 1 << ex
            nxt = dec[i + 1][1] if i + 1 < n and dec[i + 1][0] == c else last_round + 1
            Rw[i] = E.value_at(sc, nxt) - E.value_at(sc, r)
        d["A"], d["B"], d["R"] = A, B, Rw
        d["EXPLORED"] = np.array([ex >= 0 and ex != a for _, _, a, ex in dec], bool)
        return d
    except (OSError, KeyError, ValueError, zipfile.BadZipFile, json.JSONDecodeError):
        return {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=384)
    ap.add_argument("--rate", type=int, default=100)
    ap.add_argument("--hold", type=int, default=20)
    a = ap.parse_args()
    (OUT / "data").mkdir(parents=True, exist_ok=True)
    pool = [R.variant(f"_mz{i}", 1) for i in range(POOL)]
    with ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(runner.build, pool))
    train, _ = R.maps()
    opps = ["main", *R.ANCHORS, *R.mimics()]
    rng = random.Random(31)
    done = len(list((OUT / "data").glob("explore_*.npz")))
    for b in range(done, a.games // POOL):
        games, sides = [], []
        for i, bot in enumerate(pool):
            patch_explore(bot, a.rate, a.hold, 1000 * b + i + 1)
            o, m, sa = rng.choice(opps), rng.choice(train), rng.random() < 0.5
            games.append((bot, o, m) if sa else (o, bot, m))
            sides.append("a" if sa else "b")
        out = OUT / "games" / f"b{b}"
        res = runner.run_games(games, out, log_limit=R.LOG_LIMIT, fuel=R.FUEL[512], timeout=R.GAME_TIMEOUT)
        jobs = []
        for r, side in zip(res, sides):
            if r.winner is None and r.rounds == 0:
                continue
            bot = r.bot_a if side == "a" else r.bot_b
            margin = ((r.gems_a - r.gems_b) if side == "a" else (r.gems_b - r.gems_a)) / 1000.0
            jobs.append((r.replay, r.stats, side, r.score_for(bot), margin))
        with ProcessPoolExecutor(max_workers=12) as ex:
            parts = [p for p in ex.map(parse, jobs) if p]
        d = data.concat(parts)
        if d:
            np.savez_compressed(OUT / "data" / f"explore_{b:03d}.npz", **d)
        print(f"batch {b}: {len(parts)} games, {len(d.get('S', []))} decisions, "
              f"{int(d['EXPLORED'].sum()) if d else 0} explored", flush=True)
        shutil.rmtree(out, ignore_errors=True)


if __name__ == "__main__":
    main()
