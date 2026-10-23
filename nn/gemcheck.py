"""Which branch value predicts the final result? (submitted vs carried vs gold, horizon)

    python nn/gemcheck.py [--branches 128]

Base games (rules, logging) -> random decisions -> a branch forcing another goal (held 20 rounds)
played to the END. For each branch vs its base game: change in candidate measures at decision +
{150, 300, 500} rounds (from the bots' "S" score lines) and change in the final submitted-gem
margin and result. Reports the rank correlation of each measure with the final outcome.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from bctools import runner  # noqa: E402

from nn import data  # noqa: E402
from nn import expert as E  # noqa: E402
from nn import run as R  # noqa: E402

OUT = R.OUT / "gemcheck"
POOL = 32
HORIZONS = (150, 300, 500)


def series(bundle: Path, side: str) -> list[tuple[int, float, float, float]]:
    """(round, submitted ours - theirs, our carried, our earned gold) after each turn."""
    out, carried, gold = [], {}, {}
    for ln in data._lines(bundle, side):
        if ln.startswith("S "):
            o = json.loads(ln[2:])
            carried[o["c"]] = float(o["g"])
            gold[o["c"]] = float(o.get("lg", 0))
            out.append((int(o["r"]), float(o["us"]) - float(o["them"]), sum(carried.values()), sum(gold.values())))
    return out


def at(sr, t):
    v = (0.0, 0.0, 0.0)
    for r, s, c, g in sr:
        if r > t:
            break
        v = (s, c, g)
    return v


MEASURES = {
    "blend (current)": lambda s, c, g: s + 0.5 * c + 0.1 * g,
    "submitted only": lambda s, c, g: s,
    "submitted + carried/2": lambda s, c, g: s + 0.5 * c,
    "submitted + carried": lambda s, c, g: s + c,
    "blend, gold/30": lambda s, c, g: s + 0.5 * c + g / 30,
}


def final(r, side: str) -> tuple[float, float]:
    margin = (r.gems_a - r.gems_b) * (1 if side == "a" else -1)
    bot = r.bot_a if side == "a" else r.bot_b
    return float(margin), r.score_for(bot)


def spearman(x, y) -> float:
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1]) if np.std(rx) and np.std(ry) else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--branches", type=int, default=128)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    base = R.variant("_gcbase", 1)
    pool = [R.variant(f"_gc{i}", 1) for i in range(POOL)]
    with ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(runner.build, [base, *pool]))
    train, _ = R.maps()
    runner.run_games([(b, "main", train[0]) for b in pool], OUT / "warm", rounds=1, log_limit=0)
    rng = random.Random(77)
    opps = ["main", *R.ANCHORS, *R.mimics()]
    n_base = a.branches // 4
    setups = [(rng.choice(opps), rng.choice(train), rng.random() < 0.5) for _ in range(n_base)]
    res = runner.run_games([(base, o, m) if s else (o, base, m) for o, m, s in setups], OUT / "base",
                           log_limit=R.LOG_LIMIT, fuel=R.FUEL[512], timeout=R.GAME_TIMEOUT)
    specs = []
    for (o, m, s), r in zip(setups, res):
        side = "a" if s else "b"
        if r.winner is None and r.rounds == 0:
            continue
        dec = [d for d in E.decisions(Path(r.replay), side)
               if 50 <= d["r"] <= r.rounds - 520 and d["c"] in E.KIND and bin(d["m"] & ~(1 << d["a"])).count("1")]
        for d in rng.sample(dec, min(4, len(dec))):
            alt = rng.choice([g for g in range(15) if d["m"] >> g & 1 and g != d["a"]])
            specs.append(((o, m, s, side, r), d, alt))
    rows = []
    for b0 in range(0, len(specs), POOL):
        batch = specs[b0 : b0 + POOL]
        games = []
        for bot, ((o, m, s, side, r), d, alt) in zip(pool, batch):
            E.patch(bot, (d["r"], E.KIND[d["c"]], alt, E.COMMIT))
            games.append((bot, o, m) if s else (o, bot, m))
        br = runner.run_games(games, OUT / f"b{b0}", log_limit=R.LOG_LIMIT, fuel=R.FUEL[512], timeout=R.GAME_TIMEOUT)
        for ((o, m, s, side, r), d, alt), rb in zip(batch, br):
            if rb.winner is None and rb.rounds == 0:
                continue
            lines = data._lines(Path(rb.replay), side)
            if not any(ln.startswith("E ") and '"k":"branch"' in ln for ln in lines):
                continue
            sb, sx = series(Path(r.replay), side), series(Path(rb.replay), side)
            row = {"final_margin": final(rb, side)[0] - final(r, side)[0],
                   "final_win": final(rb, side)[1] - final(r, side)[1]}
            for h in HORIZONS:
                vb, vx = at(sb, d["r"] + h), at(sx, d["r"] + h)
                for name, f in MEASURES.items():
                    row[f"{name} @{h}"] = f(*vx) - f(*vb)
            rows.append(row)
        shutil.rmtree(OUT / f"b{b0}", ignore_errors=True)
        print(f"batch {b0 // POOL}: {len(rows)} branches so far", flush=True)
    shutil.rmtree(OUT / "base", ignore_errors=True)
    (OUT / "rows.json").write_text(json.dumps(rows))
    fm = np.array([r["final_margin"] for r in rows])
    fw = np.array([r["final_win"] for r in rows])
    print(f"\n{len(rows)} branches; final margin change: mean {fm.mean():+.1f}, "
          f"{np.mean(fm != 0):.0%} nonzero; result changed in {np.mean(fw != 0):.0%}")
    print(f"{'measure':34s} {'rank corr w/ final margin':>26s} {'w/ result':>10s} {'sign agrees':>12s}")
    for k in rows[0]:
        if k.startswith("final"):
            continue
        x = np.array([r[k] for r in rows])
        nz = (x != 0) & (fm != 0)
        agree = np.mean(np.sign(x[nz]) == np.sign(fm[nz])) if nz.any() else float("nan")
        print(f"{k:34s} {spearman(x, fm):26.3f} {spearman(x, fw):10.3f} {agree:12.3f}")


if __name__ == "__main__":
    main()
