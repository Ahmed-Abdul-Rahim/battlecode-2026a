"""Offline RL for the expert router: advantage-weighted regression (AWR) on logged games.

Data comes from debug games of the champion built with learned.rs EPSILON > 0: every 4th
turn the bot logs its int8 feature vector (`x`) and the expert it used (rule pick or a
random feasible exploration). Each row gets a shaped return over the next H rounds:

    R = Δlifetime gold + GEM_W·Δgems carried + TRIB_W·Δteam tributed/champions
        - DEATH_COST·deaths + WIN_BONUS·won

A ridge-regression value baseline V(x) gives advantages A = R - V(x); the router MLP is
trained to imitate each row's expert with weight exp(A / β) (clipped), so actions that
did better than expected are reinforced. Exports a BCW1 blob for bots/main/src/weights.

    python -m ml.rl.awr runs/league/rl/<cycle> --bot _lg-explore --out runs/league/router.bin
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))

from bctools import replay  # noqa: E402
from bctools.dataset import EXPERTS  # noqa: E402
from ml.distill import quant  # noqa: E402
from ml.distill.train_np import SCHEMA_HASH, adam_train, predict, quantise  # noqa: E402

HORIZON = 40
GEM_W = 3.0
TRIB_W = 4.0
DEATH_COST = 200.0
WIN_BONUS = 150.0


def rows_from_bundle(path: Path, side: str, won: bool) -> tuple[list, list, list]:
    log = replay.load(path).team(side)
    by_champ: dict[str, list[dict]] = defaultdict(list)
    for t in log.turns:
        by_champ[t["c"]].append(t)
    deaths: dict[str, list[int]] = defaultdict(list)
    last_alive: dict[str, int] = {}
    for e in log.events:
        if e["k"] == "dead" and last_alive.get(e["c"], -1) != e["r"] - 1:
            deaths[e["c"]].append(e["r"])  # first dead turn of a death
        if e["k"] == "dead":
            last_alive[e["c"]] = e["r"]
    n_champs = max(len(by_champ), 1)
    xs, ys, rs = [], [], []
    for champ, turns in by_champ.items():
        turns.sort(key=lambda t: t["r"])
        rounds = [t["r"] for t in turns]
        for t in turns:
            if "x" not in t or t.get("e") not in EXPERTS or t.get("m") == "override":
                continue
            # State H rounds later (or the last logged turn).
            j = min(np.searchsorted(rounds, t["r"] + HORIZON), len(turns) - 1)
            u = turns[j]
            died = sum(1 for d in deaths[champ] if t["r"] < d <= t["r"] + HORIZON)
            r = (u["lg"] - t["lg"]) + GEM_W * (u["gm"] - t["gm"]) + TRIB_W * (u["at"] - t["at"]) / n_champs
            r += -DEATH_COST * died + (WIN_BONUS if won else 0.0)
            xs.append(np.frombuffer(bytes.fromhex(t["x"]), dtype=np.int8))
            ys.append(EXPERTS.index(t["e"]))
            rs.append(r)
    return xs, ys, rs


def load(dirs: list[Path], bot: str) -> dict[str, np.ndarray]:
    xs, ys, rs = [], [], []
    for d in dirs:
        results = (
            json.loads((d / "results.json").read_text(encoding="utf-8"))
            if (d / "results.json").exists()
            else []
        )
        for res in results:
            side = "a" if res["bot_a"] == bot else "b"
            x, y, r = rows_from_bundle(Path(res["replay"]), side, res["winner"] == bot)
            xs += x
            ys += y
            rs += r
    if not xs:
        return {"X": np.zeros((0, 0), np.int8), "y": np.zeros(0, int), "R": np.zeros(0)}
    return {"X": np.stack(xs), "y": np.array(ys), "R": np.array(rs, dtype=float)}


def advantages(x: np.ndarray, ret: np.ndarray, ridge: float = 1.0) -> np.ndarray:
    xb = np.hstack([x, np.ones((len(x), 1))])
    w = np.linalg.solve(xb.T @ xb + ridge * np.eye(xb.shape[1]), xb.T @ ret)
    return ret - xb @ w


def train(dirs: list[Path], bot: str, out: Path, hidden: int = 32, epochs: int = 500) -> dict:
    data = load(dirs, bot)
    n = len(data["y"])
    if n < 200:
        return {"rows": n, "skipped": "too few rows"}
    xq = data["X"].astype(np.int64)
    x = xq / 127.0
    adv = advantages(x, data["R"])
    beta = max(float(adv.std()), 1e-6)
    w = np.clip(np.exp(adv / beta), 0.0, 20.0)
    params = adam_train(x, data["y"], w, hidden, epochs, 0.01, 1e-4)
    layers = quantise(params, x)
    blob = quant.encode(SCHEMA_HASH, layers)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(blob)
    agree = float((predict(params, x) == data["y"]).mean())
    report = {
        "rows": n,
        "beta": round(beta, 2),
        "mean_return": round(float(data["R"].mean()), 1),
        "label_counts": {EXPERTS[k]: int((data["y"] == k).sum()) for k in np.unique(data["y"])},
        "train_agreement": round(agree, 3),
        "blob_bytes": len(blob),
        "out": str(out),
    }
    out.with_suffix(".json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dirs", nargs="+", type=Path)
    p.add_argument("--bot", required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    print(json.dumps(train(a.dirs, a.bot, a.out), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
