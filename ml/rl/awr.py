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
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))

from bctools import replay  # noqa: E402
from bctools.dataset import EXPERTS  # noqa: E402
from ml.distill import quant  # noqa: E402
from ml.distill.train_np import SCHEMA_HASH, predict, quantise  # noqa: E402

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


def save_rows(data: dict[str, np.ndarray], path: Path) -> int:
    """A compact training shard (int8 features, expert labels, shaped returns)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, X=data["X"], y=data["y"], R=data["R"])
    return len(data["y"])


def read_shard(p: Path, keys: tuple[str, ...]) -> dict[str, np.ndarray] | None:
    """A shard's arrays, or None if it is unreadable or incomplete (e.g. written on a full disk)."""
    try:
        d = np.load(p)
        return {k: d[k] for k in keys}
    except (OSError, KeyError, ValueError, zipfile.BadZipFile, EOFError):
        return None


def load_shards(paths: list[Path]) -> dict[str, np.ndarray]:
    parts = [read_shard(p, ("X", "y", "R")) for p in paths]
    parts = [d for d in parts if d is not None and len(d["y"])]
    if not parts:
        return {"X": np.zeros((0, 0), np.int8), "y": np.zeros(0, int), "R": np.zeros(0)}
    return {k: np.concatenate([d[k] for d in parts]) for k in ("X", "y", "R")}


def train(dirs: list[Path], bot: str, out: Path, hidden: int = 32, epochs: int = 500) -> dict:
    return fit(load(dirs, bot), out, hidden, epochs)


MAX_ROWS = 300_000  # a random sample this big trains as well as all rows, in a fraction of the time
BATCH = 4096
MAX_PASSES = 20


def minibatch_train(x, y, w, hidden: int, passes: int, lr: float, l2: float, seed: int = 0) -> dict:
    """Weighted softmax MLP (n_in -> hidden ReLU -> N_EXPERTS), minibatch Adam, float32."""
    rng = np.random.default_rng(seed)
    n, n_in = x.shape
    n_out = len(EXPERTS)
    params = {
        "w1": rng.normal(0, np.sqrt(2 / n_in), (hidden, n_in)).astype(np.float32),
        "b1": np.zeros(hidden, np.float32),
        "w2": rng.normal(0, np.sqrt(1 / hidden), (n_out, hidden)).astype(np.float32),
        "b2": np.zeros(n_out, np.float32),
    }
    m = {k: np.zeros_like(v) for k, v in params.items()}
    v = {k: np.zeros_like(p) for k, p in params.items()}
    t = 0
    for _ in range(passes):
        batch = min(BATCH, max(64, n // 20))  # at least ~20 updates per pass on small data
        for idx in np.array_split(rng.permutation(n), max(1, n // batch)):
            xb, yb, wb = x[idx], y[idx], w[idx]
            h_pre = xb @ params["w1"].T + params["b1"]
            h = np.maximum(h_pre, 0)
            logits = h @ params["w2"].T + params["b2"]
            logits -= logits.max(axis=1, keepdims=True)
            p = np.exp(logits)
            p /= p.sum(axis=1, keepdims=True)
            p[np.arange(len(yb)), yb] -= 1.0
            g = p * (wb / wb.sum())[:, None]
            grads = {"w2": g.T @ h + l2 * params["w2"], "b2": g.sum(0)}
            g_h = (g @ params["w2"]) * (h_pre > 0)
            grads["w1"] = g_h.T @ xb + l2 * params["w1"]
            grads["b1"] = g_h.sum(0)
            t += 1
            for k in params:
                m[k] = 0.9 * m[k] + 0.1 * grads[k]
                v[k] = 0.999 * v[k] + 0.001 * grads[k] ** 2
                params[k] -= lr * (m[k] / (1 - 0.9**t)) / (np.sqrt(v[k] / (1 - 0.999**t)) + 1e-8)
    return {k: p.astype(np.float64) for k, p in params.items()}


def fit(data: dict[str, np.ndarray], out: Path, hidden: int = 32, epochs: int = 8) -> dict:
    """AWR on rows from `load` / `load_shards`; writes the BCW1 blob and a .json report.
    `epochs` = passes over a random sample of at most MAX_ROWS rows (minibatch Adam)."""
    n = len(data["y"])
    if n < 200:
        return {"rows": n, "skipped": "too few rows"}
    rng = np.random.default_rng(0)
    keep = rng.permutation(n)[:MAX_ROWS]
    xq, y, ret = data["X"][keep], np.asarray(data["y"])[keep], np.asarray(data["R"], dtype=float)[keep]
    x = xq.astype(np.float32) / 127.0
    adv = advantages(x.astype(np.float64), ret)
    beta = max(float(adv.std()), 1e-6)
    w = np.clip(np.exp(adv / beta), 0.0, 20.0).astype(np.float32)
    n_val = len(y) // 10
    tr, va = slice(n_val, None), slice(0, n_val)
    params = minibatch_train(x[tr], y[tr], w[tr], hidden, min(epochs, MAX_PASSES), 0.01, 1e-5)
    layers = quantise(params, x[tr].astype(np.float64))
    blob = quant.encode(SCHEMA_HASH, layers)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(blob)
    agree = float((predict(params, x[tr]) == y[tr]).mean())
    val = float((predict(params, x[va]) == y[va]).mean())
    data = {"y": y, "R": ret}
    report = {
        "rows": n,
        "rows_used": len(y),
        "val_agreement": round(val, 3),
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
