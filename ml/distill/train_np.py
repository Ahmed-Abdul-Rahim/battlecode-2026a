"""Numpy-only router distillation (no PyTorch needed): MVP of the learned MoE router.

Trains a tiny MLP (features → hidden → 9 expert logits) to imitate the expert
the rule router chose on *complex* turns (P-051 policy distillation), with rows
from won matches weighted up. Quantises to a BCW1 blob (P-052) and checks the
int8 forward pass (quant.forward, bit-exact with the Rust kernel) agrees.

    python -m ml.distill.train_np runs/ab runs/tune/<ts> --out bots/main-learned/src/weights/router.bin

Inputs: directories of replay bundles from debug-mode matches (bctools dataset rows).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from ml.distill import quant

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

from bctools import dataset  # noqa: E402

SCHEMA_HASH = 0xBDF9  # crates/bc-brain/src/features_gen.rs
N_EXPERTS = 9


def load(dirs: list[Path]) -> dict[str, np.ndarray]:
    bundles = sorted(p for d in dirs for p in Path(d).rglob("*.repl26a"))
    data = dataset.from_bundles(bundles)
    data["n_bundles"] = np.array(len(bundles))
    return data


def adam_train(x, y, w, hidden: int, epochs: int, lr: float, l2: float, seed: int = 0):
    rng = np.random.default_rng(seed)
    n_in = x.shape[1]
    params = {
        "w1": rng.normal(0, np.sqrt(2 / n_in), (hidden, n_in)),
        "b1": np.zeros(hidden),
        "w2": rng.normal(0, np.sqrt(1 / hidden), (N_EXPERTS, hidden)),
        "b2": np.zeros(N_EXPERTS),
    }
    m = {k: np.zeros_like(v) for k, v in params.items()}
    v = {k: np.zeros_like(p) for k, p in params.items()}
    onehot = np.eye(N_EXPERTS)[y]
    wn = w / w.sum()
    for t in range(1, epochs + 1):
        h_pre = x @ params["w1"].T + params["b1"]
        h = np.maximum(h_pre, 0)
        logits = h @ params["w2"].T + params["b2"]
        logits -= logits.max(axis=1, keepdims=True)
        p = np.exp(logits)
        p /= p.sum(axis=1, keepdims=True)
        g_logits = (p - onehot) * wn[:, None]
        grads = {
            "w2": g_logits.T @ h + l2 * params["w2"],
            "b2": g_logits.sum(0),
        }
        g_h = g_logits @ params["w2"] * (h_pre > 0)
        grads["w1"] = g_h.T @ x + l2 * params["w1"]
        grads["b1"] = g_h.sum(0)
        for k in params:
            m[k] = 0.9 * m[k] + 0.1 * grads[k]
            v[k] = 0.999 * v[k] + 0.001 * grads[k] ** 2
            params[k] -= lr * (m[k] / (1 - 0.9**t)) / (np.sqrt(v[k] / (1 - 0.999**t)) + 1e-8)
    return params


def predict(params, x) -> np.ndarray:
    h = np.maximum(x @ params["w1"].T + params["b1"], 0)
    return (h @ params["w2"].T + params["b2"]).argmax(1)


def quantise(params, x_float) -> list[quant.Layer]:
    in_scale = 127.0
    hidden = np.maximum(x_float @ params["w1"].T + params["b1"], 0)
    h_max = max(float(np.percentile(hidden, 99.9)), 1e-3)
    h_scale = 127.0 / h_max
    w1, b1, s1 = quant.quantize_layer(params["w1"], params["b1"], in_scale, h_scale)
    real_h_scale = (127.0 / max(float(np.abs(params["w1"]).max()), 1e-8)) * in_scale / 2**s1
    w2, b2, _ = quant.quantize_layer(params["w2"], params["b2"], real_h_scale, 1.0)
    return [
        quant.Layer(w1, b1, s1, relu=True, film_offset=None),
        quant.Layer(w2, b2, 0, relu=False, film_offset=None),
    ]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--hidden", type=int, default=16)
    p.add_argument("--epochs", type=int, default=600)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--l2", type=float, default=1e-4)
    p.add_argument("--win-weight", type=float, default=2.0, help="row weight multiplier for won matches")
    a = p.parse_args(argv)

    data = load(a.runs)
    keep = data["complex"] == 1
    xq, y = data["X"][keep].astype(np.int64), data["expert"][keep]
    w = np.where(data["won"][keep] == 1, a.win_weight, 1.0)
    x = xq / 127.0
    rng = np.random.default_rng(1)
    idx = rng.permutation(len(y))
    cut = int(0.8 * len(y))
    tr, te = idx[:cut], idx[cut:]
    params = adam_train(x[tr], y[tr], w[tr], a.hidden, a.epochs, a.lr, a.l2)
    acc_tr = float((predict(params, x[tr]) == y[tr]).mean())
    acc_te = float((predict(params, x[te]) == y[te]).mean())

    layers = quantise(params, x[tr])
    blob = quant.encode(SCHEMA_HASH, layers)
    _, decoded = quant.decode(blob)
    logits_q = np.stack([quant.forward(decoded, row) for row in xq[te]])
    acc_q = float((logits_q.argmax(1) == y[te]).mean())
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_bytes(blob)
    report = {
        "bundles": int(data["n_bundles"]),
        "rows_complex": len(y),
        "label_counts": {dataset.EXPERTS[k]: int((y == k).sum()) for k in np.unique(y)},
        "majority_baseline": round(float(np.bincount(y[te]).max() / len(te)), 4),
        "acc_train": round(acc_tr, 4),
        "acc_test_float": round(acc_te, 4),
        "acc_test_int8": round(acc_q, 4),
        "blob_bytes": len(blob),
        "shape": [x.shape[1], a.hidden, N_EXPERTS],
        "out": str(a.out),
    }
    a.out.with_suffix(".json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
