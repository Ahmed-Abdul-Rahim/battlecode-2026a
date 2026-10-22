"""Shared pieces of the learned policies' PPO; the training itself runs on the GPU (ml/rl/ppo_gpu.py).

Arm 1 "experts+": 109 features -> 32 -> 13 (9 experts + 4 fight-target modes).
Arm 2 "micro":    127 inputs   -> 32 -> 15 (9 steps + 4 targets + 2 cast).

The bot samples each head from a masked softmax and logs the log-probability (ml/rl/reward.py
turns a game into samples). Here: the arms, the bot-sized network in numpy (init, forward,
masked log-probabilities), GAE as the reference for the GPU version, and the export. Exports
are int8 BCW1 blobs whose last layer has a power-of-two accumulator scale, stored in that
layer's `shift`, so the bot's probabilities match training.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ml.distill import quant  # noqa: E402

SCHEMA_HASH = 0xBDF9  # the feature schema (crates/bc-brain features_gen.rs)


@dataclass(frozen=True)
class Arm:
    n_in: int
    heads: tuple[tuple[int, int], ...]  # (start, end) of each head in the outputs
    schema: int
    hidden: int = 32

    @property
    def n_out(self) -> int:
        return self.heads[-1][1]


ARMS = {
    1: Arm(109, ((0, 9), (9, 13)), SCHEMA_HASH),
    2: Arm(127, ((0, 9), (9, 13), (13, 15)), SCHEMA_HASH ^ 0x4D49),
}
GAMMA, LAMBDA = 0.99, 0.95
CLIP, ENTROPY, TARGET_KL = 0.2, 0.01, 0.03


def init_params(n_in: int, hidden: int, n_out: int, rng: np.random.Generator, out_std: float = 0.01) -> dict:
    return {
        "w1": rng.normal(0, np.sqrt(2 / n_in), (hidden, n_in)).astype(np.float32),
        "b1": np.zeros(hidden, np.float32),
        "w2": rng.normal(0, out_std, (n_out, hidden)).astype(np.float32),
        "b2": np.zeros(n_out, np.float32),
    }


def mlp(p: dict, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z1 = x @ p["w1"].T + p["b1"]
    h = np.maximum(z1, 0)
    return z1, h, h @ p["w2"].T + p["b2"]


def head_stats(logits: np.ndarray, masks: np.ndarray, arm: Arm):
    """Per head: masked log-softmax (n x width) and probabilities."""
    out = []
    for k, (a, b) in enumerate(arm.heads):
        allowed = ((masks[:, k : k + 1] >> np.arange(b - a)) & 1).astype(bool)
        z = np.where(allowed, logits[:, a:b], -1e9)
        z = z - z.max(axis=1, keepdims=True)
        lse = np.log(np.exp(z).sum(axis=1, keepdims=True))
        lsm = z - lse
        p = np.where(allowed, np.exp(lsm), 0.0)
        out.append((lsm, p, allowed))
    return out


def log_prob(p: dict, x: np.ndarray, acts: np.ndarray, masks: np.ndarray, arm: Arm) -> np.ndarray:
    _, _, logits = mlp(p, x)
    hs = head_stats(logits, masks, arm)
    return sum(lsm[np.arange(len(x)), acts[:, k]] for k, (lsm, _, _) in enumerate(hs))


def gae(
    rew: np.ndarray, val: np.ndarray, done: np.ndarray, traj: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Advantages and returns per trajectory (samples are in decision order within each)."""
    adv = np.zeros_like(rew)
    for t in np.unique(traj):
        idx = np.nonzero(traj == t)[0]
        last = 0.0
        for j in reversed(range(len(idx))):
            i = idx[j]
            nxt = 0.0 if done[i] or j + 1 == len(idx) else val[idx[j + 1]]
            delta = rew[i] + GAMMA * nxt - val[i]
            last = delta + GAMMA * LAMBDA * (0.0 if done[i] else last)
            adv[i] = last
    return adv, adv + val


def export(p: dict, x_int8: np.ndarray, arm: Arm) -> bytes:
    """int8 BCW1 blob: hidden layer as ml/distill, last layer with a power-of-two scale."""
    xf = x_int8.astype(np.float32) / 127.0
    hidden = np.maximum(xf @ p["w1"].T + p["b1"], 0)
    h_scale = 127.0 / max(float(np.percentile(hidden, 99.9)), 1e-3)
    w1, b1, s1 = quant.quantize_layer(p["w1"], p["b1"], 127.0, h_scale)
    real_h = (127.0 / max(float(np.abs(p["w1"]).max()), 1e-8)) * 127.0 / 2**s1
    k = int(np.floor(np.log2(127.0 * real_h / max(float(np.abs(p["w2"]).max()), 1e-8))))
    k = max(0, min(k, 30))
    w2 = np.clip(np.round(p["w2"] * (2.0**k / real_h)), -127, 127).astype(np.int8)
    b2 = np.round(p["b2"] * 2.0**k).astype(np.int32)
    layers = [
        quant.Layer(w1, b1, s1, relu=True, film_offset=None),
        quant.Layer(w2, b2, k, relu=False, film_offset=None),
    ]
    return quant.encode(arm.schema, layers)


def save(path: Path, params: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **params)


def load(path: Path) -> dict:
    d = np.load(path)
    return {k: d[k].astype(np.float32) for k in d.files}
