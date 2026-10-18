"""Fine-tune champion vectors from self-play outcomes (docs/champions.md).

Model (logistic, DraftRec-style, P-040), for team A vs team B:

    strength(T) = sum_i w . e_i  +  sum_{i<j in T} e_i^T S e_j
    P(A wins)   = sigmoid( strength(A) - strength(B) + sum_{i in A, k in B} (e_i^T C e_k - e_k^T C e_i) )

Parameters: the 12 champion vectors E (16 dims), synergy S, counter C, and w.
The stats block (dims 0-5) is regularised towards the kit projection from
descriptor.py so it keeps meaning "stats"; the strengths/weaknesses and
synergy blocks start at zero and are learned. Plain numpy: runs anywhere.

    python -m ml.champions.finetune runs/            # reads */*.json match statistics
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ml.champions.descriptor import KITS, initial_vector

NAMES = list(KITS)
# Statistics name drafts as "Cassi & Issac" etc.; map them to our kit names.
DRAFT_ALIASES = {"Vesper Vale": ["VesperVale"], "Cassi & Issac": ["Cassi", "Issac"]}


def load_matches(root: Path) -> list[tuple[list[int], list[int], float]]:
    """(team A champion indices, team B indices, 1 if A won) from match statistics JSON."""
    out = []
    for p in sorted(root.rglob("*.json")):
        try:
            s = json.loads(p.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            continue
        if "roster" not in s or s.get("match", {}).get("outcome") not in ("teamA", "teamB"):
            continue
        teams: dict[str, list[int]] = {"teamA": [], "teamB": []}
        for pick in s["roster"]["draft"]:
            for name in DRAFT_ALIASES.get(pick["champion"], [pick["champion"].replace(" ", "")]):
                if name in NAMES:
                    teams[pick["team"]].append(NAMES.index(name))
        out.append((teams["teamA"], teams["teamB"], 1.0 if s["match"]["outcome"] == "teamA" else 0.0))
    return out


class Model:
    def __init__(self, dim: int = 16, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.e0 = np.array([initial_vector(n) for n in NAMES], dtype=np.float64) / 127.0
        self.e = self.e0.copy() + rng.normal(scale=0.01, size=self.e0.shape)
        self.s = np.zeros((dim, dim))
        self.c = np.zeros((dim, dim))
        self.w = np.zeros(dim)

    def strength(self, team: list[int]) -> float:
        v = self.e[team]
        pair = sum(v[i] @ self.s @ v[j] for i in range(len(v)) for j in range(i + 1, len(v)))
        return float((v @ self.w).sum() + pair)

    def logit(self, a: list[int], b: list[int]) -> float:
        cross = sum(self.e[i] @ self.c @ self.e[k] - self.e[k] @ self.c @ self.e[i] for i in a for k in b)
        return self.strength(a) - self.strength(b) + cross

    def loss(self, matches, reg: float) -> float:
        nll = 0.0
        for a, b, y in matches:
            p = 1 / (1 + np.exp(-self.logit(a, b)))
            nll -= y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9)
        stats_drift = ((self.e[:, :6] - self.e0[:, :6]) ** 2).sum()
        return nll / max(len(matches), 1) + reg * stats_drift


def gradients(m: Model, matches, reg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Analytic gradients of the mean logistic loss (plus stats-block regulariser)."""
    ge, gs, gc, gw = np.zeros_like(m.e), np.zeros_like(m.s), np.zeros_like(m.c), np.zeros_like(m.w)
    n = max(len(matches), 1)
    for a, b, y in matches:
        g = (1 / (1 + np.exp(-m.logit(a, b))) - y) / n  # dL/dz
        for team, sign in ((a, 1.0), (b, -1.0)):
            for pos, i in enumerate(team):
                gw += sign * g * m.e[i]
                ge[i] += sign * g * m.w
                for pos2, j in enumerate(team):
                    if pos < pos2:
                        gs += sign * g * np.outer(m.e[i], m.e[j])
                        ge[i] += sign * g * (m.s @ m.e[j])
                        ge[j] += sign * g * (m.s.T @ m.e[i])
        for i in a:
            for k in b:
                gc += g * (np.outer(m.e[i], m.e[k]) - np.outer(m.e[k], m.e[i]))
                ge[i] += g * (m.c @ m.e[k] - m.c.T @ m.e[k])
                ge[k] += g * (m.c.T @ m.e[i] - m.c @ m.e[i])
    ge[:, :6] += 2 * reg * (m.e[:, :6] - m.e0[:, :6])
    return ge, gs, gc, gw


def fit(
    matches, epochs: int = 300, lr: float = 0.5, reg: float = 1.0, seed: int = 0
) -> tuple[Model, list[float]]:
    """Full-batch gradient descent with analytic gradients (~800 parameters)."""
    m = Model(seed=seed)
    history = [m.loss(matches, reg)]
    for _ in range(epochs):
        ge, gs, gc, gw = gradients(m, matches, reg)
        m.e -= lr * ge
        m.s -= lr * gs
        m.c -= lr * gc
        m.w -= lr * gw
        history.append(m.loss(matches, reg))
    return m, history


def export(m: Model, out: Path) -> dict:
    vecs = {n: [round(float(max(-1.0, min(1.0, x))) * 127) for x in m.e[i]] for i, n in enumerate(NAMES)}

    def q(a: np.ndarray) -> list:
        return np.clip(np.round(a / max(np.abs(a).max(), 1e-9) * 127), -127, 127).astype(int).tolist()

    data = {"vectors": vecs, "synergy": q(m.s), "counter": q(m.c), "w": m.w.tolist()}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=1), encoding="utf-8")
    return data


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", help="directory containing match statistics JSON (searched recursively)")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--out", default="ml/champions/champ_vec_v1.json")
    a = p.parse_args(argv)
    matches = load_matches(Path(a.runs))
    if len(matches) < 20:
        print(f"only {len(matches)} matches with drafts; self-play more varied drafts first")
    model, history = fit(matches, epochs=a.epochs)
    export(model, Path(a.out))
    print(
        json.dumps({"matches": len(matches), "loss_start": history[0], "loss_end": history[-1], "out": a.out})
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
