"""Champion descriptor parity and vector fine-tuning."""

import json
from pathlib import Path

import numpy as np

from ml.champions import finetune
from ml.champions.descriptor import KITS, initial_vector

ROOT = Path(__file__).resolve().parents[1]


def test_descriptor_matches_rust_fixture():
    fixture = json.loads((ROOT / "tests" / "fixtures" / "champion_vectors.json").read_text())
    for name in KITS:
        assert initial_vector(name) == fixture[name], name


def test_analytic_gradient_matches_numerical():
    m = finetune.Model(seed=3)
    rng = np.random.default_rng(0)
    m.s = rng.normal(scale=0.1, size=m.s.shape)
    m.c = rng.normal(scale=0.1, size=m.c.shape)
    m.w = rng.normal(scale=0.1, size=m.w.shape)
    matches = [([0, 1, 2], [3, 4, 5], 1.0), ([6, 7], [8, 9], 0.0)]
    ge, gs, gc, gw = finetune.gradients(m, matches, reg=1.0)
    eps = 1e-6
    for arr, grad, idx in ((m.e, ge, (2, 7)), (m.s, gs, (1, 4)), (m.c, gc, (5, 2)), (m.w, gw, (3,))):
        old = arr[idx]
        arr[idx] = old + eps
        up = m.loss(matches, 1.0)
        arr[idx] = old - eps
        down = m.loss(matches, 1.0)
        arr[idx] = old
        assert abs((up - down) / (2 * eps) - grad[idx]) < 1e-5


def test_fit_learns_a_planted_pair_synergy():
    rng = np.random.default_rng(1)
    names = finetune.NAMES
    x, y = names.index("Varg"), names.index("Willow")
    matches = []
    for _ in range(120):
        a = list(rng.choice(len(names), 3, replace=False))
        b = list(rng.choice(len(names), 3, replace=False))
        a_has, b_has = {x, y} <= set(a), {x, y} <= set(b)
        win = 1.0 if a_has and not b_has else 0.0 if b_has and not a_has else float(rng.random() < 0.5)
        matches.append((a, b, win))
    model, history = finetune.fit(matches, epochs=150)
    assert history[-1] < history[0]
    assert model.strength([x, y]) > model.strength([x, names.index("Tavi")])
