"""Group backward elimination on synthetic data with planted structure."""

import numpy as np

from ml.features.select import (
    Unit,
    backward_floating,
    multiresolution_select,
    nogueira_stability,
    split_units,
)


def planted(n: int = 600, seed: int = 1):
    """Groups: A informative (2 feats), B noise (4), C copy of A (redundant), D noise but expensive."""
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(n, 2))
    y = (a[:, 0] + 0.8 * a[:, 1] > 0).astype(int)
    b = rng.normal(size=(n, 4))
    c = a + rng.normal(scale=0.05, size=(n, 2))
    d = rng.normal(size=(n, 2))
    x = np.hstack([a, b, c, d])
    groups = [
        Unit("A", (0, 1), 100.0),
        Unit("B", (2, 3, 4, 5), 100.0),
        Unit("C", (6, 7), 100.0),
        Unit("D", (8, 9), 2000.0),
    ]
    return x, y, groups


def linear_scorer(x, y):
    """Deterministic proxy: R^2-like separability of a least-squares fit."""

    def score(features):
        if not features:
            return 0.5
        xs = np.hstack([x[:, features], np.ones((len(x), 1))])
        w, *_ = np.linalg.lstsq(xs, y * 2 - 1, rcond=None)
        return float(((xs @ w > 0).astype(int) == y).mean())

    return score


def test_backward_keeps_signal_drops_costly_noise():
    x, y, groups = planted()
    best, j, evals = backward_floating(groups, linear_scorer(x, y), lam=0.3, total_fuel=2300)
    names = {u.name for u in best}
    assert names & {"A", "C"}, "an informative group survives"
    assert "D" not in names, "expensive noise is dropped"
    assert not {"A", "C"} <= names, "the redundant copy is not worth its fuel"
    assert evals > 0 and j > 0.5


def test_stability_index_bounds():
    assert nogueira_stability([{0, 1}, {0, 1}, {0, 1}], 10) == 1.0
    assert nogueira_stability([{0}, {5}, {9}], 10) < 0.2


def test_split_units_shares_fuel_and_preserves_features():
    x, y, groups = planted()
    units = split_units(groups, 1, x, y)
    assert sorted(f for u in units for f in u.features) == list(range(10))
    assert abs(sum(u.fuel for u in units) - sum(g.fuel for g in groups)) < 1e-6


def test_multiresolution_refines_then_stops():
    x, y, groups = planted()
    report = multiresolution_select(x, y, groups, linear_scorer, lam=0.3, tau=0.5, bootstraps=4)
    assert report.selected, "something is selected"
    assert set(report.selected) <= {0, 1, 6, 7}, report.selected
    assert len(report.levels) >= 2, "refined below group level"
    assert report.stable_size <= 2


def test_backs_off_when_never_stable():
    x, y, groups = planted()
    report = multiresolution_select(x, y, groups, linear_scorer, lam=0.3, tau=1.01, bootstraps=3)
    assert len(report.levels) == 1
    assert report.stable_size == 4, "falls back to the starting group level"
