"""Multi-resolution group backward elimination (docs/feature-selection.md).

Starts at the schema's feature groups (each aligned to one SDK query, so
dropping a group saves its fuel), runs sequential *floating* backward selection
(Pudil et al. 1994, P-021; Kohavi & John 1997, P-020) with a fuel-cost penalty
(cf. Feature-Budgeted RF, P-026), measures stability across bootstrap reruns
(Nogueira et al. 2018, P-024), then halves the group size while the selection
stays stable. When it stops being stable, the previous level is the answer.

    J(S) = score(S) - lambda * sum(fuel(S)) / total_fuel

`score` is pluggable. The offline proxy (`cv_scorer`) uses cross-validated
accuracy on logged telemetry (bctools dataset). Coarse levels can instead use
in-game win rate via `bctools gate`, since a full sweep takes ~20 s.

    python -m ml.features.select data.npz --target expert --lam 0.2
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Unit:
    """A set of features removed or kept together."""

    name: str
    features: tuple[int, ...]
    fuel: float


@dataclass
class LevelResult:
    size: int
    units: int
    selected: list[int]
    objective: float
    stability: float
    evals: int


@dataclass
class Report:
    levels: list[LevelResult] = field(default_factory=list)
    selected: list[int] = field(default_factory=list)
    stable_size: int = 0

    def to_dict(self) -> dict:
        return {
            "selected": self.selected,
            "stable_size": self.stable_size,
            "levels": [vars(level) for level in self.levels],
        }


Scorer = Callable[[list[int]], float]


def backward_floating(
    units: list[Unit],
    score: Scorer,
    lam: float,
    total_fuel: float,
    min_units: int = 1,
) -> tuple[list[Unit], float, int]:
    """Sequential floating backward selection over units. Returns (best subset, J, evals)."""
    evals = 0
    cache: dict[frozenset[str], float] = {}

    def objective(subset: list[Unit]) -> float:
        nonlocal evals
        key = frozenset(u.name for u in subset)
        if key not in cache:
            feats = sorted({f for u in subset for f in u.features})
            fuel = sum(u.fuel for u in subset)
            cache[key] = score(feats) - lam * fuel / max(total_fuel, 1e-9)
            evals += 1
        return cache[key]

    current = list(units)
    best_by_size: dict[int, tuple[float, list[Unit]]] = {len(current): (objective(current), list(current))}
    while len(current) > min_units:
        # Exclusion step: drop the unit whose removal helps most.
        candidates = [[u for u in current if u is not drop] for drop in current]
        current = max(candidates, key=objective)
        j = objective(current)
        k = len(current)
        if k not in best_by_size or j > best_by_size[k][0]:
            best_by_size[k] = (j, list(current))
        # Conditional inclusion (the floating step): re-add while it beats the best at that size.
        while True:
            excluded = [u for u in units if u not in current]
            if not excluded:
                break
            trial = max(([*current, u] for u in excluded), key=objective)
            jt, kt = objective(trial), len(trial)
            if kt in best_by_size and jt <= best_by_size[kt][0] + 1e-12:
                break
            best_by_size[kt] = (jt, list(trial))
            current = trial
    best_j, best = max(best_by_size.values(), key=lambda v: v[0])
    return best, best_j, evals


def nogueira_stability(selections: list[set[int]], d: int) -> float:
    """Stability index of Nogueira, Sechidis & Brown (JMLR 2018). 1 = identical selections."""
    m = len(selections)
    if m < 2 or d == 0:
        return 1.0
    z = np.zeros((m, d))
    for i, s in enumerate(selections):
        z[i, list(s)] = 1.0
    k_bar = z.sum(axis=1).mean()
    if k_bar in (0, d):
        return 1.0
    p = z.mean(axis=0)
    s2 = m / (m - 1) * p * (1 - p)
    return float(1 - s2.mean() / ((k_bar / d) * (1 - k_bar / d)))


def mrmr_order(x: np.ndarray, y: np.ndarray, features: tuple[int, ...]) -> list[int]:
    """Order features by minimum-redundancy maximum-relevance (Peng et al. 2005, P-022)."""
    from sklearn.feature_selection import mutual_info_classif

    feats = list(features)
    if len(feats) <= 1:
        return feats
    relevance = mutual_info_classif(x[:, feats], y, random_state=0)
    corr = np.nan_to_num(np.abs(np.corrcoef(x[:, feats], rowvar=False)))
    chosen: list[int] = []
    remaining = list(range(len(feats)))
    while remaining:

        def gain(i: int) -> float:
            redundancy = corr[i, chosen].mean() if chosen else 0.0
            return relevance[i] - redundancy

        best = max(remaining, key=gain)
        chosen.append(best)
        remaining.remove(best)
    return [feats[i] for i in chosen]


def split_units(units: list[Unit], size: int, x: np.ndarray, y: np.ndarray) -> list[Unit]:
    """Split each unit into chunks of at most `size` features, mRMR-ordered.
    Fuel is shared out by feature count (the SDK call is paid once per group,
    so finer units over-estimate savings; the in-game check corrects that)."""
    out = []
    for u in units:
        if len(u.features) <= size:
            out.append(u)
            continue
        order = mrmr_order(x, y, u.features)
        chunks = [order[i : i + size] for i in range(0, len(order), size)]
        for ci, chunk in enumerate(chunks):
            out.append(Unit(f"{u.name}/{ci}", tuple(chunk), u.fuel * len(chunk) / len(u.features)))
    return out


def multiresolution_select(
    x: np.ndarray,
    y: np.ndarray,
    groups: list[Unit],
    scorer_factory: Callable[[np.ndarray, np.ndarray], Scorer],
    lam: float = 0.2,
    tau: float = 0.75,
    bootstraps: int = 5,
    plateau: float = 0.005,
    seed: int = 0,
) -> Report:
    """Coarse-to-fine backward elimination with stability-driven group sizing."""
    rng = np.random.default_rng(seed)
    total_fuel = sum(g.fuel for g in groups)
    d = x.shape[1]
    units = list(groups)
    size = max(len(g.features) for g in groups)
    report = Report()
    previous: LevelResult | None = None
    while True:
        selections, objectives, evals = [], [], 0
        for _ in range(bootstraps):
            idx = rng.integers(0, len(x), len(x))
            score = scorer_factory(x[idx], y[idx])
            best, j, n = backward_floating(units, score, lam, total_fuel)
            selections.append({f for u in best for f in u.features})
            objectives.append(j)
            evals += n
        stability = nogueira_stability(selections, d)
        # Consensus: features chosen in a majority of reruns.
        counts = np.zeros(d)
        for s in selections:
            counts[list(s)] += 1
        consensus = sorted(int(f) for f in np.flatnonzero(counts > bootstraps / 2))
        level = LevelResult(size, len(units), consensus, float(np.mean(objectives)), stability, evals)
        report.levels.append(level)
        stable = stability >= tau
        improved = previous is None or level.objective > previous.objective + plateau
        if not stable:
            # Back off: the previous (coarser) level is the stable point.
            chosen = previous or level
            report.selected, report.stable_size = chosen.selected, chosen.size
            return report
        previous = level
        if size <= 1 or (not improved and len(report.levels) > 1):
            report.selected, report.stable_size = level.selected, level.size
            return report
        survivors = [u for u in units if set(u.features) & set(consensus)]
        size = max(1, math.ceil(size / 2))
        units = split_units(survivors, size, x, y)


def cv_scorer(n_splits: int = 3) -> Callable[[np.ndarray, np.ndarray], Scorer]:
    """Offline proxy: cross-validated accuracy of a small logistic regression."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    def factory(x: np.ndarray, y: np.ndarray) -> Scorer:
        def score(features: list[int]) -> float:
            if not features:
                return float(np.bincount(y).max() / len(y))
            model = LogisticRegression(max_iter=300)
            return float(cross_val_score(model, x[:, features], y, cv=n_splits).mean())

        return score

    return factory


def schema_units() -> list[Unit]:
    from ml.features import schema_gen

    return [Unit(g["name"], tuple(range(g["start"], g["end"])), float(g["fuel"])) for g in schema_gen.GROUPS]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset", help=".npz from `bctools dataset`")
    p.add_argument("--target", default="expert", choices=["expert", "complex", "won"])
    p.add_argument("--lam", type=float, default=0.2)
    p.add_argument("--tau", type=float, default=0.75)
    p.add_argument("--bootstraps", type=int, default=5)
    p.add_argument("--out", default="ml/features/selected/report.json")
    a = p.parse_args(argv)
    data = np.load(a.dataset)
    x, y = data["X"].astype(np.float32) / 127.0, data[a.target]
    report = multiresolution_select(x, y, schema_units(), cv_scorer(), a.lam, a.tau, a.bootstraps)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report.to_dict(), indent=1), encoding="utf-8")
    print(json.dumps({"selected": len(report.selected), "stable_size": report.stable_size, "out": str(out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
