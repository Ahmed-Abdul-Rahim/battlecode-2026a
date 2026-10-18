"""Domain-adversarial training (DANN, Ganin et al. JMLR 2016, P-010) scaffold.

Source domain: another MOBA (Hokoff Honor of Kings trajectories, P-005) mapped
into our feature schema. Target domain: our cambc self-play telemetry, which
is unlabelled for the transferred task. A gradient-reversal layer makes the
shared encoder's features indistinguishable between domains, so a head trained
on source labels (e.g. fight-or-flee, target choice) transfers to cambc. The
trained encoder + head is then distilled to an int8 student (ml/distill).

Only `inv`-tagged schema features (egocentric, map-size invariant) can be
filled from source data; `map`-tagged ones are zero-filled for the source.

Status: scaffold for the GPU box; Hokoff ingestion (`hokoff_to_schema`) is a
backlog item. NOT exercised in CI (no PyTorch there).
"""

from __future__ import annotations

import numpy as np


def invariant_mask() -> np.ndarray:
    """Boolean mask of schema features tagged `inv` (transferable across MOBAs)."""
    from pathlib import Path

    import yaml

    schema = yaml.safe_load((Path(__file__).resolve().parents[1] / "features" / "schema.yaml").read_text())
    mask = []
    for g in schema["groups"]:
        for f in g["features"]:
            mask += [f.get("domain") == "inv"] * int(f.get("count", 1))
    return np.array(mask, dtype=bool)


def hokoff_to_schema(trajectory) -> np.ndarray:  # pragma: no cover - backlog
    """Map one Hokoff frame (hero/unit vectors) onto our schema's `inv` features."""
    raise NotImplementedError("backlog: Hokoff ingestion (see GitHub issue 'UDA: Hokoff ingestion')")


def build(torch, n_features: int, n_classes: int, hidden: int = 128):  # pragma: no cover - GPU box only
    class GradReverse(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, lam):
            ctx.lam = lam
            return x.view_as(x)

        @staticmethod
        def backward(ctx, grad):
            return -ctx.lam * grad, None

    class Dann(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Sequential(
                torch.nn.Linear(n_features, hidden),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden, hidden),
                torch.nn.ReLU(),
            )
            self.label_head = torch.nn.Linear(hidden, n_classes)
            self.domain_head = torch.nn.Sequential(
                torch.nn.Linear(hidden, 64), torch.nn.ReLU(), torch.nn.Linear(64, 2)
            )

        def forward(self, x, lam: float = 1.0):
            z = self.encoder(x)
            return self.label_head(z), self.domain_head(GradReverse.apply(z, lam))

    return Dann()


def train_step(torch, model, opt, xs, ys, xt, lam: float):  # pragma: no cover - GPU box only
    """One DANN step: label loss on source + domain confusion on source and target."""
    ce = torch.nn.functional.cross_entropy
    opt.zero_grad()
    logits_s, dom_s = model(xs, lam)
    _, dom_t = model(xt, lam)
    domain_y = torch.cat([torch.zeros(len(xs)), torch.ones(len(xt))]).long().to(xs.device)
    loss = ce(logits_s, ys) + ce(torch.cat([dom_s, dom_t]), domain_y)
    loss.backward()
    opt.step()
    return float(loss)


def lambda_schedule(progress: float) -> float:
    """Ganin et al.'s ramp: 2 / (1 + exp(-10 p)) - 1."""
    return float(2.0 / (1.0 + np.exp(-10.0 * progress)) - 1.0)
