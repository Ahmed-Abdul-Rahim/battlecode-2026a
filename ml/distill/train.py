"""Teacher -> student distillation and int8 export (P-051 policy distillation,
P-052 integer-only quantisation, P-054 soft targets).

Runs on the GPU box (teammate's 5060 Ti). NOT exercised in CI or on the team
laptop (no PyTorch there); the int8 export path it calls (quant.py) is tested.

Pipeline:
  1. `bctools run ...` self-play  ->  `bctools dataset runs/X data.npz`
  2. teacher: wide MLP on all features, trained on heuristic labels weighted by
     match outcome (imitate what won)
  3. student: tiny MLP on the features chosen by ml/features/select.py,
     trained on the teacher's soft targets (temperature T)
  4. quantise -> crates/bc-brain/weights/<head>.bin (BCW1 blob, schema hash checked
     by the bot at load)

    python -m ml.distill.train data.npz --head router --features ml/features/selected/report.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ml.distill import quant
from ml.features import schema_gen


def _torch():
    try:
        import torch
    except ImportError as e:  # pragma: no cover - GPU box only
        raise SystemExit("PyTorch missing: pip install -r ml/requirements-ml.txt (CUDA 12.8+ build)") from e
    return torch


def mlp(torch, sizes: list[int]):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(torch.nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(torch.nn.ReLU())
    return torch.nn.Sequential(*layers)


def train(model, x, y, weights, epochs: int, lr: float, torch, soft=None, temperature: float = 3.0):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        logits = model(x)
        if soft is None:
            loss = (torch.nn.functional.cross_entropy(logits, y, reduction="none") * weights).mean()
        else:
            log_p = torch.nn.functional.log_softmax(logits / temperature, dim=1)
            loss = torch.nn.functional.kl_div(log_p, soft, reduction="batchmean") * temperature**2
        loss.backward()
        opt.step()
    return model


def export_int8(student, features: list[int], out: Path, film_offset: int | None = 0) -> None:
    """Quantise a float student whose input is the feature subset (int8 / 127)."""
    layers = []
    in_scale = 127.0  # int8 feature = real value * 127
    linear = [m for m in student if hasattr(m, "weight")]
    for li, lin in enumerate(linear):
        w = lin.weight.detach().cpu().numpy()
        b = lin.bias.detach().cpu().numpy()
        if li == 0:
            # Expand to the full schema width so the bot can feed its whole feature vector.
            full = np.zeros((w.shape[0], schema_gen.N_FEATURES), dtype=w.dtype)
            full[:, features] = w
            w = full
        last = li == len(linear) - 1
        out_scale = 1.0 if last else 127.0 / 4.0  # hidden activations assumed within ~[0, 4]
        wq, bq, shift = quant.quantize_layer(w, b, in_scale, out_scale)
        layers.append(quant.Layer(wq, bq, 0 if last else shift, not last, None if last else film_offset))
        film_offset = None if film_offset is None else film_offset + w.shape[0]
        in_scale = out_scale
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(quant.encode(schema_gen.SCHEMA_HASH, layers))


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - GPU box only
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset")
    p.add_argument("--head", choices=["router", "gate"], default="router")
    p.add_argument("--features", help="selection report from ml/features/select.py (default: all)")
    p.add_argument("--hidden", type=int, nargs="*", default=[32, 16])
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    torch = _torch()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = np.load(a.dataset)
    target = data["expert"] if a.head == "router" else data["complex"]
    n_out = 9 if a.head == "router" else 2
    features = (
        json.loads(Path(a.features).read_text())["selected"]
        if a.features
        else list(range(data["X"].shape[1]))
    )
    x_all = torch.tensor(data["X"].astype(np.float32) / 127.0, device=device)
    y = torch.tensor(target, dtype=torch.long, device=device)
    w = torch.tensor(0.5 + data["won"].astype(np.float32), device=device)  # winners count 3x as much

    teacher = mlp(torch, [x_all.shape[1], 256, 256, n_out]).to(device)
    train(teacher, x_all, y, w, a.epochs, 1e-3, torch)
    with torch.no_grad():
        soft = torch.softmax(teacher(x_all) / 3.0, dim=1)
    student = mlp(torch, [len(features), *a.hidden, n_out]).to(device)
    train(student, x_all[:, features], y, w, a.epochs, 3e-3, torch, soft=soft)
    with torch.no_grad():
        acc = (student(x_all[:, features]).argmax(1) == y).float().mean().item()
    out = Path(a.out or f"crates/bc-brain/weights/{a.head}.bin")
    export_int8(student.cpu(), features, out, film_offset=0 if a.head == "router" else None)
    print(
        json.dumps(
            {
                "head": a.head,
                "student_acc": round(acc, 3),
                "params": sum(p.numel() for p in student.parameters()),
                "out": str(out),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
