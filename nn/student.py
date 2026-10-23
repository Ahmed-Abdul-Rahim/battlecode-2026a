"""Distil the teacher into the shipped student: int8 BCW1, goal head only (GPU venv).

    python nn/student.py --teacher <ckpt> --data <shards...> --out <dir> [--hidden 16]

The student reads the first STUDENT_IN int8 scalars the bot already computes (teacher.rs
STUDENT_IN: self, economy, allies, enemies, camps, goal slots, the rules' suggestion) and
picks the goal; fighting stays with the executor (the teacher's step is "auto"). Trained on the
teacher's goal probabilities (KL), quantised with ml/distill/quant.py, and scored with its
bit-exact mirror of the bot's int8 kernel. Writes <out>/student.bin and prints JSON metrics.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.distill import quant  # noqa: E402
from nn import data, model, spec  # noqa: E402

_sp = importlib.util.spec_from_file_location("nntrain", ROOT / "nn" / "train.py")
T = importlib.util.module_from_spec(_sp)
_sp.loader.exec_module(T)

STUDENT_IN = 256
SCHEMA = (spec.SPEC & 0xFFFF) ^ 0x5354
G = spec.HEADS[0]


def goal_mask(m: torch.Tensor) -> torch.Tensor:
    return ((m[:, 0:1] >> torch.arange(G, device=m.device)) & 1).bool()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", required=True)
    p.add_argument("--data", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--hidden", type=int, default=16)
    p.add_argument("--epochs", type=int, default=8)
    a = p.parse_args(argv)
    torch.manual_seed(0)
    dev = T.DEV
    d = data.load([Path(x) for x in a.data])
    b = T.Batches(d)
    teacher, _, _ = T.load_nets(Path(a.teacher), "mlp", 512)
    teacher.eval()
    held = (d["TRAJ"] % 10) == 0
    tr = torch.from_numpy(np.flatnonzero(~held)).to(dev)
    te = torch.from_numpy(np.flatnonzero(held)).to(dev)
    S8 = b.S[:, :STUDENT_IN]  # int8 on the device

    def xs(idx):
        return S8[idx].float() / 127.0

    def t_logp(idx):
        with torch.no_grad():
            z = teacher(b.x(idx))[:, :G].masked_fill(~goal_mask(b.M[idx]), -1e9)
            return torch.log_softmax(z, 1)

    stu = nn.Sequential(nn.Linear(STUDENT_IN, a.hidden), nn.ReLU(), nn.Linear(a.hidden, G)).to(dev)
    opt = torch.optim.AdamW(stu.parameters(), lr=3e-3)
    for _ in range(a.epochs):
        for idx in tr[torch.randperm(len(tr), device=dev)].split(4096):
            lt = t_logp(idx)
            ls = torch.log_softmax(stu(xs(idx)).masked_fill(~goal_mask(b.M[idx]), -1e9), 1)
            loss = (lt.exp() * (lt - ls)).clamp(min=0).sum(1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

    # ---- int8: calibrate the hidden scale on training states, then quantise ----
    l1, l2 = stu[0], stu[2]
    with torch.no_grad():
        h = torch.relu(l1(xs(tr[:200000]))).cpu().numpy()
    h_max = float(np.percentile(h, 99.9)) or 1.0
    w1, b1 = l1.weight.detach().cpu().numpy(), l1.bias.detach().cpu().numpy()
    w2, b2 = l2.weight.detach().cpu().numpy(), l2.bias.detach().cpu().numpy()
    w1q, b1q, s1 = quant.quantize_layer(w1, b1, 127.0, 127.0 / h_max)
    hidden_scale = (127.0 / max(float(np.abs(w1).max()), 1e-8)) * 127.0 / 2**s1  # int8 units per real
    w2q, b2q, _ = quant.quantize_layer(w2, b2, hidden_scale, 1.0)
    layers = [quant.Layer(w1q, b1q, s1, True, None), quant.Layer(w2q, b2q, 0, False, None)]
    blob = quant.encode(SCHEMA, layers)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "student.bin").write_bytes(blob)

    # ---- scoring on held-out states: teacher, float student, int8 student (bit-exact) ----
    idx = te[:30000]
    with torch.no_grad():
        mask = goal_mask(b.M[idx])
        tg = t_logp(idx).argmax(1).cpu().numpy()
        fg = stu(xs(idx)).masked_fill(~mask, -1e9).argmax(1).cpu().numpy()
    s_np = S8[idx].cpu().numpy().astype(np.int64)
    m_np = mask.cpu().numpy()
    qg = np.array([np.where(m_np[i], quant.forward(layers, s_np[i]).astype(np.int64), -(2**40)).argmax() for i in range(len(s_np))])
    rule = s_np[:, spec.RULE_GOAL : spec.RULE_GOAL + G].argmax(1)
    off = tg != rule
    res = {
        "hidden": a.hidden,
        "macs": STUDENT_IN * a.hidden + a.hidden * G,
        "bytes": len(blob),
        "float_vs_teacher": round(float((fg == tg).mean()), 5),
        "int8_vs_teacher": round(float((qg == tg).mean()), 5),
        "int8_vs_float": round(float((qg == fg).mean()), 5),
        "teacher_overrules_rules": int(off.sum()),
        "int8_vs_teacher_where_overruled": round(float((qg[off] == tg[off]).mean()), 4) if off.any() else None,
        "teacher": str(a.teacher),
    }
    (out / "metrics.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
