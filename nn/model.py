"""Teacher networks (torch) and the BCN1 export the bot reads (crates/bc-brain/src/nn.rs).

Architectures share one interface: inputs [n, spec.INPUT] → logits [n, sum(HEADS)]; the value
network is the same body with one output. v1: MLP. Phase 4 adds GRU / LSTM / SSM / transformer
behind the same `make(arch, width)`.
"""

from __future__ import annotations

import struct

import numpy as np
import torch
from torch import nn

from nn import spec


class MLP(nn.Module):
    def __init__(self, n_in: int, width: int, n_out: int, depth: int = 2):
        super().__init__()
        layers: list[nn.Module] = []
        d = n_in
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.ReLU()]
            d = width
        layers.append(nn.Linear(d, n_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def make(arch: str, width: int, n_out: int) -> nn.Module:
    if arch == "mlp":
        return MLP(spec.INPUT, width, n_out)
    raise ValueError(f"unknown arch {arch}")


def export_bcn1(model: nn.Module, path) -> None:
    """Write an MLP as a BCN1 blob (f32, little endian)."""
    if not isinstance(model, MLP):
        raise TypeError("only MLPs export to the bot")
    lin = [m for m in model.net if isinstance(m, nn.Linear)]
    out = bytearray(b"BCN1")
    out += struct.pack("<II", spec.SPEC, len(lin))
    for i, l in enumerate(lin):
        w = l.weight.detach().float().cpu().numpy().astype("<f4")
        b = l.bias.detach().float().cpu().numpy().astype("<f4")
        out += struct.pack("<III", w.shape[1], w.shape[0], int(i + 1 < len(lin)))
        out += w.tobytes(order="C") + b.tobytes()
    out += struct.pack("<I", len(spec.HEADS))
    out += struct.pack(f"<{len(spec.HEADS)}I", *spec.HEADS)
    with open(path, "wb") as f:
        f.write(bytes(out))


def head_logp(logits: torch.Tensor, masks: torch.Tensor, acts: torch.Tensor):
    """Per head: masked log-softmax; returns (sum of chosen log-probs over used heads, entropy
    sum, per-head log-softmax list). Heads with action -1 are unused for that sample."""
    lp_sum = torch.zeros(logits.shape[0], device=logits.device)
    ent = torch.zeros_like(lp_sum)
    lsms = []
    start = 0
    for h, size in enumerate(spec.HEADS):
        z = logits[:, start : start + size]
        allowed = ((masks[:, h : h + 1] >> torch.arange(size, device=z.device)) & 1).bool()
        z = z.masked_fill(~allowed, -1e9)
        lsm = torch.log_softmax(z, dim=1)
        lsms.append(lsm)
        a = acts[:, h]
        used = a >= 0
        chosen = lsm.gather(1, a.clamp(min=0).unsqueeze(1)).squeeze(1)
        lp_sum = lp_sum + torch.where(used, chosen, torch.zeros_like(chosen))
        p = lsm.exp()
        h_ent = -(p * lsm.masked_fill(~allowed, 0.0)).sum(1)
        ent = ent + torch.where(used, h_ent, torch.zeros_like(h_ent))
        start += size
    return lp_sum, ent, lsms


def np_forward(model: MLP, x: np.ndarray) -> np.ndarray:
    """Reference forward in numpy (matches the bot's f32 loop up to summation order)."""
    h = x
    lin = [m for m in model.net if isinstance(m, nn.Linear)]
    for i, l in enumerate(lin):
        h = h @ l.weight.detach().cpu().numpy().T + l.bias.detach().cpu().numpy()
        if i + 1 < len(lin):
            h = np.maximum(h, 0)
    return h
