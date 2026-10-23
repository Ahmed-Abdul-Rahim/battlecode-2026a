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
    """`rank` > 0 factorises the first layer (n_in -> rank, no activation, then rank -> width):
    the first layer is ~80% of the work in the bot, and rank 64 kept 99.95% of the goal choices
    of a full 512 teacher (2026-10-23)."""

    def __init__(self, n_in: int, width: int, n_out: int, depth: int = 2, rank: int = 0):
        super().__init__()
        self.rank = rank
        layers: list[nn.Module] = []
        d = n_in
        if rank:
            layers.append(nn.Linear(n_in, rank, bias=False))
            d = rank
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.ReLU()]
            d = width
        layers.append(nn.Linear(d, n_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def make(arch: str, width: int, n_out: int, rank: int = 0) -> nn.Module:
    if arch == "mlp":
        return MLP(spec.INPUT, width, n_out, rank=rank)
    raise ValueError(f"unknown arch {arch}")


def export_bcn1(model: nn.Module, path) -> None:
    """Write an MLP as a BCN1 blob (f32, little endian)."""
    if not isinstance(model, MLP):
        raise TypeError("only MLPs export to the bot")
    mods = list(model.net)
    lin = [(m, i + 1 < len(mods) and isinstance(mods[i + 1], nn.ReLU)) for i, m in enumerate(mods) if isinstance(m, nn.Linear)]
    out = bytearray(b"BCN1")
    out += struct.pack("<II", spec.SPEC, len(lin))
    for l, relu in lin:
        w = l.weight.detach().float().cpu().numpy().astype("<f4")
        b = (l.bias.detach().float().cpu().numpy() if l.bias is not None else np.zeros(w.shape[0])).astype("<f4")
        out += struct.pack("<III", w.shape[1], w.shape[0], int(relu))
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
    mods = list(model.net)
    for i, m in enumerate(mods):
        if isinstance(m, nn.Linear):
            h = h @ m.weight.detach().cpu().numpy().T
            if m.bias is not None:
                h = h + m.bias.detach().cpu().numpy()
        elif isinstance(m, nn.ReLU):
            h = np.maximum(h, 0)
    return h


def factorize(full: MLP, rank: int) -> MLP:
    """A rank-`rank` copy of a full MLP: truncated SVD of the first layer, the rest copied."""
    lin = [m for m in full.net if isinstance(m, nn.Linear)]
    w0 = lin[0].weight.detach()
    u, s, vt = torch.linalg.svd(w0.float(), full_matrices=False)
    out = MLP(lin[0].in_features, lin[0].out_features, lin[-1].out_features, depth=len(lin) - 1, rank=rank)
    out = out.to(w0.device)
    new = [m for m in out.net if isinstance(m, nn.Linear)]
    with torch.no_grad():
        new[0].weight.copy_(vt[:rank])
        new[1].weight.copy_(u[:, :rank] * s[:rank])
        new[1].bias.copy_(lin[0].bias)
        for a, b in zip(new[2:], lin[1:]):
            a.weight.copy_(b.weight)
            a.bias.copy_(b.bias)
    return out
