"""Architecture comparison: which network type extracts the most signal from the game state?

    python nn/zoo.py [--steps 3000] [--shards N]     (torch venv)

Every model gets the same logged games (whole games held out), the same budget, and the same
task: at each decision, predict whether this champion's team wins (log-loss) and its discounted
return-to-go (the critic's target). A model that predicts clearly better sees something the
others miss, which is what an RL policy needs to beat the rules. Prints a table and writes
runs/nn/zoo.json.

Models: mlp (current step only), mlp_hist (the teacher's input: 3 previous turns stacked),
gru, lstm, ssm (selective diagonal state space, Mamba-style), transformer (causal, over time),
entity (transformer over the units of the current step), encdec (GRU encoder + decoder that
also predicts the next 8 rewards).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nn import data, spec  # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GAMMA = 0.999
L = 64  # sequence chunk length
D = 192  # model width
IN = spec.S + spec.GRID_BITS


class Data:
    def __init__(self, shards: list[Path]):
        d = data.load(shards)
        self.S = torch.from_numpy(d["S"]).to(DEV)
        self.G = torch.from_numpy(d["G"]).to(DEV)
        traj, r, done = d["TRAJ"], d["R"], d["DONE"]
        # per trajectory: rows [start, end), win label (its last reward carries +-3 for the result)
        starts = np.flatnonzero(np.r_[True, traj[1:] != traj[:-1]])
        ends = np.r_[starts[1:], len(traj)]
        ret = np.zeros_like(r)
        win = np.zeros_like(r)
        for a, b in zip(starts, ends):
            g = 0.0
            for i in range(b - 1, a - 1, -1):
                g = r[i] + GAMMA * g
                ret[i] = g
            win[a:b] = float(r[b - 1] > 0)
        self.ret = torch.from_numpy(ret).to(DEV)
        self.win = torch.from_numpy(win).to(DEV)
        self.frac = torch.zeros(len(traj), device=DEV)  # position within the game (0..1)
        for a, b in zip(starts, ends):
            self.frac[a:b] = torch.linspace(0, 1, b - a, device=DEV)
        self.r = torch.from_numpy(r).to(DEV)
        held = (traj[starts] % 10) == 0
        self.train = [(a, b) for a, b, h in zip(starts, ends, held) if not h and b - a >= L]
        self.test = [(a, b) for a, b, h in zip(starts, ends, held) if h and b - a >= L]
        self.shifts = torch.arange(8, device=DEV, dtype=torch.uint8)
        self.ret_scale = float(self.ret.std())

    def x(self, rows: torch.Tensor) -> torch.Tensor:
        s = self.S[rows].float() / 127.0
        bits = ((self.G[rows].unsqueeze(-1) >> self.shifts) & 1).flatten(-2)[..., : spec.GRID_BITS].float()
        return torch.cat([s, bits], -1)

    def batch(self, trajs, n: int, g: torch.Generator):
        pick = torch.randint(len(trajs), (n,), generator=g)
        rows = []
        for i in pick.tolist():
            a, b = trajs[i]
            st = a + int(torch.randint(b - a - L + 1, (1,), generator=g))
            rows.append(torch.arange(st, st + L))
        rows = torch.stack(rows).to(DEV)
        return rows, self.x(rows)


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.win = nn.Linear(D, 1)
        self.ret = nn.Linear(D, 1)

    def forward(self, h):
        return self.win(h).squeeze(-1), self.ret(h).squeeze(-1)


def embed() -> nn.Module:
    return nn.Sequential(nn.Linear(IN, D), nn.ReLU(), nn.Linear(D, D), nn.ReLU())


class MLP(nn.Module):
    def __init__(self, hist: int = 0):
        super().__init__()
        self.hist = hist
        self.net = nn.Sequential(nn.Linear(IN + hist * spec.S, D), nn.ReLU(), nn.Linear(D, D), nn.ReLU())
        self.head = Head()

    def forward(self, x):
        if self.hist:
            s = x[..., : spec.S]
            prev = [F.pad(s, (0, 0, k, 0))[:, :-k] for k in range(1, self.hist + 1)]
            x = torch.cat([x, *prev], -1)
        return self.head(self.net(x))


class RNN(nn.Module):
    def __init__(self, cell: str):
        super().__init__()
        self.emb = embed()
        self.rnn = (nn.GRU if cell == "gru" else nn.LSTM)(D, D, batch_first=True)
        self.head = Head()

    def forward(self, x):
        h, _ = self.rnn(self.emb(x))
        return self.head(h)


class SSM(nn.Module):
    """Selective diagonal state space (Mamba-style): h_t = a_t * h_{t-1} + (1 - a_t) * u_t with
    an input-dependent decay a_t, two layers with residuals."""

    def __init__(self, layers: int = 2):
        super().__init__()
        self.emb = embed()
        self.u = nn.ModuleList(nn.Linear(D, D) for _ in range(layers))
        self.a = nn.ModuleList(nn.Linear(D, D) for _ in range(layers))
        self.o = nn.ModuleList(nn.Linear(D, D) for _ in range(layers))
        self.head = Head()

    def forward(self, x):
        z = self.emb(x)
        for u, a, o in zip(self.u, self.a, self.o):
            uu, aa = u(z), torch.sigmoid(a(z) + 2.0)
            h = torch.zeros_like(uu[:, 0])
            hs = []
            for t in range(z.shape[1]):
                h = aa[:, t] * h + (1 - aa[:, t]) * uu[:, t]
                hs.append(h)
            z = z + F.relu(o(torch.stack(hs, 1)))
        return self.head(z)


class TimeTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = embed()
        self.pos = nn.Parameter(torch.zeros(L, D))
        layer = nn.TransformerEncoderLayer(D, 4, 2 * D, dropout=0.0, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, 2)
        self.head = Head()
        self.register_buffer("mask", torch.triu(torch.full((L, L), float("-inf")), 1))

    def forward(self, x):
        z = self.emb(x) + self.pos[: x.shape[1]]
        return self.head(self.enc(z, mask=self.mask[: x.shape[1], : x.shape[1]]))


# unit token slices of the scalar vector (teacher.rs layout): self(37), allies 2x18, enemies 3x26,
# ghosts 4, camps 4x6, groups 4x13, rule suggestion 15, steps 54; economy+map 10 go to "self"
TOKENS = [(0, 47)] + [(47 + 18 * i, 47 + 18 * (i + 1)) for i in range(2)] + \
    [(83 + 26 * i, 83 + 26 * (i + 1)) for i in range(3)] + [(161, 165)] + \
    [(165 + 6 * i, 165 + 6 * (i + 1)) for i in range(4)] + [(189 + 13 * i, 189 + 13 * (i + 1)) for i in range(4)] + \
    [(241, 256), (256, 310)]


class EntityTransformer(nn.Module):
    """Transformer over the units of the current step (relations between them), plus a GRU over
    time on the pooled result."""

    def __init__(self):
        super().__init__()
        self.tok = nn.ModuleList(nn.Linear(b - a, D) for a, b in TOKENS)
        self.grid = nn.Sequential(nn.Linear(spec.GRID_BITS, D), nn.ReLU())
        self.kind = nn.Parameter(torch.zeros(len(TOKENS) + 1, D))
        layer = nn.TransformerEncoderLayer(D, 4, 2 * D, dropout=0.0, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, 2)
        self.rnn = nn.GRU(D, D, batch_first=True)
        self.head = Head()

    def forward(self, x):
        B, T, _ = x.shape
        toks = [f(x[..., a:b]) for f, (a, b) in zip(self.tok, TOKENS)] + [self.grid(x[..., spec.S :])]
        z = torch.stack(toks, 2) + self.kind  # [B, T, K, D]
        z = self.enc(z.view(B * T, -1, D)).mean(1).view(B, T, D)
        h, _ = self.rnn(z)
        return self.head(h)


class EncDec(nn.Module):
    """GRU encoder over the past; a decoder GRU unrolled 8 steps predicts the next rewards (an
    auxiliary target that forces the state to carry the near future)."""

    K = 8

    def __init__(self):
        super().__init__()
        self.emb = embed()
        self.enc = nn.GRU(D, D, batch_first=True)
        self.dec = nn.GRUCell(1, D)
        self.out = nn.Linear(D, 1)
        self.head = Head()

    def forward(self, x):
        h, _ = self.enc(self.emb(x))
        return self.head(h)

    def future(self, h):
        """Predicted next K rewards from each state h [N, D]."""
        y, preds = torch.zeros(h.shape[0], 1, device=h.device), []
        for _ in range(self.K):
            h = self.dec(y, h)
            y = self.out(h)
            preds.append(y.squeeze(-1))
        return torch.stack(preds, -1)


def make(name: str) -> nn.Module:
    return {
        "mlp": lambda: MLP(0), "mlp_hist": lambda: MLP(spec.HIST), "gru": lambda: RNN("gru"),
        "lstm": lambda: RNN("lstm"), "ssm": SSM, "transformer": TimeTransformer, "entity": EntityTransformer,
        "encdec": EncDec,
    }[name]()


def run(name: str, dd: Data, steps: int, bs: int = 64, seed: int = 0) -> dict:
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    m = make(name).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=0.01)
    t0 = time.time()
    for it in range(steps):
        rows, x = dd.batch(dd.train, bs, g)
        wl, rl = m(x)
        # score only the second half of each chunk: every model gets some context first
        sl = slice(L // 2, None)
        loss = F.binary_cross_entropy_with_logits(wl[:, sl], dd.win[rows][:, sl]) + \
            F.smooth_l1_loss(rl[:, sl], dd.ret[rows][:, sl] / dd.ret_scale)
        if isinstance(m, EncDec):
            h, _ = m.enc(m.emb(x))
            fut = m.future(h[:, sl].reshape(-1, D))
            idx = rows[:, sl].reshape(-1, 1) + torch.arange(1, EncDec.K + 1, device=DEV)
            tgt = dd.r[idx.clamp(max=len(dd.r) - 1)] / dd.ret_scale
            loss = loss + 0.5 * F.smooth_l1_loss(fut, tgt)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
    m.eval()
    out = evaluate(m, dd)
    out.update(model=name, seconds=round(time.time() - t0, 1), params=sum(p.numel() for p in m.parameters()))
    return out


@torch.no_grad()
def evaluate(m, dd: Data, n: int = 400) -> dict:
    g = torch.Generator().manual_seed(123)
    ll, acc, rerr, frac = [], [], [], []
    for _ in range(n // 50):
        rows, x = dd.batch(dd.test, 50, g)
        wl, rl = m(x)
        sl = slice(L // 2, None)
        y = dd.win[rows][:, sl]
        ll.append(F.binary_cross_entropy_with_logits(wl[:, sl], y, reduction="none").flatten())
        acc.append(((wl[:, sl] > 0).float() == y).float().flatten())
        rerr.append((rl[:, sl] - dd.ret[rows][:, sl] / dd.ret_scale).abs().flatten())
        frac.append(dd.frac[rows][:, sl].flatten())
    ll, acc, rerr, frac = map(torch.cat, (ll, acc, rerr, frac))
    out = {"logloss": round(ll.mean().item(), 4), "acc": round(acc.mean().item(), 4),
           "ret_mae": round(rerr.mean().item(), 4)}
    for nm, lo, hi in (("early", 0, 0.33), ("mid", 0.33, 0.66), ("late", 0.66, 1.01)):
        sel = (frac >= lo) & (frac < hi)
        out[f"logloss_{nm}"] = round(ll[sel].mean().item(), 4) if sel.any() else None
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--shards", type=int, default=6)
    p.add_argument("--models", nargs="*", default=["mlp", "mlp_hist", "gru", "lstm", "ssm", "transformer", "entity", "encdec"])
    a = p.parse_args(argv)
    shards = sorted((ROOT / "runs" / "nn" / "data" / "imitation").glob("*.npz"))[: a.shards]
    dd = Data(shards)
    print(f"{len(dd.train)} train / {len(dd.test)} held-out games-champions; base rate win {dd.win.mean().item():.3f}", flush=True)
    base = -(dd.win.mean() * dd.win.mean().log() + (1 - dd.win.mean()) * (1 - dd.win.mean()).log()).item()
    print(f"log-loss of always guessing the base rate: {base:.4f}", flush=True)
    res = []
    for name in a.models:
        r = run(name, dd, a.steps)
        res.append(r)
        print(json.dumps(r), flush=True)
    (ROOT / "runs" / "nn" / "zoo.json").write_text(json.dumps({"base_logloss": base, "results": res}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
