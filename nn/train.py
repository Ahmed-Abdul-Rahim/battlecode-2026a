"""Teacher training on the GPU (runs in the torch venv; nn/run.py calls it).

    python nn/train.py bc  --data <shards...> --out <ckpt dir> [--arch mlp --width 512]
    python nn/train.py ppo --data <shards...> --init <ckpt dir> --out <ckpt dir>

`bc` imitates the rules (logged with TEACHER=1) and fits the value network to the returns.
`ppo` is one PPO update on rollouts the current teacher played (TEACHER=2, sampling).
Each run writes <out>/policy.pt, value.pt, teacher.bin (the bot's BCN1 blob) and prints one
JSON line of metrics.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nn import data, model, spec  # noqa: E402

GAMMA, LAMBDA = 0.999, 0.95
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Batches:
    """Samples kept compact on the device (int8 scalars, packed grid bits); inputs are built
    per minibatch, in the same order as the bot."""

    def __init__(self, d: dict[str, np.ndarray]):
        self.n = len(d["S"])
        self.S = torch.from_numpy(d["S"]).to(DEV)
        self.G = torch.from_numpy(d["G"]).to(DEV)
        self.PREV = torch.from_numpy(d["PREV"]).to(DEV)
        self.A = torch.from_numpy(d["A"]).to(DEV)
        self.M = torch.from_numpy(d["M"]).to(DEV)
        self.shifts = torch.arange(8, device=DEV, dtype=torch.uint8)

    def x(self, idx: torch.Tensor) -> torch.Tensor:
        parts = [self.S[idx].float() / 127.0]
        for k in range(spec.HIST):
            p = self.PREV[idx, k]
            prev = self.S[p.clamp(min=0)].float() / 127.0
            parts.append(prev * (p >= 0).float().unsqueeze(1))
        bits = (self.G[idx].unsqueeze(2) >> self.shifts) & 1
        parts.append(bits.reshape(len(idx), -1)[:, : spec.GRID_BITS].float())
        return torch.cat(parts, 1)


def returns_and_adv(r: np.ndarray, done: np.ndarray, traj: np.ndarray, v: np.ndarray):
    """GAE per trajectory (rows of one trajectory are consecutive, in time order)."""
    adv = np.zeros_like(r)
    last = 0.0
    for i in range(len(r) - 1, -1, -1):
        end = done[i] or i + 1 == len(r) or traj[i + 1] != traj[i]
        nv = 0.0 if end else v[i + 1]
        if end:
            last = 0.0
        delta = r[i] + GAMMA * nv - v[i]
        last = delta + GAMMA * LAMBDA * last
        adv[i] = last
    return adv + v, adv


def values(vnet, b: Batches, bs: int = 8192) -> np.ndarray:
    out = []
    with torch.no_grad():
        for s in range(0, b.n, bs):
            idx = torch.arange(s, min(s + bs, b.n), device=DEV)
            out.append(vnet(b.x(idx)).squeeze(1).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, np.float32)


def fit_value(vnet, opt, b: Batches, target: np.ndarray, epochs: int, bs: int) -> float:
    t = torch.from_numpy(target).to(DEV)
    loss_sum, k = 0.0, 0
    for _ in range(epochs):
        for idx in torch.randperm(b.n, device=DEV).split(bs):
            loss = F.smooth_l1_loss(vnet(b.x(idx)).squeeze(1), t[idx])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vnet.parameters(), 1.0)
            opt.step()
            loss_sum, k = loss_sum + loss.item(), k + 1
    return loss_sum / max(k, 1)


def load_nets(init: Path | None, arch: str, width: int):
    pol = model.make(arch, width, sum(spec.HEADS)).to(DEV)
    val = model.make(arch, width, 1).to(DEV)
    meta = {"arch": arch, "width": width}
    if init:
        meta = json.loads((init / "meta.json").read_text())
        pol = model.make(meta["arch"], meta["width"], sum(spec.HEADS)).to(DEV)
        val = model.make(meta["arch"], meta["width"], 1).to(DEV)
        pol.load_state_dict(torch.load(init / "policy.pt", map_location=DEV))
        val.load_state_dict(torch.load(init / "value.pt", map_location=DEV))
    return pol, val, meta


def save(out: Path, pol, val, meta: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    torch.save(pol.state_dict(), out / "policy.pt")
    torch.save(val.state_dict(), out / "value.pt")
    (out / "meta.json").write_text(json.dumps(meta))
    model.export_bcn1(pol, out / "teacher.bin")


def bc(a) -> dict:
    d = data.load([Path(p) for p in a.data])
    pol, val, meta = load_nets(Path(a.init) if a.init else None, a.arch, a.width)
    traj = d["TRAJ"]
    held = (traj % 10) == 0  # every 10th champion-game held out
    b = Batches(d)
    tr = torch.from_numpy(np.flatnonzero(~held)).to(DEV)
    te = torch.from_numpy(np.flatnonzero(held)).to(DEV)
    opt = torch.optim.AdamW(pol.parameters(), lr=a.lr, weight_decay=0.01)
    hist = []
    best = (-1.0, None)
    for ep in range(a.epochs):
        pol.train()
        for idx in tr[torch.randperm(len(tr), device=DEV)].split(a.batch):
            lp, _, _ = model.head_logp(pol(b.x(idx)), b.M[idx], b.A[idx])
            loss = -lp.mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(pol.parameters(), 1.0)
            opt.step()
        acc = accuracy(pol, b, te)
        hist.append(acc)
        if acc["goal"] > best[0]:
            best = (acc["goal"], {k: v.detach().clone() for k, v in pol.state_dict().items()})
        # Plateau: no goal-accuracy gain over the last 3 epochs.
        if len(hist) > 3 and max(h["goal"] for h in hist[-3:]) <= max(h["goal"] for h in hist[:-3]) + 1e-3:
            break
    pol.load_state_dict(best[1])
    # Value warm start: Monte Carlo-ish returns (GAE with V = 0) of the rules' games.
    ret, _ = returns_and_adv(d["R"], d["DONE"], d["TRAJ"], np.zeros_like(d["R"]))
    vopt = torch.optim.AdamW(val.parameters(), lr=a.lr, weight_decay=0.01)
    vloss = fit_value(val, vopt, b, ret.astype(np.float32), 3, a.batch)
    meta.update({"stage": "bc", "samples": int(b.n)})
    save(Path(a.out), pol, val, meta)
    return {"mode": "bc", "samples": int(b.n), "epochs": len(hist), "held_out": hist, "value_loss": vloss}


def accuracy(pol, b: Batches, idx_all: torch.Tensor) -> dict:
    pol.eval()
    hits = np.zeros(len(spec.HEADS))
    tot = np.zeros(len(spec.HEADS))
    with torch.no_grad():
        for idx in idx_all.split(8192):
            _, _, lsms = model.head_logp(pol(b.x(idx)), b.M[idx], b.A[idx])
            for h, lsm in enumerate(lsms):
                a = b.A[idx, h]
                used = a >= 0
                hits[h] += ((lsm.argmax(1) == a) & used).sum().item()
                tot[h] += used.sum().item()
    return {n: round(float(hits[h] / tot[h]), 4) if tot[h] else None for h, n in enumerate(spec.HEAD_NAMES)}


def ppo(a) -> dict:
    d = data.load([Path(p) for p in a.data])
    pol, val, meta = load_nets(Path(a.init), a.arch, a.width)
    b = Batches(d)
    v = values(val, b)
    ret, adv = returns_and_adv(d["R"], d["DONE"], d["TRAJ"], v)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    adv_t = torch.from_numpy(adv.astype(np.float32)).to(DEV)
    old_lp = torch.from_numpy(d["LP"]).to(DEV)
    opt = torch.optim.AdamW(pol.parameters(), lr=a.lr, weight_decay=0.0)
    kls, clipfrac, ents, steps = [], [], [], 0
    stop = False
    for _ in range(a.epochs):
        for idx in torch.randperm(b.n, device=DEV).split(a.batch):
            lp, ent, _ = model.head_logp(pol(b.x(idx)), b.M[idx], b.A[idx])
            ratio = torch.exp(lp - old_lp[idx])
            s1 = ratio * adv_t[idx]
            s2 = ratio.clamp(1 - a.clip, 1 + a.clip) * adv_t[idx]
            loss = -torch.min(s1, s2).mean() - a.entropy * ent.mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(pol.parameters(), 0.5)
            opt.step()
            steps += 1
            with torch.no_grad():
                kl = (old_lp[idx] - lp).mean().item()
                kls.append(kl)
                clipfrac.append(((ratio - 1).abs() > a.clip).float().mean().item())
                ents.append(ent.mean().item())
            if kl > a.target_kl:
                stop = True
                break
        if stop:
            break
    vopt = torch.optim.AdamW(val.parameters(), lr=a.lr, weight_decay=0.01)
    vloss = fit_value(val, vopt, b, ret.astype(np.float32), 2, a.batch)
    meta.update({"stage": "ppo", "samples": int(b.n)})
    save(Path(a.out), pol, val, meta)
    return {
        "mode": "ppo",
        "samples": int(b.n),
        "steps": steps,
        "kl": round(float(np.mean(kls)), 5) if kls else 0.0,
        "clipfrac": round(float(np.mean(clipfrac)), 4) if clipfrac else 0.0,
        "entropy": round(float(np.mean(ents)), 4) if ents else 0.0,
        "early_stop": stop,
        "value_loss": round(vloss, 5),
        "reward_mean": round(float(d["R"].mean()), 5),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["bc", "ppo"])
    p.add_argument("--data", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--init")
    p.add_argument("--arch", default="mlp")
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch", type=int, default=4096)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--entropy", type=float, default=0.003)
    p.add_argument("--target-kl", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    if a.lr is None:
        a.lr = 1e-3 if a.mode == "bc" else 1e-4
    if a.epochs is None:
        a.epochs = 30 if a.mode == "bc" else 4
    t = time.time()
    m = bc(a) if a.mode == "bc" else ppo(a)
    m["seconds"] = round(time.time() - t, 1)
    print(json.dumps(m))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
