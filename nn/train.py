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


def load_nets(init: Path | None, arch: str, width: int, rank: int = 0):
    """Policy (first layer factorised to `rank` if > 0) and value network (always full: it
    never runs in the bot)."""
    pol = model.make(arch, width, sum(spec.HEADS), rank).to(DEV)
    val = model.make(arch, width, 1).to(DEV)
    meta = {"arch": arch, "width": width, "rank": rank}
    if init:
        meta = json.loads((init / "meta.json").read_text())
        pol = model.make(meta["arch"], meta["width"], sum(spec.HEADS), meta.get("rank", 0)).to(DEV)
        val = model.make(meta["arch"], meta["width"], 1).to(DEV)
        pol.load_state_dict(torch.load(init / "policy.pt", map_location=DEV))
        val.load_state_dict(torch.load(init / "value.pt", map_location=DEV))
    return pol, val, meta


def save(out: Path, pol, val, meta: dict) -> None:
    """Write into <out>.tmp, flush every file to disk, then rename: a power cut mid-save leaves
    the previous checkpoint intact instead of a zero-filled one (seen 2026-10-23)."""
    import os
    import shutil

    tmp = out.with_name(out.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    torch.save(pol.state_dict(), tmp / "policy.pt")
    torch.save(val.state_dict(), tmp / "value.pt")
    (tmp / "meta.json").write_text(json.dumps(meta))
    model.export_bcn1(pol, tmp / "teacher.bin")
    for f in tmp.iterdir():
        with open(f, "rb+") as h:
            os.fsync(h.fileno())
    shutil.rmtree(out, ignore_errors=True)
    tmp.rename(out)


#: Imitation: the rules' fights go through the executor (step "auto"); copying their moves
#: through the manual heads lost ~120 gems a game (2026-10-23). Manual control stays available
#: with a little probability (label smoothing) so PPO tries it and keeps it where it wins.
STEP_SMOOTH = 0.1


def bc(a) -> dict:
    d = data.load([Path(p) for p in a.data])
    d["A"][:, 1] = 0
    d["A"][:, 2:] = -1
    pol, val, meta = load_nets(Path(a.init) if a.init else None, a.arch, a.width, a.rank)
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
            lp, _, lsms = model.head_logp(pol(b.x(idx)), b.M[idx], b.A[idx])
            allowed = ((b.M[idx, 1:2] >> torch.arange(spec.HEADS[1], device=DEV)) & 1).bool()
            manual = allowed.clone()
            manual[:, 0] = False
            n_manual = manual.sum(1)
            smooth = (lsms[1] * manual).sum(1) / n_manual.clamp(min=1)
            has = (n_manual > 0).float()
            loss = -lp.mean() - STEP_SMOOTH * (has * (smooth - lsms[1][:, 0])).mean()
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
    # Labels that differ from the rules' suggestion (nn/expert.py relabels): does the policy adopt them?
    pol.eval()
    hit = tot = 0
    with torch.no_grad():
        for idx in torch.arange(b.n, device=DEV).split(8192):
            x = b.x(idx)
            lab = b.A[idx, 0]
            off = lab != x[:, spec.RULE_GOAL : spec.RULE_GOAL + spec.HEADS[0]].argmax(1)
            if off.any():
                _, _, lsms = model.head_logp(pol(x), b.M[idx], b.A[idx])
                hit += ((lsms[0].argmax(1) == lab) & off).sum().item()
                tot += off.sum().item()
    off_rule = {"rows": int(tot), "adopted": round(hit / tot, 4) if tot else None}
    # Value warm start: Monte Carlo-ish returns (GAE with V = 0) of the rules' games.
    ret, _ = returns_and_adv(d["R"], d["DONE"], d["TRAJ"], np.zeros_like(d["R"]))
    vopt = torch.optim.AdamW(val.parameters(), lr=a.lr, weight_decay=0.01)
    vloss = fit_value(val, vopt, b, ret.astype(np.float32), 3, a.batch)
    meta.update({"stage": "bc", "samples": int(b.n)})
    save(Path(a.out), pol, val, meta)
    return {"mode": "bc", "samples": int(b.n), "epochs": len(hist), "held_out": hist, "value_loss": vloss,
            "off_rule_labels": off_rule}


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


def anchor_kl(logits: torch.Tensor, ref_logits: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """Per-sample KL(policy || anchor) over the goal and step heads (always used)."""
    acts = torch.zeros(len(logits), len(spec.HEADS), dtype=torch.long, device=logits.device)
    _, _, lp = model.head_logp(logits, masks, acts)
    _, _, lr = model.head_logp(ref_logits, masks, acts)
    kl = torch.zeros(len(logits), device=logits.device)
    for h in (0, 1):
        p = lp[h].exp()
        kl = kl + (p * (lp[h] - lr[h])).clamp(min=0.0).sum(1)
    return kl


def ppo(a) -> dict:
    d = data.load([Path(p) for p in a.data])
    pol, val, meta = load_nets(Path(a.init), a.arch, a.width)
    # Anchor (2026-10-23): without it PPO drifted away from the imitation policy's goals at random
    # (goal agreement 100% -> 91%, win rate vs the rules 0.50 -> 0.29 in 16 iterations). The
    # anchor's weight adapts so the KL to the imitation policy stays near --anchor-target.
    anchor = None
    beta = float(meta.get("anchor_beta", a.anchor_beta))
    if a.anchor:
        anchor, _, _ = load_nets(Path(a.anchor), a.arch, a.width)
        anchor.eval()
    b = Batches(d)
    v = values(val, b)
    ret, adv = returns_and_adv(d["R"], d["DONE"], d["TRAJ"], v)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    adv_t = torch.from_numpy(adv.astype(np.float32)).to(DEV)
    old_lp = torch.from_numpy(d["LP"]).to(DEV)
    opt = torch.optim.AdamW(pol.parameters(), lr=a.lr, weight_decay=0.0)
    kls, clipfrac, ents, akls, steps = [], [], [], [], 0
    stop = False
    for _ in range(a.epochs):
        for idx in torch.randperm(b.n, device=DEV).split(a.batch):
            x = b.x(idx)
            logits = pol(x)
            lp, ent, _ = model.head_logp(logits, b.M[idx], b.A[idx])
            ratio = torch.exp(lp - old_lp[idx])
            s1 = ratio * adv_t[idx]
            s2 = ratio.clamp(1 - a.clip, 1 + a.clip) * adv_t[idx]
            loss = -torch.min(s1, s2).mean() - a.entropy * ent.mean()
            if anchor is not None:
                with torch.no_grad():
                    ref = anchor(x)
                akl = anchor_kl(logits, ref, b.M[idx]).mean()
                loss = loss + beta * akl
                akls.append(akl.item())
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
    akl_mean = float(np.mean(akls[-20:])) if akls else 0.0
    if anchor is not None:
        if akl_mean > 1.5 * a.anchor_target:
            beta = min(beta * 2, 100.0)
        elif akl_mean < a.anchor_target / 1.5:
            beta = max(beta / 2, 1e-3)
        meta["anchor_beta"] = beta
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
        "anchor_kl": round(akl_mean, 5),
        "anchor_beta": round(beta, 4),
    }


def soft_kl(logits: torch.Tensor, t_logits: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """Per-sample KL(teacher || student): goal and step heads, plus attack and cast weighted by
    the teacher's probability of a manual step (they only act then)."""
    acts = torch.zeros(len(logits), len(spec.HEADS), dtype=torch.long, device=logits.device)
    _, _, ls = model.head_logp(logits, masks, acts)
    _, _, lt = model.head_logp(t_logits, masks, acts)
    manual = 1.0 - lt[1][:, 0].exp()
    kl = torch.zeros(len(logits), device=logits.device)
    for h in range(len(spec.HEADS)):
        pt = lt[h].exp()
        k = (pt * (lt[h] - ls[h])).clamp(min=0.0).sum(1)
        kl = kl + (k if h < 2 else manual * k)
    return kl


def fake_int8(pol) -> None:
    """Round the policy's weights to per-row int8 in place (evaluation of int8 inference)."""
    with torch.no_grad():
        for m in pol.modules():
            if isinstance(m, torch.nn.Linear):
                s = m.weight.abs().amax(1, keepdim=True).clamp(min=1e-8) / 127
                m.weight.copy_((m.weight / s).round().clamp(-127, 127) * s)


def plasticity_stats(pol, b: "Batches", idx: torch.Tensor) -> dict:
    """Dead ReLU units (never active on these states) and the effective rank of the last hidden
    layer's features: both fall when a network loses the ability to learn new things."""
    feats = []

    def hook(_m, _i, out):
        feats.append(out.detach())

    relus = [m for m in pol.net if isinstance(m, torch.nn.ReLU)]
    h = relus[-1].register_forward_hook(hook)
    with torch.no_grad():
        for i in idx.split(8192):
            pol(b.x(i))
    h.remove()
    f = torch.cat(feats)
    dead = float(((f > 0).sum(0) == 0).float().mean())
    sv = torch.linalg.svdvals(f[:20000] - f[:20000].mean(0))
    p = sv / sv.sum().clamp(min=1e-12)
    erank = float(torch.exp(-(p * p.clamp(min=1e-12).log()).sum()))
    return {"dead_units": round(dead, 4), "effective_rank": round(erank, 1), "units": int(f.shape[1])}


def compare(pol, teacher, b: "Batches", idx: torch.Tensor) -> dict:
    """Agreement of argmax picks with the teacher (goal; step; attack/cast where the teacher
    steps manually) and the mean KL."""
    agree = np.zeros(4)
    tot = np.zeros(4)
    kl_sum = 0.0
    off_hit = off_tot = 0  # states where the teacher overrules the rules' suggestion
    rule_at = spec.RULE_GOAL
    with torch.no_grad():
        for i in idx.split(8192):
            x = b.x(i)
            acts = torch.zeros(len(i), len(spec.HEADS), dtype=torch.long, device=DEV)
            ls_, lt_ = pol(x), teacher(x)
            _, _, ls = model.head_logp(ls_, b.M[i], acts)
            _, _, lt = model.head_logp(lt_, b.M[i], acts)
            manual = lt[1].argmax(1) > 0
            tg = lt[0].argmax(1)
            off = tg != x[:, rule_at : rule_at + spec.HEADS[0]].argmax(1)
            off_hit += ((ls[0].argmax(1) == tg) & off).sum().item()
            off_tot += off.sum().item()
            for h in range(4):
                use = torch.ones_like(manual) if h < 2 else manual
                agree[h] += ((ls[h].argmax(1) == lt[h].argmax(1)) & use).sum().item()
                tot[h] += use.sum().item()
            kl_sum += soft_kl(ls_, lt_, b.M[i]).sum().item()
    out = {n: (round(float(agree[h] / tot[h]), 5) if tot[h] else None) for h, n in enumerate(spec.HEAD_NAMES)}
    out["kl"] = round(kl_sum / len(idx), 6)
    out["off_rules_states"] = int(off_tot)
    out["goal_off_rules"] = round(off_hit / off_tot, 4) if off_tot else None
    return out


def distill(a) -> dict:
    """Student (width/rank of the arguments) trained to match the teacher's action probabilities
    on the given states; the held-out tenth of the games measures fidelity."""
    d = data.load([Path(p) for p in a.data])
    teacher, _, tmeta = load_nets(Path(a.teacher), "mlp", 512)
    teacher.eval()
    b = Batches(d)
    held = (d["TRAJ"] % 10) == 0
    tr_idx = torch.from_numpy(np.flatnonzero(~held)).to(DEV)
    te_idx = torch.from_numpy(np.flatnonzero(held)).to(DEV)
    if a.int8_only:
        pol = teacher
        ref, _, _ = load_nets(Path(a.teacher), "mlp", 512)
        ref.eval()
        fake_int8(pol)
        return {"mode": "distill", "candidate": "teacher int8", **compare(pol, ref, b, te_idx)}
    pol = model.make("mlp", a.width, sum(spec.HEADS), a.rank).to(DEV)
    opt = torch.optim.AdamW(pol.parameters(), lr=a.lr, weight_decay=0.0)
    hist = []
    for _ in range(a.epochs):
        pol.train()
        for idx in tr_idx[torch.randperm(len(tr_idx), device=DEV)].split(a.batch):
            x = b.x(idx)
            with torch.no_grad():
                t = teacher(x)
            loss = soft_kl(pol(x), t, b.M[idx]).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(pol.parameters(), 1.0)
            opt.step()
        pol.eval()
        hist.append(compare(pol, teacher, b, te_idx))
        if len(hist) > 2 and hist[-1]["kl"] > min(h["kl"] for h in hist[:-1]) * 0.97:
            break
    m = hist[-1]
    lin = [l for l in pol.net if isinstance(l, torch.nn.Linear)]
    macs = sum(l.in_features * l.out_features for l in lin)
    meta = {"arch": "mlp", "width": a.width, "rank": a.rank, "stage": "distill", "teacher": str(a.teacher)}
    if a.out:
        _, val, _ = load_nets(Path(a.teacher), "mlp", 512)
        save(Path(a.out), pol, val, meta)
    return {"mode": "distill", "candidate": f"w{a.width} r{a.rank}", "macs": macs, "epochs": len(hist),
            **m, **plasticity_stats(pol, b, te_idx[:40000])}


def factorize(a) -> dict:
    """Turn a checkpoint's policy into a rank-`a.rank` one (the value network stays full)."""
    pol, val, meta = load_nets(Path(a.init), a.arch, a.width)
    if meta.get("rank", 0):
        raise SystemExit(f"{a.init} is already rank {meta['rank']}")
    small = model.factorize(pol, a.rank)
    meta["rank"] = a.rank
    save(Path(a.out), small, val, meta)
    return {"mode": "factorize", "rank": a.rank}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["bc", "ppo", "factorize", "distill"])
    p.add_argument("--data", nargs="+", default=[])
    p.add_argument("--out", default="")
    p.add_argument("--init")
    p.add_argument("--arch", default="mlp")
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch", type=int, default=4096)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--entropy", type=float, default=0.003)
    p.add_argument("--target-kl", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--teacher", help="distill: the teacher checkpoint")
    p.add_argument("--int8-only", action="store_true", help="distill: score the teacher with int8 weights")
    p.add_argument("--anchor", help="checkpoint whose policy PPO stays close to (KL penalty)")
    p.add_argument("--anchor-target", type=float, default=0.03)
    p.add_argument("--anchor-beta", type=float, default=0.5)
    a = p.parse_args(argv)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    if a.lr is None:
        a.lr = 1e-3 if a.mode in ("bc", "distill") else 1e-4
    if a.epochs is None:
        a.epochs = {"bc": 30, "distill": 12}.get(a.mode, 4)
    t = time.time()
    m = {"bc": bc, "ppo": ppo, "factorize": factorize, "distill": distill}[a.mode](a)
    m["seconds"] = round(time.time() - t, 1)
    print(json.dumps(m))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
