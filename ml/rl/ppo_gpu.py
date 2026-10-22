"""PPO on the GPU with a big teacher network, distilled into the small network the bot runs.

The bot can only afford a 32-unit int8 network per turn, but training doesn't have that limit:
a 512x512 teacher (policy) and a 512x512 value network learn from every rollout on the GPU
(torch, CUDA if present), then the bot's small network is trained to copy the teacher's
probabilities on the same game states and exported exactly like ml/rl/ppo.py does.

All the training maths runs here on the device: advantages (GAE), the PPO update, the value
fit, distillation and iteration 0 (--init). Only the int8 export stays in numpy, because its
integer rounding has to match the bot exactly.

Policy loss:  clipped PPO - ENTROPY * entropy + ANCHOR[arm] * KL(teacher || iteration-0 network)
              with AdamW weight decay (L2) on the teacher and on the distilled student.
Value loss:   Huber (smooth L1), so the odd large return does not dominate the fit.

Run by harness/mljobs.py in a separate environment that has torch:

    <gpu python> ml/rl/ppo_gpu.py --arm 1 --it 7 --store <ml-data worktree>
    <gpu python> ml/rl/ppo_gpu.py --arm 1 --init --store <ml-data worktree>

Prints one JSON line of metrics. Needs only torch + numpy (no repo tooling).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ml.rl import ppo, reward  # noqa: E402  (numpy only)

KEYS = ("X", "A", "LP", "M", "R", "DONE", "TRAJ")
WIDTH = 512
EPOCHS, BATCH, LR, DISTILL_EPOCHS = 6, 1024, 3e-4, 6
#: AdamW decoupled weight decay (shrink per step = lr * WEIGHT_DECAY; 0.01 is torch's default).
WEIGHT_DECAY = 0.01
#: Pull towards the iteration-0 network: arm 1 starts at the rules' choices, so drifting far
#: from them on ~1000 games per iteration is more likely noise than skill. Arm 2 starts random.
ANCHOR = {1: 0.02, 2: 0.0}
INIT_ROWS, INIT_EPOCHS, INIT_LR = 200_000, 4, 3e-3


def read(p: Path) -> dict | None:
    return reward.read_shard(p, KEYS)


def net(n_in: int, n_out: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(n_in, WIDTH), nn.ReLU(), nn.Linear(WIDTH, WIDTH), nn.ReLU(), nn.Linear(WIDTH, n_out)
    )


class Student(nn.Module):
    """The bot's network shape (n_in -> 32 relu -> n_out), initialised from ppo.py params."""

    def __init__(self, p: dict):
        super().__init__()
        self.l1 = nn.Linear(p["w1"].shape[1], p["w1"].shape[0])
        self.l2 = nn.Linear(p["w2"].shape[1], p["w2"].shape[0])
        with torch.no_grad():
            for lin, w, b in ((self.l1, "w1", "b1"), (self.l2, "w2", "b2")):
                lin.weight.copy_(torch.from_numpy(p[w]))
                lin.bias.copy_(torch.from_numpy(p[b]))

    def forward(self, x):
        return self.l2(torch.relu(self.l1(x)))

    def params(self) -> dict:
        return {
            "w1": self.l1.weight.detach().cpu().numpy(),
            "b1": self.l1.bias.detach().cpu().numpy(),
            "w2": self.l2.weight.detach().cpu().numpy(),
            "b2": self.l2.bias.detach().cpu().numpy(),
        }


def head_logp(logits, masks, arm: ppo.Arm):
    """Per head: masked log-softmax (list of tensors)."""
    out = []
    for k, (a, b) in enumerate(arm.heads):
        bits = (masks[:, k : k + 1] >> torch.arange(b - a, device=logits.device)) & 1
        z = logits[:, a:b].masked_fill(bits == 0, -1e9)
        out.append((torch.log_softmax(z, dim=1), bits.bool()))
    return out


def kl_heads(teacher_logits, student_logits, masks, arm):
    """Sum over heads of KL(teacher || student), masked."""
    total = 0.0
    for (lt, ok), (ls, _) in zip(
        head_logp(teacher_logits, masks, arm), head_logp(student_logits, masks, arm), strict=True
    ):
        pt = lt.exp() * ok
        total = total + (pt * (lt - ls) * ok).sum(dim=1).mean()
    return total


def gae(rew, val, done, traj, gamma: float = ppo.GAMMA, lam: float = ppo.LAMBDA):
    """Advantages and returns per trajectory, on the tensors' device; same result as ppo.gae.
    Samples are in decision order within each trajectory. The trajectories are padded into a
    [n_traj, max_len] grid and swept backwards once over time."""
    dev, n = rew.device, len(rew)
    traj = torch.unique(traj, return_inverse=True)[1]  # 0..n_traj-1
    order = torch.sort(traj, stable=True).indices  # grouped by trajectory, order kept
    t_sorted = traj[order]
    counts = torch.bincount(t_sorted)
    starts = torch.cumsum(counts, 0) - counts
    pos = torch.arange(n, device=dev) - starts[t_sorted]
    shape = (len(counts), int(counts.max()) + 1)  # one padding column after the longest
    R = torch.zeros(shape, device=dev)
    V = torch.zeros(shape, device=dev)
    D = torch.ones(shape, dtype=torch.bool, device=dev)  # padding counts as an end
    R[t_sorted, pos], V[t_sorted, pos], D[t_sorted, pos] = rew[order], val[order], done[order]
    valid = torch.zeros(shape, dtype=torch.bool, device=dev)
    valid[t_sorted, pos] = True
    A = torch.zeros(shape, device=dev)
    carry = torch.zeros(shape[0], device=dev)
    for j in range(shape[1] - 2, -1, -1):
        nxt = torch.where(D[:, j] | ~valid[:, j + 1], 0.0, V[:, j + 1])
        delta = R[:, j] + gamma * nxt - V[:, j]
        carry = torch.where(valid[:, j], delta + gamma * lam * torch.where(D[:, j], 0.0, carry), 0.0)
        A[:, j] = carry
    adv = torch.empty(n, device=dev)
    adv[order] = A[t_sorted, pos]
    return adv, adv + val


def batches(n: int, dev):
    return torch.randperm(n, device=dev).split(BATCH)


def export(student: Student, x_int8: np.ndarray, arm: ppo.Arm, base: Path, it: int) -> None:
    params = student.params()
    ppo.save(base / f"policy-{it:04d}.npz", params)
    (base / f"policy-{it:04d}.bin").write_bytes(ppo.export(params, x_int8, arm))


def init(a: argparse.Namespace, arm: ppo.Arm, base: Path, dev) -> dict:
    """Iteration 0. Arm 1 imitates the rules' expert choices (the logged rule decisions in the
    store's data/ shards) on the device; arm 2 (no rule data for its inputs) starts near-uniform."""
    rng = np.random.default_rng(a.arm)
    parts = (
        [reward.read_shard(s, ("X", "y")) for s in sorted((a.store / "data").glob("*.npz"))]
        if a.arm == 1
        else []
    )
    parts = [d for d in parts if d is not None and len(d["y"])]
    if not parts:
        x8 = rng.integers(-127, 128, (2000, arm.n_in)).astype(np.int8)
        student = Student(ppo.init_params(arm.n_in, arm.hidden, arm.n_out, rng))
        export(student, x8, arm, base, 0)
        return {"arm": a.arm, "it": 0, "warm_rows": 0, "device": str(dev)}
    x8, y = np.concatenate([d["X"] for d in parts]), np.concatenate([d["y"] for d in parts])
    keep = rng.permutation(len(x8))[:INIT_ROWS]
    x8, y = x8[keep], y[keep]
    student = Student(ppo.init_params(arm.n_in, arm.hidden, arm.n_out, rng, out_std=0.1)).to(dev)
    x = torch.from_numpy(x8.astype(np.float32) / 127.0).to(dev)
    acts = torch.from_numpy(np.stack([y, np.zeros_like(y)], axis=1).astype(np.int64)).to(dev)
    masks = torch.tensor([(1 << 9) - 1, 0b1], device=dev).expand(len(x), 2)
    opt = torch.optim.Adam(student.parameters(), lr=INIT_LR)
    for _ in range(INIT_EPOCHS):
        for idx in batches(len(x), dev):
            heads = head_logp(student(x[idx]), masks[idx], arm)
            loss = -sum(lsm.gather(1, acts[idx, k : k + 1]).mean() for k, (lsm, _) in enumerate(heads))
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        agree = float((student(x)[:, :9].argmax(1) == acts[:, 0]).float().mean())
    export(student.cpu(), x8, arm, base, 0)
    return {"arm": a.arm, "it": 0, "warm_rows": len(x8), "agree": round(agree, 4), "device": str(dev)}


def update(a: argparse.Namespace, arm: ppo.Arm, base: Path, dev) -> dict:
    it = a.it
    data = reward.concat([d for d in (read(p) for p in sorted(base.glob(f"it{it:04d}-p*.npz"))) if d])
    if not data:
        return {"arm": a.arm, "it": it, "skipped": "no decisions logged"}
    x = torch.from_numpy(data["X"].astype(np.float32) / 127.0).to(dev)
    acts = torch.from_numpy(data["A"]).to(dev)
    masks = torch.from_numpy(data["M"]).to(dev)
    lp_old = torch.from_numpy(data["LP"]).to(dev)
    student = Student(ppo.load(base / f"policy-{it:04d}.npz")).to(dev)

    teacher, value = net(arm.n_in, arm.n_out).to(dev), net(arm.n_in, 1).to(dev)
    tp, vp = base / f"teacher-{it:04d}.pt", base / f"tvalue-{it:04d}.pt"
    if tp.exists():
        teacher.load_state_dict(torch.load(tp, map_location=dev))
        value.load_state_dict(torch.load(vp, map_location=dev))
    else:  # first GPU iteration: the teacher starts where the bot's network is
        opt = torch.optim.Adam(teacher.parameters(), lr=1e-3)
        with torch.no_grad():
            s_logits = student(x)
        for _ in range(DISTILL_EPOCHS):
            for idx in batches(len(x), dev):
                loss = kl_heads(s_logits[idx], teacher(x[idx]), masks[idx], arm)
                opt.zero_grad()
                loss.backward()
                opt.step()

    anchor_logits = None
    a0 = base / "policy-0000.npz"
    if ANCHOR[a.arm] > 0 and a0.exists():
        with torch.no_grad():
            anchor_logits = Student(ppo.load(a0)).to(dev)(x)

    # Advantages from the value network (GAE per champion trajectory), normalised.
    with torch.no_grad():
        v = value(x)[:, 0]
        rew = torch.from_numpy(data["R"]).float().to(dev)
        done = torch.from_numpy(data["DONE"]).to(dev)
        adv, ret = gae(rew, v, done, torch.from_numpy(data["TRAJ"]).to(dev))
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    popt = torch.optim.AdamW(teacher.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    vopt = torch.optim.Adam(value.parameters(), lr=LR)
    kl, epochs, vloss_sum, vloss_n, anchor_kl = 0.0, 0, 0.0, 0, 0.0
    for _ in range(EPOCHS):
        epochs += 1
        for idx in batches(len(x), dev):
            logits = teacher(x[idx])
            heads = head_logp(logits, masks[idx], arm)
            lp = sum(lsm.gather(1, acts[idx, k : k + 1])[:, 0] for k, (lsm, _) in enumerate(heads))
            ratio = torch.exp(lp - lp_old[idx])
            pg = -torch.min(ratio * adv[idx], ratio.clamp(1 - ppo.CLIP, 1 + ppo.CLIP) * adv[idx]).mean()
            ent = -sum(((lsm.exp() * lsm) * ok).sum(1).mean() for lsm, ok in heads)
            loss = pg - ppo.ENTROPY * ent
            if anchor_logits is not None:
                loss = loss + ANCHOR[a.arm] * kl_heads(logits, anchor_logits[idx], masks[idx], arm)
            popt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(teacher.parameters(), 1.0)
            popt.step()
            vloss = F.smooth_l1_loss(value(x[idx])[:, 0], ret[idx])
            vopt.zero_grad()
            vloss.backward()
            vopt.step()
            vloss_sum, vloss_n = vloss_sum + vloss.item(), vloss_n + 1
        with torch.no_grad():
            t_logits = teacher(x)
            heads = head_logp(t_logits, masks, arm)
            lp = sum(lsm.gather(1, acts[:, k : k + 1])[:, 0] for k, (lsm, _) in enumerate(heads))
            log_r = lp - lp_old
            kl = float((torch.exp(log_r) - 1 - log_r).mean())  # >= 0, unlike mean(lp_old - lp)
            if anchor_logits is not None:
                anchor_kl = float(kl_heads(t_logits, anchor_logits, masks, arm))
        if kl > ppo.TARGET_KL:
            break

    # Distil: the bot's small network copies the teacher on these game states.
    sopt = torch.optim.AdamW(student.parameters(), lr=1e-3, weight_decay=WEIGHT_DECAY)
    for _ in range(DISTILL_EPOCHS):
        for idx in batches(len(x), dev):
            loss = kl_heads(t_logits[idx], student(x[idx]), masks[idx], arm)
            sopt.zero_grad()
            loss.backward()
            sopt.step()
    with torch.no_grad():
        gap = float(kl_heads(t_logits, student(x), masks, arm))

    nxt = it + 1
    export(student.cpu(), data["X"], arm, base, nxt)
    torch.save(teacher.state_dict(), base / f"teacher-{nxt:04d}.pt")
    torch.save(value.state_dict(), base / f"tvalue-{nxt:04d}.pt")
    return {
        "samples": len(x),
        "kl": round(kl, 4),
        "epochs_run": epochs,
        "vloss": round(vloss_sum / max(vloss_n, 1), 4),
        "anchor_kl": round(anchor_kl, 4),
        "mean_reward": round(float(data["R"].mean()), 4),
        "distill_gap": round(gap, 4),
        "device": str(dev),
        "teacher": f"{WIDTH}x{WIDTH}",
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", type=int, required=True)
    ap.add_argument("--it", type=int, default=0)
    ap.add_argument("--init", action="store_true", help="iteration 0 (the rlinit job)")
    ap.add_argument("--store", type=Path, required=True)
    ap.add_argument("--cpu", action="store_true", help="tests only: run without CUDA")
    a = ap.parse_args(argv)
    if not (a.cpu or torch.cuda.is_available()):
        print("ppo_gpu: no CUDA device; training runs on the GPU only", file=sys.stderr)
        return 2
    dev = torch.device("cpu" if a.cpu else "cuda")
    torch.manual_seed(a.it)
    arm, base = ppo.ARMS[a.arm], a.store / "rl" / f"arm{a.arm}"
    base.mkdir(parents=True, exist_ok=True)
    print(json.dumps(init(a, arm, base, dev) if a.init else update(a, arm, base, dev)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
