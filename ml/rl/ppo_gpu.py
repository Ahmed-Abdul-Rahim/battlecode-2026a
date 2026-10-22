"""PPO on the GPU with a big teacher network, distilled into the small network the bot runs.

The bot can only afford a 32-unit int8 network per turn, but training doesn't have that limit:
a 512x512 teacher (policy) and a 512x512 value network learn from every rollout on the GPU
(torch, CUDA if present), then the bot's small network is trained to copy the teacher's
probabilities on the same game states and exported exactly like ml/rl/ppo.py does.

Run by harness/mljobs.py (update jobs) in a separate environment that has torch:

    <gpu python> ml/rl/ppo_gpu.py --arm 1 --it 7 --store <ml-data worktree>

Prints one JSON line of metrics. Needs only torch + numpy (no repo tooling).
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ml.rl import ppo, reward  # noqa: E402  (numpy only)

KEYS = ("X", "A", "LP", "M", "R", "DONE", "TRAJ")
WIDTH = 512
EPOCHS, BATCH, LR, DISTILL_EPOCHS = 6, 1024, 3e-4, 6


def read(p: Path) -> dict | None:
    try:
        d = np.load(p)
        return {k: d[k] for k in KEYS}
    except (OSError, KeyError, ValueError, zipfile.BadZipFile, EOFError):
        return None


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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", type=int, required=True)
    ap.add_argument("--it", type=int, required=True)
    ap.add_argument("--store", type=Path, required=True)
    a = ap.parse_args(argv)
    arm, it, base = ppo.ARMS[a.arm], a.it, a.store / "rl" / f"arm{a.arm}"
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(it)

    data = reward.concat([d for d in (read(p) for p in sorted(base.glob(f"it{it:04d}-p*.npz"))) if d])
    if not data:
        print(json.dumps({"arm": a.arm, "it": it, "skipped": "no decisions logged"}))
        return 0
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
            for idx in torch.randperm(len(x), device=dev).split(BATCH):
                loss = kl_heads(s_logits[idx], teacher(x[idx]), masks[idx], arm)
                opt.zero_grad()
                loss.backward()
                opt.step()

    # Advantages from the value network (GAE per champion trajectory, numpy as in ppo.py).
    with torch.no_grad():
        v = value(x)[:, 0].cpu().numpy()
    adv, ret = ppo.gae(data["R"], v, data["DONE"], data["TRAJ"])
    adv = torch.from_numpy((adv - adv.mean()) / (adv.std() + 1e-8)).float().to(dev)
    ret = torch.from_numpy(ret).float().to(dev)

    popt = torch.optim.Adam(teacher.parameters(), lr=LR)
    vopt = torch.optim.Adam(value.parameters(), lr=LR)
    kl = 0.0
    for _ in range(EPOCHS):
        for idx in torch.randperm(len(x), device=dev).split(BATCH):
            heads = head_logp(teacher(x[idx]), masks[idx], arm)
            lp = sum(lsm.gather(1, acts[idx, k : k + 1])[:, 0] for k, (lsm, _) in enumerate(heads))
            ratio = torch.exp(lp - lp_old[idx])
            pg = -torch.min(ratio * adv[idx], ratio.clamp(1 - ppo.CLIP, 1 + ppo.CLIP) * adv[idx]).mean()
            ent = -sum(((lsm.exp() * lsm) * ok).sum(1).mean() for lsm, ok in heads)
            loss = pg - ppo.ENTROPY * ent
            popt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(teacher.parameters(), 1.0)
            popt.step()
            vloss = ((value(x[idx])[:, 0] - ret[idx]) ** 2).mean()
            vopt.zero_grad()
            vloss.backward()
            vopt.step()
        with torch.no_grad():
            heads = head_logp(teacher(x), masks, arm)
            lp = sum(lsm.gather(1, acts[:, k : k + 1])[:, 0] for k, (lsm, _) in enumerate(heads))
            kl = float((lp_old - lp).mean())
        if kl > ppo.TARGET_KL:
            break

    # Distil: the bot's small network copies the teacher on these game states.
    with torch.no_grad():
        t_logits = teacher(x)
    sopt = torch.optim.Adam(student.parameters(), lr=1e-3)
    for _ in range(DISTILL_EPOCHS):
        for idx in torch.randperm(len(x), device=dev).split(BATCH):
            loss = kl_heads(t_logits[idx], student(x[idx]), masks[idx], arm)
            sopt.zero_grad()
            loss.backward()
            sopt.step()
    with torch.no_grad():
        gap = float(kl_heads(t_logits, student(x), masks, arm))

    nxt = it + 1
    params = student.params()
    ppo.save(base / f"policy-{nxt:04d}.npz", params)
    (base / f"policy-{nxt:04d}.bin").write_bytes(ppo.export(params, data["X"], arm))
    torch.save(teacher.state_dict(), base / f"teacher-{nxt:04d}.pt")
    torch.save(value.state_dict(), base / f"tvalue-{nxt:04d}.pt")
    # The numpy value net file keeps the CPU path usable if a later update runs without a GPU.
    vnp = base / f"value-{it:04d}.npz"
    if vnp.exists():
        ppo.save(base / f"value-{nxt:04d}.npz", ppo.load(vnp))
    print(
        json.dumps(
            {
                "samples": len(x),
                "kl": round(kl, 4),
                "mean_reward": round(float(data["R"].mean()), 4),
                "distill_gap": round(gap, 4),
                "device": str(dev),
                "teacher": f"{WIDTH}x{WIDTH}",
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
