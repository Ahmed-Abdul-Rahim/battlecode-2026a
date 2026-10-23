"""MuZero-style learned model of the game, tested against measured branch outcomes.

    python nn/muzero.py [--steps 6000] [--unroll 5]      (GPU venv)

Model (MuZero, with EfficientZero's consistency loss):
  h: observation -> latent z;  g: (z, goal) -> (z', reward);  f: z -> (goal policy, value).
Trained on the logged games (runs/nn/data/imitation + nn/expert.py data): each row's next row in
the same champion trajectory is the next decision. Targets: the goal taken, the shaped reward,
the discounted return (gamma 0.99, roughly the 150-round horizon of a branch), and h(next).

Test: the decisions nn/expert.py searched (it*.q.json: measured gain of each alternative over
the policy's own choice, from real branch games). Planning with the model gives
Q(a) = r(z, a) + gamma * v(g(z, a)); reported:
- pair agreement: sign(Q(alt) - Q(own)) == sign(measured gain), over pairs with |gain| > 4;
- top-1: the model's best of {own, alternatives} is the measured best.
Chance is 0.5 for pairs; "the policy's own choice is always best" is the baseline to beat.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "nn"))

from nn import data, spec  # noqa: E402
from nn.train import Batches  # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GAMMA = 0.99
Z = 256
G = spec.HEADS[0]


class MuZero(nn.Module):
    def __init__(self, n_in: int):
        super().__init__()
        self.h = nn.Sequential(nn.Linear(n_in, 512), nn.ReLU(), nn.Linear(512, Z), nn.LayerNorm(Z))
        self.g = nn.Sequential(nn.Linear(Z + G, 512), nn.ReLU(), nn.Linear(512, Z), nn.LayerNorm(Z))
        self.r = nn.Sequential(nn.Linear(Z + G, 128), nn.ReLU(), nn.Linear(128, 1))
        self.f = nn.Sequential(nn.Linear(Z, 256), nn.ReLU(), nn.Linear(256, G + 1))

    def step(self, z: torch.Tensor, a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        za = torch.cat([z, nn.functional.one_hot(a.clamp(0, G - 1), G).float()], 1)
        return self.g(za), self.r(za).squeeze(1)

    def pv(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        o = self.f(z)
        return o[:, :G], o[:, G]


def load() -> tuple[dict, list[dict], int]:
    shards = sorted((ROOT / "runs" / "nn" / "data" / "imitation").glob("*.npz"))
    exp = sorted((ROOT / "runs" / "nn" / "expert" / "data").glob("it*.npz"))
    exp = [p for p in exp if not p.name.endswith(".q.json")]
    parts, qrecs, off, n_imit = [], [], 0, 0
    for p in shards + exp:
        d = data.load([p])
        if not d:
            continue
        qp = p.with_suffix(".q.json")
        if qp.exists():
            for q in json.loads(qp.read_text()):
                qrecs.append({**q, "row": q["row"] + off})
        parts.append(d)
        off += len(d["S"])
        if p in shards:
            n_imit = off
    return data.concat(parts), qrecs, n_imit


def targets(d: dict) -> tuple[np.ndarray, np.ndarray]:
    n = len(d["S"])
    nxt = np.full(n, -1, np.int64)
    same = (d["TRAJ"][1:] == d["TRAJ"][:-1]) & ~d["DONE"][:-1]
    nxt[:-1][same] = np.arange(1, n)[same]
    ret = np.zeros(n, np.float32)
    acc = 0.0
    for i in range(n - 1, -1, -1):
        acc = d["R"][i] + (GAMMA * acc if nxt[i] >= 0 else 0.0)
        ret[i] = acc
    return nxt, ret


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--unroll", type=int, default=5)
    ap.add_argument("--batch", type=int, default=1024)
    a = ap.parse_args()
    torch.manual_seed(0)
    d, qrecs, n_imit = load()
    nxt_np, ret_np = targets(d)
    b = Batches(d)
    nxt = torch.from_numpy(nxt_np).to(DEV)
    ret = torch.from_numpy(ret_np).to(DEV)
    rew = torch.from_numpy(d["R"]).to(DEV)
    A = b.A[:, 0]
    rs = float(ret.std()) + 1e-6
    q_rows = {q["row"] for q in qrecs}
    # transitions from the imitation games only (expert data has oversampled copies)
    pool = torch.tensor([i for i in range(n_imit) if nxt_np[i] >= 0], device=DEV)
    model = MuZero(b.x(torch.zeros(1, dtype=torch.long, device=DEV)).shape[1]).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    print(f"rows {b.n}, searched decisions {len(qrecs)}, return std {rs:.3f}", flush=True)
    for it in range(a.steps):
        idx = pool[torch.randint(len(pool), (a.batch,), device=DEV)]
        z = model.h(b.x(idx))
        loss = torch.zeros((), device=DEV)
        alive = torch.ones(len(idx), dtype=torch.bool, device=DEV)
        cur = idx
        for k in range(a.unroll + 1):
            logits, v = model.pv(z)
            m = ((b.M[cur, 0:1] >> torch.arange(G, device=DEV)) & 1).bool()
            lp = torch.log_softmax(logits.masked_fill(~m, -1e9), 1)
            w = alive.float()
            loss = loss + (-(lp.gather(1, A[cur].clamp(0, G - 1).unsqueeze(1)).squeeze(1)) * w).mean()
            loss = loss + (((v - ret[cur] / rs) ** 2) * w).mean()
            if k == a.unroll:
                break
            z2, r = model.step(z, A[cur])
            loss = loss + (((r - rew[cur] / rs) ** 2) * w).mean()
            nx = nxt[cur]
            alive = alive & (nx >= 0)
            cur = nx.clamp(min=0)
            with torch.no_grad():
                zt = model.h(b.x(cur))
            cons = 1 - nn.functional.cosine_similarity(z2, zt, dim=1)
            loss = loss + (cons * alive.float()).mean()
            z = z2
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (it + 1) % 1000 == 0:
            print(f"step {it + 1}: loss {loss.item():.4f}", flush=True)
            evaluate(model, b, qrecs)


@torch.no_grad()
def evaluate(model: MuZero, b: Batches, qrecs: list[dict]) -> None:
    if not qrecs:
        print("  no searched decisions yet (nn/expert.py writes it*.q.json)")
        return
    model.eval()
    pairs = hits = top = top_n = own_best = 0
    for q in qrecs:
        acts = [q["own"]] + [int(x) for x in q["gains"] if int(x) != q["own"]]
        gains = {q["own"]: 0.0, **{int(k): v for k, v in q["gains"].items() if int(k) != q["own"]}}
        row = torch.tensor([q["row"]] * len(acts), device=DEV)
        z = model.h(b.x(row))
        z2, r = model.step(z, torch.tensor(acts, device=DEV))
        _, v = model.pv(z2)
        Q = (r + GAMMA * v).cpu().numpy()
        for j, a_ in enumerate(acts[1:], 1):
            g = gains[a_]
            if abs(g) > 4:
                pairs += 1
                hits += (Q[j] - Q[0] > 0) == (g > 0)
        if len(acts) > 1:
            top_n += 1
            best = max(gains, key=gains.get)
            top += acts[int(np.argmax(Q))] == best
            own_best += best == q["own"]
    model.train()
    print(f"  pair agreement {hits / max(pairs, 1):.3f} over {pairs} pairs | top-1 {top / max(top_n, 1):.3f}"
          f" (own-choice-always {own_best / max(top_n, 1):.3f}, {top_n} decisions)", flush=True)


if __name__ == "__main__":
    main()
