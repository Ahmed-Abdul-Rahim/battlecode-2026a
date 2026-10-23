"""MuZero-style learned model of the game: train on exploration games, test, relabel.

    python nn/muzero.py train   [--steps 20000]   # GPU venv; model -> runs/nn/muzero/model.pt
    python nn/muzero.py relabel [--margin 0.5]    # planned goals -> runs/nn/muzero/relabel/*.npz

Model (MuZero, with EfficientZero's consistency loss):
  h: observation -> latent z;  g: (z, goal) -> (z', reward);  f: z -> (prior over goals, value).
Data: nn/muzero_collect.py games (rules + 10% random held goals, so the model sees the outcome of
choices the rules didn't make). Unrolled K steps along each champion's decisions with the goal
actually executed. Targets: the rules' choice (prior), the team-value change to the next
decision (reward, gems), the discounted return (gamma 0.99 per decision), and h(next state).

Tests printed during training:
1. held-out reward / value error vs predicting the mean (does it model the game at all?);
2. explored decisions in held-out games: does Q(executed) - Q(rule's choice) have the sign of the
   realised advantage (reward + discounted return of the executed choice vs the model-free
   value baseline)? (weak, one sample per state);
3. branch decisions (nn/expert.py it*.q.json, measured in real branch games): pair agreement and
   top-1 vs "the policy's own choice is always best".
`relabel` writes the imitation data with the planned goal (argmax Q, when it beats the rules'
choice by `margin` gems) for the play test: distil into the teacher, evaluate vs the rules.
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
OUT = ROOT / "runs" / "nn" / "muzero"
GAMMA = 0.99
Z = 256
G = spec.HEADS[0]
KEYS = ("S", "G", "PREV", "A", "M", "R", "DONE", "TRAJ", "B", "EXPLORED")


class MuZero(nn.Module):
    def __init__(self, n_in: int):
        super().__init__()
        self.h = nn.Sequential(nn.Linear(n_in, 512), nn.ReLU(), nn.Linear(512, Z), nn.LayerNorm(Z))
        self.g = nn.Sequential(nn.Linear(Z + G, 512), nn.ReLU(), nn.Linear(512, Z), nn.LayerNorm(Z))
        self.r = nn.Sequential(nn.Linear(Z + G, 256), nn.ReLU(), nn.Linear(256, 1))
        self.f = nn.Sequential(nn.Linear(Z, 256), nn.ReLU(), nn.Linear(256, G + 1))

    def step(self, z: torch.Tensor, a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        za = torch.cat([z, nn.functional.one_hot(a.clamp(0, G - 1), G).float()], 1)
        return self.g(za), self.r(za).squeeze(1)

    def pv(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        o = self.f(z)
        return o[:, :G], o[:, G]

    def q(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        z2, r = self.step(z, a)
        return r + GAMMA * self.pv(z2)[1]


def load_npz(p: Path) -> dict:
    with np.load(p) as z:
        d = {k: z[k] for k in z.files if k in KEYS}
    d.setdefault("B", d["A"][:, 0].copy())
    d.setdefault("EXPLORED", np.zeros(len(d["S"]), bool))
    return d


def dataset() -> tuple[dict, int, int, list[dict]]:
    """Explore shards (last ~10% held out), then the expert data (only for the branch test)."""
    shards = sorted((OUT / "data").glob("explore_*.npz"))
    n_held = max(1, len(shards) // 10)
    parts = [load_npz(p) for p in shards]
    n_train = sum(len(p["S"]) for p in parts[:-n_held])
    n_explore = sum(len(p["S"]) for p in parts)
    qrecs, off = [], n_explore
    for p in sorted((ROOT / "runs" / "nn" / "expert" / "data").glob("it*.npz")):
        qp = p.with_suffix(".q.json")
        if not qp.exists():
            continue
        d = load_npz(p)
        d["R"] = np.zeros(len(d["S"]), np.float32)  # rewards unused for these rows
        for q in json.loads(qp.read_text()):
            qrecs.append({**q, "row": q["row"] + off})
        parts.append(d)
        off += len(d["S"])
    return data.concat(parts), n_train, n_explore, qrecs


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


class Data:
    def __init__(self):
        d, self.n_train, self.n_explore, self.qrecs = dataset()
        nxt, ret = targets(d)
        self.b = Batches(d)
        self.nxt = torch.from_numpy(nxt).to(DEV)
        self.ret = torch.from_numpy(ret).to(DEV)
        self.rew = torch.from_numpy(d["R"]).to(DEV)
        self.A = self.b.A[:, 0]
        self.Bh = torch.from_numpy(d["B"]).to(DEV)
        self.explored = torch.from_numpy(d["EXPLORED"]).to(DEV)
        self.scale = float(np.std(ret[: self.n_explore])) + 1e-6
        self.train_rows = torch.from_numpy(np.flatnonzero(nxt[: self.n_train] >= 0)).to(DEV)
        self.held_rows = torch.arange(self.n_train, self.n_explore, device=DEV)
        print(f"rows {self.b.n} (train {self.n_train}, held-out {self.n_explore - self.n_train}, "
              f"branch-test {len(self.qrecs)} decisions), explored {int(d['EXPLORED'][: self.n_explore].sum())}, "
              f"return std {self.scale:.2f} gems", flush=True)

    def mask(self, rows: torch.Tensor) -> torch.Tensor:
        return ((self.b.M[rows, 0:1] >> torch.arange(G, device=DEV)) & 1).bool()


def train(a) -> None:
    torch.manual_seed(0)
    D = Data()
    model = MuZero(D.b.x(torch.zeros(1, dtype=torch.long, device=DEV)).shape[1]).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
    best = (-1.0, None)
    for it in range(a.steps):
        idx = D.train_rows[torch.randint(len(D.train_rows), (a.batch,), device=DEV)]
        z = model.h(D.b.x(idx))
        loss = torch.zeros((), device=DEV)
        alive = torch.ones(len(idx), dtype=torch.bool, device=DEV)
        cur = idx
        for k in range(a.unroll + 1):
            w = alive.float()
            logits, v = model.pv(z)
            lp = torch.log_softmax(logits.masked_fill(~D.mask(cur), -1e9), 1)
            loss = loss + 0.5 * (-(lp.gather(1, D.Bh[cur].clamp(0, G - 1).unsqueeze(1)).squeeze(1)) * w).mean()
            loss = loss + (((v - D.ret[cur] / D.scale) ** 2) * w).mean()
            if k == a.unroll:
                break
            z2, r = model.step(z, D.A[cur])
            loss = loss + 4.0 * (((r - D.rew[cur] / D.scale) ** 2) * w).mean()
            nx = D.nxt[cur]
            alive = alive & (nx >= 0)
            cur = nx.clamp(min=0)
            with torch.no_grad():
                zt = model.h(D.b.x(cur))
            loss = loss + 2.0 * ((1 - nn.functional.cosine_similarity(z2, zt, dim=1)) * alive.float()).mean()
            z = 0.5 * z2 + 0.5 * z2.detach()  # MuZero: halve the gradient through the dynamics
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if (it + 1) % a.eval_every == 0:
            m = evaluate(model, D)
            print(f"step {it + 1}: loss {loss.item():.3f} | {m}", flush=True)
            score = m.get("branch_pairs", 0.0)
            if score > best[0]:
                best = (score, {k: v.detach().clone() for k, v in model.state_dict().items()})
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.state_dict(), "n_in": model.h[0].in_features, "scale": D.scale}, OUT / "model.pt")
    print(f"saved {OUT / 'model.pt'} (final); best branch pair agreement {best[0]:.3f}")


@torch.no_grad()
def evaluate(model: MuZero, D: Data) -> dict:
    model.eval()
    out = {}
    # 1. held-out reward / value error vs the mean predictor
    rows = D.held_rows[: 50_000]
    z = model.h(D.b.x(rows))
    _, r = model.step(z, D.A[rows])
    _, v = model.pv(z)
    tr, tv = D.rew[rows] / D.scale, D.ret[rows] / D.scale
    out["reward_r2"] = round(1 - float(((r - tr) ** 2).mean() / tr.var().clamp(min=1e-9)), 3)
    out["value_r2"] = round(1 - float(((v - tv) ** 2).mean() / tv.var().clamp(min=1e-9)), 3)
    # 2. explored held-out decisions: sign of Q(ex) - Q(rule) vs realised advantage
    ex = rows[D.explored[rows] & (D.A[rows] != D.Bh[rows])]
    if len(ex):
        z = model.h(D.b.x(ex))
        dq = model.q(z, D.A[ex]) - model.q(z, D.Bh[ex])
        _, vb = model.pv(z)
        adv = D.ret[ex] / D.scale - vb  # realised return of the executed choice vs the state's value
        out["explored_sign"] = round(float(((dq > 0) == (adv > 0)).float().mean()), 3)
        out["explored_n"] = len(ex)
    # 3. branch decisions measured in real games
    pairs = hits = top = top_n = own_best = 0
    for q in D.qrecs:
        acts = [q["own"]] + [int(x) for x in q["gains"] if int(x) != q["own"]]
        gains = {q["own"]: 0.0, **{int(k): v for k, v in q["gains"].items() if int(k) != q["own"]}}
        if len(acts) < 2:
            continue
        row = torch.tensor([q["row"]] * len(acts), device=DEV)
        Q = model.q(model.h(D.b.x(row)), torch.tensor(acts, device=DEV)).cpu().numpy()
        for j, a_ in enumerate(acts[1:], 1):
            if abs(gains[a_]) > 4:
                pairs += 1
                hits += (Q[j] - Q[0] > 0) == (gains[a_] > 0)
        top_n += 1
        best = max(gains, key=gains.get)
        top += acts[int(np.argmax(Q))] == best
        own_best += best == q["own"]
    if top_n:
        out["branch_pairs"] = round(hits / max(pairs, 1), 3)
        out["branch_n_pairs"] = pairs
        out["branch_top1"] = round(top / top_n, 3)
        out["own_always_top1"] = round(own_best / top_n, 3)
    model.train()
    return out


@torch.no_grad()
def relabel(a) -> None:
    """Imitation data with the planned goal where it beats the rules' choice by `margin` gems."""
    ck = torch.load(OUT / "model.pt", map_location=DEV)
    model = MuZero(ck["n_in"]).to(DEV)
    model.load_state_dict(ck["state"])
    model.eval()
    scale = ck["scale"]
    dst = OUT / "relabel"
    dst.mkdir(parents=True, exist_ok=True)
    changed = total = 0
    for p in sorted((ROOT / "runs" / "nn" / "data" / "imitation").glob("*.npz")):
        d = data.load([p])
        b = Batches(d)
        A = d["A"].copy()
        for rows in torch.arange(b.n, device=DEV).split(8192):
            z = model.h(b.x(rows))
            m = ((b.M[rows, 0:1] >> torch.arange(G, device=DEV)) & 1).bool()
            Q = torch.stack([model.q(z, torch.full((len(rows),), g, device=DEV)) for g in range(G)], 1)
            Q = Q.masked_fill(~m, -1e9)
            rule = b.A[rows, 0].clamp(0, G - 1)
            q_rule = Q.gather(1, rule.unsqueeze(1)).squeeze(1)
            best_q, best = Q.max(1)
            take = (best_q - q_rule) * scale > a.margin
            new = torch.where(take, best, rule).cpu().numpy()
            A[rows.cpu().numpy(), 0] = new
            changed += int(take.sum())
            total += len(rows)
        d["A"] = A
        data.save(dst / p.name, d)
    print(json.dumps({"relabelled": changed, "rows": total, "frac": round(changed / max(total, 1), 4),
                      "margin": a.margin}))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "relabel"])
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--unroll", type=int, default=5)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--margin", type=float, default=0.5)
    a = ap.parse_args()
    if a.cmd == "train":
        train(a)
    else:
        relabel(a)


if __name__ == "__main__":
    main()
