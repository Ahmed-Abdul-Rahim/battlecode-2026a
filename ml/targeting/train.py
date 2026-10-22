"""Learned target choice on the GPU: teacher, backward feature elimination, int8 student.

1. Teacher: every candidate group's inputs (plus the decision's global inputs) go through a
   shared encoder and attention across the groups; it predicts the outcome of picking each
   group (ml/targeting/data.py), trained on the group actually picked.
2. Backward elimination: drop the input whose removal hurts least (permutation importance on
   held-out games, minus a fuel credit), retrain, repeat; stop once held-out loss is more than
   TOL above the best twice in a row. The last set within TOL is the selection.
3. Student: per group, selected inputs -> 16 relu -> 1, trained to match the teacher's scores
   on every candidate, exported as an int8 BCW1 blob (same format as ml/rl, bot int8.rs).

    <gpu python> -m ml.targeting.train data.npz --out runs/target
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ml.rl import ppo  # noqa: E402
from ml.targeting.features import FUEL, N_GLOBAL, N_GROUP, NAMES  # noqa: E402

D_MODEL, HEADS, LAYERS = 128, 4, 2
BATCH, LR = 512, 1e-3
EPOCHS_FULL, EPOCHS_SEL = 12, 3
TOL, LAM = 0.01, 0.02
STUDENT_HIDDEN, STUDENT_EPOCHS = 32, 30
RANK_TEMP = 0.1
TARGET_SCHEMA = 0x7A61  # bot: weights/target.bin


class Teacher(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.inp = nn.Sequential(
            nn.Linear(N_GROUP + N_GLOBAL, D_MODEL), nn.GELU(), nn.Linear(D_MODEL, D_MODEL)
        )
        layer = nn.TransformerEncoderLayer(
            D_MODEL, HEADS, 2 * D_MODEL, dropout=0.1, batch_first=True, norm_first=True
        )
        self.enc = nn.TransformerEncoder(layer, LAYERS)
        self.head = nn.Sequential(nn.LayerNorm(D_MODEL), nn.Linear(D_MODEL, 1))

    def forward(self, g, x, m):
        h = torch.cat([g, x[:, None, :].expand(-1, g.shape[1], -1)], dim=-1)
        h = self.enc(self.inp(h), src_key_padding_mask=~m)
        return self.head(h)[..., 0]


class Data:
    """The dataset on the device (int8), a split by game, and normalised outcomes."""

    def __init__(self, path: Path, dev: torch.device) -> None:
        d = np.load(path)
        self.dev = dev
        self.G = torch.from_numpy(d["G"]).to(dev)
        self.M = torch.from_numpy(d["M"]).to(dev)
        self.X = torch.from_numpy(d["X"]).to(dev)
        self.A = torch.from_numpy(d["A"].astype(np.int64)).to(dev)
        y = d["Y"].astype(np.float32)
        game = d["GAME"]
        self.val = torch.from_numpy(np.nonzero(game % 5 == 0)[0]).to(dev)
        self.train = torch.from_numpy(np.nonzero(game % 5 != 0)[0]).to(dev)
        tr = y[game % 5 != 0]
        self.y_mean, self.y_std = float(tr.mean()), float(tr.std() + 1e-6)
        self.Y = torch.from_numpy((y - self.y_mean) / self.y_std).to(dev)
        self.n = len(y)

    def batch(self, idx, fmask):
        g = self.G[idx].float() / 127.0 * fmask[None, None, :N_GROUP]
        x = self.X[idx].float() / 127.0 * fmask[None, N_GROUP:]
        return g, x, self.M[idx]


def loss_on(model, data: Data, idx, fmask, perm: int | None = None) -> float:
    """Huber loss on the picked groups; `perm` shuffles that input across the rows first."""
    model.eval()
    total, n = 0.0, 0
    gen = torch.Generator(device=data.dev).manual_seed(0)
    with torch.no_grad():
        for b in idx.split(4096):
            g, x, m = data.batch(b, fmask)
            if perm is not None:
                p = torch.randperm(len(b), device=data.dev, generator=gen)
                if perm < N_GROUP:
                    g[:, :, perm] = g[p][:, :, perm]
                else:
                    x[:, perm - N_GROUP] = x[p, perm - N_GROUP]
            q = model(g, x, m).gather(1, data.A[b][:, None])[:, 0]
            total += float(F.smooth_l1_loss(q, data.Y[b], reduction="sum"))
            n += len(b)
    return total / max(n, 1)


def fit(data: Data, fmask, epochs: int, seed: int = 0) -> tuple[Teacher, float]:
    torch.manual_seed(seed)
    model = Teacher().to(data.dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    steps = epochs * (len(data.train) // BATCH + 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=steps)
    for _ in range(epochs):
        model.train()
        for b in data.train[torch.randperm(len(data.train), device=data.dev)].split(BATCH):
            g, x, m = data.batch(b, fmask)
            q = model(g, x, m).gather(1, data.A[b][:, None])[:, 0]
            loss = F.smooth_l1_loss(q, data.Y[b])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
    return model, loss_on(model, data, data.val, fmask)


def baselines(data: Data) -> dict:
    """Held-out loss of: the mean outcome; a linear fit on the rule score alone."""
    yv = data.Y[data.val]
    mean = float(F.smooth_l1_loss(torch.zeros_like(yv), yv))
    rule = NAMES.index("g.rule_score")

    def score(idx):
        return data.G[idx, :, rule].float().gather(1, data.A[idx][:, None])[:, 0] / 127.0

    s, y = score(data.train), data.Y[data.train]
    a = torch.stack([s, torch.ones_like(s)], 1)
    coef = torch.linalg.lstsq(a, y[:, None]).solution[:, 0]
    sv = score(data.val)
    lin = float(F.smooth_l1_loss(coef[0] * sv + coef[1], yv))
    return {"mean": round(mean, 5), "rule_score_linear": round(lin, 5)}


def eliminate(data: Data, log) -> tuple[list[int], list[dict]]:
    n_feat = N_GROUP + N_GLOBAL
    fuel = np.array(FUEL, np.float64)
    fuel_norm = fuel / max(fuel.sum(), 1.0)
    active = list(range(n_feat))
    history = []
    best, strikes, selected = None, 0, list(active)
    while len(active) > 1:
        fmask = torch.zeros(n_feat, device=data.dev)
        fmask[active] = 1.0
        model, val = fit(data, fmask, EPOCHS_SEL)
        history.append({"n": len(active), "val": round(val, 5), "active": [NAMES[i] for i in active]})
        log(f"  {len(active):2d} inputs: held-out loss {val:.5f}")
        if best is None or val < best:
            best = val
        if val <= best * (1 + TOL):
            strikes, selected = 0, list(active)
        else:
            strikes += 1
            if strikes >= 2:
                break
        base = loss_on(model, data, data.val, fmask)
        imp = {f: loss_on(model, data, data.val, fmask, perm=f) - base for f in active}
        # While many inputs remain, drop about a tenth of them per step (each step retrains).
        n_drop = max(1, len(active) // 10)
        drops = sorted(active, key=lambda f: imp[f] - LAM * fuel_norm[f])[:n_drop]
        history[-1]["dropped"] = [NAMES[f] for f in drops]
        history[-1]["importance"] = {
            NAMES[f]: round(v, 5) for f, v in sorted(imp.items(), key=lambda kv: -kv[1])
        }
        for f in drops:
            active.remove(f)
    return selected, history


class Student(nn.Module):
    def __init__(self, n_in: int, hidden: int = STUDENT_HIDDEN) -> None:
        super().__init__()
        self.l1, self.l2 = nn.Linear(n_in, hidden), nn.Linear(hidden, 1)

    def forward(self, z):
        return self.l2(torch.relu(self.l1(z)))[..., 0]


def distill(
    data: Data, teacher: Teacher, sel: list[int], fmask, hidden: int = STUDENT_HIDDEN
) -> tuple[Student, dict]:
    """Student on every candidate group: selected group inputs + selected global inputs."""
    gsel = [f for f in sel if f < N_GROUP]
    xsel = [f - N_GROUP for f in sel if f >= N_GROUP]
    student = Student(len(gsel) + len(xsel), hidden).to(data.dev)
    opt = torch.optim.Adam(student.parameters(), lr=3e-3)

    def inputs(b):
        g, x, m = data.batch(b, fmask)
        z = torch.cat([g[:, :, gsel], x[:, None, xsel].expand(-1, g.shape[1], -1)], dim=-1)
        return g, x, m, z

    teacher.eval()
    for _ in range(STUDENT_EPOCHS):
        for b in data.train[torch.randperm(len(data.train), device=data.dev)].split(BATCH):
            g, x, m, z = inputs(b)
            with torch.no_grad():
                t = teacher(g, x, m)
            # Rank, not level: match the teacher's choice distribution over this decision's
            # groups (temperature RANK_TEMP) plus scores relative to the decision's mean.
            st = student(z)
            mf = m.float()
            n_g = mf.sum(1, keepdim=True).clamp(min=1)
            tc = (t - (t * mf).sum(1, keepdim=True) / n_g) * mf
            sc = (st - (st * mf).sum(1, keepdim=True) / n_g) * mf
            lt = (t / RANK_TEMP).masked_fill(~m, -1e9).log_softmax(1)
            ls = (st / RANK_TEMP).masked_fill(~m, -1e9).log_softmax(1)
            kl = (lt.exp() * (lt - ls) * mf).sum(1).mean()
            loss = kl + ((sc - tc) ** 2).sum() / mf.sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
    student.eval()
    agree, agree3, n, picked = 0, 0, 0, 0.0
    rule = NAMES.index("g.rule_score")
    with torch.no_grad():
        for b in data.val.split(4096):
            g, x, m, z = inputs(b)
            t = teacher(g, x, m).masked_fill(~m, -1e9)
            s = student(z).masked_fill(~m, -1e9)
            agree += int((t.argmax(1) == s.argmax(1)).sum())
            # Within the rules' top 3 (TARGET_NET 2): same choice as the teacher?
            r = data.G[b, :, rule].float().masked_fill(~m, -1e9)
            top = r.topk(min(3, r.shape[1]), dim=1).indices
            agree3 += int((t.gather(1, top).argmax(1) == s.gather(1, top).argmax(1)).sum())
            picked += float(
                F.smooth_l1_loss(s.gather(1, data.A[b][:, None])[:, 0], data.Y[b], reduction="sum")
            )
            n += len(b)
    return student, {
        "argmax_agree_with_teacher": round(agree / max(n, 1), 4),
        "top3_agree_with_teacher": round(agree3 / max(n, 1), 4),
        "held_out_loss": round(picked / max(n, 1), 5),
    }


def export(student: Student, data: Data, sel: list[int], fmask, out: Path) -> None:
    gsel = [f for f in sel if f < N_GROUP]
    xsel = [f - N_GROUP for f in sel if f >= N_GROUP]
    p = {
        k: v.detach().cpu().numpy()
        for k, v in (
            ("w1", student.l1.weight),
            ("b1", student.l1.bias),
            ("w2", student.l2.weight),
            ("b2", student.l2.bias),
        )
    }
    idx = data.train[:20000]
    g, x, m = data.batch(idx, fmask)
    z = torch.cat([g[:, :, gsel], x[:, None, xsel].expand(-1, g.shape[1], -1)], dim=-1)[m]
    z8 = np.round(z.cpu().numpy() * 127).astype(np.int8)
    arm = ppo.Arm(len(sel), ((0, 1),), TARGET_SCHEMA, student.l1.out_features)
    (out / "target.bin").write_bytes(ppo.export(p, z8, arm))
    (out / "target_inputs.json").write_text(
        json.dumps({"group": gsel, "global": xsel, "names": [NAMES[f] for f in sel]}, indent=1)
    )
    # bot.rs target_net(): n_group u8, group indices, n_global u8, global indices.
    (out / "target_inputs.bin").write_bytes(bytes([len(gsel), *gsel, len(xsel), *xsel]))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--no-select", action="store_true", help="keep every input")
    ap.add_argument("--inputs", type=Path, help="a previous report.json: reuse its selection")
    ap.add_argument("--hidden", type=int, default=STUDENT_HIDDEN, help="student hidden units (fuel)")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    log_f = (a.out / "train.log").open("w", encoding="utf-8")

    def log(s: str) -> None:
        print(s, flush=True)
        log_f.write(s + "\n")
        log_f.flush()

    data = Data(a.data, dev)
    log(f"{data.n} decisions ({len(data.train)} train / {len(data.val)} held out) on {dev}")
    report: dict = {"baselines": baselines(data)}
    log(f"baselines (held-out loss): {report['baselines']}")
    n_feat = N_GROUP + N_GLOBAL
    if a.inputs:
        sel, hist = [NAMES.index(nm) for nm in json.loads(a.inputs.read_text())["selected"]], []
    elif a.no_select:
        sel, hist = list(range(n_feat)), []
    else:
        log("backward elimination:")
        sel, hist = eliminate(data, log)
    report["selected"] = [NAMES[f] for f in sel]
    report["elimination"] = hist
    fmask = torch.zeros(n_feat, device=dev)
    fmask[sel] = 1.0
    teacher, val = fit(data, fmask, EPOCHS_FULL)
    report["teacher_held_out_loss"] = round(val, 5)
    log(f"teacher on {len(sel)} inputs: held-out loss {val:.5f}")
    student, st = distill(data, teacher, sel, fmask, a.hidden)
    report["student"] = st
    log(f"student: {st}")
    torch.save(teacher.state_dict(), a.out / "teacher.pt")
    export(student, data, sel, fmask, a.out)
    report["y_mean"], report["y_std"] = data.y_mean, data.y_std
    report["minutes"] = round((time.time() - t0) / 60, 1)
    (a.out / "report.json").write_text(json.dumps(report, indent=1))
    log(f"done in {report['minutes']} min -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
