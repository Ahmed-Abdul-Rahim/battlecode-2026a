"""GRPO-style training of a small goal-head adapter (plan: paired groups + LoRA-like adapter).

    python nn/grpo.py run [--exploit mimic-slug]   # forever (or until stopped)
    python nn/grpo.py status

Uses the deterministic engine: every perturbed teacher plays the same setups (map, opponent,
side), so differences in result come only from the perturbation. Per setup, scores are compared
with the group mean (group-relative advantage, no critic); the adapter steps along the
antithetic estimate (evolution strategies). Exploration is persistent: one perturbation plays a
whole game. The adapter is a change to the goal head of the teacher's last layer (15 x 512
weights + 15 biases), explored along random rank-1 directions; merged into the weights, so the
bot and the student don't change.

`--exploit` trains against one opponent only (an exploiter: proof that the model can find wins).
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import nn.run as R  # noqa: E402
from nn import bcn1, data, spec  # noqa: E402

PAIRS = 4  # antithetic pairs per iteration (8 perturbed teachers)
SETUPS = 7  # (map, opponent, side) per iteration; every teacher plays all of them
EVAL_EVERY = 4
TARGET_FLIP = 0.03  # a perturbation should change about this share of goal choices
LR = 0.5  # step length, in units of a typical perturbation's length
RESET_DROP = 0.10
WIDTH = 512


def out_dir(exploit: str | None) -> Path:
    return R.OUT / ("grpo" if not exploit else f"grpo-x-{exploit}")


def log(d: Path, msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(d / "grpo.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_state(d: Path) -> dict:
    p = d / "state.json"
    return json.loads(p.read_text()) if p.exists() else {"it": 0, "evals": [], "games": 0}


def save_state(d: Path, st: dict) -> None:
    tmp = d / "state.tmp"
    tmp.write_text(json.dumps(st, indent=1))
    tmp.replace(d / "state.json")


def calibrate(base: bcn1.Blob, rng: np.random.Generator) -> float:
    """sigma such that a random rank-1 perturbation flips ~TARGET_FLIP of argmax goals."""
    shard = sorted((R.OUT / "data" / "imitation").glob("*.npz"))[0]
    d = data.load([shard])
    idx = np.arange(min(20000, len(d["S"])))
    x = spec.build_inputs(d["S"][idx], d["G"][idx], np.full((len(idx), spec.HIST), -1))
    h = base.forward_hidden(x)
    g = base.heads[0]
    last = base.layers[-1]
    mask = spec.mask_bits(d["M"][idx], 0)
    z0 = np.where(mask, h @ last.w[:g].T + last.b[:g], -1e9)
    a0 = z0.argmax(1)
    lo, hi = 1e-4, 10.0
    for _ in range(20):
        s = (lo * hi) ** 0.5
        flips = []
        for _ in range(4):
            dw, db = direction(rng, g, h.shape[1])
            z = np.where(mask, h @ (last.w[:g] + s * dw).T + last.b[:g] + s * db, -1e9)
            flips.append((z.argmax(1) != a0).mean())
        if np.mean(flips) > TARGET_FLIP:
            hi = s
        else:
            lo = s
    return float((lo * hi) ** 0.5)


def direction(rng: np.random.Generator, g: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    """A random rank-1 direction for the goal head (unit-ish scale) plus bias noise."""
    u = rng.standard_normal(g)
    v = rng.standard_normal(n) / np.sqrt(n)
    return np.outer(u, v).astype(np.float32), (0.1 * rng.standard_normal(g)).astype(np.float32)


def score(r, bot: str) -> float:
    ours, theirs = (r.gems_a, r.gems_b) if r.bot_a == bot else (r.gems_b, r.gems_a)
    return r.score_for(bot) + 0.5 * (ours - theirs) / max(ours + theirs, 1)


def setups(rng: random.Random, exploit: str | None) -> list[tuple[str, Path, bool]]:
    train, _ = R.maps()
    pool = ["main"] * 3 + R.ANCHORS[:3] + R.mimics()
    out = []
    for _ in range(SETUPS):
        o = exploit or rng.choice(pool)
        out.append((o, rng.choice(train), rng.random() < 0.5))
    return out


def cmd_run(a) -> int:
    d = out_dir(a.exploit)
    d.mkdir(parents=True, exist_ok=True)
    st = load_state(d)
    base_path = R.OUT / "ckpt" / "bc512" / "teacher.bin"
    base = bcn1.read(base_path)
    g, n = base.heads[0], base.layers[-1].w.shape[1]
    if (d / "theta.npz").exists():
        z = np.load(d / "theta.npz")
        dw, db, m, v = z["dw"], z["db"], z["m"], z["v"]
    else:
        dw, db = np.zeros((g, n), np.float32), np.zeros(g, np.float32)
        m, v = np.zeros(g * n + g), np.zeros(g * n + g)
    nrng = np.random.default_rng(1234 + st["it"])
    if "sigma" not in st:
        st["sigma"] = calibrate(base, nrng)
        log(d, f"sigma {st['sigma']:.4f} (flips ~{TARGET_FLIP:.0%} of goal choices)")
        save_state(d, st)
    sigma = st["sigma"]
    if st["it"] == 0 and not st["evals"]:
        e = evaluate(d, base, dw, db, "it0", a.exploit)
        st["evals"].append({"it": 0, "games": 0, **e})
        np.savez(d / "best.npz", dw=dw, db=db)
        save_state(d, st)
        log(d, f"start (imitation teacher): {e}")
    while True:
        it = st["it"]
        rng = random.Random(77 + it)
        nrng = np.random.default_rng(1000 + it)
        dirs = [direction(nrng, g, n) for _ in range(PAIRS)]
        bots = []
        for j, (ew, eb) in enumerate(dirs):
            for sgn in (+1, -1):
                name = f"_grpo{'x' if a.exploit else ''}{2 * j + (sgn < 0)}"
                w = d / f"{name}.bin"
                bcn1.write(bcn1.with_goal_delta(base, dw + sgn * sigma * ew, db + sgn * sigma * eb), w)
                bots.append((R.variant(name, 2, 0, w), j, sgn))
        sets = setups(rng, a.exploit)
        games = []
        for o, mp, a_side in sets:
            for b, _, _ in bots:
                games.append((b, o, mp) if a_side else (o, b, mp))
        t0 = time.time()
        res = R.play_games(games, R.OUT / "games" / f"grpo-{it}", WIDTH)
        shutil.rmtree(R.OUT / "games" / f"grpo-{it}", ignore_errors=True)
        # group-relative advantage per setup, then fitness per perturbation
        fit = np.zeros(len(bots))
        wins = []
        for s_i, (o, mp, a_side) in enumerate(sets):
            sc = []
            for k, (b, _, _) in enumerate(bots):
                r = next(x for x in res if x.map == mp.stem and b in (x.bot_a, x.bot_b) and o in (x.bot_a, x.bot_b)
                         and (x.bot_a == b) == a_side)
                sc.append(score(r, b))
                wins.append(r.score_for(b))
            sc = np.array(sc)
            if sc.std() > 1e-9:
                fit += (sc - sc.mean()) / sc.std()
        grad_w = np.zeros_like(dw)
        grad_b = np.zeros_like(db)
        for k, (_, j, sgn) in enumerate(bots):
            ew, eb = dirs[j]
            grad_w += sgn * fit[k] * ew
            grad_b += sgn * fit[k] * eb
        grad = np.concatenate([grad_w.ravel(), grad_b]) / (len(bots) * SETUPS)
        # Normalised step with momentum: as long as LR x a typical perturbation, along the
        # (smoothed) gradient. Adam's per-coordinate scaling turned one rank-1 nudge into a
        # full-rank jump (|theta| 130 after one step vs ~6 per perturbation).
        m = 0.7 * m + 0.3 * grad
        typical = sigma * np.sqrt(np.mean([np.sum(ew**2) + np.sum(eb**2) for ew, eb in dirs]))
        step = LR * typical * m / (np.linalg.norm(m) + 1e-12)
        dw = dw + step[: g * n].reshape(g, n).astype(np.float32)
        db = db + step[g * n :].astype(np.float32)
        st["it"], st["games"] = it + 1, st["games"] + len(res)
        np.savez(d / "theta.npz", dw=dw, db=db, m=m, v=v)
        log(d, f"it {it + 1}: {len(res)} games in {time.time() - t0:.0f}s, perturbed win {np.mean(wins):.3f}, "
               f"fitness spread {fit.std():.2f}, |theta| {np.linalg.norm(dw):.3f}")
        if st["it"] % EVAL_EVERY == 0:
            e = evaluate(d, base, dw, db, f"it{st['it']}", a.exploit)
            st["evals"].append({"it": st["it"], "games": st["games"], **e})
            best = max(st["evals"], key=lambda x: (x["win"], x["margin"]))
            log(d, f"eval it {st['it']}: {e} (best it {best['it']}: {best['win']})")
            if best["it"] == st["it"]:
                np.savez(d / "best.npz", dw=dw, db=db)
                shutil.copyfile(d / "eval.bin", d / "best.bin")
            elif e["win"] < best["win"] - RESET_DROP:
                z = np.load(d / "best.npz")
                dw, db = z["dw"], z["db"]
                log(d, f"rolled back to it {best['it']}")
        save_state(d, st)


def evaluate(d: Path, base: bcn1.Blob, dw, db, tag: str, exploit: str | None) -> dict:
    """Unperturbed adapter: vs the rule bot on every map and side (or, exploiter: vs its target
    on every map and side)."""
    w = d / "eval.bin"
    bcn1.write(bcn1.with_goal_delta(base, dw, db), w)
    if not exploit:
        e = R.evaluate(w, WIDTH, f"grpo-{tag}")
        return {"win": e["train"], "held_out": e["held_out"], "margin": e["gem_margin"]}
    bot = R.variant("_grpox_eval", 2, 0, w)
    train, held = R.maps()
    res = R.play([(bot, exploit), (exploit, bot)], train + held, R.OUT / "games" / f"grpox-{tag}", WIDTH)
    shutil.rmtree(R.OUT / "games" / f"grpox-{tag}", ignore_errors=True)
    win = sum(r.score_for(bot) for r in res) / max(len(res), 1)
    mg = sum(((r.gems_a - r.gems_b) if r.bot_a == bot else (r.gems_b - r.gems_a)) for r in res) / max(len(res), 1)
    return {"win": round(win, 3), "held_out": None, "margin": round(mg, 1)}


def cmd_status(a) -> int:
    d = out_dir(a.exploit)
    st = load_state(d)
    print(f"iteration {st['it']}, {st['games']} games, sigma {st.get('sigma')}")
    for e in st["evals"][-12:]:
        print(f"  it {e['it']:4d} ({e['games']} games): win {e['win']}, held-out {e.get('held_out')}, margin {e['margin']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["run", "status"])
    p.add_argument("--exploit", help="train against this one opponent (bots/<name>)")
    a = p.parse_args(argv)
    return cmd_run(a) if a.cmd == "run" else cmd_status(a)


if __name__ == "__main__":
    raise SystemExit(main())
