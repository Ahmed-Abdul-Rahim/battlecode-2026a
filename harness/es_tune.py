"""Evolution strategies over the bot's decision constants (bots/main/src/learned.rs).

The ladder's top teams share economic tempo and *survival* (0.3-1.8 deaths per game
against our ~17; harness/ladder_study.py). Hand-set constants (camp scoring weights,
fight margin, retreat HP, assist HP, tribute and shop thresholds) are searched as
log-multipliers on their defaults with a (mu, lambda)-ES. Fitness on the training maps:

    win rate vs a pool + 0.25 x gem margin  - DEATH_W x our deaths per game
    + SPEED_W x relative speedrun improvement (rounds to 1000 vs bots/idle)

Held-out maps (maps/heldout) are only used for the periodic validation line.

    python harness/es_tune.py --gens 20 --lam 8          # train (hours)
    python harness/es_tune.py --apply runs/es/<ts>/best.json   # write learned.rs
Local only: nothing here touches the Battlecode platform.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import speedrun

from bctools import ROOT, gate, runner

LEARNED = ROOT / "bots" / "main" / "src" / "learned.rs"
POOL = ["main-v12", "main-v11", "main-v10", "main-v9"]
DEATH_W = 0.01
SPEED_W = 0.3
MARGIN_W = 0.25
CLIP = (math.log(0.25), math.log(4.0))


def n_params() -> int:
    for line in LEARNED.read_text(encoding="utf-8").splitlines():
        if line.startswith("pub const N: usize"):
            return int(line.split("=")[1].strip(" ;"))
    raise ValueError("N not found in learned.rs")


def learned_rs(w: list[float]) -> str:
    text = LEARNED.read_text(encoding="utf-8")
    head = text.split("pub const W:")[0]
    body = ", ".join(f"{x:.4f}" for x in w)
    return head + f"pub const W: [f32; N] = [{body}];\n"


def make_bot(name: str, w: list[float]) -> str:
    bot = f"_es-{name}"
    d = ROOT / "bots" / bot
    if d.exists():
        shutil.rmtree(d / "src", ignore_errors=True)
        shutil.copytree(ROOT / "bots" / "main" / "src", d / "src")
    else:
        shutil.copytree(ROOT / "bots" / "main", d, ignore=shutil.ignore_patterns(".cambc", "target"))
    (d / "src" / "learned.rs").write_text(learned_rs(w), encoding="utf-8", newline="\n")
    return bot


def maps(heldout: bool) -> list[Path]:
    held = gate.heldout_names()
    return [p for p in runner.list_maps() if (p.stem in held) == heldout]


def evaluate(bot: str, out: Path, map_list: list[Path], base_speed: float | None) -> dict:
    games = deaths = 0
    win = margin = 0.0
    for opp in POOL:
        for r in runner.run_matches(bot, opp, maps=map_list, out_dir=out / opp, release=True):
            ours, theirs = (r.gems_a, r.gems_b) if r.bot_a == bot else (r.gems_b, r.gems_a)
            win += r.score_for(bot)
            margin += (ours - theirs) / max(ours + theirs, 1)
            deaths += r.deaths_a if r.bot_a == bot else r.deaths_b
            games += 1
    speed = speedrun.evaluate(bot, out / "speed", release=True, maps=map_list)["mean"]
    win, margin, deaths = win / games, margin / games, deaths / games
    fit = win + MARGIN_W * margin - DEATH_W * deaths
    if base_speed:
        fit += SPEED_W * (base_speed - speed) / base_speed
    return {"fit": round(fit, 4), "win": round(win, 3), "deaths": round(deaths, 2), "speed": speed}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gens", type=int, default=20)
    p.add_argument("--lam", type=int, default=8)
    p.add_argument("--mu", type=int, default=3)
    p.add_argument("--sigma", type=float, default=0.3)
    p.add_argument("--validate-every", type=int, default=4)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--apply", type=Path, help="write best.json's multipliers into learned.rs")
    a = p.parse_args(argv)
    if a.apply:
        w = json.loads(a.apply.read_text(encoding="utf-8"))["w"]
        LEARNED.write_text(learned_rs(w), encoding="utf-8", newline="\n")
        print(f"wrote {LEARNED}: {w}")
        return 0
    rng = random.Random(a.seed)
    n = n_params()
    out = ROOT / "runs" / "es" / time.strftime("%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    train, held = maps(False), maps(True)
    mean = [0.0] * n
    sigma = a.sigma
    base = evaluate(make_bot("base", [1.0] * n), out / "base", train, None)
    base_speed = base["speed"]
    base = evaluate("_es-base", out / "base2", train, base_speed)
    best = {"fit": base["fit"], "w": [1.0] * n, "gen": 0, **base}
    log = [{"gen": 0, "base": base}]
    print(f"base {base}", flush=True)
    for g in range(1, a.gens + 1):
        pop = []
        improved = False
        for k in range(a.lam):
            z = [rng.gauss(0, 1) for _ in range(n)]
            m = [min(max(mean[i] + sigma * z[i], CLIP[0]), CLIP[1]) for i in range(n)]
            w = [math.exp(x) for x in m]
            res = evaluate(make_bot(f"c{k}", w), out / f"g{g}" / f"c{k}", train, base_speed)
            pop.append((res["fit"], m, w, res))
            if res["fit"] > best["fit"]:
                improved = True
                best = {"fit": res["fit"], "w": [round(x, 4) for x in w], "gen": g, **res}
                (out / "best.json").write_text(json.dumps(best, indent=1), encoding="utf-8")
        pop.sort(key=lambda x: -x[0])
        elite = pop[: a.mu]
        new_mean = [sum(e[1][i] for e in elite) / a.mu for i in range(n)]
        # Step control: widen after a new best, narrow otherwise.
        sigma = min(0.6, sigma * 1.1) if improved else max(0.05, sigma * 0.85)
        mean = new_mean
        entry = {"gen": g, "sigma": round(sigma, 3), "top": [e[3] for e in elite], "best_fit": best["fit"]}
        print(f"gen {g:2d} sigma {sigma:.3f} top {elite[0][3]} best {best['fit']}", flush=True)
        if g % a.validate_every == 0:
            wm = [math.exp(x) for x in mean]
            entry["validate_mean"] = evaluate(make_bot("val", wm), out / f"val{g}", held, None)
            entry["validate_base"] = evaluate("_es-base", out / f"valbase{g}", held, None)
            print(f"   held-out: mean {entry['validate_mean']}  base {entry['validate_base']}", flush=True)
        log.append(entry)
        (out / "log.json").write_text(json.dumps(log, indent=1), encoding="utf-8")
    (out / "best.json").write_text(json.dumps(best, indent=1), encoding="utf-8")
    print(f"best {best}\nsaved {out / 'best.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
