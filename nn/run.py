"""Loop 3: the teacher pipeline (this PC only; `scripts\\train.ps1 ml`).

    python nn/run.py run        # resume / continue the protocol below, forever
    python nn/run.py status     # where it is, the learning curves
    python nn/run.py gate       # the full gate for the best teacher so far

Protocol (docs/architecture.md, Phase 2): imitation data from the rules (TEACHER=1) -> imitation
per width -> PPO runs (one config each) until their learning curve is flat, judged against the
rule bot. Nothing is thrown away: every evaluation is appended to runs/nn/state.json.
Games are deterministic, so evaluations use every (map, side) once.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "harness"))
sys.path.insert(0, str(ROOT / "tools"))

import experiments as X  # noqa: E402
from bctools import runner  # noqa: E402

from nn import data  # noqa: E402

OUT = ROOT / "runs" / "nn"
STATE = OUT / "state.json"
GPU_PY = Path(os.environ.get("NN_GPU_PY", r"D:\Code\ml-gpu-venv\Scripts\python.exe"))
HELD_OUT = ["00contest", "00monolith", "01islands", "01corridors"]
LOG_LIMIT = 64 * 1024 * 1024
FUEL = {512: 50_000_000, 1024: 200_000_000}
GAME_TIMEOUT = 900
ANCHORS = [
    "baseline", "claude-pillager", "claude-ambusher", "claude-focus",
    "exploit-qwen-rush", "exploit-qwen-pillager", "exploit-qwen-spawn_camper",
]
IMIT_GAMES = 288
ROLL_GAMES = 56
EVAL_EVERY = 4  # PPO iterations between evaluations
#: Teacher's first layer factorised to this rank: games ~2x faster (90 s -> 47 s per 56), no loss
#: measured (80-game evaluation 0.528 vs the full teacher's 0.486, within noise; 2026-10-23).
RANK = 64
PLATEAU_EVALS = 5
MIN_GAMES = 20_000  # PPO training games before a run may be called flat
# The sweep, in order: early signal first. Each runs until its curve is flat.
# Entropy 0 and an anchor to the imitation policy (KL near `anchor`): the first run without them
# drifted at random and lost (0.50 -> 0.29 vs the rules in 16 iterations, 2026-10-23).
SWEEP = [
    {"width": 512, "seed": 0, "lr": 1e-4, "entropy": 0.0, "reward": "shaped", "anchor": 0.03},
    {"width": 512, "seed": 0, "lr": 1e-4, "entropy": 0.003, "reward": "win", "anchor": 0.1},  # explores more
    {"width": 512, "seed": 1, "lr": 1e-4, "entropy": 0.0, "reward": "shaped", "anchor": 0.03},
    {"width": 512, "seed": 0, "lr": 1e-4, "entropy": 0.0, "reward": "win", "anchor": 0.03},
    {"width": 512, "seed": 0, "lr": 3e-5, "entropy": 0.0, "reward": "shaped", "anchor": 0.01},
    {"width": 1024, "seed": 0, "lr": 1e-4, "entropy": 0.0, "reward": "shaped", "anchor": 0.03},
    {"width": 512, "seed": 2, "lr": 1e-4, "entropy": 0.0, "reward": "shaped", "anchor": 0.03},
    {"width": 1024, "seed": 1, "lr": 1e-4, "entropy": 0.0, "reward": "win", "anchor": 0.03},
]
#: An evaluation this far below the run's best rolls the run back to its best checkpoint.
RESET_DROP = 0.10
PATCH_EVERY = 6 * 3600
REWARDS = {
    "shaped": dict(data.WEIGHTS),
    "win": {k: (v if k in ("win", "margin", "fault") else 0.0) for k, v in data.WEIGHTS.items()},
}


def log(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}", flush=True)


def load_state() -> dict:
    if STATE.exists():
        return json.loads(STATE.read_text(encoding="utf-8"))
    return {"stage": "imitation_data", "bc": {}, "runs": [], "snapshots": []}


def save_state(st: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(st, f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(STATE)


def valid_ckpt(d: Path) -> bool:
    """A checkpoint survived intact (a power cut can leave zero-filled files)."""
    try:
        json.loads((d / "meta.json").read_text(encoding="utf-8"))
        with open(d / "teacher.bin", "rb") as f:
            ok = f.read(4) == b"BCN1"
        return ok and all((d / n).stat().st_size > 0 for n in ("policy.pt", "value.pt"))
    except (OSError, ValueError):
        return False


def ensure_rank(ck: Path) -> Path:
    """Factorise a full checkpoint in place to RANK (older checkpoints were full)."""
    meta = json.loads((ck / "meta.json").read_text(encoding="utf-8"))
    if meta.get("rank", 0) == RANK:
        return ck
    tmp = ck.with_name(ck.name + ".r")
    gpu(["factorize", "--init", str(ck), "--out", str(tmp), "--rank", str(RANK)])
    shutil.rmtree(ck)
    tmp.rename(ck)
    log(f"{ck.name}: first layer factorised to rank {RANK}")
    return ck


def last_valid(rdir: Path, it: int) -> int:
    """The newest intact checkpoint at or before iteration `it` (broken ones are removed)."""
    for k in range(it, -1, -1):
        d = rdir / f"it{k:04d}"
        if d.exists() and valid_ckpt(d):
            return k
        shutil.rmtree(d, ignore_errors=True)
    return -1


def maps() -> tuple[list[Path], list[Path]]:
    all_maps = runner.list_maps()
    train = [m for m in all_maps if m.stem not in HELD_OUT]
    held = [m for m in all_maps if m.stem in HELD_OUT]
    return train, held


def all_mimics() -> list[str]:
    return sorted(p.name for p in (ROOT / "bots").glob("mimic-*") if (p / "Cargo.toml").exists())


#: Opponents never used in training (overfitting check): every 7th mimic, at most 8.
def held_mimics() -> list[str]:
    return all_mimics()[::7][:8]


def mimics() -> list[str]:
    held = set(held_mimics())
    return [m for m in all_mimics() if m not in held]


HELD_OPP_MAPS = 6  # maps (spread over the train pool) for the held-out opponent check
HELD_OPP_EVERY = 3  # every n-th evaluation


def _stamp(teacher: int, seed: int, weights: Path | None) -> str:
    """What a bot copy was built from: bots/main's sources, the knobs and the weights."""
    import hashlib

    h = hashlib.sha256(f"{teacher}/{seed}".encode())
    for f in sorted((ROOT / "bots" / "main" / "src").rglob("*")):
        if f.is_file() and f.name != "teacher.bin":
            st = f.stat()
            h.update(f"{f.relative_to(ROOT)}:{st.st_size}:{st.st_mtime_ns}".encode())
    if weights is not None:
        h.update(Path(weights).read_bytes())
    return h.hexdigest()


def variant(name: str, teacher: int, seed: int = 0, weights: Path | None = None) -> str:
    """bots/<name> = bots/main with these knobs and weights; reused (no rebuild) if unchanged."""
    stamp = _stamp(teacher, seed, weights)
    mark = ROOT / "bots" / name / ".nn-stamp"
    if mark.exists() and mark.read_text() == stamp:
        return name
    X.knob_variant(name, {"TEACHER": teacher, "RL_SEED": seed})
    if weights is not None:
        shutil.copyfile(weights, ROOT / "bots" / name / "src" / "weights" / "teacher.bin")
    mark.write_text(stamp)
    return name


def _parse(job: tuple[str, str, str, float, float, str]) -> dict:
    replay, stats, side, won, margin, reward = job
    try:
        return data.game_samples(Path(replay), Path(stats), side, won, margin, REWARDS[reward])
    except (OSError, KeyError, ValueError) as e:
        return {"error": str(e)}


def play(pairs: list[tuple[str, str]], maps_: list[Path], out: Path, width: int) -> list:
    return runner.run_pairs(
        pairs, maps_, out, log_limit=LOG_LIMIT, fuel=FUEL[width], timeout=GAME_TIMEOUT
    )


def play_games(games: list[tuple[str, str, Path]], out: Path, width: int) -> list:
    return runner.run_games(games, out, log_limit=LOG_LIMIT, fuel=FUEL[width], timeout=GAME_TIMEOUT)


def samples(results: list, bot: str, reward: str) -> tuple[dict, dict]:
    """Training samples of `bot`'s side in every finished game, parsed in parallel."""
    jobs = []
    for r in results:
        if r.winner is None and r.rounds == 0:
            continue
        side = "a" if r.bot_a == bot else "b"
        ours, theirs = (r.gems_a, r.gems_b) if side == "a" else (r.gems_b, r.gems_a)
        jobs.append((r.replay, r.stats, side, r.score_for(bot), (ours - theirs) / max(ours + theirs, 1), reward))
    with ProcessPoolExecutor(max_workers=max(2, (os.cpu_count() or 4) - 4)) as ex:
        parts = list(ex.map(_parse, jobs))
    errors = [p["error"] for p in parts if "error" in p]
    d = data.concat([p for p in parts if p and "error" not in p])
    wins = sum(r.score_for(bot) for r in results)
    return d, {"games": len(results), "win": round(wins / max(len(results), 1), 3), "parse_errors": len(errors)}


def gpu(args: list[str]) -> dict:
    p = subprocess.run([str(GPU_PY), "-X", "utf8", str(ROOT / "nn" / "train.py"), *args],
                       cwd=ROOT, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"train.py failed: {p.stderr[-2000:]}")
    return json.loads(p.stdout.strip().splitlines()[-1])


# ---------------------------------------------------------------- stages


def imitation_data(st: dict) -> None:
    train, _ = maps()
    opps = ["main", *ANCHORS, *mimics()]
    rng = random.Random(1)
    bot = variant("_nn_rules", 1)
    games = []
    for k in range(IMIT_GAMES // 2):
        m = train[k % len(train)]
        o = rng.choice(opps)
        games += [(bot, o, m), (o, bot, m)]
    shard_dir = OUT / "data" / "imitation"
    out = OUT / "games" / "imitation"
    log(f"imitation data: {len(games)} games")
    # Played and parsed in chunks so the replays don't pile up on disk.
    for c in range(0, len(games), 56):
        if (shard_dir / f"chunk{c // 56:03d}.npz").exists():
            continue
        res = play_games(games[c : c + 56], out, 512)
        d, m = samples(res, bot, "shaped")
        if d:
            data.save(shard_dir / f"chunk{c // 56:03d}.npz", d)
        log(f"  chunk {c // 56}: {m}, {len(d.get('R', []))} decisions")
        if m["parse_errors"] == m["games"]:
            raise RuntimeError("no game could be parsed: see nn/data.py")
        shutil.rmtree(out, ignore_errors=True)
    st["stage"] = "runs"
    save_state(st)


def bc_for(st: dict, width: int) -> Path:
    key = str(width)
    ck = OUT / "ckpt" / f"bc{width}"
    if key in st["bc"] and valid_ckpt(ck):
        return ensure_rank(ck)
        
    shards = sorted(str(p) for p in (OUT / "data" / "imitation").glob("*.npz"))
    log(f"imitation training, width {width}, {len(shards)} shards")
    m = gpu(["bc", "--data", *shards, "--out", str(ck), "--width", str(width), "--rank", str(RANK)])
    st["bc"][key] = m
    save_state(st)
    log(f"  held-out accuracy {m['held_out'][-1]} after {m['epochs']} epochs")
    return ck


def evaluate(weights: Path, width: int, tag: str) -> dict:
    """Teacher (argmax) vs the rule bot on every (map, side): train maps and held-out maps."""
    train, held = maps()
    bot = variant("_nn_eval", 2, 0, weights)
    out = OUT / "games" / f"eval-{tag}"
    res = play([(bot, "main"), ("main", bot)], train + held, out, width)
    shutil.rmtree(out, ignore_errors=True)
    tr = [r for r in res if Path(r.replay).name.split("__")[0] not in HELD_OUT]
    ho = [r for r in res if Path(r.replay).name.split("__")[0] in HELD_OUT]

    def score(rs: list) -> float:
        return round(sum(r.score_for(bot) for r in rs) / max(len(rs), 1), 3)

    margin = [((r.gems_a - r.gems_b) if r.bot_a == bot else (r.gems_b - r.gems_a)) for r in res]
    return {"train": score(tr), "held_out": score(ho), "games": len(res),
            "gem_margin": round(sum(margin) / max(len(margin), 1), 1)}


def held_opponents(st: dict, weights: Path, width: int, tag: str) -> dict:
    """Paired overfitting check: the teacher and the rule bot each vs the held-out mimics on the
    same maps and sides. Reports win rates and the gem-margin difference (teacher - rules)."""
    train, _ = maps()
    ms = train[:: max(1, len(train) // HELD_OPP_MAPS)][:HELD_OPP_MAPS]
    opps = held_mimics()
    bot = variant("_nn_eval", 2, 0, weights)

    def games(b: str) -> list:
        return [(b, o, m) for o in opps for m in ms] + [(o, b, m) for o in opps for m in ms]

    def summary(res: list, b: str) -> tuple[float, float]:
        win = sum(r.score_for(b) for r in res) / max(len(res), 1)
        mg = sum(((r.gems_a - r.gems_b) if r.bot_a == b else (r.gems_b - r.gems_a)) for r in res) / max(len(res), 1)
        return round(win, 3), round(mg, 1)

    key = "|".join(opps) + "|" + "|".join(m.stem for m in ms)
    base = st.get("held_rules")
    if not base or base.get("key") != key:
        out = OUT / "games" / "held-rules"
        res = play_games(games("main"), out, width)
        shutil.rmtree(out, ignore_errors=True)
        w, mg = summary(res, "main")
        base = st["held_rules"] = {"key": key, "win": w, "margin": mg}
    out = OUT / "games" / f"held-{tag}"
    res = play_games(games(bot), out, width)
    shutil.rmtree(out, ignore_errors=True)
    w, mg = summary(res, bot)
    return {"held_opp_win": w, "held_opp_rules_win": base["win"], "held_opp_margin_gain": round(mg - base["margin"], 1)}


def ppo_run(st: dict, idx: int) -> None:
    cfg = SWEEP[idx]
    if idx >= len(st["runs"]):
        st["runs"].append({"cfg": cfg, "it": 0, "games": 0, "evals": [], "done": False})
        save_state(st)
    run = st["runs"][idx]
    width = cfg["width"]
    base = bc_for(st, width)
    rdir = OUT / "ckpt" / f"run{idx}"
    good = last_valid(rdir, run["it"])
    if good != run["it"]:
        good = max(good, 0)
        log(f"run {idx}: checkpoint it {run['it']} is damaged; resuming from it {good}")
        run["it"] = good
        run["evals"] = [e for e in run["evals"] if e["it"] <= good]
        save_state(st)
    cur = rdir / f"it{run['it']:04d}"
    if cur.exists():
        ensure_rank(cur)
    if run["it"] == 0 and not cur.exists():
        shutil.copytree(base, cur)
        e = evaluate(cur / "teacher.bin", width, f"r{idx}-it0")
        run["evals"].append({"it": 0, "games": 0, **e})
        save_state(st)
        log(f"run {idx} {cfg}: imitation teacher vs rules {e}")
    train, _ = maps()
    rng = random.Random(1000 * idx + run["it"])
    while not run["done"]:
        patch_check(st)
        it = run["it"]
        seed = (it * 7 + idx * 31) % 250 + 1
        bot = variant("_nn_play", 2, seed, cur / "teacher.bin")
        pool = ["main"] * 4 + ANCHORS[:3] + rng.sample(mimics(), k=min(3, len(mimics())))
        snaps = [s for s in st["snapshots"] if Path(s).exists()]
        games = []
        for k in range(ROLL_GAMES // 2):
            m = rng.choice(train)
            if snaps and rng.random() < 0.15:
                o = variant(f"_nn_snap{snaps.index(s := rng.choice(snaps))}", 2, 0, Path(s))
            else:
                o = rng.choice(pool)
            games += [(bot, o, m), (o, bot, m)]
        out = OUT / "games" / f"r{idx}-it{it:04d}"
        t0 = time.time()
        res = play_games(games, out, width)
        t1 = time.time()
        d, m = samples(res, bot, cfg["reward"])
        shutil.rmtree(out, ignore_errors=True)
        if not d:
            log(f"run {idx} it {it}: no samples ({m}); retrying")
            continue
        shard = OUT / "data" / f"run{idx}" / "rollout.npz"
        data.save(shard, d)
        t2 = time.time()
        nxt = rdir / f"it{it + 1:04d}"
        u = gpu(["ppo", "--data", str(shard), "--init", str(cur), "--out", str(nxt),
                 "--lr", str(cfg["lr"]), "--entropy", str(cfg["entropy"]), "--seed", str(cfg["seed"]),
                 "--anchor", run.get("anchor_ckpt", str(base)), "--anchor-target", str(cfg.get("anchor", 0.03))])
        run["it"], run["games"] = it + 1, run["games"] + m["games"]
        cur = nxt
        log(f"run {idx} it {it + 1}: rollout {m}, update kl {u['kl']} ent {u['entropy']} "
            f"clip {u['clipfrac']} reward {u['reward_mean']} anchor kl {u.get('anchor_kl')} beta {u.get('anchor_beta')} | build+games {t1 - t0:.0f}s, "
            f"parse+save {t2 - t1:.0f}s, update {time.time() - t2:.0f}s (gpu {u.get('seconds')}s)")
        # Keep only the evaluated checkpoints and the latest one.
        old = rdir / f"it{it:04d}"
        if it % EVAL_EVERY and old.exists():
            shutil.rmtree(old, ignore_errors=True)
        if run["it"] % EVAL_EVERY == 0:
            e = evaluate(cur / "teacher.bin", width, f"r{idx}-it{run['it']}")
            if (run["it"] // EVAL_EVERY) % HELD_OPP_EVERY == 0:
                e.update(held_opponents(st, cur / "teacher.bin", width, f"r{idx}-it{run['it']}"))
            run["evals"].append({"it": run["it"], "games": run["games"], **e})
            best = max(run["evals"], key=lambda x: (x["train"], x["gem_margin"]))
            log(f"run {idx} eval it {run['it']}: {e} (best it {best['it']}: {best['train']})")
            if best["it"] == run["it"] and e["train"] > 0.5:
                st["snapshots"] = [*st["snapshots"], str(cur / "teacher.bin")][-6:]
                # Re-anchor to the new best: the policy stays near the best model found so far,
                # not near the rules, so it can keep moving away from imitation as it wins more.
                keep = rdir / f"anchor-it{run['it']:04d}"
                shutil.rmtree(keep, ignore_errors=True)
                shutil.copytree(cur, keep)
                run["anchor_ckpt"] = str(keep)
                log(f"run {idx}: anchor moved to it {run['it']} (new best {e['train']})")
            if e["train"] < best["train"] - RESET_DROP and best["it"] != run["it"]:
                src = rdir / f"it{best['it']:04d}"
                if valid_ckpt(src):
                    shutil.rmtree(cur, ignore_errors=True)
                    shutil.copytree(src, cur)
                    run.setdefault("resets", []).append({"at": run["it"], "to": best["it"]})
                    log(f"run {idx}: eval {e['train']} is {RESET_DROP}+ below the best {best['train']}: "
                        f"rolled back to it {best['it']}'s weights")
            recent = run["evals"][-PLATEAU_EVALS:]
            flat = len(run["evals"]) > PLATEAU_EVALS and best not in recent
            if flat and run["games"] >= MIN_GAMES:
                run["done"] = True
                log(f"run {idx}: flat after {run['games']} games; best it {best['it']} {best['train']}")
        save_state(st)


def patch_check(st: dict) -> None:
    """Every PATCH_EVERY: apply a new game patch (SDK, maps) and note it; the runs continue
    from their current best on the new rules (fine-tune, not restart)."""
    if time.time() - st.get("patch_checked", 0) < PATCH_EVERY:
        return
    st["patch_checked"] = time.time()
    try:
        from nn import patch

        rep = patch.run_check(do_apply=True)
        if rep.get("changed"):
            st.setdefault("patches", []).append({"id": rep.get("applied_id"), "time": rep["time"],
                                                 "constants": rep["constants"], "maps": rep["maps"]})
            log(f"game patch applied: {rep.get('applied_id')} ({len(rep['constants'])} constant changes, maps {rep['maps']})")
    except Exception as e:  # noqa: BLE001 - a failed check must not stop training
        log(f"patch check failed: {e}")
    save_state(st)


def cmd_run(_: argparse.Namespace) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    st = load_state()
    if st["stage"] == "imitation_data":
        imitation_data(st)
    for idx in range(len(SWEEP)):
        if idx < len(st["runs"]) and st["runs"][idx]["done"]:
            continue
        ppo_run(st, idx)
    log("sweep finished")
    return 0


def cmd_status(_: argparse.Namespace) -> int:
    st = load_state()
    print(f"stage: {st['stage']}")
    for w, m in st["bc"].items():
        print(f"imitation width {w}: held-out accuracy {m['held_out'][-1]} ({m['samples']} samples)")
    for i, r in enumerate(st["runs"]):
        ev = r["evals"]
        best = max(ev, key=lambda x: x["train"]) if ev else None
        state = "done" if r["done"] else "running"
        print(f"run {i} {r['cfg']}: {state}, it {r['it']}, {r['games']} games")
        if best:
            print(f"   best: it {best['it']} vs rules {best['train']} (held-out {best['held_out']}, margin {best['gem_margin']})")
        print("   curve (games: win vs rules): " + ", ".join(f"{e['games']}: {e['train']}" for e in ev[-10:]))
        held = [e for e in ev if "held_opp_win" in e]
        if held:
            h = held[-1]
            print(f"   held-out opponents (it {h['it']}): {h['held_opp_win']} vs rules' {h['held_opp_rules_win']}, "
                  f"margin gain {h['held_opp_margin_gain']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("run").set_defaults(fn=cmd_run)
    sub.add_parser("status").set_defaults(fn=cmd_status)
    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
