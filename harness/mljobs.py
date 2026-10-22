"""Neural-net jobs shared by every PC (league/ml_jobs.json): networks that learn by playing.

Two arms (ml/rl/ppo.py):
    arm 1  experts + fight target (knob2 ROUTER, weights/router.bin)
    arm 2  micro control on fight turns (knob2 MICRO_NET, weights/micro.bin), experimental

Per arm and iteration:
    rlinit   iteration 0: arm 1 starts from the rules' choices, arm 2 near-uniform
    rollout  RL_PARTS jobs: the current network plays (sampling) against the mimics and the map's
             exploiter; its decisions and rewards (ml/rl/reward.py) go to branch ml-data
    update   PPO: on the GPU with a big teacher distilled into the bot's network (ml/rl/ppo_gpu.py)
    eval     every RL_EVAL_EVERY iterations, a fair check (fixed maps, no sampling); the best network
             is kept and training resets to it when a network falls behind; a new best is queued
             as an experiment, so it only reaches the bot through the usual gate. Training runs on the GPU only: rlinit and update jobs go to PCs with
a CUDA torch Python (GPU_PYTHON) and fail (to be retried) rather than fall back to the CPU; any
PC plays rollouts (claims, as for experiments).

    python harness/mljobs.py status | list
    python harness/mljobs.py run [--forever] [--no-push]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import experiments as X

import league as L
from bctools import ROOT, runner
from ml.rl import reward

REGISTRY = L.LEAGUE / "ml_jobs.json"
STORE_BRANCH = "ml-data"
STORE = ROOT.parent / f"{ROOT.name}-mldata"
OUT = L.SCRATCH / "ml"
KINDS = ("rlinit", "rollout", "update", "eval")
#: Fair checks: every RL_EVAL_EVERY iterations (before its rollouts) the network plays without
#: sampling on the same RL_EVAL_MAPS maps against the faithful mimics, both sides. The best
#: score is kept; a network more than RL_EVAL_MARGIN below it is replaced by the best one, so
#: training goes on from the best network. A new best goes to the experiment queue.
RL_EVAL_EVERY = 2
RL_EVAL_MAPS = 3
RL_EVAL_MARGIN = 0.05
RL_PARTS = 2  # rollout jobs per iteration
RL_MAPS = 3  # maps per rollout job (rotating through the pool)
RL_ARMS = (1, 2)
RL_ARM_SHARE = 2 / 3  # arm 1's share of the jobs
RL_WEIGHTS = {1: ("ROUTER", "router.bin"), 2: ("MICRO_NET", "micro.bin")}
#: Keep every n-th decision per champion (arm 1 decides every turn; its full logs are too big
#: to share through git each iteration; the rewards in between are kept).
RL_KEEP_EVERY = {1: 8, 2: 1}
SHARD_KEYS = ("X", "A", "LP", "M", "R", "DONE", "TRAJ")
FAILURES_SHOWN_FOR = 24 * 3600
CLAIM_TTL = X.CLAIM_TTL
#: A Python with CUDA torch for ml/rl/ppo_gpu.py; PCs without one only play rollouts.
GPU_PYTHON = Path(
    os.environ.get("ML_GPU_PYTHON")
    or ROOT.parent / "ml-gpu-venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
)


#: Job kinds that train networks: GPU only.
GPU_KINDS = ("rlinit", "update")
GPU_RETRY_WAIT = 600  # seconds after a failed training job before the next try
_gpu: bool | None = None


def gpu_ready() -> bool:
    """This PC can train: GPU_PYTHON exists and sees a CUDA device (checked once per process)."""
    global _gpu
    if _gpu is None:
        check = "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)"
        _gpu = (
            GPU_PYTHON.exists()
            and subprocess.run([str(GPU_PYTHON), "-c", check], capture_output=True).returncode == 0
        )
    return _gpu


def load() -> dict:
    return {"jobs": [], **L.load_json(REGISTRY, {})}


def save(reg: dict) -> None:
    L.save_json(REGISTRY, reg)


def add(reg: dict, kind: str, params: dict | None = None) -> dict:
    assert kind in KINDS, kind
    j = {
        "id": 1 + max((x["id"] for x in reg["jobs"]), default=0),
        "kind": kind,
        "params": params or {},
        "status": "queued",
        "result": {},
        "added": L.now_iso(),
    }
    reg["jobs"].append(j)
    return j


def available(j: dict, now: float) -> bool:
    """Queued, or a claim of ours / gone stale. Old job kinds (the retired imitation loop) never."""
    if j["kind"] not in KINDS:
        return False
    if j["kind"] in GPU_KINDS and not gpu_ready():
        return False
    if j["status"] == "queued":
        return True
    if j["status"] != "running" or "claimed" not in j:
        return False
    return j.get("host") == L.HOST or now - L.ts(j["claimed"]) > CLAIM_TTL


def next_job(reg: dict, now: float | None = None) -> dict | None:
    now = time.time() if now is None else now
    return next((j for j in reg["jobs"] if available(j, now)), None)


def rl_state(reg: dict, arm: int) -> dict:
    """Where arm `arm` is: initialised?, iteration (= finished updates), rollouts done/running for
    it, whether its update is running."""
    jobs = [j for j in reg["jobs"] if j["kind"] in KINDS and j["params"].get("arm") == arm]
    done = [j for j in jobs if j["status"] == "done"]
    it = sum(1 for j in done if j["kind"] == "update" and not j["result"].get("duplicate"))
    cur = [j for j in jobs if j["params"].get("it") == it]
    return {
        "init": any(j["kind"] == "rlinit" for j in done),
        "init_busy": any(j["kind"] == "rlinit" and j["status"] in ("queued", "running") for j in jobs),
        "it": it,
        "rolled": sum(1 for j in cur if j["kind"] == "rollout" and j["status"] == "done"),
        "rolling": sum(1 for j in cur if j["kind"] == "rollout" and j["status"] in ("queued", "running")),
        "updating": any(j["kind"] == "update" and j["status"] in ("queued", "running") for j in cur),
        "evaluated": any(j["kind"] == "eval" and j["status"] == "done" for j in cur),
        "eval_busy": any(j["kind"] == "eval" and j["status"] in ("queued", "running") for j in cur),
    }


def best_eval(reg: dict, arm: int) -> tuple[float, int] | None:
    """(win, iteration) of the arm's best fair check so far (the later one on a tie)."""
    evals = [
        j
        for j in reg["jobs"]
        if j["kind"] == "eval"
        and j["params"].get("arm") == arm
        and j["status"] == "done"
        and "win" in j["result"]
    ]
    return max(((j["result"]["win"], j["params"]["it"]) for j in evals), default=None)


def plan_rl(reg: dict, gpu: bool = True) -> tuple[str, dict] | None:
    """The next job, arm 1 first by share; None when both arms wait for running jobs (or, on a PC
    without a GPU, for training jobs it can't run)."""
    work = sum(1 for j in reg["jobs"] if j["params"].get("arm") == 1)
    total = sum(1 for j in reg["jobs"] if j["params"].get("arm") in RL_ARMS)
    order = (1, 2) if work <= RL_ARM_SHARE * max(total, 1) else (2, 1)
    for arm in order:
        s = rl_state(reg, arm)
        if not s["init"]:
            if not s["init_busy"] and gpu:
                return "rlinit", {"arm": arm}
            continue
        # A fair check comes before an iteration's rollouts (once they started, the network
        # can't be swapped any more).
        if s["it"] % RL_EVAL_EVERY == 0 and not s["evaluated"] and s["rolled"] + s["rolling"] == 0:
            if not s["eval_busy"]:
                return "eval", {"arm": arm, "it": s["it"]}
            continue
        if s["rolled"] + s["rolling"] < RL_PARTS:
            return "rollout", {"arm": arm, "it": s["it"], "part": s["rolled"] + s["rolling"]}
        if s["rolled"] >= RL_PARTS and not s["updating"] and gpu:
            return "update", {"arm": arm, "it": s["it"]}
    return None


def publish(paths: list[Path], msg: str, push: bool) -> None:
    if push and L.git_ready():
        L.publish(paths, msg, L.h2h_decider(OUT, L.load_champion()["maps"]))


def claim(j: dict, reg: dict, push: bool) -> bool:
    j.update(status="running", host=L.HOST, claimed=L.now_iso())
    save(reg)
    publish([REGISTRY], f"league: ml job {j['id']} ({j['kind']}) claimed by {L.HOST}", push)
    now = next((x for x in load()["jobs"] if x["id"] == j["id"]), j)
    return now.get("host") == L.HOST


# ----------------------------------------------------------------- data store (branch ml-data)


def git(*args: str, cwd: Path = STORE) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)


def store_ready() -> Path:
    """A worktree of branch ml-data next to the checkout (created, orphan, on first use)."""
    git("fetch", "-q", "origin", STORE_BRANCH, cwd=ROOT)
    remote = git("rev-parse", "--verify", "-q", f"origin/{STORE_BRANCH}", cwd=ROOT).returncode == 0
    if not STORE.exists():
        if remote:
            git("worktree", "add", "-q", "-B", STORE_BRANCH, str(STORE), f"origin/{STORE_BRANCH}", cwd=ROOT)
        else:
            git("worktree", "add", "-q", "--detach", str(STORE), cwd=ROOT)
            git("checkout", "-q", "--orphan", STORE_BRANCH)
            git("rm", "-rq", "--cached", ".")
            for p in STORE.iterdir():
                if p.name != ".git":
                    shutil.rmtree(p) if p.is_dir() else p.unlink()
            (STORE / "README.md").write_text(
                "Neural-net rollouts and networks (rl/arm<N>/), written by harness/mljobs.py.\n"
                "Not a code branch.\n",
                encoding="utf-8",
            )
            git("add", "README.md")
            git("commit", "-qm", "ml-data: start")
            git("push", "-q", "-u", "origin", STORE_BRANCH)
    elif remote:
        git("pull", "-q", "--rebase", "origin", STORE_BRANCH)
    return STORE


def store_push(files: list[Path], msg: str, push: bool) -> bool:
    git("add", "--", *[str(f.relative_to(STORE)) for f in files])
    git("commit", "-qm", f"{msg}\n\nml host: {L.HOST}")
    if not push:
        return True
    for _ in range(3):
        if git("push", "-q", "origin", f"HEAD:{STORE_BRANCH}").returncode == 0:
            return True
        git("pull", "-q", "--rebase", "origin", STORE_BRANCH)  # file names are unique per job
    L.log("ml: push to ml-data failed; the commit stays local and goes out with the next push")
    return False


def rl_path(arm: int, name: str) -> Path:
    return STORE / "rl" / f"arm{arm}" / name


# ----------------------------------------------------------------- jobs


def run_rlinit(j: dict, push: bool) -> dict:
    """Iteration 0 of an arm, on the GPU (ml/rl/ppo_gpu.py --init): arm 1 imitates the rules'
    expert choices (the logged rule decisions in ml-data data/), arm 2 starts near-uniform."""
    arm = j["params"]["arm"]
    store = store_ready()
    m = gpu_train(arm, ["--init"], store)
    files = [rl_path(arm, "policy-0000.npz"), rl_path(arm, "policy-0000.bin")]
    store_push(files, f"ml-data: rl arm {arm} iteration 0", push)
    return m


def run_rollout(j: dict, push: bool) -> dict:
    """Play the arm's current network (sampling, seeded per iteration and part) against the
    league: faithful mimics on rotating maps, plus each map's exploiter. Store the decisions."""
    arm, it, part = j["params"]["arm"], j["params"]["it"], j["params"]["part"]
    store_ready()
    knob, wfile = RL_WEIGHTS[arm]
    name = f"_rl{arm}"
    X.knob_variant(name, {knob: 1, "RL_SEED": (it * 7 + part) % 250 + 1})
    (ROOT / "bots" / name / "src" / "weights" / wfile).write_bytes(
        rl_path(arm, f"policy-{it:04d}.bin").read_bytes()
    )
    led = L.load_ledger()
    faithful, _ = L.ensure_mimics(led)
    pool = sorted(L.trainable_maps(led).items())
    k = (it * RL_PARTS + part) * RL_MAPS
    maps = [pool[(k + i) % len(pool)] for i in range(min(RL_MAPS, len(pool)))]
    out = OUT / f"rl{arm}-{it:04d}-{part}"
    res = runner.run_pairs([p for o in faithful for p in ((name, o), (o, name))], [m for _, m in maps], out)
    xpl = L.ensure_exploiters()
    for mp, path in maps:
        if mp in xpl:
            res += runner.run_pairs([(name, xpl[mp]), (xpl[mp], name)], [path], out / f"xpl-{mp}")
    parts, wins, terms = [], 0.0, {}
    for r in res:
        side = "a" if r.bot_a == name else "b"
        ours, theirs = (r.gems_a, r.gems_b) if side == "a" else (r.gems_b, r.gems_a)
        won = r.score_for(name)
        wins += won
        margin = (ours - theirs) / max(ours + theirs, 1)
        try:
            parts.append(
                reward.game_samples(
                    Path(r.replay), Path(r.stats), side, arm, won, margin, RL_KEEP_EVERY[arm], terms
                )
            )
        except (OSError, KeyError, ValueError):
            continue
    data = reward.concat(parts)
    shard = rl_path(arm, f"it{it:04d}-p{part}-{L.HOST}.npz")
    if data:
        shard.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(shard, **data)
        store_push([shard], f"ml-data: rl arm {arm} it {it} part {part} ({len(data['R'])} decisions)", push)
    shutil.rmtree(out, ignore_errors=True)
    return {
        "arm": arm,
        "it": it,
        "games": len(res),
        "win": round(wins / max(len(res), 1), 3),
        "decisions": len(data.get("R", [])),
        "maps": [m for m, _ in maps],
        "reward_terms": term_shares(terms),
    }


def eval_maps(pool: list[tuple[str, Path]]) -> list[tuple[str, Path]]:
    """The fixed maps of the fair checks: RL_EVAL_MAPS spread evenly over the sorted pool."""
    n = min(RL_EVAL_MAPS, len(pool))
    return [pool[k * len(pool) // n] for k in range(n)]


def run_eval(j: dict, push: bool) -> dict:
    """Fair check of the arm's current network: no sampling, fixed maps, every faithful mimic on
    both sides. Below the best check by more than RL_EVAL_MARGIN -> its files become copies of
    the best network's, so the next rollouts and update start from the best one."""
    arm, it = j["params"]["arm"], j["params"]["it"]
    store_ready()
    knob, wfile = RL_WEIGHTS[arm]
    name = f"_rl{arm}e"
    X.knob_variant(name, {knob: 1})  # RL_SEED stays 0: the argmax, as shipped
    (ROOT / "bots" / name / "src" / "weights" / wfile).write_bytes(
        rl_path(arm, f"policy-{it:04d}.bin").read_bytes()
    )
    led = L.load_ledger()
    faithful, _ = L.ensure_mimics(led)
    maps = eval_maps(sorted(L.trainable_maps(led).items()))
    out = OUT / f"eval{arm}-{it:04d}"
    res = runner.run_pairs([p for o in faithful for p in ((name, o), (o, name))], [m for _, m in maps], out)
    shutil.rmtree(out, ignore_errors=True)
    if not res:
        raise RuntimeError("fair check played no games")
    win = round(sum(r.score_for(name) for r in res) / len(res), 3)
    m = {"arm": arm, "it": it, "games": len(res), "win": win, "maps": [mp for mp, _ in maps]}
    best = best_eval(load(), arm)
    if best is None or win >= best[0]:
        m["best"] = True
        if it > 0:
            notes = f"New best fair check: win {win} over {len(res)} games on {', '.join(m['maps'])}."
            m["experiment"] = promote(arm, it, notes)
        return m
    m["best_so_far"] = {"win": best[0], "it": best[1]}
    if win < best[0] - RL_EVAL_MARGIN:
        files = []
        for stem, ext in (("policy", "npz"), ("policy", "bin"), ("teacher", "pt"), ("tvalue", "pt")):
            src, dst = rl_path(arm, f"{stem}-{best[1]:04d}.{ext}"), rl_path(arm, f"{stem}-{it:04d}.{ext}")
            if src.exists():
                shutil.copyfile(src, dst)
            elif dst.exists():  # no teacher at the best iteration: the update rebuilds it
                dst.unlink()
            files.append(dst)
        store_push(files, f"ml-data: rl arm {arm} iteration {it} reset to iteration {best[1]}", push)
        m["reset_to"] = best[1]
    return m


def term_shares(terms: dict[str, float]) -> dict:
    """Each reward term's share of |reward| before the fault cap and the clip, biggest first,
    plus how many decisions were capped or clipped."""
    counts = {k: int(terms.pop(k, 0)) for k in ("faults_capped", "clipped")}
    total = sum(terms.values()) or 1.0
    shares = {k: round(v / total, 3) for k, v in sorted(terms.items(), key=lambda kv: -kv[1]) if v}
    return {**shares, **counts}


def gpu_train(arm: int, args: list[str], store: Path) -> dict:
    """Run ml/rl/ppo_gpu.py (an update, or --init) and return its metrics. Raises on failure, so
    the job fails and is planned again: training never falls back to the CPU."""
    if not gpu_ready():
        raise RuntimeError(f"no CUDA torch Python at {GPU_PYTHON}")
    cmd = [str(GPU_PYTHON), str(ROOT / "ml" / "rl" / "ppo_gpu.py"), "--arm", str(arm), *args]
    p = subprocess.run([*cmd, "--store", str(store)], cwd=ROOT, capture_output=True, text=True)
    lines = [ln for ln in p.stdout.splitlines() if ln.startswith("{")]
    if p.returncode != 0 or not lines:
        raise RuntimeError(f"GPU training failed: {(p.stderr or p.stdout)[-300:]}")
    return json.loads(lines[-1])


def promote(arm: int, it: int, notes: str) -> int:
    """Queue the arm's network as the next experiment (first in line); its id."""
    knob, wfile = RL_WEIGHTS[arm]
    xreg = X.load()
    e = X.add(
        xreg,
        f"RL arm {arm} ({'experts+target' if arm == 1 else 'micro control'}) iteration {it}",
        "learning",
        knobs={knob: 1},
        files={f"bots/main/src/weights/{wfile}": f"ml-data:rl/arm{arm}/policy-{it:04d}.bin"},
        notes=notes,
    )
    xreg["experiments"].remove(e)
    i = next(
        (k for k, x in enumerate(xreg["experiments"]) if x["status"] == "queued"), len(xreg["experiments"])
    )
    xreg["experiments"].insert(i, e)
    X.save(xreg)
    return e["id"]


def run_update(j: dict, reg: dict, push: bool) -> dict:
    """PPO on the iteration's rollouts -> the next network (GPU teacher if available)."""
    arm, it = j["params"]["arm"], j["params"]["it"]
    if any(
        x is not j
        and x["kind"] == "update"
        and x["params"] == j["params"]
        and x["status"] in ("running", "done")
        and not x["result"].get("duplicate")
        for x in reg["jobs"]
    ):
        return {"arm": arm, "it": it, "duplicate": True}
    store = store_ready()
    shards = sorted((store / "rl" / f"arm{arm}").glob(f"it{it:04d}-p*.npz"))
    if not any(reward.read_shard(s, SHARD_KEYS) for s in shards):
        return {"arm": arm, "it": it, "skipped": "no decisions logged"}
    nxt = it + 1
    m = gpu_train(arm, ["--it", str(it)], store)  # big teacher on the GPU, distilled into the bot's net
    names = (("policy", "npz"), ("policy", "bin"), ("teacher", "pt"), ("tvalue", "pt"))
    files = [rl_path(arm, f"{name}-{nxt:04d}.{ext}") for name, ext in names]
    store_push([f for f in files if f.exists()], f"ml-data: rl arm {arm} iteration {nxt}", push)
    rolls = [
        x
        for x in reg["jobs"]
        if x["kind"] == "rollout"
        and x["params"].get("arm") == arm
        and x["params"].get("it") == it
        and x["status"] == "done"
    ]
    win = sum(x["result"].get("win", 0) for x in rolls) / max(len(rolls), 1)
    m.update(arm=arm, it=nxt, shards=len(shards), rollout_win=round(win, 3))
    return m


RUNNERS = {
    "rlinit": lambda j, reg, push: run_rlinit(j, push),
    "rollout": lambda j, reg, push: run_rollout(j, push),
    "update": run_update,
    "eval": lambda j, reg, push: run_eval(j, push),
}


def finish(j: dict, reg: dict, push: bool) -> None:
    j["finished"] = L.now_iso()
    save(reg)
    L.log(f"ml job {j['id']} {j['kind']}: {j['status']} {j['result']}")
    paths = [REGISTRY] + ([X.REGISTRY] if "experiment" in j["result"] else [])
    publish(paths, f"league: ml job {j['id']} ({j['kind']}) -> {j['status']}", push)


def cmd_run(a: argparse.Namespace) -> int:
    while True:
        L.ensure_space()
        if a.push and L.git_ready():
            L.integrate(L.h2h_decider(OUT, L.load_champion()["maps"]))
            L.apply_champion(L.load_champion())
        reg = load()
        j = next_job(reg)
        if j is None:
            planned = plan_rl(reg, gpu_ready())
            if planned is None:  # both arms wait for jobs running on other PCs
                time.sleep(300)
                continue
            j = add(reg, *planned)
            save(reg)
        if not claim(j, reg, a.push):
            L.log(f"ml: job {j['id']} was claimed by another PC; next")
            continue
        try:
            j["result"] = RUNNERS[j["kind"]](j, reg, a.push)
            j["status"] = "done"
        except Exception as ex:  # one broken job must not stop the queue
            j["status"] = "failed"
            j["result"] = {"error": repr(ex)[:300]}
        finish(j, reg, a.push)
        if not a.forever:
            return 0
        if j["status"] == "failed" and j["kind"] in GPU_KINDS:
            time.sleep(GPU_RETRY_WAIT)  # don't re-plan a failing GPU job in a tight loop


# ----------------------------------------------------------------- reporting


def shared(path: str, default: dict) -> dict:
    """A registry file as on GitHub (every PC's work), falling back to the local copy."""
    subprocess.run(["git", "fetch", "-q", "origin", L.BRANCH], cwd=ROOT, capture_output=True)
    p = subprocess.run(
        ["git", "show", f"origin/{L.BRANCH}:{path}"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return {**default, **json.loads(p.stdout)} if p.returncode == 0 and p.stdout.strip() else default


def status_report(reg: dict, xreg: dict, now: float | None = None) -> str:
    now = time.time() if now is None else now
    jobs = [j for j in reg["jobs"] if j["kind"] in KINDS]
    running = [j for j in jobs if j["status"] == "running"]

    def fixed(f: dict) -> bool:  # a later job of the same kind and arm succeeded
        return any(
            j["id"] > f["id"]
            and j["kind"] == f["kind"]
            and j["params"].get("arm") == f["params"].get("arm")
            and j["status"] == "done"
            for j in jobs
        )

    failed = [
        j
        for j in jobs
        if j["status"] == "failed"
        and not fixed(j)
        and now - L.ts(j.get("finished") or j.get("added")) < FAILURES_SHOWN_FOR
    ]
    names = {1: "arm 1 (experts + fight target)", 2: "arm 2 (micro control)"}
    by_id = {e["id"]: e for e in xreg.get("experiments", [])}
    lines = [
        f"LEARNING BY PLAYING (PPO; fair check every {RL_EVAL_EVERY} iterations, a new best goes to the experiment queue; "
        f"{'training on the GPU here' if gpu_ready() else 'no GPU here: rollouts only'})",
        "",
    ]
    for arm in RL_ARMS:
        s = rl_state(reg, arm)
        done = [j for j in jobs if j["params"].get("arm") == arm and j["status"] == "done"]
        if not s["init"]:
            lines.append(f"  {names[arm]}: not started")
            continue
        rolls = [j for j in done if j["kind"] == "rollout"]
        ups = [j for j in done if j["kind"] == "update" and not j["result"].get("duplicate")]
        games = sum(j["result"].get("games", 0) for j in rolls)
        last = ups[-1]["result"] if ups else {}
        win = rolls[-1]["result"].get("win") if rolls else None
        lines.append(
            f"  {names[arm]}: iteration {s['it']}, {games} games played, "
            f"latest rollout win {win if win is not None else '-'}, "
            f"reward {last.get('mean_reward', '-')}, rollouts this iteration {s['rolled']}/{RL_PARTS}"
        )
        evals = [j for j in done if j["kind"] == "eval"]
        best = best_eval(reg, arm)
        if best:
            lines.append(f"    best network: iteration {best[1]}, fair check win {best[0]}")
        for j in evals[-3:]:
            r = j["result"]
            how = (
                "new best"
                if r.get("best")
                else f"reset to iteration {r['reset_to']}"
                if "reset_to" in r
                else "kept"
            )
            lines.append(
                f"    fair check iteration {r.get('it')}: win {r.get('win')} ({r.get('games')} games), {how}"
            )
        for j in ups + evals:
            e = by_id.get(j["result"].get("experiment"))
            if e:
                r = e.get("results", {}).get("full") or e.get("results", {}).get("screen") or {}
                score = f", win {r['live']['win']:.2f} -> {r['variant']['win']:.2f}" if "variant" in r else ""
                lines.append(
                    f"    iteration {j['result']['it']} -> experiment #{e['id']}: {e['status']}{score}"
                )
    if running:
        lines += ["", "RUNNING NOW"]
        for j in running:
            what = {
                "rollout": "RL games",
                "update": "PPO update",
                "rlinit": "RL start",
                "eval": "fair check",
            }[j["kind"]]
            when = time.strftime("%d %b %H:%M", time.localtime(L.ts(j.get("claimed"))))
            arm = j["params"].get("arm")
            lines.append(f"  job {j['id']:<4} arm {arm} {what:10s} on {j.get('host', '?')}, since {when}")
    if failed:
        lines += ["", "UNRESOLVED FAILURES (last 24 h)"]
        lines += [f"  job {j['id']} ({j['kind']}): {j['result'].get('error', '')[:90]}" for j in failed[-3:]]
    return "\n".join(lines).rstrip() + "\n"


def cmd_status(_: argparse.Namespace) -> int:
    reg = shared("league/ml_jobs.json", {"jobs": []})
    xreg = shared("league/experiments.json", {"experiments": []})
    print(status_report(reg, xreg), end="")
    return 0


def cmd_list(_: argparse.Namespace) -> int:
    reg = load()
    for j in reg["jobs"]:
        if j["kind"] not in KINDS:
            continue
        r = j.get("result", {})
        info = r.get("error") or (f"{r.get('games')} games, win {r.get('win')}" if "games" in r else "")
        print(
            f"{j['id']:4d} {j['kind']:8s} arm {j['params'].get('arm')} {j['status']:8s} {j.get('host', ''):18s} {info}"
        )
    print(f"next when the queue is empty: {plan_rl(reg)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    sub.add_parser("status", help="human-readable report")
    rn = sub.add_parser("run")
    rn.add_argument("--forever", action="store_true")
    rn.add_argument("--no-push", dest="push", action="store_false")
    a = p.parse_args(argv)
    return {"list": cmd_list, "status": cmd_status, "run": cmd_run}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
