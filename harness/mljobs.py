"""Neural-net jobs shared by every PC (league/ml_jobs.json): networks that learn by playing.

Two arms (ml/rl/ppo.py):
    arm 1  experts + fight target (knob2 ROUTER, weights/router.bin)
    arm 2  micro control on fight turns (knob2 MICRO_NET, weights/micro.bin), experimental

Per arm and iteration:
    rlinit   iteration 0: arm 1 starts from the rules' choices, arm 2 near-uniform
    rollout  RL_PARTS jobs: the current network plays (sampling) against the mimics and the map's
             exploiter; its decisions and rewards (ml/rl/reward.py) go to branch ml-data
    update   PPO: on the GPU with a big teacher distilled into the bot's network
             (ml/rl/ppo_gpu.py, needs a torch Python), else in numpy (ml/rl/ppo.py)
Every RL_PROMOTE_EVERY iterations the network is queued as an experiment, so it only reaches the
bot through the usual gate. Any PC can take any job (claims, as for experiments).

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
from ml.rl import ppo, reward

REGISTRY = L.LEAGUE / "ml_jobs.json"
STORE_BRANCH = "ml-data"
STORE = ROOT.parent / f"{ROOT.name}-mldata"
OUT = L.SCRATCH / "ml"
KINDS = ("rlinit", "rollout", "update")
RL_PARTS = 2  # rollout jobs per iteration
RL_MAPS = 3  # maps per rollout job (rotating through the pool)
RL_PROMOTE_EVERY = 5  # iterations between networks sent to the experiment queue
RL_ARMS = (1, 2)
RL_ARM_SHARE = 2 / 3  # arm 1's share of the jobs
RL_WEIGHTS = {1: ("ROUTER", "router.bin"), 2: ("MICRO_NET", "micro.bin")}
#: Keep every n-th decision per champion (arm 1 decides every turn; its full logs are too big
#: to share through git each iteration; the rewards in between are kept).
RL_KEEP_EVERY = {1: 8, 2: 1}
SHARD_KEYS = ("X", "A", "LP", "M", "R", "DONE", "TRAJ")
FAILURES_SHOWN_FOR = 24 * 3600
CLAIM_TTL = X.CLAIM_TTL
#: A Python with torch (CUDA if there's a GPU) for ml/rl/ppo_gpu.py; without it updates run in numpy.
GPU_PYTHON = Path(
    os.environ.get("ML_GPU_PYTHON")
    or ROOT.parent / "ml-gpu-venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
)


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
    }


def plan_rl(reg: dict) -> tuple[str, dict] | None:
    """The next job, arm 1 first by share; None when both arms wait for running jobs."""
    work = sum(1 for j in reg["jobs"] if j["params"].get("arm") == 1)
    total = sum(1 for j in reg["jobs"] if j["params"].get("arm") in RL_ARMS)
    order = (1, 2) if work <= RL_ARM_SHARE * max(total, 1) else (2, 1)
    for arm in order:
        s = rl_state(reg, arm)
        if not s["init"]:
            if not s["init_busy"]:
                return "rlinit", {"arm": arm}
            continue
        if s["rolled"] + s["rolling"] < RL_PARTS:
            return "rollout", {"arm": arm, "it": s["it"], "part": s["rolled"] + s["rolling"]}
        if s["rolled"] >= RL_PARTS and not s["updating"]:
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
    """Iteration 0 of an arm: arm 1 imitates the rules' expert choices (the logged rule decisions
    in ml-data data/), arm 2 starts near-uniform. Plus a fresh value network."""
    arm = j["params"]["arm"]
    spec = ppo.ARMS[arm]
    store = store_ready()
    rng = np.random.default_rng(arm)
    parts = (
        [reward.read_shard(s, ("X", "y")) for s in sorted((store / "data").glob("*.npz"))] if arm == 1 else []
    )
    parts = [d for d in parts if d is not None and len(d["y"])]
    if parts:
        x, y = np.concatenate([d["X"] for d in parts]), np.concatenate([d["y"] for d in parts])
        keep = rng.permutation(len(x))[:200_000]
        x, y = x[keep], y[keep]
        policy = ppo.warm_start(x, y, spec, rng)
    else:
        x = rng.integers(-127, 128, (2000, spec.n_in)).astype(np.int8)
        policy = ppo.init_params(spec.n_in, spec.hidden, spec.n_out, rng)
    value = ppo.init_params(spec.n_in, 64, 1, rng)
    files = [rl_path(arm, "policy-0000.npz"), rl_path(arm, "value-0000.npz"), rl_path(arm, "policy-0000.bin")]
    ppo.save(files[0], policy)
    ppo.save(files[1], value)
    files[2].write_bytes(ppo.export(policy, x, spec))
    store_push(files, f"ml-data: rl arm {arm} iteration 0", push)
    return {"arm": arm, "it": 0, "warm_rows": len(x)}


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
    parts, wins = [], 0.0
    for r in res:
        side = "a" if r.bot_a == name else "b"
        ours, theirs = (r.gems_a, r.gems_b) if side == "a" else (r.gems_b, r.gems_a)
        won = r.score_for(name)
        wins += won
        margin = (ours - theirs) / max(ours + theirs, 1)
        try:
            parts.append(
                reward.game_samples(Path(r.replay), Path(r.stats), side, arm, won, margin, RL_KEEP_EVERY[arm])
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
    }


def gpu_update(arm: int, it: int, store: Path) -> dict | None:
    """The PPO update with the big teacher (ml/rl/ppo_gpu.py) when a torch Python exists; its
    metrics, or None to fall back to numpy (no torch Python, or it failed)."""
    if not GPU_PYTHON.exists():
        return None
    cmd = [str(GPU_PYTHON), str(ROOT / "ml" / "rl" / "ppo_gpu.py"), "--arm", str(arm), "--it", str(it)]
    p = subprocess.run([*cmd, "--store", str(store)], cwd=ROOT, capture_output=True, text=True)
    lines = [ln for ln in p.stdout.splitlines() if ln.startswith("{")]
    if p.returncode != 0 or not lines:
        L.log(f"ml: GPU update failed, using numpy: {(p.stderr or p.stdout)[-300:]}")
        return None
    return json.loads(lines[-1])


def promote(arm: int, it: int, m: dict) -> int:
    """Queue the arm's network as the next experiment (first in line); its id."""
    knob, wfile = RL_WEIGHTS[arm]
    xreg = X.load()
    e = X.add(
        xreg,
        f"RL arm {arm} ({'experts+target' if arm == 1 else 'micro control'}) iteration {it}",
        "learning",
        knobs={knob: 1},
        files={f"bots/main/src/weights/{wfile}": f"ml-data:rl/arm{arm}/policy-{it:04d}.bin"},
        notes=f"PPO on {m['samples']} decisions; rollout win {m['rollout_win']}; reward {m['mean_reward']}.",
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
    spec = ppo.ARMS[arm]
    store = store_ready()
    shards = sorted((store / "rl" / f"arm{arm}").glob(f"it{it:04d}-p*.npz"))
    data = reward.concat([d for d in (reward.read_shard(s, SHARD_KEYS) for s in shards) if d])
    if not data:
        return {"arm": arm, "it": it, "skipped": "no decisions logged"}
    nxt = it + 1
    files = [
        rl_path(arm, f"policy-{nxt:04d}.npz"),
        rl_path(arm, f"value-{nxt:04d}.npz"),
        rl_path(arm, f"policy-{nxt:04d}.bin"),
    ]
    m = gpu_update(arm, it, store)
    if m is not None:  # big teacher on the GPU, distilled into the bot's network
        files += [rl_path(arm, f"teacher-{nxt:04d}.pt"), rl_path(arm, f"tvalue-{nxt:04d}.pt")]
    else:
        policy = ppo.load(rl_path(arm, f"policy-{it:04d}.npz"))
        value = ppo.load(rl_path(arm, f"value-{it:04d}.npz"))
        m = ppo.update(policy, value, data, spec, np.random.default_rng(it))
        m["device"] = "cpu (numpy)"
        ppo.save(files[0], policy)
        ppo.save(files[1], value)
        files[2].write_bytes(ppo.export(policy, data["X"], spec))
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
    if nxt % RL_PROMOTE_EVERY == 0:
        m["experiment"] = promote(arm, nxt, m)
    return m


RUNNERS = {
    "rlinit": lambda j, reg, push: run_rlinit(j, push),
    "rollout": lambda j, reg, push: run_rollout(j, push),
    "update": run_update,
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
            planned = plan_rl(reg)
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
        f"LEARNING BY PLAYING (PPO; a network goes to the experiment queue every {RL_PROMOTE_EVERY} iterations; "
        f"updates on {'the GPU' if GPU_PYTHON.exists() else 'the CPU'} here)",
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
        for j in ups:
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
            what = {"rollout": "RL games", "update": "PPO update", "rlinit": "RL start"}[j["kind"]]
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
