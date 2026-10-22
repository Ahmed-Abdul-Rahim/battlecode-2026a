"""Neural-net job queue shared by every PC (league/ml_jobs.json), like the experiment autopilot.

    gen    play logged games of the live bot that explores a random feasible expert on some
           turns (knob2 EXPLORE) against the faithful mimics; the rows (int8 features, the
           expert used, a shaped return) become a data shard on branch ml-data
    train  offline RL (advantage-weighted regression, ml/rl/awr.py) on every shard -> a
           candidate router blob on ml-data, queued as an autopilot experiment (knob2 ROUTER
           on + that blob), so it only ships through the usual gate

Games are the slow part (CPU); training this small network takes minutes on any CPU, so a
GPU is not needed. With an empty queue a PC adds the next job itself: a training run once
TRAIN_EVERY new shards exist, otherwise another data job.

    python harness/mljobs.py list
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
from ml.rl import awr

REGISTRY = L.LEAGUE / "ml_jobs.json"
STORE_BRANCH = "ml-data"
STORE = ROOT.parent / f"{ROOT.name}-mldata"
OUT = L.SCRATCH / "ml"
GEN_EXPLORE = 10  # percent of logged turns that explore
GEN_MAPS = 3  # maps per data job (rotating through the pool)
TRAIN_EVERY = 6  # new data shards per training run
# On-policy RL (ml/rl): per iteration RL_PARTS rollout jobs, then one PPO update; every
# RL_PROMOTE_EVERY iterations the network goes to the experiment queue. Arm 1 (experts + fight
# target) gets RL_ARM_SHARE of the jobs, arm 2 (micro control, experimental) the rest.
RL_PARTS = 2
RL_PROMOTE_EVERY = 5
RL_ARMS = (1, 2)
RL_ARM_SHARE = 2 / 3
RL_WEIGHTS = {1: ("ROUTER", "router.bin"), 2: ("MICRO_NET", "micro.bin")}
#: Keep every n-th decision per champion (arm 1 decides every turn; its full logs are too big
#: to share through git each iteration; the rewards in between are kept).
RL_KEEP_EVERY = {1: 8, 2: 1}
CLAIM_TTL = X.CLAIM_TTL


def load() -> dict:
    return {"jobs": [], **L.load_json(REGISTRY, {})}


def save(reg: dict) -> None:
    L.save_json(REGISTRY, reg)


def add(reg: dict, kind: str, params: dict | None = None) -> dict:
    assert kind in ("gen", "train", "rlinit", "rollout", "update"), kind
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
    jobs = [j for j in reg["jobs"] if j["params"].get("arm") == arm]
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
    """The next RL job, arm 1 first by share; None when both arms wait for running jobs."""
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


def plan_next(reg: dict) -> tuple[str, dict]:
    """What to add when the queue is empty: RL work first (ml/rl), else the older AWR loop."""
    rl = plan_rl(reg)
    if rl:
        return rl
    return plan_awr(reg)


def plan_awr(reg: dict) -> tuple[str, dict]:
    """Imitation + offline RL (AWR): train once enough new shards exist, else gen."""
    trains = [j["id"] for j in reg["jobs"] if j["kind"] == "train" and j["status"] == "done"]
    last = max(trains, default=0)
    fresh = [j for j in reg["jobs"] if j["kind"] == "gen" and j["status"] == "done" and j["id"] > last]
    if len(fresh) >= TRAIN_EVERY:
        return "train", {"hidden": 32, "epochs": 8}
    n_gen = sum(1 for j in reg["jobs"] if j["kind"] == "gen")
    return "gen", {"explore": GEN_EXPLORE, "offset": n_gen * GEN_MAPS}


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
                "Neural-net data shards (data/) and candidate router blobs (candidates/),\n"
                "written by harness/mljobs.py. Not a code branch.\n",
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


# ----------------------------------------------------------------- jobs


def run_gen(j: dict, push: bool) -> dict:
    led = L.load_ledger()
    faithful, _ = L.ensure_mimics(led)
    pool = sorted(L.trainable_maps(led).items())
    k = j["params"].get("offset", 0)
    maps = [pool[(k + i) % len(pool)][1] for i in range(min(GEN_MAPS, len(pool)))]
    var = X.knob_variant("_mlgen", {"EXPLORE": int(j["params"].get("explore", GEN_EXPLORE))})
    out = OUT / f"gen-{j['id']}"
    pairs = [p for o in faithful for p in ((var, o), (o, var))]
    res = runner.run_pairs(pairs, maps, out, release=False)  # debug build: logs the features
    xs, ys, rs, wins = [], [], [], 0.0
    for r in res:
        side = "a" if r.bot_a == var else "b"
        x, y, ret = awr.rows_from_bundle(Path(r.replay), side, r.winner == var)
        xs += x
        ys += y
        rs += ret
        wins += r.score_for(var)
    store = store_ready()
    shard = store / "data" / f"gen-{j['id']:05d}-{L.HOST}.npz"
    rows = awr.save_rows(
        {"X": np.stack(xs) if xs else np.zeros((0, 0), np.int8), "y": np.array(ys), "R": np.array(rs)}, shard
    )
    store_push([shard], f"ml-data: shard {shard.name} ({rows} rows)", push)
    shutil.rmtree(out, ignore_errors=True)  # bundles are large; the shard keeps what training needs
    return {
        "games": len(res),
        "rows": rows,
        "win": round(wins / max(len(res), 1), 3),
        "maps": [m.stem for m in maps],
        "shard": shard.name,
    }


def run_train(j: dict, push: bool) -> dict:
    store = store_ready()
    shards = sorted((store / "data").glob("*.npz"))
    data = awr.load_shards(shards)
    out = store / "candidates" / f"router-{j['id']:05d}.bin"
    rep = awr.fit(data, out, int(j["params"].get("hidden", 32)), int(j["params"].get("epochs", 500)))
    rep["shards"] = len(shards)
    if "skipped" in rep:
        return rep
    store_push([out, out.with_suffix(".json")], f"ml-data: candidate {out.name}", push)
    xreg = X.load()
    e = X.add(
        xreg,
        f"Learned router #{j['id']} ({rep['rows']} rows, AWR)",
        "learning",
        knobs={"ROUTER": 1},
        files={"bots/main/src/weights/router.bin": f"ml-data:candidates/{out.name}"},
        notes=f"Offline RL on {len(shards)} shards; train agreement {rep['train_agreement']}.",
    )
    # Test it next: a new network matters more than the old switch ideas waiting in the queue.
    xreg["experiments"].remove(e)
    i = next(
        (k for k, x in enumerate(xreg["experiments"]) if x["status"] == "queued"), len(xreg["experiments"])
    )
    xreg["experiments"].insert(i, e)
    X.save(xreg)
    rep["experiment"] = e["id"]
    return rep


def rl_path(arm: int, name: str) -> Path:
    return STORE / "rl" / f"arm{arm}" / name


def run_rlinit(j: dict, push: bool) -> dict:
    """Iteration 0 of an arm: arm 1 imitates the rules' expert choices (warm start from the AWR
    data shards), arm 2 starts near-uniform. Plus a fresh value network."""
    from ml.rl import ppo

    arm = j["params"]["arm"]
    spec = ppo.ARMS[arm]
    store = store_ready()
    rng = np.random.default_rng(arm)
    shards = sorted((store / "data").glob("*.npz"))
    data = awr.load_shards(shards) if arm == 1 else {"X": np.zeros((0, 0), np.int8)}
    if arm == 1 and len(data["X"]):
        keep = rng.permutation(len(data["X"]))[:200_000]
        x, y = data["X"][keep], np.asarray(data["y"])[keep]
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
    from ml.rl import reward

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
    k = (it * RL_PARTS + part) * GEN_MAPS
    maps = [pool[(k + i) % len(pool)] for i in range(min(GEN_MAPS, len(pool)))]
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
        try:
            parts.append(
                reward.game_samples(
                    Path(r.replay),
                    Path(r.stats),
                    side,
                    arm,
                    won,
                    (ours - theirs) / max(ours + theirs, 1),
                    RL_KEEP_EVERY[arm],
                )
            )
        except (OSError, KeyError, ValueError):
            continue
    data = reward.concat(parts)
    shard = rl_path(arm, f"it{it:04d}-p{part}-{L.HOST}.npz")
    shard.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(shard, **data) if data else None
    if data:
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


#: A Python with torch (CUDA if there's a GPU) for ml/rl/ppo_gpu.py; without it updates run in numpy.
GPU_PYTHON = Path(
    os.environ.get("ML_GPU_PYTHON")
    or ROOT.parent / "ml-gpu-venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
)


def gpu_update(arm: int, it: int, store: Path) -> dict | None:
    """The PPO update with the big teacher (ml/rl/ppo_gpu.py) when a torch Python exists; its
    metrics, or None to fall back to numpy (no torch Python, or it failed)."""
    if not GPU_PYTHON.exists():
        return None
    p = subprocess.run(
        [
            str(GPU_PYTHON),
            str(ROOT / "ml" / "rl" / "ppo_gpu.py"),
            "--arm",
            str(arm),
            "--it",
            str(it),
            "--store",
            str(store),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    lines = [ln for ln in p.stdout.splitlines() if ln.startswith("{")]
    if p.returncode != 0 or not lines:
        L.log(f"ml: GPU update failed, using numpy: {(p.stderr or p.stdout)[-300:]}")
        return None
    return json.loads(lines[-1])


def run_update(j: dict, reg: dict, push: bool) -> dict:
    """PPO on the iteration's rollouts -> the next network; every RL_PROMOTE_EVERY iterations it
    is queued as an experiment (first in line), so it only ships through the gate."""
    from ml.rl import ppo, reward

    arm, it = j["params"]["arm"], j["params"]["it"]
    dup = [
        x
        for x in reg["jobs"]
        if x is not j
        and x["kind"] == "update"
        and x["params"] == j["params"]
        and x["status"] in ("running", "done")
        and not x["result"].get("duplicate")
    ]
    if dup:
        return {"arm": arm, "it": it, "duplicate": True}
    spec = ppo.ARMS[arm]
    store = store_ready()
    shards = sorted((store / "rl" / f"arm{arm}").glob(f"it{it:04d}-p*.npz"))
    keys = ("X", "A", "LP", "M", "R", "DONE", "TRAJ")
    data = reward.concat([d for d in (awr.read_shard(s, keys) for s in shards) if d])
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
    m.update(
        arm=arm,
        it=nxt,
        shards=len(shards),
        rollout_win=round(sum(x["result"].get("win", 0) for x in rolls) / max(len(rolls), 1), 3),
    )
    if nxt % RL_PROMOTE_EVERY == 0:
        knob, wfile = RL_WEIGHTS[arm]
        xreg = X.load()
        e = X.add(
            xreg,
            f"RL arm {arm} ({'experts+target' if arm == 1 else 'micro control'}) iteration {nxt}",
            "learning",
            knobs={knob: 1},
            files={f"bots/main/src/weights/{wfile}": f"ml-data:rl/arm{arm}/policy-{nxt:04d}.bin"},
            notes=f"PPO on {m['samples']} decisions; rollout win {m['rollout_win']}; reward {m['mean_reward']}.",
        )
        xreg["experiments"].remove(e)
        i = next(
            (k for k, x in enumerate(xreg["experiments"]) if x["status"] == "queued"),
            len(xreg["experiments"]),
        )
        xreg["experiments"].insert(i, e)
        X.save(xreg)
        m["experiment"] = e["id"]
    return m


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
            kind, params = plan_next(reg)
            j = add(reg, kind, params)
            save(reg)
        if not claim(j, reg, a.push):
            L.log(f"ml: job {j['id']} was claimed by another PC; next")
            continue
        try:
            kind = j["kind"]
            if kind == "gen":
                j["result"] = run_gen(j, a.push)
            elif kind == "train":
                j["result"] = run_train(j, a.push)
            elif kind == "rlinit":
                j["result"] = run_rlinit(j, a.push)
            elif kind == "rollout":
                j["result"] = run_rollout(j, a.push)
            else:
                j["result"] = run_update(j, reg, a.push)
            j["status"] = "done"
        except Exception as ex:  # one broken job must not stop the queue
            j["status"] = "failed"
            j["result"] = {"error": repr(ex)[:300]}
        finish(j, reg, a.push)
        if not a.forever:
            return 0


def shared(path: str, default: dict) -> dict:
    """A registry file as on GitHub (every PC's work), falling back to the local copy."""
    import json

    subprocess.run(["git", "fetch", "-q", "origin", L.BRANCH], cwd=ROOT, capture_output=True)
    p = subprocess.run(
        ["git", "show", f"origin/{L.BRANCH}:{path}"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return {**default, **json.loads(p.stdout)} if p.returncode == 0 and p.stdout.strip() else default


def status_report(reg: dict, xreg: dict) -> str:
    jobs = reg["jobs"]
    gens = [j for j in jobs if j["kind"] == "gen" and j["status"] == "done"]
    trains = [
        j for j in jobs if j["kind"] == "train" and j["status"] == "done" and "experiment" in j["result"]
    ]
    running = [j for j in jobs if j["status"] == "running"]
    failed = [j for j in jobs if j["status"] == "failed"]
    rows = sum(j["result"].get("rows", 0) for j in gens)
    games = sum(j["result"].get("games", 0) for j in gens)
    last = max((j["id"] for j in jobs if j["kind"] == "train" and j["status"] == "done"), default=0)
    fresh = sum(1 for j in gens if j["id"] > last)
    lines = [
        f"NEURAL NET: {games} data games, {rows:,} training rows in {len(gens)} shards | "
        f"{len(trains)} networks trained | {len(running)} running | {len(failed)} failed",
        "",
    ]
    if running:
        lines.append("RUNNING NOW")
        for j in running:
            what = {
                "gen": "data games",
                "rollout": "RL games",
                "update": "PPO update",
                "rlinit": "RL start",
            }.get(j["kind"], "training")
            when = time.strftime("%d %b %H:%M", time.localtime(L.ts(j.get("claimed"))))
            lines.append(f"  job {j['id']:<4} {what:11s} on {j.get('host', '?')}, since {when}")
        lines.append("")
    by_id = {e["id"]: e for e in xreg.get("experiments", [])}
    if trains:
        lines.append("TRAINED NETWORKS (each is tested by the experiment autopilot)")
        for j in trains:
            r, e = j["result"], by_id.get(j["result"]["experiment"], {})
            res = e.get("results", {}).get("full") or e.get("results", {}).get("screen") or {}
            score = (
                f", win {res['live']['win']:.2f} -> {res['variant']['win']:.2f}" if "variant" in res else ""
            )
            agree = r.get("val_agreement", r.get("train_agreement"))
            lines.append(
                f"  job {j['id']:<4} {r.get('rows', 0):>9,} rows, copies the rules {agree:.0%} | "
                f"experiment #{e.get('id', '?')}: {e.get('status', '?')}{score}"
            )
        lines.append("")
    else:
        lines.append(f"No network trained yet: the first one trains after {TRAIN_EVERY} data jobs.")
        lines.append("")
    lines.append(
        f"NEXT TRAINING after {max(TRAIN_EVERY - fresh, 0)} more data jobs ({fresh} of {TRAIN_EVERY} new shards)"
    )
    if gens:
        lines.append("")
        lines.append("RECENT DATA JOBS (win = the exploring bot vs the mimics, lower than live on purpose)")
        for j in gens[-6:]:
            r = j["result"]
            lines.append(
                f"  job {j['id']:<4} {r.get('games', 0):>3} games, {r.get('rows', 0):>8,} rows, "
                f"win {r.get('win', 0):.2f}  {', '.join(r.get('maps', []))}"
            )
    lines += rl_report(reg, xreg)
    if failed:
        lines.append("")
        lines.append("RECENT FAILURES")
    for j in failed[-3:]:
        lines.append(f"  job {j['id']} ({j['kind']}): {j['result'].get('error', '')[:90]}")
    return "\n".join(lines).rstrip() + "\n"


def rl_report(reg: dict, xreg: dict) -> list[str]:
    """The learning-by-playing arms: iteration, latest rollout win rate and reward, experiments."""
    names = {1: "arm 1 (experts + fight target)", 2: "arm 2 (micro control)"}
    by_id = {e["id"]: e for e in xreg.get("experiments", [])}
    out = [
        "",
        f"LEARNING BY PLAYING (PPO; a network goes to the experiment queue every {RL_PROMOTE_EVERY} iterations)",
    ]
    for arm in RL_ARMS:
        s = rl_state(reg, arm)
        jobs = [j for j in reg["jobs"] if j["params"].get("arm") == arm and j["status"] == "done"]
        if not s["init"]:
            out.append(f"  {names[arm]}: not started")
            continue
        rolls = [j for j in jobs if j["kind"] == "rollout"]
        ups = [j for j in jobs if j["kind"] == "update" and not j["result"].get("duplicate")]
        games = sum(j["result"].get("games", 0) for j in rolls)
        last = ups[-1]["result"] if ups else {}
        win = rolls[-1]["result"].get("win") if rolls else None
        out.append(
            f"  {names[arm]}: iteration {s['it']}, {games} games played, "
            f"latest rollout win {win if win is not None else '-'}, "
            f"reward {last.get('mean_reward', '-')}, rollouts this iteration {s['rolled']}/{RL_PARTS}"
        )
        for j in ups:
            e = by_id.get(j["result"].get("experiment"))
            if e:
                r = e.get("results", {}).get("full") or e.get("results", {}).get("screen") or {}
                score = f", win {r['live']['win']:.2f} -> {r['variant']['win']:.2f}" if "variant" in r else ""
                out.append(
                    f"    iteration {j['result']['it']} -> experiment #{e['id']}: {e['status']}{score}"
                )
    return out


def cmd_status(_: argparse.Namespace) -> int:
    reg = shared("league/ml_jobs.json", {"jobs": []})
    xreg = shared("league/experiments.json", {"experiments": []})
    print(status_report(reg, xreg), end="")
    return 0


def cmd_list(_: argparse.Namespace) -> int:
    reg = load()
    for j in reg["jobs"]:
        r = j.get("result", {})
        info = r.get("error") or (f"{r.get('rows')} rows" if "rows" in r else "")
        print(f"{j['id']:4d} {j['kind']:5s} {j['status']:8s} {j.get('host', ''):18s} {info}")
    kind, params = plan_next(reg)
    print(f"next when the queue is empty: {kind} {params}")
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
