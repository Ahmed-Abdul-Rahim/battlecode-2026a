"""League trainer: a daemon that keeps improving the bot the longer it runs.

Every cycle a bandit picks one of three competing learner tracks; its candidate goes
through the same gate, and only what beats the champion survives:

  es      sep-ES over bots/main/src/learned.rs (W multipliers, TARGET and FIGHT scorer
          weights) on contested fitness against a PFSP-sampled league
  rl      offline RL: exploration games of the champion (learned.rs EPSILON), advantage-
          weighted regression of the expert router (ml/rl/awr.py), ROUTER on
  es+rl   the ES track's current mean with the latest RL router

League: the champion, every past champion (auto-vN, v9-v13b), the claude/qwen
exploiters. PFSP: opponents we beat less are sampled more (w = (1 - p)^2).

Gate (all 17 maps, both sides, deterministic): candidate beats the champion head to
head (>= GATE_H2H), does not lose to the sampled pool, and stays within fuel. On
promotion: bots/auto-vN snapshot, bots/main updated, `cambc bundle` zip in
submission/, verified build, git commit + push; with --auto-submit also `cambc submit`.

Ladder watcher (each cycle, read-only): downloads our new ranked/unranked matches to
runs/ladder_live, tracks win rate and deaths per submitted version, raises the death
weight while ladder deaths stay high, and (with --auto-submit) reverts a submitted
version whose ladder win rate drops below the previous one's.

    python harness/league_train.py --forever                 # train until stopped
    python harness/league_train.py --cycles 1 --quick         # smoke test
    python harness/league_train.py --forever --auto-submit    # also upload (team must agree)
Never runs `cambc login`; submit only with --auto-submit.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import speedrun
from league_gate import suspended

from bctools import ROOT, gate, runner
from ml.rl import awr

CAMBC = ROOT / ".venv" / "Scripts" / "cambc.exe"
OUT = ROOT / "runs" / "league"
STATE = OUT / "state.json"
LOCK = OUT / "lock"
LEARNED = ROOT / "bots" / "main" / "src" / "learned.rs"
BLOB = ROOT / "bots" / "main" / "src" / "weights" / "router.bin"
OUR_TEAM = "1ec259ba-7596-4a78-9fd6-090c77f543bf"

FIXED_POOL = ["main-v13b", "main-v12", "main-v11", "main-v10", "main-v9", "claude-pillager", "claude-focus"]
TRACKS = ["es", "rl", "es+rl"]
N_W, N_T, N_F = 18, 10, 9
GATE_H2H = 0.56
MARGIN_W = 0.25
SUBMIT_MIN_HOURS = 3.0


# ----------------------------------------------------------------- learned.rs I/O


def read_params() -> list[float]:
    """Current champion parameters: [log W..., TARGET..., FIGHT...]."""
    text = LEARNED.read_text(encoding="utf-8")

    def arr(name: str, n: int, default: float) -> list[float]:
        m = re.search(rf"pub const {name}: \[f32; \w+\] = \[(.*?)\];", text, re.S)
        if not m or ";" in m.group(1):
            return [default] * n
        vals = [float(v) for v in m.group(1).split(",") if v.strip()]
        return vals if len(vals) == n else [default] * n

    return [math.log(max(w, 1e-3)) for w in arr("W", N_W, 1.0)] + arr("TARGET", N_T, 0.0) + arr("FIGHT", N_F, 0.0)


def learned_rs(params: list[float], router: bool, epsilon: float) -> str:
    text = LEARNED.read_text(encoding="utf-8")
    w = [math.exp(min(max(x, math.log(0.2)), math.log(5.0))) for x in params[:N_W]]
    tgt, fight = params[N_W:N_W + N_T], params[N_W + N_T:]
    fmt = lambda v: ", ".join(f"{x:.4f}" for x in v)  # noqa: E731
    text = re.sub(r"pub const W: \[f32; N\] = \[.*?\];", f"pub const W: [f32; N] = [{fmt(w)}];", text, flags=re.S)
    text = re.sub(r"pub const TARGET: \[f32; T_N\] = \[.*?\];", f"pub const TARGET: [f32; T_N] = [{fmt(tgt)}];", text, flags=re.S)
    text = re.sub(r"pub const FIGHT: \[f32; F_N\] = \[.*?\];", f"pub const FIGHT: [f32; F_N] = [{fmt(fight)}];", text, flags=re.S)
    text = re.sub(r"pub const ROUTER: bool = \w+;", f"pub const ROUTER: bool = {'true' if router else 'false'};", text)
    text = re.sub(r"pub const EPSILON: f32 = [\d.]+;", f"pub const EPSILON: f32 = {epsilon:.3f};", text)
    return text


def make_bot(name: str, params: list[float], router: bool, blob: Path | None, epsilon: float = 0.0) -> str:
    bot = f"_lg-{name}"
    d = ROOT / "bots" / bot
    if d.exists():
        shutil.rmtree(d / "src", ignore_errors=True)
        shutil.copytree(ROOT / "bots" / "main" / "src", d / "src")
    else:
        shutil.copytree(ROOT / "bots" / "main", d, ignore=shutil.ignore_patterns(".cambc", "target"))
    (d / "src" / "learned.rs").write_text(learned_rs(params, router, epsilon), encoding="utf-8", newline="\n")
    if blob and blob.exists():
        shutil.copy(blob, d / "src" / "weights" / "router.bin")
    return bot


# ----------------------------------------------------------------- evaluation


def maps(heldout: bool | None) -> list[Path]:
    held = gate.heldout_names()
    return [p for p in runner.list_maps() if heldout is None or (p.stem in held) == heldout]


def margin(r: runner.MatchResult, bot: str) -> float:
    ours, theirs = (r.gems_a, r.gems_b) if r.bot_a == bot else (r.gems_b, r.gems_a)
    return (ours - theirs) / max(ours + theirs, 1)


def play(bot: str, opps: list[str], map_list: list[Path], out: Path, cache: dict, release: bool = True) -> dict:
    """Aggregate results of `bot` vs each opponent (deterministic: cached per pair+maps)."""
    tot = {"games": 0, "win": 0.0, "margin": 0.0, "deaths": 0.0, "per": {}}
    for opp in opps:
        key = f"{bot}|{opp}|{','.join(p.stem for p in map_list)}|{int(release)}"
        if key not in cache or bot.startswith("_lg-"):
            rs = runner.run_matches(bot, opp, maps=map_list, out_dir=out / opp, release=release)
            g = len(rs)
            res = {
                "games": g,
                "win": sum(r.score_for(bot) for r in rs),
                "margin": sum(margin(r, bot) for r in rs),
                "deaths": sum(r.deaths_a if r.bot_a == bot else r.deaths_b for r in rs),
                "susp": suspended(rs, bot)[0],
            }
            if not bot.startswith("_lg-"):
                cache[key] = res
        else:
            res = cache[key]
        tot["games"] += res["games"]
        tot["win"] += res["win"]
        tot["margin"] += res["margin"]
        tot["deaths"] += res["deaths"]
        tot["per"][opp] = round(res["win"] / max(res["games"], 1), 3)
        tot["susp"] = tot.get("susp", 0) + res.get("susp", 0)
    g = max(tot["games"], 1)
    return {"win": tot["win"] / g, "margin": tot["margin"] / g, "deaths": tot["deaths"] / g, "games": tot["games"],
            "per": tot["per"], "susp": tot.get("susp", 0)}


def fitness(res: dict, speed_gain: float, w: dict) -> float:
    return res["win"] + MARGIN_W * res["margin"] - w["death"] * res["deaths"] + w["speed"] * speed_gain


# ----------------------------------------------------------------- state


def load_state() -> dict:
    if STATE.exists():
        return json.loads(STATE.read_text(encoding="utf-8"))
    params = read_params()
    return {
        "champion": "main-v13b", "champ_params": params, "champ_router": False, "version": 0,
        "pool": list(FIXED_POOL), "pwin": {}, "cache": {},
        "tracks": {t: {"tries": 0, "promos": 0} for t in TRACKS},
        "es": {"mean": params, "sigma": [0.25] * N_W + [0.15] * (N_T + N_F)},
        "weights": {"death": 0.01, "speed": 0.3},
        "rl_dirs": [], "router_blob": None,
        "ladder": {"seen": [], "submissions": [], "results": []},
        "history": [],
    }


def save_state(st: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, indent=1), encoding="utf-8")
    tmp.replace(STATE)


def pfsp(st: dict, k: int, rng: random.Random) -> list[str]:
    mimics = sorted(q.name for q in (ROOT / "bots").glob("mimic-*") if q.is_dir())
    pool = [b for b in dict.fromkeys(st["pool"] + mimics) if b != st["champion"] and (ROOT / "bots" / b).exists()]
    weights = [(1.0 - st["pwin"].get(b, 0.5)) ** 2 + 0.05 for b in pool]
    chosen = []
    while pool and len(chosen) < k:
        i = rng.choices(range(len(pool)), weights=weights)[0]
        chosen.append(pool.pop(i))
        weights.pop(i)
    return chosen


def pick_track(st: dict) -> str:
    total = sum(t["tries"] for t in st["tracks"].values()) + 1
    best, best_u = TRACKS[0], -1.0
    for name in TRACKS:
        t = st["tracks"][name]
        if t["tries"] == 0:
            return name
        u = (t["promos"] + 0.5) / (t["tries"] + 1) + math.sqrt(2 * math.log(total) / t["tries"])
        if u > best_u:
            best, best_u = name, u
    return best


# ----------------------------------------------------------------- tracks


def track_es(st: dict, cyc: Path, rng: random.Random, quick: bool) -> tuple[str, dict] | None:
    train = maps(False)
    opps = [st["champion"]] + pfsp(st, 1 if quick else 2, rng)
    champ = play(st["champion"], opps[1:], train, cyc / "champ", st["cache"]) if len(opps) > 1 else None
    base_speed = speedrun.evaluate(st["champion"], cyc / "speed", release=True, maps=train)["mean"]
    mean, sigma = st["es"]["mean"], st["es"]["sigma"]
    lam = 2 if quick else 6
    scored = []
    for k in range(lam // 2):
        eps = [rng.gauss(0, 1) for _ in mean]
        for sgn in (1, -1):
            params = [m + sgn * s * e for m, s, e in zip(mean, sigma, eps)]
            bot = make_bot(f"es{k}{'p' if sgn > 0 else 'n'}", params, st["champ_router"], Path(st["router_blob"]) if st["router_blob"] else None)
            res = play(bot, opps, train, cyc / bot, st["cache"])
            sp = speedrun.evaluate(bot, cyc / "speed", release=True, maps=train)["mean"]
            f = fitness(res, (base_speed - sp) / base_speed, st["weights"])
            scored.append((f, params, eps, sgn, res, sp))
            print(f"    {bot}: fit {f:.3f} win {res['win']:.3f} deaths {res['deaths']:.1f} speed {sp}", flush=True)
    scored.sort(key=lambda s: -s[0])
    # Rank-weighted recombination of the better half; sigma follows the elite spread.
    mu = max(1, len(scored) // 2)
    ws = [math.log(mu + 0.5) - math.log(i + 1) for i in range(mu)]
    ws = [w / sum(ws) for w in ws]
    st["es"]["mean"] = [sum(w * s[1][j] for w, s in zip(ws, scored[:mu])) for j in range(len(mean))]
    st["es"]["sigma"] = [max(0.03, min(0.6, 0.8 * sg + 0.2 * math.sqrt(sum(w * (s[1][j] - mean[j]) ** 2 for w, s in zip(ws, scored[:mu])))))
                         for j, sg in enumerate(sigma)]
    best = scored[0]
    return make_bot("cand", best[1], st["champ_router"], Path(st["router_blob"]) if st["router_blob"] else None), {
        "params": best[1], "router": st["champ_router"], "fit": round(best[0], 4)}


def track_rl(st: dict, cyc: Path, rng: random.Random, quick: bool, with_es_mean: bool) -> tuple[str, dict] | None:
    train = maps(False)
    params = st["es"]["mean"] if with_es_mean else st["champ_params"]
    explore = make_bot("explore", st["champ_params"], st["champ_router"], Path(st["router_blob"]) if st["router_blob"] else None, epsilon=0.15)
    opps = pfsp(st, 1 if quick else 2, rng) + [st["champion"]]
    data_dir = cyc / "rl"
    for opp in opps:
        runner.run_matches(explore, opp, maps=train[:4] if quick else train, out_dir=data_dir / opp, release=False)
    st["rl_dirs"] = (st["rl_dirs"] + [str(data_dir / o) for o in opps])[-9:]
    blob = OUT / "router" / f"{cyc.name}.bin"
    report = awr.train([Path(d) for d in st["rl_dirs"]], explore, blob)
    print(f"    awr: {report}", flush=True)
    if "skipped" in report:
        return None
    bot = make_bot("cand", params, True, blob)
    return bot, {"params": params, "router": True, "blob": str(blob), "awr": report}


# ----------------------------------------------------------------- gate / promote


def gate_candidate(st: dict, bot: str, cyc: Path, rng: random.Random, quick: bool) -> dict:
    allm = maps(None)[:4] if quick else maps(None)
    h2h = play(bot, [st["champion"]], allm, cyc / "gate", st["cache"])
    opps = pfsp(st, 1 if quick else 3, rng)
    cand_pool = play(bot, opps, allm, cyc / "gate", st["cache"]) if opps else None
    champ_pool = play(st["champion"], opps, allm, cyc / "gate-champ", st["cache"]) if opps else None
    for o, p in (champ_pool or {"per": {}})["per"].items():
        st["pwin"][o] = p
    ok = h2h["win"] >= GATE_H2H
    if cand_pool and champ_pool:
        ok = ok and cand_pool["win"] >= champ_pool["win"] - 0.02
        ok = ok and cand_pool["susp"] <= 2 * champ_pool["susp"] + 20
    return {"promote": ok, "h2h": round(h2h["win"], 3), "pool": cand_pool and round(cand_pool["win"], 3),
            "champ_pool": champ_pool and round(champ_pool["win"], 3), "deaths": round(h2h["deaths"], 1)}


def sh(args: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(args, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", **kw)


def promote(st: dict, bot: str, info: dict, auto_submit: bool) -> str:
    st["version"] += 1
    name = f"auto-v{st['version']}"
    dst = ROOT / "bots" / name
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(ROOT / "bots" / bot, dst, ignore=shutil.ignore_patterns(".cambc", "target"))
    # bots/main carries the champion's parameters (and router) from now on.
    shutil.copy(dst / "src" / "learned.rs", LEARNED)
    if info.get("blob"):
        shutil.copy(info["blob"], BLOB)
    zip_path = ROOT / "submission" / f"{name}.zip"
    sh([str(CAMBC), "bundle", name, "--output", str(zip_path)])
    built = sh([str(CAMBC), "build", name, "--release"]).returncode == 0
    st["champion"], st["champ_params"], st["champ_router"] = name, info["params"], info["router"]
    if info.get("blob"):
        st["router_blob"] = info["blob"]
    st["pool"] = [name] + [b for b in st["pool"] if b != name][:12]
    msg = f"auto: promote {name} ({info.get('track')}, gate {info.get('gate')})\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
    sh(["git", "add", str(LEARNED.relative_to(ROOT)), str(BLOB.relative_to(ROOT))])
    sh(["git", "commit", "-q", "-m", msg])
    sh(["git", "push", "-q"])
    if auto_submit and built and zip_path.exists():
        last = st["ladder"]["submissions"][-1]["time"] if st["ladder"]["submissions"] else 0
        if time.time() - last >= SUBMIT_MIN_HOURS * 3600:
            r = sh([str(CAMBC), "submit", name])
            st["ladder"]["submissions"].append({"time": time.time(), "version": name, "ok": r.returncode == 0})
    return name


# ----------------------------------------------------------------- ladder watcher


def ladder_watch(st: dict, auto_submit: bool) -> str:
    r = sh([str(CAMBC), "matches", "list", "--team", OUR_TEAM, "--limit", "40", "--json"], timeout=60)
    try:
        matches = json.loads(r.stdout)["matches"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return "ladder: offline"
    live = ROOT / "runs" / "ladder_live"
    live.mkdir(parents=True, exist_ok=True)
    seen = set(st["ladder"]["seen"])
    new = 0
    for m in matches:
        if m["status"] != "finished" or m["matchId"] in seen:
            continue
        us = [t for t in m["teams"] if t["teamId"] == OUR_TEAM]
        if not us:
            continue
        out = live / f"{m['matchId']}.json"
        sh([str(CAMBC), "matches", "summary", m["matchId"], "--json", "-o", str(out)], timeout=120)
        deaths = None
        if out.exists():
            d = json.loads(out.read_text(encoding="utf-8"))
            team = "team" + us[0]["side"].upper()
            deaths = sum(int(float(c["metrics"]["deaths"])) for c in d["champions"] if c["team"] == team)
        t = time.mktime(time.strptime(m["createdAt"][:19], "%Y-%m-%dT%H:%M:%S"))
        st["ladder"]["results"].append({"id": m["matchId"], "time": t, "kind": m["kind"], "win": us[0]["result"] == "win",
                                        "deaths": deaths, "map": m["map"]})
        seen.add(m["matchId"])
        new += 1
    st["ladder"]["seen"] = list(seen)[-500:]
    recent = [x for x in st["ladder"]["results"][-20:]]
    if not recent:
        return f"ladder: {new} new"
    wr = sum(x["win"] for x in recent) / len(recent)
    dl = [x["deaths"] for x in recent if x["deaths"] is not None]
    avg_d = sum(dl) / len(dl) if dl else 0.0
    # Ladder feedback: while we keep dying on the ladder, value survival more in training.
    if avg_d > 8:
        st["weights"]["death"] = min(0.05, st["weights"]["death"] * 1.1)
    elif avg_d < 4:
        st["weights"]["death"] = max(0.005, st["weights"]["death"] * 0.95)
    msg = f"ladder: {new} new, last {len(recent)} win {wr:.2f} deaths {avg_d:.1f}, death_w {st['weights']['death']:.3f}"
    subs = st["ladder"]["submissions"]
    if auto_submit and len(subs) >= 2:
        cur, prev = subs[-1], subs[-2]
        def rate(sub, nxt):
            xs = [x for x in st["ladder"]["results"] if x["kind"] == "ranked" and sub["time"] <= x["time"] < nxt]
            return (sum(x["win"] for x in xs) / len(xs), len(xs)) if xs else (None, 0)
        r_cur, n_cur = rate(cur, 1e18)
        r_prev, _ = rate(prev, cur["time"])
        if n_cur >= 10 and r_prev is not None and r_cur < r_prev:
            sh([str(CAMBC), "submit", prev["version"]])
            subs.append({"time": time.time(), "version": prev["version"], "ok": True, "revert_of": cur["version"]})
            msg += f" | reverted {cur['version']} -> {prev['version']}"
    return msg


# ----------------------------------------------------------------- main loop


def report(st: dict) -> None:
    lines = ["# League trainer", "", f"Champion: **{st['champion']}** (version {st['version']})", "",
             "| track | tries | promotions |", "|---|---|---|"]
    lines += [f"| {k} | {v['tries']} | {v['promos']} |" for k, v in st["tracks"].items()]
    lines += ["", "## Recent cycles", ""]
    lines += [f"- {h['time']} {h['track']}: {h.get('gate')} {h.get('promoted', '')}" for h in st["history"][-20:]]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--forever", action="store_true")
    p.add_argument("--cycles", type=int, default=1)
    p.add_argument("--hours", type=float, default=0.0)
    p.add_argument("--quick", action="store_true", help="tiny cycle for smoke tests")
    p.add_argument("--auto-submit", action="store_true", help="upload promoted zips (team must agree)")
    p.add_argument("--track", choices=TRACKS)
    a = p.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    if LOCK.exists() and time.time() - LOCK.stat().st_mtime < 6 * 3600:
        print(f"another trainer holds {LOCK}; delete it if that run is dead")
        return 1
    LOCK.write_text(str(os.getpid()))
    try:
        st = load_state()
        rng = random.Random(len(st["history"]) + 17)
        t_end = time.time() + a.hours * 3600 if a.hours else None
        n = 0
        while a.forever or n < a.cycles or (t_end and time.time() < t_end):
            n += 1
            LOCK.write_text(str(os.getpid()))
            track = a.track or pick_track(st)
            cyc = OUT / "cycles" / f"{time.strftime('%m%d-%H%M%S')}-{track}"
            print(f"[{time.strftime('%H:%M')}] cycle {len(st['history']) + 1} track {track} champion {st['champion']}", flush=True)
            st["tracks"][track]["tries"] += 1
            entry = {"time": time.strftime("%m-%d %H:%M"), "track": track}
            try:
                if track == "es":
                    out = track_es(st, cyc, rng, a.quick)
                else:
                    out = track_rl(st, cyc, rng, a.quick, with_es_mean=track == "es+rl")
                if out:
                    bot, info = out
                    info["track"] = track
                    g = gate_candidate(st, bot, cyc, rng, a.quick)
                    info["gate"] = g
                    entry["gate"] = g
                    if g["promote"]:
                        name = promote(st, bot, info, a.auto_submit)
                        st["tracks"][track]["promos"] += 1
                        entry["promoted"] = name
                    print(f"    gate {g} {'-> PROMOTED ' + entry.get('promoted', '') if g['promote'] else ''}", flush=True)
            except Exception as e:  # keep the daemon alive; log and move on
                entry["error"] = repr(e)[:300]
                print(f"    error: {e!r}", flush=True)
            entry["ladder"] = ladder_watch(st, a.auto_submit)
            print(f"    {entry['ladder']}", flush=True)
            st["history"].append(entry)
            save_state(st)
            report(st)
            if t_end and time.time() >= t_end and not a.forever:
                break
    finally:
        LOCK.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
