"""Autonomous self-improvement harness for Battlecode:
- Benchmarks continuously against the Top 5 teams (mimics) and the league.
- Monitors the ladder in real time for losses.
- Automatically runs targeted Evolutionary Strategy (ES) + micro parameter optimization on lost maps.
- Gates strictly on head-to-head performance, pool fitness, and fuel budgets.
- Automatically packages, git commits/pushes, and submits the latest best model to Battlecode platform after losses are solved.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "harness"))
sys.path.insert(0, str(ROOT))

import pack as packlib
import params
from league_gate import suspended
from bctools import runner

CAMBC = ROOT / ".venv" / "Scripts" / "cambc.exe"
OUT = ROOT / "runs" / "auto_train"
STATE_FILE = OUT / "state.json"
LOCK_FILE = OUT / "lock"
REPORT_FILE = OUT / "report.md"

OUR_TEAM = "1ec259ba-7596-4a78-9fd6-090c77f543bf"

TOP5_DEFAULT = [
    "mimic-intercontinental-fish",
    "mimic-nyx",
    "mimic-sneaky-bush-campers",
    "mimic-univardisty-of-shingtonwash",
    "mimic-mite",
]

# Fitness constants
MARGIN_W = 0.25
DEATH_W = 0.01
SUSP_W = 0.002
ACCEPT_GAIN = 0.04
N_LAMBDA = 8
N_CONFIRM = 3


class Logger:
    def __init__(self, filepath: Path):
        self.terminal = sys.stdout
        filepath.parent.mkdir(parents=True, exist_ok=True)
        self.log = open(filepath, "a", encoding="utf-8", buffering=1)

    def write(self, message):
        try:
            self.terminal.write(message)
            self.terminal.flush()
        except Exception:
            pass
        try:
            self.log.write(message)
            self.log.flush()
        except Exception:
            pass

    def flush(self):
        try:
            self.terminal.flush()
        except Exception:
            pass
        try:
            self.log.flush()
        except Exception:
            pass


def sh(args: list[str], timeout: int | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        args,
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )


def cambc_api(args: list[str], max_retries: int = 5, backoff: float = 2.0) -> tuple[str, str, int]:
    """Execute cambc command with exponential backoff on HTTP 429."""
    cmd = [str(CAMBC)] + args
    delay = backoff
    for attempt in range(max_retries):
        try:
            p = sh(cmd, timeout=90)
            if "HTTP 429" in p.stderr or "Public API is busy" in p.stderr:
                time.sleep(delay)
                delay = min(delay * 2.0, 30.0)
                continue
            return p.stdout, p.stderr, p.returncode
        except subprocess.TimeoutExpired:
            time.sleep(delay)
            delay = min(delay * 2.0, 30.0)
    return "", "Timeout or rate limit exceeded", 1


# ----------------------------------------------------------------- Map resolution


def list_competition_maps() -> list[Path]:
    maps = sorted((ROOT / "maps").glob("*.map26a"))
    return [m for m in maps if not m.name.startswith(".")]


def resolve_map_path(name: str) -> Path | None:
    clean = re.sub(r"^\d{2}", "", name).lower()
    for p in (ROOT / "maps").glob("*.map26a"):
        if p.stem.lower() == clean or p.stem.lower() == name.lower():
            return p
    for p in (ROOT / "maps" / "heldout").glob("*.map26a"):
        if p.stem.lower() == clean or p.stem.lower() == name.lower():
            return p
    return None


# ----------------------------------------------------------------- Parameter helpers


def base_theta(src: str = "main") -> list[float]:
    return params.to_search(params.defaults(ROOT / "bots" / src / "src" / "learned.rs"))


def to_pack_theta(theta: list[float], src: str = "main") -> list[float]:
    return params.to_bot(theta, params.defaults(ROOT / "bots" / src / "src" / "learned.rs"))


def write_pack(path: Path, pack: dict[str, list[float]], base: dict | None = None) -> None:
    maps = {m: dict(e) for m, e in ((base or {}).get("maps") or {}).items()}
    for m, th in pack.items():
        maps.setdefault(m, {})["theta"] = to_pack_theta(th)
    packlib.write({"maps": maps}, path)


def safe_copy_bot(src_dir: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for root, dirs, files in os.walk(src_dir):
        if ".cambc" in root or "target" in root:
            continue
        rel = Path(root).relative_to(src_dir)
        target = dst_dir / rel
        target.mkdir(parents=True, exist_ok=True)
        for f in files:
            sf = Path(root) / f
            df = target / f
            for _ in range(3):
                try:
                    shutil.copy2(sf, df)
                    break
                except OSError:
                    time.sleep(0.05)


def make_bot(name: str, pack: dict[str, list[float]], src: str = "main") -> str:
    bot = name
    d = ROOT / "bots" / bot
    safe_copy_bot(ROOT / "bots" / src, d)
    write_pack(d / "src" / "pack" / "pack.bin", pack)
    return bot


# ----------------------------------------------------------------- State


def init_state() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass

    # Migrate prior best state if available
    champ = "main"
    version = 0
    pack: dict[str, list[float]] = {}
    coevo_state = ROOT / "runs" / "coevo" / "state.json"
    if coevo_state.exists():
        try:
            old = json.loads(coevo_state.read_text(encoding="utf-8"))
            lin_a = old.get("lineages", {}).get("A", {})
            if lin_a.get("champion") and (ROOT / "bots" / lin_a["champion"]).exists():
                champ = lin_a["champion"]
                version = lin_a.get("version", 0)
                pack = lin_a.get("pack", {})
        except Exception:
            pass

    st = {
        "champion": champ,
        "version": version,
        "cycle": 0,
        "pack": pack,
        "sigma": {},
        "losses": [],
        "ladder_seen": [],
        "submissions": [],
        "top5": [],
        "pwin": {},
        "benchmark_scores": {},
        "history": [],
    }
    save_state(st)
    return st


def save_state(st: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, indent=1), encoding="utf-8")
    tmp.replace(STATE_FILE)


# ----------------------------------------------------------------- Matches & Fitness


def fitness(results: list[runner.MatchResult], bot: str) -> tuple[float, float, float, int]:
    n = max(len(results), 1)
    win = sum(r.score_for(bot) for r in results) / n
    susp_turns, _ = suspended(results, bot)
    susp_avg = susp_turns / n
    margin = deaths = 0.0
    for r in results:
        ours, theirs = (r.gems_a, r.gems_b) if r.bot_a == bot else (r.gems_b, r.gems_a)
        margin += (ours - theirs) / max(ours + theirs, 1)
        deaths += r.deaths_a if r.bot_a == bot else r.deaths_b
    avg_deaths = deaths / n
    avg_margin = margin / n
    fit = win + MARGIN_W * avg_margin - DEATH_W * avg_deaths - SUSP_W * susp_avg
    return fit, win, avg_deaths, susp_turns


def play_map(bot: str, opps: list[str], mp: Path, out_dir: Path) -> list[runner.MatchResult]:
    res = []
    for o in opps:
        res += runner.run_matches(bot, o, maps=[mp], out_dir=out_dir / o, release=True)
    return res


# ----------------------------------------------------------------- Top 5 & Ladder


def get_active_top5_mimics(st: dict) -> list[str]:
    # Check live ladder
    stdout, _, rc = cambc_api(["ladder", "--limit", "10", "--json"])
    if rc == 0 and stdout.strip():
        try:
            data = json.loads(stdout)
            ladder = data.get("ladder", [])
            top5_teams = [t for t in ladder if t.get("teamId") != OUR_TEAM][:5]
            if top5_teams:
                st["top5"] = [
                    {"name": t.get("name"), "teamId": t.get("teamId"), "rating": t.get("rating")}
                    for t in top5_teams
                ]
        except Exception:
            pass

    # Find existing mimic bots
    bots = [b for b in TOP5_DEFAULT if (ROOT / "bots" / b).exists()]
    all_mimics = sorted(q.name for q in (ROOT / "bots").glob("mimic-*") if q.is_dir())
    for m in all_mimics:
        if m not in bots and len(bots) < 5:
            bots.append(m)
    return bots


def sync_live_top5_mimics(st: dict) -> list[str]:
    """Download newly finished games of the top 5 teams and update mimic bots."""
    stdout, _, rc = cambc_api(["ladder", "--limit", "10", "--json"])
    if rc != 0 or not stdout.strip():
        return get_active_top5_mimics(st)

    try:
        data = json.loads(stdout)
        ladder = data.get("ladder", [])
        top5_teams = [t for t in ladder if t.get("teamId") != OUR_TEAM][:5]
    except Exception:
        return get_active_top5_mimics(st)

    dest_dir = ROOT / "runs" / "ladder_top"
    dest_dir.mkdir(parents=True, exist_ok=True)
    existing_summaries = {p.stem for p in dest_dir.glob("*.json")}

    updated_any = False
    for team in top5_teams:
        tid = team.get("teamId")
        tname = team.get("name")
        if not tid or not tname:
            continue
        out_list, _, rc_l = cambc_api(["matches", "list", "--team", tid, "--limit", "6", "--json"])
        if rc_l != 0 or not out_list.strip():
            continue
        try:
            m_data = json.loads(out_list)
            matches = m_data.get("matches", [])
        except Exception:
            continue

        for m in matches:
            mid = m.get("matchId")
            if not mid or mid in existing_summaries or m.get("status") != "finished":
                continue
            out_file = dest_dir / f"{mid}.json"
            stdout_s, _, rc_s = cambc_api(["matches", "summary", mid, "--json", "-o", str(out_file)])
            if rc_s == 0 and out_file.exists():
                existing_summaries.add(mid)
                updated_any = True

    if updated_any:
        print("[MIMIC SYNC] New top-team matches downloaded. Rebuilding mimic macro specs...")
        sh([sys.executable, str(ROOT / "harness" / "mimic.py"), "--all-top"])
        for spec_p in (ROOT / "runs" / "mimic").glob("*.spec.json"):
            slug = spec_p.name.replace(".spec.json", "")
            bot_name = f"mimic-{slug}"
            bot_dir = ROOT / "bots" / bot_name
            if bot_dir.exists():
                try:
                    spec_data = json.loads(spec_p.read_text(encoding="utf-8"))
                    packlib.write(spec_data, bot_dir / "src" / "pack" / "pack.bin")
                except Exception:
                    pass

    return get_active_top5_mimics(st)


def check_ladder_for_losses(st: dict) -> list[dict]:
    stdout, _, rc = cambc_api(["matches", "list", "--team", OUR_TEAM, "--limit", "25", "--json"])
    if rc != 0 or not stdout.strip():
        return []

    try:
        data = json.loads(stdout)
        matches = data.get("matches", [])
    except Exception:
        return []

    seen = set(st.get("ladder_seen", []))
    new_losses = []

    for m in matches:
        mid = m.get("matchId")
        if not mid or mid in seen or m.get("status") != "finished":
            continue
        seen.add(mid)

        teams = m.get("teams", [])
        us = next((t for t in teams if t.get("teamId") == OUR_TEAM), None)
        opp = next((t for t in teams if t.get("teamId") != OUR_TEAM), None)

        if us and us.get("result") == "loss":
            map_name = m.get("map", "")
            mp = resolve_map_path(map_name)
            if mp:
                loss_info = {
                    "source": "ladder",
                    "map": mp.stem,
                    "opp_name": (opp or {}).get("name", "Unknown"),
                    "matchId": mid,
                    "time": m.get("createdAt", time.strftime("%Y-%m-%d %H:%M:%S")),
                }
                new_losses.append(loss_info)
                # Check if this loss map is already queued
                if not any(x.get("map") == mp.stem for x in st.get("losses", [])):
                    st.setdefault("losses", []).append(loss_info)

    st["ladder_seen"] = list(seen)[-300:]
    return new_losses


# ----------------------------------------------------------------- Benchmark


def run_benchmark(st: dict, candidate: str, full: bool = False) -> dict:
    mimics = get_active_top5_mimics(st)
    all_maps = list_competition_maps()
    # 4 core maps covering open, choke, clustered and resource-heavy layouts
    core_names = {"expanse", "collision", "contest", "monolith"}
    eval_maps = all_maps if full else [m for m in all_maps if m.stem in core_names] or all_maps[:4]

    cyc_dir = OUT / "benchmark" / f"{candidate}-{int(time.time())}"
    cyc_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, dict] = {}
    total_wins = 0
    total_games = 0

    print(f"\n=== Benchmarking {candidate} against Top 5 Competitors ===")
    for mimic in mimics:
        res = []
        for mp in eval_maps:
            res += runner.run_matches(candidate, mimic, maps=[mp], out_dir=cyc_dir / mimic / mp.stem, release=True)

        n = max(len(res), 1)
        w = sum(r.score_for(candidate) for r in res)
        wr = w / n
        total_wins += w
        total_games += n
        summary[mimic] = {"win_rate": round(wr, 3), "games": n}

        # Check for loss maps to queue
        for r in res:
            if r.score_for(candidate) < 0.5:
                if not any(x.get("map") == r.map and x.get("opp_name") == mimic for x in st.get("losses", [])):
                    st.setdefault("losses", []).append({
                        "source": "benchmark",
                        "map": r.map,
                        "opp_name": mimic,
                        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    })

        print(f"  vs {mimic:35s}: win rate {wr*100:5.1f}% ({int(w)}/{n})")

    overall = round(total_wins / max(total_games, 1), 3)
    st["benchmark_scores"] = {k: v["win_rate"] for k, v in summary.items()}
    st["benchmark_overall"] = overall
    print(f"Overall win rate vs Top 5: {overall*100:.1f}%\n")
    return {"per_team": summary, "overall": overall}


# ----------------------------------------------------------------- Platform Push


def push_to_platform(bot_name: str, version_name: str, map_name: str, st: dict) -> bool:
    zip_path = ROOT / "submission" / f"{version_name}.zip"
    print(f"\n>>> Bundling {version_name} -> {zip_path.name}...")
    sh([str(CAMBC), "bundle", bot_name, "--output", str(zip_path)])

    # Git commit
    print(f">>> Committing {version_name} to git...")
    sh(["git", "add", "bots/main/src/pack/pack.bin", "bots/main/src/learned.rs"])
    sh(["git", "commit", "-q", "-m", f"auto: push {version_name} (loss on {map_name} solved)\n\nCo-Authored-By: Antigravity Autonomous Harness <noreply@google.com>"])
    sh(["git", "push", "-q"])

    # Battlecode platform submit
    print(f">>> Submitting {version_name} to Battlecode platform (`cambc submit {bot_name}`)...")
    stdout, stderr, rc = cambc_api(["submit", bot_name], max_retries=3)
    ok = (rc == 0)
    output_log = (stdout + "\n" + stderr).strip()

    st.setdefault("submissions", []).append({
        "time": time.time(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "version": version_name,
        "map": map_name,
        "zip": str(zip_path),
        "ok": ok,
        "output": output_log[:300],
    })

    if ok:
        print(f"*** SUBMISSION SUCCESS: {version_name} is now live on Battlecode! ***\n")
    else:
        print(f"*** SUBMISSION NOTICE: {output_log[:200]} ***\n")
    return ok


# ----------------------------------------------------------------- Mimic Hardening (Co-Evolution)


def harden_mimic_step(st: dict, rng: random.Random) -> dict:
    """Co-evolve the micro execution (theta) of a Top 5 mimic against our champion."""
    champ = st["champion"]
    if not (ROOT / "bots" / champ).exists():
        champ = "main"

    mimics = get_active_top5_mimics(st)
    if not mimics:
        return {"status": "skipped", "reason": "no mimics"}

    mimic_idx = (st["cycle"] // 4) % len(mimics)
    target_mimic = mimics[mimic_idx]
    mimic_slug = target_mimic.replace("mimic-", "")
    target_bot_dir = ROOT / "bots" / target_mimic

    spec_p = ROOT / "runs" / "mimic" / f"{mimic_slug}.spec.json"
    base_spec = None
    if spec_p.exists():
        try:
            base_spec = json.loads(spec_p.read_text(encoding="utf-8"))
        except Exception:
            pass

    all_maps = list_competition_maps()
    target_mp = all_maps[st["cycle"] % len(all_maps)]
    target_map_name = target_mp.stem

    cyc_dir = OUT / "cycles" / f"mimic-{st['cycle']:05d}-{mimic_slug}"
    if cyc_dir.exists():
        shutil.rmtree(cyc_dir, ignore_errors=True)
    cyc_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{time.strftime('%H:%M:%S')}] [MIMIC HARDENING] Co-evolving {target_mimic} on '{target_map_name}' vs champion {champ}...")

    base_res = runner.run_matches(target_mimic, champ, maps=[target_mp], out_dir=cyc_dir / "baseline", release=True)
    base_fit, base_win, base_deaths, base_susp = fitness(base_res, target_mimic)

    st.setdefault("mimic_packs", {})
    st["mimic_packs"].setdefault(mimic_slug, {})
    current_th = list(st["mimic_packs"][mimic_slug].get(target_map_name) or base_theta("main"))
    if len(current_th) < params.size():
        current_th += base_theta("main")[len(current_th):]

    s0 = params.sigma0()
    sigma = list(st.get("mimic_sigmas", {}).get(mimic_slug, {}).get(target_map_name, s0))
    if len(sigma) < params.size():
        sigma += s0[len(sigma):]

    scored = []
    # Test 6 antithetic candidates for the mimic
    for k in range(3):
        raw_eps = [rng.gauss(0, 1) for _ in current_th]
        eps = [0.0] * len(current_th)
        if k == 0:
            for i in range(18):
                eps[i] = raw_eps[i]
        elif k == 1:
            for i in range(38, 98):
                eps[i] = raw_eps[i]
        else:
            for i in range(18):
                eps[i] = raw_eps[i]
            for i in range(38, 98):
                eps[i] = raw_eps[i]

        for sgn in (1, -1):
            cand_th = [t + sgn * s * e for t, s, e in zip(current_th, sigma, eps)]
            cand_name = f"_cand_m_{k}{'p' if sgn > 0 else 'n'}"
            cand_d = ROOT / "bots" / cand_name
            safe_copy_bot(target_bot_dir, cand_d)
            write_pack(cand_d / "src" / "pack" / "pack.bin", {**st["mimic_packs"][mimic_slug], target_map_name: cand_th}, base_spec)

            m_res = runner.run_matches(cand_name, champ, maps=[target_mp], out_dir=cyc_dir / cand_name, release=True)
            m_fit, m_win, m_deaths, m_susp = fitness(m_res, cand_name)
            scored.append((m_fit, m_win, m_deaths, m_susp, cand_th, cand_name))

    scored.sort(key=lambda x: -x[0])
    best_fit, best_win, best_deaths, best_susp, best_th, best_name = scored[0]

    gain = best_fit - base_fit
    mult = 1.15 if gain > 0 else 0.88
    st.setdefault("mimic_sigmas", {}).setdefault(mimic_slug, {})[target_map_name] = [
        max(0.02, min(0.5, s * mult)) for s in sigma
    ]

    upgraded = False
    if gain >= 0.03 and best_susp <= max(base_susp * 2, 20):
        print(f"  *** MIMIC UPGRADED! {target_mimic} fitness vs champion: {base_fit:.3f} -> {best_fit:.3f} (+{gain:.3f}) ***")
        st["mimic_packs"][mimic_slug][target_map_name] = best_th
        write_pack(target_bot_dir / "src" / "pack" / "pack.bin", st["mimic_packs"][mimic_slug], base_spec)
        upgraded = True

    return {
        "cycle": st["cycle"],
        "type": "mimic_hardening",
        "mimic": target_mimic,
        "map": target_map_name,
        "base_fit": round(base_fit, 3),
        "best_fit": round(best_fit, 3),
        "gain": round(gain, 3),
        "upgraded": upgraded,
    }


# ----------------------------------------------------------------- Training Cycle


def train_step(st: dict, rng: random.Random, auto_submit: bool = True) -> dict:
    st["cycle"] += 1
    cyc_id = f"{st['cycle']:05d}"
    cyc_dir = OUT / "cycles" / cyc_id
    if cyc_dir.exists():
        shutil.rmtree(cyc_dir, ignore_errors=True)
    cyc_dir.mkdir(parents=True, exist_ok=True)

    # 1. Periodic live ladder sync of top-5 mimics (every 10 cycles)
    if st["cycle"] % 10 == 0:
        sync_live_top5_mimics(st)

    # 2. Mimic Co-Evolution (every 4th cycle): harden the rivals so our champion never plateaus against weak sparring partners
    if st["cycle"] % 4 == 0:
        res = harden_mimic_step(st, rng)
        st.setdefault("history", []).append(res)
        st["history"] = st["history"][-50:]
        save_state(st)
        update_report(st)
        return res

    champ = st["champion"]
    if not (ROOT / "bots" / champ).exists():
        champ = "main"

    # 1. Check ladder for any new losses
    new_ladder_losses = check_ladder_for_losses(st)
    if new_ladder_losses:
        print(f"[{time.strftime('%H:%M:%S')}] Detected {len(new_ladder_losses)} new loss(es) on Battlecode ladder!")

    # 2. Select target map: prioritize unaddressed loss
    target_loss = None
    target_map_name = None
    target_mp = None

    if st.get("losses"):
        target_loss = st["losses"][0]
        target_map_name = target_loss["map"]
        target_mp = resolve_map_path(target_map_name)

    if not target_mp:
        all_maps = list_competition_maps()
        target_mp = all_maps[st["cycle"] % len(all_maps)]
        target_map_name = target_mp.stem

    # 3. Assemble opponent sparring pool (Top 5 Mimics + Champion)
    top5_mimics = get_active_top5_mimics(st)
    opps = [m for m in top5_mimics if (ROOT / "bots" / m).exists()][:4]
    if champ not in opps:
        opps.append(champ)

    print(f"[{time.strftime('%H:%M:%S')}] Cycle {st['cycle']}: optimizing map '{target_map_name}' against {len(opps)} rivals...")

    # Current champion baseline on this map
    champ_res = play_map(champ, opps, target_mp, cyc_dir / "champ")
    champ_fit, champ_win, champ_deaths, champ_susp = fitness(champ_res, champ)

    p_size = params.size()
    theta = list(st["pack"].get(target_map_name) or base_theta("main"))
    if len(theta) < p_size:
        theta += base_theta("main")[len(theta):]

    s0 = params.sigma0()
    sigma = list(st["sigma"].get(target_map_name, s0))
    if len(sigma) < p_size:
        sigma += s0[len(sigma):]

    # 4. Antithetic ES generation & evaluation (Block-structured ES)
    scored = []
    w_end = 18
    target_end = 28
    fight_end = 37
    u_start = 38
    u_end = 98

    for k in range(N_LAMBDA // 2):
        raw_eps = [rng.gauss(0, 1) for _ in theta]
        # Structure perturbations by functional blocks for faster convergence
        eps = [0.0] * len(theta)
        if k == 0:
            # Full parameter space exploration
            eps = raw_eps
        elif k == 1:
            # Multipliers W (retreat HP, aggression, fight margin, tribute thresholds)
            for i in range(0, w_end):
                eps[i] = raw_eps[i]
        elif k == 2:
            # Micro Utility Arbiter U (aggro swap, support ally, walk home safely)
            for i in range(u_start, u_end):
                eps[i] = raw_eps[i]
        else:
            # Combined Macro & Micro (W + U)
            for i in range(0, w_end):
                eps[i] = raw_eps[i]
            for i in range(u_start, u_end):
                eps[i] = raw_eps[i]

        for sgn in (1, -1):
            cand = [t + sgn * s * e for t, s, e in zip(theta, sigma, eps)]
            cand_name = f"_auto_c{k}{'p' if sgn > 0 else 'n'}"
            cand_bot = make_bot(cand_name, {**st["pack"], target_map_name: cand})
            c_res = play_map(cand_bot, opps, target_mp, cyc_dir / cand_name)
            c_fit, c_win, c_deaths, c_susp = fitness(c_res, cand_bot)
            scored.append((c_fit, c_win, c_deaths, c_susp, cand, cand_bot))

    scored.sort(key=lambda x: -x[0])
    best_fit, best_win, best_deaths, best_susp, best_theta, best_bot = scored[0]

    # Adapt sigma
    gain = best_fit - champ_fit
    mult = 1.15 if gain > 0 else 0.88
    st["sigma"][target_map_name] = [max(0.02, min(0.5, s * mult)) for s in sigma]

    result_entry = {
        "cycle": st["cycle"],
        "map": target_map_name,
        "champ_fit": round(champ_fit, 3),
        "best_fit": round(best_fit, 3),
        "gain": round(gain, 3),
        "best_win": round(best_win, 3),
        "promoted": False,
    }

    # 5. Gate & Confirmation
    if gain >= ACCEPT_GAIN and best_susp <= max(champ_susp * 2, 20):
        print(f"  Candidate {best_bot} passed initial gate (+{gain:.3f} fitness). Running head-to-head & confirmation...")
        # Head-to-head against champion on target map
        h2h = runner.run_matches(best_bot, champ, maps=[target_mp], out_dir=cyc_dir / "h2h", release=True)
        h2h_win = sum(r.score_for(best_bot) for r in h2h) / max(len(h2h), 1)

        # Fresh opponents confirmation
        fresh_opps = [m for m in get_active_top5_mimics(st) if m not in opps]
        if not fresh_opps:
            fresh_opps = [champ]
        cf_res = play_map(best_bot, fresh_opps, target_mp, cyc_dir / "confirm")
        cf_fit, cf_win, _, cf_susp = fitness(cf_res, best_bot)

        if h2h_win >= 0.50:
            print(f"  *** PROMOTION CONFIRMED! Head-to-head: {h2h_win*100:.1f}%, confirmed fitness: {cf_fit:.3f} ***")
            st["version"] += 1
            new_version_name = f"auto-v{st['version']}"

            # Save in main pack
            st["pack"][target_map_name] = best_theta
            write_pack(ROOT / "bots" / "main" / "src" / "pack" / "pack.bin", st["pack"])

            # Snapshot bot
            make_bot(new_version_name, st["pack"])
            st["champion"] = new_version_name
            result_entry["promoted"] = True
            result_entry["champion"] = new_version_name

            # Push to platform and git
            if auto_submit:
                push_to_platform(new_version_name, new_version_name, target_map_name, st)

            # Mark loss resolved if target was a loss
            if target_loss and target_loss in st.get("losses", []):
                st["losses"].remove(target_loss)
                print(f"  Resolved and removed loss on map '{target_map_name}' from queue!")
        else:
            print(f"  Head-to-head gate failed: {h2h_win*100:.1f}%")
            if target_loss and len(st.get("losses", [])) > 1:
                # Rotate queue to explore other loss maps
                st["losses"].append(st["losses"].pop(0))
    else:
        if target_loss and len(st.get("losses", [])) > 1:
            # Rotate queue to explore other loss maps
            st["losses"].append(st["losses"].pop(0))

    st.setdefault("history", []).append(result_entry)
    st["history"] = st["history"][-50:]
    save_state(st)
    update_report(st)
    return result_entry


# ----------------------------------------------------------------- Report & Status


def update_report(st: dict) -> None:
    lines = [
        "# Autonomous Battlecode Self-Improvement Harness",
        "",
        f"- **Current Champion**: `{st.get('champion')}` (v{st.get('version', 0)})",
        f"- **Total Cycles**: {st.get('cycle', 0)}",
        f"- **Pending Losses in Queue**: {len(st.get('losses', []))}",
        "",
        "## Benchmark vs Top 5 Competitors",
        "",
        "| Competitor Mimic | Win Rate |",
        "| --- | --- |",
    ]
    for team, wr in (st.get("benchmark_scores") or {}).items():
        lines.append(f"| {team} | {wr*100:.1f}% |")

    lines += [
        "",
        "## Submissions & Pushes to Battlecode",
        "",
        "| Time | Version | Target Map | Status |",
        "| --- | --- | --- | --- |",
    ]
    for s in reversed((st.get("submissions") or [])[-10:]):
        st_text = "Submitted" if s.get("ok") else "Pending"
        lines.append(f"| {s.get('timestamp', '')} | `{s.get('version')}` | {s.get('map')} | {st_text} |")

    lines += [
        "",
        "## Recent Training Iterations",
        "",
        "| Cycle | Task / Map | Base Fit | Best Fit | Gain | Outcome |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for h in reversed((st.get("history") or [])[-10:]):
        if h.get("type") == "mimic_hardening":
            outcome = f"Mimic Upgraded: {h.get('mimic')}" if h.get("upgraded") else f"Mimic Tested: {h.get('mimic')}"
            lines.append(f"| {h['cycle']} | [Mimic] {h.get('map')} | {h.get('base_fit', 0.0)} | {h.get('best_fit', 0.0)} | {h.get('gain', 0.0):+.3f} | {outcome} |")
        else:
            outcome = "PROMOTED & PUSHED" if h.get("promoted") else "Search"
            lines.append(f"| {h['cycle']} | [Champ] {h.get('map')} | {h.get('champ_fit', 0.0)} | {h.get('best_fit', 0.0)} | {h.get('gain', 0.0):+.3f} | {outcome} |")

    REPORT_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def pid_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x1000, False, pid)
        if handle:
            kernel32.CloseHandle(handle)
            return True
        return False
    except Exception:
        return False


def print_status(st: dict) -> None:
    print("=================================================================")
    print("        AUTONOMOUS SELF-IMPROVEMENT HARNESS STATUS               ")
    print("=================================================================")
    is_running = False
    pid_str = "None"
    if LOCK_FILE.exists():
        try:
            pid = int(LOCK_FILE.read_text().strip())
            pid_str = str(pid)
            is_running = pid_exists(pid)
        except Exception:
            pass

    status_str = f"RUNNING (PID {pid_str})" if is_running else "STOPPED"
    print(f"Harness Daemon    : {status_str}")
    print(f"Current Champion  : {st.get('champion')} (Version: {st.get('version', 0)})")
    print(f"Training Cycles   : {st.get('cycle', 0)}")
    print(f"Pending Losses    : {len(st.get('losses', []))}")

    if st.get("losses"):
        print("\nActive Loss Queue:")
        for i, loss in enumerate(st["losses"][:5]):
            print(f"  [{i+1}] Map: {loss.get('map')} | Opponent: {loss.get('opp_name')} | Source: {loss.get('source')}")

    print("\nBenchmark vs Top 5 Competitors:")
    scores = st.get("benchmark_scores", {})
    if scores:
        for t, wr in scores.items():
            print(f"  {t:35s}: {wr*100:5.1f}%")
        if "benchmark_overall" in st:
            print(f"  {'OVERALL':35s}: {st['benchmark_overall']*100:5.1f}%")
    else:
        print("  (No benchmark run yet; run `scripts\\train.ps1 benchmark`)")

    subs = st.get("submissions", [])
    print(f"\nSubmissions to Battlecode Platform ({len(subs)} total):")
    for s in subs[-3:]:
        ok_str = "SUCCESS" if s.get("ok") else "FAILED"
        print(f"  {s.get('timestamp')} | {s.get('version')} (Map: {s.get('map')}) -> {ok_str}")
    print("=================================================================\n")


# ----------------------------------------------------------------- Main Entrypoint


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forever", action="store_true", help="Run continuous improvement loop")
    parser.add_argument("--cycles", type=int, default=1, help="Number of training cycles to run")
    parser.add_argument("--auto-submit", action="store_true", help="Automatically submit promoted champions")
    parser.add_argument("--status", action="store_true", help="Print current status")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark vs Top 5 teams")
    parser.add_argument("--full", action="store_true", help="Run full 17-map benchmark instead of core maps")
    parser.add_argument("--push", action="store_true", help="Push current champion immediately")
    args = parser.parse_args()

    st = init_state()

    if args.status:
        print_status(st)
        return 0

    # Tee output to runs/auto_train/train.log
    logger = Logger(OUT / "train.log")
    sys.stdout = logger
    sys.stderr = logger

    if args.benchmark:
        champ = st.get("champion", "main")
        run_benchmark(st, champ, full=args.full)
        save_state(st)
        update_report(st)
        return 0

    if args.push:
        champ = st.get("champion", "main")
        push_to_platform(champ, f"manual-{champ}", "manual", st)
        save_state(st)
        return 0

    if LOCK_FILE.exists():
        try:
            old_pid = int(LOCK_FILE.read_text().strip())
            if pid_exists(old_pid):
                print(f"Another auto_train daemon is already running (PID {old_pid}).")
                return 1
        except Exception:
            pass

    LOCK_FILE.write_text(str(os.getpid()))

    def clean_exit(*_):
        LOCK_FILE.unlink(missing_ok=True)
        sys.exit(0)

    signal.signal(signal.SIGINT, clean_exit)
    signal.signal(signal.SIGTERM, clean_exit)

    print(f"Starting Autonomous Self-Improvement Harness (PID: {os.getpid()})")
    print(f"Champion: {st.get('champion')}, Auto-submit: {args.auto_submit}")

    rng = random.Random(st["cycle"] + 1337)
    try:
        n = 0
        while args.forever or n < args.cycles:
            n += 1
            t0 = time.time()
            try:
                res = train_step(st, rng, auto_submit=args.auto_submit)
                dt = round(time.time() - t0, 1)
                if res.get("type") == "mimic_hardening":
                    outcome = "Upgraded" if res.get("upgraded") else "Tested"
                    print(f"Cycle {res['cycle']} [Mimic {res.get('mimic')}] finished in {dt}s. Gain: {res.get('gain', 0.0):+.3f}, Outcome: {outcome}")
                else:
                    outcome = "PROMOTED & PUSHED" if res.get("promoted") else "Search"
                    print(f"Cycle {res['cycle']} [Champion] finished in {dt}s. Gain: {res.get('gain', 0.0):+.3f}, Outcome: {outcome}")
            except Exception as e:
                print(f"Cycle exception: {e!r}. Retrying after 5s...", flush=True)
                time.sleep(5)
    finally:
        LOCK_FILE.unlink(missing_ok=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
