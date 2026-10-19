"""Co-evolution of two lineages with per-map ES (map-aware policy packs).

Lineage A is our engine (bots/main); lineage B is the same engine seeded with the top
ladder teams' comp and build (DRAFT_BEST + BUILD_PROFILE 13), source in bots/lin-b.
Each lineage owns a policy pack (bots/<src>/src/pack/pack.bin): one learned θ per map
(learned.rs layout: log-W multipliers, TARGET, FIGHT, HOTSPOT_W). Every cycle one
lineage evolves one map (round-robin): antithetic ES samples around that map's θ, each
candidate plays that map only (both sides) against the *other* lineage's hall of fame
(PFSP-weighted) plus its own champion. Only that map's behaviour changes, so the
per-map comparison is exact; a candidate is accepted if its fitness beats the
champion's on the same games and it does not lose the head-to-head on that map.

Accepted packs make a new champion snapshot (bots/coA-vN / coB-vN) that joins the
hall of fame; A's pack is committed and both lineages' zips are bundled. Every
CROSS_EVERY cycles the two champions play all maps, and a Bradley-Terry Elo over the
hall is written to runs/coevo/report.md, so progress is measured without the ladder.

    python harness/coevo.py --forever
    python harness/coevo.py --cycles 2       # smoke
Local only (bundles zips, commits, pushes; never submits).
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

import pack as packlib
import v6_trials

from bctools import ROOT, runner

CAMBC = ROOT / ".venv" / "Scripts" / "cambc.exe"
OUT = ROOT / "runs" / "coevo"
STATE = OUT / "state.json"
LOCK = OUT / "lock"
N_W, N_T, N_F = 18, 10, 9
P_N = N_W + N_T + N_F + 1
LAMBDA = 8
SIGMA0 = [0.25] * N_W + [0.2] * (N_T + N_F) + [0.0]
ACCEPT_GAIN = 0.05
DEATH_W = 0.01
MARGIN_W = 0.25
CROSS_EVERY = 10
FIXED = ["main-v13b", "main-v12"]
LINEAGE_B_KNOBS = {"DRAFT_BEST": 1, "BUILD_PROFILE": 13}


def sh(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(args, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")


def base_theta(src: str) -> list[float]:
    """The lineage's compiled defaults as a θ vector (log W, TARGET, FIGHT, HOTSPOT_W)."""
    text = (ROOT / "bots" / src / "src" / "learned.rs").read_text(encoding="utf-8")

    def arr(name: str, n: int, d: float) -> list[float]:
        m = re.search(rf"pub const {name}: \[f32; \w+\] = \[(.*?)\];", text, re.S)
        if not m or ";" in m.group(1):
            return [d] * n
        v = [float(x) for x in m.group(1).split(",") if x.strip()]
        return v if len(v) == n else [d] * n

    return [math.log(max(w, 1e-3)) for w in arr("W", N_W, 1.0)] + arr("TARGET", N_T, 0.0) + arr("FIGHT", N_F, 0.0) + [0.0]


def to_pack_theta(theta: list[float]) -> list[float]:
    w = [math.exp(min(max(x, math.log(0.2)), math.log(5.0))) for x in theta[:N_W]]
    rest = [min(max(x, -3.0), 3.0) for x in theta[N_W:]]
    return w + rest


def write_pack(path: Path, pack: dict[str, list[float]]) -> None:
    packlib.write({"maps": {m: {"theta": to_pack_theta(t)} for m, t in pack.items()}}, path)


def make(lin: dict, name: str, pack: dict[str, list[float]]) -> str:
    bot = name
    d = ROOT / "bots" / bot
    src = ROOT / "bots" / lin["src"]
    if d.exists():
        shutil.rmtree(d / "src", ignore_errors=True)
        shutil.copytree(src / "src", d / "src")
    else:
        shutil.copytree(src, d, ignore=shutil.ignore_patterns(".cambc", "target"))
    write_pack(d / "src" / "pack" / "pack.bin", pack)
    return bot


def fitness(results: list[runner.MatchResult], bot: str) -> tuple[float, float]:
    n = max(len(results), 1)
    win = sum(r.score_for(bot) for r in results) / n
    margin = deaths = 0.0
    for r in results:
        ours, theirs = (r.gems_a, r.gems_b) if r.bot_a == bot else (r.gems_b, r.gems_a)
        margin += (ours - theirs) / max(ours + theirs, 1)
        deaths += r.deaths_a if r.bot_a == bot else r.deaths_b
    return win + MARGIN_W * margin / n - DEATH_W * deaths / n, win


def play_map(bot: str, opps: list[str], mp: Path, out: Path) -> list[runner.MatchResult]:
    res = []
    for o in opps:
        res += runner.run_matches(bot, o, maps=[mp], out_dir=out / o, release=True)
    return res


def init_state() -> dict:
    if STATE.exists():
        return json.loads(STATE.read_text(encoding="utf-8"))
    linb = ROOT / "bots" / "lin-b"
    if linb.exists():
        shutil.rmtree(linb)
    shutil.copytree(ROOT / "bots" / "main", linb, ignore=shutil.ignore_patterns(".cambc", "target"))
    mem = linb / "src" / "bc_core" / "memory.rs"
    mem.write_text(v6_trials.set_knobs(mem.read_text(encoding="utf-8"), LINEAGE_B_KNOBS), encoding="utf-8", newline="\n")
    st = {"cycle": 0, "pwin": {}, "elo": {}, "pairs": {}, "lineages": {}}
    for key, src in (("A", "main"), ("B", "lin-b")):
        lin = {"src": src, "pack": {}, "version": 0, "map_rr": 0, "sigma": {}}
        snap = make(lin, f"co{key}-v0", {})
        lin["champion"] = snap
        lin["hall"] = [snap]
        st["lineages"][key] = lin
    return st


def save(st: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, indent=1), encoding="utf-8")
    tmp.replace(STATE)


def pfsp(st: dict, me: str, other: dict, k: int, rng: random.Random) -> list[str]:
    cands = [b for b in other["hall"][-6:] + FIXED if (ROOT / "bots" / b).exists()]
    cands = list(dict.fromkeys(cands))
    w = [(1.0 - st["pwin"].get(f"{me}|{b}", 0.5)) ** 2 + 0.05 for b in cands]
    out = []
    while cands and len(out) < k:
        i = rng.choices(range(len(cands)), weights=w)[0]
        out.append(cands.pop(i))
        w.pop(i)
    return out


def evolve_map(st: dict, key: str, rng: random.Random, cyc: Path) -> dict:
    lin, other = st["lineages"][key], st["lineages"]["B" if key == "A" else "A"]
    all_maps = runner.list_maps()
    mp = all_maps[lin["map_rr"] % len(all_maps)]
    lin["map_rr"] += 1
    name = mp.stem
    theta = lin["pack"].get(name) or base_theta(lin["src"])
    sigma = lin["sigma"].get(name, SIGMA0)
    opps = pfsp(st, lin["champion"], other, 3, rng)
    champ_res = play_map(lin["champion"], opps, mp, cyc / "champ")
    champ_fit, _ = fitness(champ_res, lin["champion"])
    for o in opps:
        st["pwin"][f"{lin['champion']}|{o}"] = sum(r.score_for(lin["champion"]) for r in champ_res
                                                  if o in (r.bot_a, r.bot_b)) / 2
    scored = []
    for k in range(LAMBDA // 2):
        eps = [rng.gauss(0, 1) for _ in theta]
        for sgn in (1, -1):
            cand = [t + sgn * s * e for t, s, e in zip(theta, sigma, eps)]
            bot = make(lin, f"_co{key}-{k}{'p' if sgn > 0 else 'n'}", {**lin["pack"], name: cand})
            res = play_map(bot, opps, mp, cyc / bot)
            f, w = fitness(res, bot)
            scored.append((f, w, cand, bot))
    scored.sort(key=lambda s: -s[0])
    best_f, best_w, best_theta, best_bot = scored[0]
    # Sigma follows success (narrow when nothing beats the champion on this map).
    lin["sigma"][name] = [max(0.03, min(0.6, s * (1.15 if best_f > champ_fit else 0.85))) for s in sigma]
    entry = {"lineage": key, "map": name, "opps": opps, "champ_fit": round(champ_fit, 3), "best_fit": round(best_f, 3)}
    if best_f >= champ_fit + ACCEPT_GAIN:
        h2h = runner.run_matches(best_bot, lin["champion"], maps=[mp], out_dir=cyc / "h2h", release=True)
        hw = sum(r.score_for(best_bot) for r in h2h) / max(len(h2h), 1)
        entry["h2h"] = hw
        if hw >= 0.5:
            lin["pack"][name] = best_theta
            lin["version"] += 1
            snap = make(lin, f"co{key}-v{lin['version']}", lin["pack"])
            write_pack(ROOT / "bots" / lin["src"] / "src" / "pack" / "pack.bin", lin["pack"])
            lin["champion"] = snap
            lin["hall"].append(snap)
            entry["accepted"] = snap
            zip_path = ROOT / "submission" / f"{snap}.zip"
            sh([str(CAMBC), "bundle", snap, "--output", str(zip_path)])
            if key == "A":
                sh(["git", "add", "bots/main/src/pack/pack.bin"])
                sh(["git", "commit", "-q", "-m", f"auto: coevo {snap} ({name}: fit {champ_fit:.3f} -> {best_f:.3f})\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"])
                sh(["git", "push", "-q"])
    return entry


def cross_check(st: dict, cyc: Path) -> str:
    a, b = st["lineages"]["A"]["champion"], st["lineages"]["B"]["champion"]
    rs = runner.run_matches(a, b, out_dir=cyc / "cross", release=True)
    wa = sum(r.score_for(a) for r in rs) / max(len(rs), 1)
    st["pairs"][f"{a}|{b}"] = wa
    elo = bradley_terry(st["pairs"])
    st["elo"] = elo
    lines = ["# Co-evolution", "", f"Cross-check {a} vs {b}: {wa:.3f}", "", "| bot | Elo |", "|---|---|"]
    lines += [f"| {k} | {v:.0f} |" for k, v in sorted(elo.items(), key=lambda kv: -kv[1])]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return f"cross {a} vs {b}: {wa:.3f}"


def bradley_terry(pairs: dict[str, float], iters: int = 200) -> dict[str, float]:
    players = sorted({p for k in pairs for p in k.split("|")})
    r = {p: 0.0 for p in players}
    for _ in range(iters):
        for k, s in pairs.items():
            a, b = k.split("|")
            e = 1 / (1 + 10 ** ((r[b] - r[a]) / 400))
            r[a] += 8 * (s - e)
            r[b] -= 8 * (s - e)
    return {p: 1500 + v for p, v in r.items()}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--forever", action="store_true")
    p.add_argument("--cycles", type=int, default=2)
    a = p.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    if LOCK.exists() and time.time() - LOCK.stat().st_mtime < 6 * 3600:
        print(f"another coevo run holds {LOCK}")
        return 1
    LOCK.write_text(str(os.getpid()))
    try:
        st = init_state()
        save(st)
        rng = random.Random(st["cycle"] + 99)
        n = 0
        while a.forever or n < a.cycles:
            n += 1
            st["cycle"] += 1
            LOCK.write_text(str(os.getpid()))
            key = "A" if st["cycle"] % 2 else "B"
            cyc = OUT / "cycles" / f"{st['cycle']:05d}-{key}"
            t0 = time.time()
            try:
                e = evolve_map(st, key, rng, cyc)
            except Exception as ex:  # keep running
                e = {"lineage": key, "error": repr(ex)[:300]}
            print(f"[{time.strftime('%H:%M')}] {st['cycle']} {e} ({round(time.time() - t0)}s)", flush=True)
            if st["cycle"] % CROSS_EVERY == 0:
                try:
                    print("   " + cross_check(st, cyc), flush=True)
                except Exception as ex:
                    print(f"   cross error {ex!r}", flush=True)
            save(st)
    finally:
        LOCK.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
