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
import params
import v6_trials
from league_gate import suspended

from bctools import ROOT, runner

CAMBC = ROOT / ".venv" / "Scripts" / "cambc.exe"
OUT = ROOT / "runs" / "coevo"
STATE = OUT / "state.json"
LOCK = OUT / "lock"
N_W = 18
LAMBDA = 8
#: Fitness penalty per suspended (out-of-fuel) turn per game: a learned policy must not buy
#: strength by risking a frozen champion.
SUSP_W = 0.002
ACCEPT_GAIN = 0.05
DEATH_W = 0.01
MARGIN_W = 0.25
CROSS_EVERY = 10
FIXED = ["main-v13b", "main-v12"]
#: Opponents per candidate evaluation (x2 sides on the map). The engine is deterministic
#: but chaotic (a no-op code change flips ~25% of winners), so few games are noise.
N_OPPS = 5
#: Re-check an accepted candidate on this many fresh opponents before it counts.
N_CONFIRM = 4
MIMIC_REFRESH_EVERY = 30
LINEAGES = ("A", "B", "C")
#: Lineage C: imitation-initialised from a top-team mimic (harness/mimic.py).
LINEAGE_C_SPEC = next((p for p in (ROOT / "runs" / "mimic" / "nyx.spec.json", ROOT / "data" / "mimic" / "nyx.spec.json") if p.exists()), ROOT / "data" / "mimic" / "nyx.spec.json")
LINEAGE_B_KNOBS = {"DRAFT_BEST": 1, "BUILD_PROFILE": 13}


def sh(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(args, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")


def base_theta(src: str) -> list[float]:
    """The lineage's compiled defaults as a search-space θ (harness/params.py layout)."""
    return params.to_search(params.defaults(ROOT / "bots" / src / "src" / "learned.rs"))


def full_theta(theta: list[float], src: str) -> list[float]:
    """Pad an older, shorter θ with the lineage defaults for the parameters added since."""
    base = base_theta(src)
    return list(theta) + base[len(theta):]


def to_pack_theta(theta: list[float], src: str = "main") -> list[float]:
    return params.to_bot(theta, params.defaults(ROOT / "bots" / src / "src" / "learned.rs"))


def write_pack(path: Path, pack: dict[str, list[float]], base: dict | None = None) -> None:
    """Evolved per-map theta on top of an optional base spec (lineage C keeps the mimic's
    draft / build / opening sections)."""
    maps = {m: dict(e) for m, e in ((base or {}).get("maps") or {}).items()}
    for m, th in pack.items():
        maps.setdefault(m, {})["theta"] = to_pack_theta(th)  # noqa: B023
    packlib.write({"maps": maps}, path)


def mimic_bots() -> list[str]:
    return sorted(q.name for q in (ROOT / "bots").glob("mimic-*") if q.is_dir())


def make(lin: dict, name: str, pack: dict[str, list[float]]) -> str:
    bot = name
    d = ROOT / "bots" / bot
    src = ROOT / "bots" / lin["src"]
    if d.exists():
        shutil.rmtree(d / "src", ignore_errors=True)
        shutil.copytree(src / "src", d / "src")
    else:
        shutil.copytree(src, d, ignore=shutil.ignore_patterns(".cambc", "target"))
    write_pack(d / "src" / "pack" / "pack.bin", pack, lin.get("base"))
    return bot


def fitness(results: list[runner.MatchResult], bot: str) -> tuple[float, float]:
    n = max(len(results), 1)
    win = sum(r.score_for(bot) for r in results) / n
    susp = suspended(results, bot)[0] / n
    margin = deaths = 0.0
    for r in results:
        ours, theirs = (r.gems_a, r.gems_b) if r.bot_a == bot else (r.gems_b, r.gems_a)
        margin += (ours - theirs) / max(ours + theirs, 1)
        deaths += r.deaths_a if r.bot_a == bot else r.deaths_b
    return win + MARGIN_W * margin / n - DEATH_W * deaths / n - SUSP_W * susp, win


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


def ensure_lineage_c(st: dict) -> None:
    if "C" in st["lineages"] or not LINEAGE_C_SPEC.exists():
        return
    linc = ROOT / "bots" / "lin-c"
    if linc.exists():
        shutil.rmtree(linc)
    shutil.copytree(ROOT / "bots" / "main", linc, ignore=shutil.ignore_patterns(".cambc", "target"))
    base = json.loads(LINEAGE_C_SPEC.read_text(encoding="utf-8"))
    # The mimic's theta seeds lineage C's per-map theta (log space for the W multipliers).
    dflt = params.defaults(ROOT / "bots" / "lin-c" / "src" / "learned.rs")
    pack = {m: params.to_search(list(e["theta"]) + dflt[len(e["theta"]):])
            for m, e in base["maps"].items() if e.get("theta")}
    lin = {"src": "lin-c", "pack": pack, "version": 0, "map_rr": 0, "sigma": {}, "base": base}
    snap = make(lin, "coC-v0", pack)
    lin["champion"] = snap
    lin["hall"] = [snap]
    st["lineages"]["C"] = lin


def refresh_mimics() -> str:
    """Re-extract the top teams from newly downloaded games and rebuild the mimic bots."""
    r = sh([sys.executable, str(ROOT / "harness" / "mimic.py"), "--all-top"])
    n = 0
    for spec in (ROOT / "runs" / "mimic").glob("*.spec.json"):
        bot = ROOT / "bots" / ("mimic-" + spec.name.replace(".spec.json", ""))
        if bot.exists():
            shutil.rmtree(bot)
        shutil.copytree(ROOT / "bots" / "main", bot, ignore=shutil.ignore_patterns(".cambc", "target"))
        packlib.write(json.loads(spec.read_text(encoding="utf-8")), bot / "src" / "pack" / "pack.bin")
        n += 1
    return f"mimics refreshed: {n} (rc {r.returncode})"


def save(st: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, indent=1), encoding="utf-8")
    tmp.replace(STATE)


def pfsp(st: dict, me: str, others: list[dict], k: int, rng: random.Random, exclude: set | None = None) -> list[str]:
    pool = [b for o in others for b in o["hall"][-6:]] + FIXED + mimic_bots()
    cands = [b for b in dict.fromkeys(pool) if (ROOT / "bots" / b).exists() and b not in (exclude or set())]
    w = [(1.0 - st["pwin"].get(f"{me}|{b}", 0.5)) ** 2 + 0.05 for b in cands]
    out = []
    while cands and len(out) < k:
        i = rng.choices(range(len(cands)), weights=w)[0]
        out.append(cands.pop(i))
        w.pop(i)
    return out


def evolve_map(st: dict, key: str, rng: random.Random, cyc: Path) -> dict:
    lin = st["lineages"][key]
    others = [v for k2, v in st["lineages"].items() if k2 != key]
    all_maps = runner.list_maps()
    mp = all_maps[lin["map_rr"] % len(all_maps)]
    lin["map_rr"] += 1
    name = mp.stem
    theta = full_theta(lin["pack"].get(name) or base_theta(lin["src"]), lin["src"])
    s0 = params.sigma0()
    sigma = list(lin["sigma"].get(name, s0)) + s0[len(lin["sigma"].get(name, s0)):]
    opps = pfsp(st, lin["champion"], others, N_OPPS, rng)
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
        # Confirmation on fresh opponents (guards against chaos noise) + head-to-head.
        fresh = pfsp(st, lin["champion"], others, N_CONFIRM, rng, exclude=set(opps))
        cf, chf = best_f, champ_fit
        if fresh:
            cf, _ = fitness(play_map(best_bot, fresh, mp, cyc / "confirm"), best_bot)
            chf, _ = fitness(play_map(lin["champion"], fresh, mp, cyc / "confirm-champ"), lin["champion"])
        h2h = runner.run_matches(best_bot, lin["champion"], maps=[mp], out_dir=cyc / "h2h", release=True)
        hw = sum(r.score_for(best_bot) for r in h2h) / max(len(h2h), 1)
        entry.update({"h2h": hw, "confirm": round(cf, 3), "confirm_champ": round(chf, 3)})
        if hw >= 0.5 and cf >= chf:
            lin["pack"][name] = best_theta
            lin["version"] += 1
            snap = make(lin, f"co{key}-v{lin['version']}", lin["pack"])
            write_pack(ROOT / "bots" / lin["src"] / "src" / "pack" / "pack.bin", lin["pack"], lin.get("base"))
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
    champs = [st["lineages"][k]["champion"] for k in LINEAGES if k in st["lineages"]]
    msgs = []
    for i, a in enumerate(champs):
        for b in champs[i + 1:] + ["main-v13b"]:
            if a == b:
                continue
            rs = runner.run_matches(a, b, out_dir=cyc / "cross" / f"{a}-{b}", release=True)
            wa = sum(r.score_for(a) for r in rs) / max(len(rs), 1)
            st["pairs"][f"{a}|{b}"] = wa
            msgs.append(f"{a} vs {b}: {wa:.3f}")
    elo = bradley_terry(st["pairs"])
    st["elo"] = elo
    lines = ["# Co-evolution", "", *[f"- {m}" for m in msgs], "", "| bot | Elo |", "|---|---|"]
    lines += [f"| {k} | {v:.0f} |" for k, v in sorted(elo.items(), key=lambda kv: -kv[1])]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return "cross: " + "; ".join(msgs)


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
        ensure_lineage_c(st)
        save(st)
        rng = random.Random(st["cycle"] + 99)
        n = 0
        while a.forever or n < a.cycles:
            n += 1
            st["cycle"] += 1
            LOCK.write_text(str(os.getpid()))
            keys = [k for k in LINEAGES if k in st["lineages"]]
            key = keys[st["cycle"] % len(keys)]
            if st["cycle"] % MIMIC_REFRESH_EVERY == 0:
                print("   " + refresh_mimics(), flush=True)
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
