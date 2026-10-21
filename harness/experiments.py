"""Experiment autopilot: a shared queue of ideas, each screened and checked against the live bot.

An experiment is a knob set (`{"EN_ROUTE": 2}`) applied to a copy of bots/main, or a git ref
whose bots/main is the variant (code changes). Every experiment goes through the same gate:

  screen  7 maps x every faithful mimic, both sides, variant and live bot on the same games;
          passes if fitness >= live and win >= live - SCREEN_SLACK
  full    every active map x every mimic + head-to-head against the live bot;
          passes if win >= live, fitness >= live and head-to-head >= H2H_MIN

Results go to league/experiments.json (git, shared), so the registry of what was tried, what
passed and what was rejected is the same on every PC and in the PDF (`tex`).

    python harness/experiments.py list
    python harness/experiments.py add --title "Focus fire" --area teamwork --knobs FIGHT_V6=1
    python harness/experiments.py add --title "Spacing v1" --area combat --ref feat/spacing
    python harness/experiments.py run [--forever] [--no-push]
    python harness/experiments.py tex > docs/experiments.tex
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import league as L
from bctools import ROOT, runner

REGISTRY = L.LEAGUE / "experiments.json"
OUT = L.SCRATCH / "experiments"
SCREEN_MAPS = ("contest", "lanes", "islands", "hooks", "big-world", "ambush", "expanse")
SCREEN_SLACK = 0.01
H2H_MIN = 0.45
STATUSES = ("queued", "screening", "full", "passed", "rejected", "merged")


def load() -> dict:
    return {"experiments": [], **L.load_json(REGISTRY, {})}


def save(reg: dict) -> None:
    L.save_json(REGISTRY, reg)


def add(
    reg: dict,
    title: str,
    area: str,
    knobs: dict | None = None,
    ref: str | None = None,
    notes: str = "",
    status: str = "queued",
    results: dict | None = None,
) -> dict:
    assert knobs is not None or ref is not None, "an experiment is a knob set, a git ref, or both"
    n = 1 + max((e["id"] for e in reg["experiments"]), default=0)
    e = {
        "id": n,
        "title": title,
        "area": area,
        "spec": {k: v for k, v in (("ref", ref), ("knobs", knobs)) if v is not None},
        "status": status,
        "results": results or {},
        "added": L.now_iso(),
        "notes": notes,
    }
    reg["experiments"].append(e)
    return e


# ----------------------------------------------------------------- variants


def knob_variant(name: str, knobs: dict[str, int], copy: bool = True) -> str:
    """bots/<name> = bots/main with these knob defaults (knob and knob2 names, in the bot's
    vendored memory.rs). copy=False only patches an existing bots/<name> (a ref variant)."""
    src, dst = ROOT / "bots" / "main", ROOT / "bots" / name
    if not copy:
        pass
    elif dst.exists():
        shutil.rmtree(dst / "src", ignore_errors=True)
        shutil.copytree(src / "src", dst / "src")
    else:
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns(".cambc", "target"))
    mem = dst / "src" / "bc_core" / "memory.rs"
    t = mem.read_text(encoding="utf-8")
    for knob, value in knobs.items():
        t, n = re.subn(rf"\({knob}, \d+\),", f"({knob}, {int(value)}),", t)
        if n != 1:
            raise ValueError(f"unknown knob {knob}")
    mem.write_text(t, encoding="utf-8", newline="\n")
    return name


def ref_variant(name: str, ref: str) -> str:
    """bots/<name> = bots/main as it is at git `ref` (a branch or commit with code changes)."""
    dst = ROOT / "bots" / name
    tmp = OUT / f"src-{name}"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    arch = subprocess.run(
        ["git", "archive", ref, "bots/main"], cwd=ROOT, capture_output=True, check=True
    ).stdout
    subprocess.run(["tar", "xf", "-", "-C", str(tmp)], input=arch, check=True)
    if dst.exists():
        shutil.rmtree(dst / "src", ignore_errors=True)
    shutil.copytree(tmp / "bots" / "main", dst, dirs_exist_ok=True)
    shutil.copy2(ROOT / "bots" / "main" / "src" / "pack" / "pack.bin", dst / "src" / "pack" / "pack.bin")
    return name


def build_variant(e: dict) -> str:
    name = f"_x{e['id']}"
    spec = e["spec"]
    if "ref" in spec:
        ref_variant(name, spec["ref"])
        return knob_variant(name, spec["knobs"], copy=False) if spec.get("knobs") else name
    return knob_variant(name, spec["knobs"])


# ----------------------------------------------------------------- measurement


def stats(results: list[runner.MatchResult], bot: str) -> dict:
    rs = [r for r in results if bot in (r.bot_a, r.bot_b)]
    n = max(len(rs), 1)
    margin = sum(
        ((r.gems_a - r.gems_b) if r.bot_a == bot else (r.gems_b - r.gems_a)) / max(r.gems_a + r.gems_b, 1)
        for r in rs
    )
    return {
        "games": len(rs),
        "win": round(sum(r.score_for(bot) for r in rs) / n, 3),
        "deaths": round(sum(r.deaths_a if r.bot_a == bot else r.deaths_b for r in rs) / n, 2),
        "gem_margin": round(margin / n, 3),
        "fitness": round(L.fitness(rs, bot)[0], 3),
    }


def screen_passes(live: dict, var: dict) -> bool:
    return var["fitness"] >= live["fitness"] and var["win"] >= live["win"] - SCREEN_SLACK


def full_passes(live: dict, var: dict, h2h: float) -> bool:
    return var["win"] >= live["win"] and var["fitness"] >= live["fitness"] and h2h >= H2H_MIN


def evaluate(e: dict, reg: dict, push: bool) -> dict:
    """Screen, then (if it passes) the full check. Updates `e` in place and saves as it goes."""
    led = L.load_ledger()
    faithful, _ = L.ensure_mimics(led)
    maps = L.trainable_maps(led)
    var = build_variant(e)
    out = OUT / f"x{e['id']}-{time.strftime('%Y%m%d-%H%M%S')}"

    e["status"] = "screening"
    save(reg)
    screen = [p for n, p in maps.items() if n in SCREEN_MAPS]
    res = L.play(["main", var], faithful, screen, out / "screen")
    live, mine = stats(res, "main"), stats(res, var)
    e["results"]["screen"] = {"live": live, "variant": mine}
    if not screen_passes(live, mine):
        e["status"] = "rejected"
        e["notes"] = (
            e.get("notes", "") + f" Screen: win {mine['win']} vs {live['win']}, "
            f"fitness {mine['fitness']} vs {live['fitness']}."
        ).strip()
        return finish(e, reg, push)

    e["status"] = "full"
    save(reg)
    res = L.play(["main", var], faithful, list(maps.values()), out / "full")
    live, mine = stats(res, "main"), stats(res, var)
    h2h_res = runner.run_matches(var, "main", maps=list(maps.values()), out_dir=out / "h2h", release=True)
    h2h = round(sum(r.score_for(var) for r in h2h_res) / max(len(h2h_res), 1), 3)
    e["results"]["full"] = {"live": live, "variant": mine, "h2h": h2h, "h2h_games": len(h2h_res)}
    e["status"] = "passed" if full_passes(live, mine, h2h) else "rejected"
    return finish(e, reg, push)


def finish(e: dict, reg: dict, push: bool) -> dict:
    e["finished"] = L.now_iso()
    save(reg)
    L.log(f"experiment {e['id']} {e['title']}: {e['status']}")
    if push and L.git_ready():
        L.publish(
            [REGISTRY],
            f"league: experiment {e['id']} {e['title']} -> {e['status']}",
            L.h2h_decider(OUT, L.load_champion()["maps"]),
        )
    return e


def next_queued(reg: dict) -> dict | None:
    return next((e for e in reg["experiments"] if e["status"] == "queued"), None)


# ----------------------------------------------------------------- reporting


def tex(reg: dict) -> str:
    """A LaTeX table of every experiment (\\input from docs/league.tex)."""

    def esc(s: str) -> str:
        return re.sub(r"([_&%#$])", r"\\\1", s)

    def nums(e: dict) -> str:
        r = e["results"].get("full") or e["results"].get("screen")
        if not r:
            return "--"
        v, live = r["variant"], r["live"]
        s = f"win {live['win']:.2f}$\\to${v['win']:.2f}, fit {live['fitness']:.2f}$\\to${v['fitness']:.2f}"
        return s + (f", h2h {r['h2h']:.2f}" if "h2h" in r else "")

    rows = [
        f"{e['id']} & {esc(e['title'])} & {esc(e['area'])} & {e['status']} & {nums(e)} & {esc(e.get('notes', ''))}\\\\"
        for e in reg["experiments"]
    ]
    return "\n".join(
        [
            "\\begin{tabularx}{\\linewidth}{@{}rp{3.2cm}lp{1.3cm}p{4.3cm}X@{}}",
            "\\toprule",
            "\\# & Experiment & Area & Status & Result (live$\\to$variant) & Notes\\\\",
            "\\midrule",
            *rows,
            "\\bottomrule",
            "\\end{tabularx}",
        ]
    )


def cmd_list(_: argparse.Namespace) -> int:
    for e in load()["experiments"]:
        r = e["results"].get("full") or e["results"].get("screen") or {}
        w = f"{r['live']['win']:.3f} -> {r['variant']['win']:.3f}" if r else ""
        print(f"{e['id']:3d} {e['status']:9s} {e['area']:10s} {e['title'][:44]:44s} {w}")
    return 0


def cmd_add(a: argparse.Namespace) -> int:
    reg = load()
    knobs = dict(kv.split("=") for kv in a.knobs) if a.knobs else None
    if knobs:
        knobs = {k: int(v) for k, v in knobs.items()}
    e = add(reg, a.title, a.area, knobs=knobs, ref=a.ref, notes=a.notes or "")
    save(reg)
    print(f"queued experiment {e['id']}: {e['title']}")
    return 0


def cmd_run(a: argparse.Namespace) -> int:
    while True:
        if a.push and L.git_ready():
            L.integrate(L.h2h_decider(OUT, L.load_champion()["maps"]))  # test against the latest live bot
            L.apply_champion(L.load_champion())
        reg = load()
        e = next_queued(reg)
        if not e:
            L.log("experiments: queue empty")
            if not a.forever:
                return 0
            time.sleep(600)
            continue
        try:
            evaluate(e, reg, a.push)
        except Exception as ex:  # one broken experiment must not stop the queue
            e["status"] = "rejected"
            e["notes"] = f"error: {ex!r}"[:300]
            finish(e, reg, a.push)
        if not a.forever:
            return 0


def cmd_tex(_: argparse.Namespace) -> int:
    print(tex(load()))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    ad = sub.add_parser("add")
    ad.add_argument("--title", required=True)
    ad.add_argument("--area", required=True)
    ad.add_argument("--knobs", nargs="*", help="KNOB=value ...")
    ad.add_argument("--ref", help="git ref whose bots/main is the variant")
    ad.add_argument("--notes")
    rn = sub.add_parser("run")
    rn.add_argument("--forever", action="store_true")
    rn.add_argument("--no-push", dest="push", action="store_false")
    sub.add_parser("tex")
    a = p.parse_args(argv)
    return {"list": cmd_list, "add": cmd_add, "run": cmd_run, "tex": cmd_tex}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
