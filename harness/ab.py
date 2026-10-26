"""Paired A/B of two builds of our bot: the same games (map, opponent, side) for both, so the
difference is the change alone.

    python -m harness.ab NAME --b-knob K2_JUMP_HOLES=1 [--maps a,b] [--no-table]
    python -m harness.ab NAME --a-dir bots/x --b-dir bots/y

A defaults to bots/main as it is. --b-knob builds B as a copy of A with one compiled default
changed (knob or K2_ knob name from memory.rs, e.g. FIGHT_V6=0); --no-table gives both the
reference pack without the per-map plan table (data/search/live-pack.bin), to test a rule
rather than the searched plans. Opponents: our rule bot + the ladder top mimics (league
ledger pool), both sides. Maps default to the pool's old (non-mimic-blind) maps: on maps the
mimics have never seen they all play the same base bot, so prefer maps with ladder data.

Writes runs/ab/NAME.json and prints per-map and per-opponent win rates and the paired
difference (dwin = A - B, with its standard error).
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harness.search import common as C  # noqa: E402

OUT = ROOT / "runs" / "ab"


def copy_bot(src: Path, dst: Path) -> Path:
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    for name in ("Cargo.toml", "Cargo.lock", "rust-toolchain.toml"):
        if (src / name).exists():
            shutil.copy2(src / name, dst / name)
    for name in ("src", ".cargo"):
        shutil.copytree(src / name, dst / name)
    return dst


def set_default(bot: Path, knob: str, value: int) -> None:
    """Change one entry of knob::DEFAULTS / knob2::DEFAULTS in the bot's vendored memory.rs."""
    mem = bot / "src" / "bc_core" / "memory.rs"
    s = mem.read_text(encoding="utf-8")
    mod, name = ("knob2", knob[3:]) if knob.startswith("K2_") else ("knob", knob)
    start = s.index(f"pub mod {mod} {{")
    end = s.index("\n}", start)
    body, n = re.subn(rf"\(({name}), \d+\),", rf"(\1, {value}),", s[start:end])
    if n != 1:
        raise SystemExit(f"{knob}: not found in {mod}::DEFAULTS")
    mem.write_text(s[:start] + body + s[end:], encoding="utf-8")


def run(name: str, a: Path, b: Path, maps: list[str], opps: dict[str, str]) -> dict:
    mp = C.maps()
    games, meta = [], []
    for m in maps:
        for o, od in opps.items():
            for s in "ab":
                for tag, bot in (("A", str(a)), ("B", str(b))):
                    games.append((bot, od, mp[m]) if s == "a" else (od, bot, mp[m]))
                    meta.append((tag, m, o, s, bot))
    tmp = C.RUNS / "tmp" / f"ab-{name}"
    res = C.runner.run_games(games, tmp, release=True, log_limit=0)
    rows = {}
    for r, (tag, m, o, s, bot) in zip(res, meta):
        rows[(tag, m, o, s)] = C._row(r, C.Game(tag, m, o, s), "ab", bot)
    shutil.rmtree(tmp, ignore_errors=True)
    rep: dict = {"a": str(a), "b": str(b), "maps": maps, "opps": list(opps), "per_map": {}, "per_opp": {}}
    dw_all, df_all = [], []
    keys = [(o, s) for o in opps for s in "ab"]
    for m in maps:
        dw = [rows[("A", m, o, s)]["res"] - rows[("B", m, o, s)]["res"] for o, s in keys]
        df = [C.fitness(rows[("A", m, o, s)]) - C.fitness(rows[("B", m, o, s)]) for o, s in keys]
        st = sum(C.stalls(rows[("A", m, o, s)]) - C.stalls(rows[("B", m, o, s)]) for o, s in keys)
        rep["per_map"][m] = {
            "winA": round(sum(rows[("A", m, o, s)]["res"] for o, s in keys) / len(keys), 3),
            "winB": round(sum(rows[("B", m, o, s)]["res"] for o, s in keys) / len(keys), 3),
            **C._mean_se(dw, "dwin"), **C._mean_se(df, "dfit"), "dstalls": st,
        }
        dw_all += dw
        df_all += df
    for o in opps:
        rep["per_opp"][o] = {t: round(sum(rows[(t, m, o, s)]["res"] for m in maps for s in "ab") / (2 * len(maps)), 3)
                             for t in "AB"}
    rep["all"] = {"n": len(dw_all), **C._mean_se(dw_all, "dwin"), **C._mean_se(df_all, "dfit")}
    OUT.mkdir(parents=True, exist_ok=True)
    C.save_json(OUT / f"{name}.json", rep)
    return rep


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("name")
    p.add_argument("--a-dir", type=Path, default=ROOT / "bots" / "main")
    p.add_argument("--b-dir", type=Path)
    p.add_argument("--b-knob", help="KNOB=VALUE: B = A with this compiled default")
    p.add_argument("--maps", help="comma-separated (default: every map in splits.json's old pool)")
    p.add_argument("--no-table", action="store_true", help="both without the per-map plan table")
    a = p.parse_args(argv)
    if not a.b_dir and not a.b_knob:
        p.error("give --b-dir or --b-knob")
    work = ROOT / "bots"
    A = copy_bot(a.a_dir, work / f"_ab-{a.name}-A")
    B = copy_bot(a.b_dir or a.a_dir, work / f"_ab-{a.name}-B")
    if a.b_knob:
        k, v = a.b_knob.split("=")
        set_default(B, k.strip(), int(v))
    if a.no_table:
        ref = ROOT / "data" / "search" / "live-pack.bin"
        for d in (A, B):
            shutil.copy2(ref, d / "src" / "pack" / "pack.bin")
    led = C.load_json(ROOT / "league" / "ledger.json", {})
    C.ensure_opponents(["rules"] + [f"top-{x['slug']}" for x in led.get("pool", [])])
    opps = {"rules": str(ROOT / "bots" / "_rules"), **C.fresh_top()}
    if a.maps:
        maps = a.maps.split(",")
    else:
        splits = C.load_json(C.RUNS / "splits.json", {})
        old = json.loads((ROOT / "data" / "fingerprints.json").read_text(encoding="utf-8"))
        maps = [m for m in C.maps() if m in old and m in set(splits.get("maps_train", [])) | set(splits.get("maps_heldout", []))]
    rep = run(a.name, A, B, maps, opps)
    al = rep["all"]
    print(f"{a.name}: A - B over {al['n']} paired games: dwin {al['dwin']:+.3f} +- {al['dwin_se']:.3f}, "
          f"dfit {al['dfit']:+.3f} +- {al['dfit_se']:.3f}")
    for m, v in rep["per_map"].items():
        print(f"  {m:24s} A {v['winA']:.3f} B {v['winB']:.3f} dwin {v['dwin']:+.3f} +- {v['dwin_se']:.3f} dstalls {v['dstalls']:+d}")
    for o, v in rep["per_opp"].items():
        print(f"  vs {o:38s} A {v['A']:.3f} B {v['B']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
