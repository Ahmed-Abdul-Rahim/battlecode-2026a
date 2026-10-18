"""Per-map strategy tuning: pick the best knob set for every map and write map_plans.rs.

The ladder plays exactly our 17 maps (every ladder mapHash seen matches a file in
maps/ byte for byte), and the draft identifies the map exactly (world::fingerprint
-> map_plans::lookup -> knob overrides in θ_match for every champion). So the best
strategy can be chosen per map instead of one universal rule.

For each candidate (knob overrides on top of bots/main's defaults) and each map:
  speed  harness/speedrun.py score vs bots/idle (rounds to 1000; both sides)
  h2h    win rate vs the submitted bots (v6_trials SUBMISSION; both sides)
A map takes a candidate over `base` only if
  h2h_gain + SPEED_WEIGHT * speed_gain >= MIN_GAIN   and   h2h >= base h2h - H2H_SLACK,
with speed_gain = (base_speed - speed) / base_speed.

    python harness/map_tune.py                     # run every candidate, then select + write
    python harness/map_tune.py --select runs/map_tune/<ts>   # re-select from saved rows
Local only: nothing here touches the Battlecode platform.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import speedrun
import v6_trials

from bctools import ROOT

SPEED_WEIGHT = 0.5
MIN_GAIN = 0.1
H2H_SLACK = 0.1
FINGERPRINTS = ROOT / "runs" / "variants" / "fingerprints.json"
PLANS_RS = ROOT / "bots" / "main" / "src" / "map_plans.rs"
MEMORY_RS = ROOT / "crates" / "bc-core" / "src" / "memory.rs"

# name -> (knob overrides for the test bot, draft profile for the plan).
# DRAFT_BEST is read from the defaults at draft time, so per map it becomes draft 8.
CANDIDATES: dict[str, tuple[dict[str, int], int]] = {
    "base": ({}, 0),
    "shop-first": ({"SHOP_FIRST": 1}, 0),
    "feasible": ({"CAMP_FEASIBLE": 1}, 0),
    "escape": ({"CAMP_SAFE": 1}, 0),
    "gather": ({"CAMP_SAFE": 2}, 0),
    "build-jungle": ({"BUILD_PROFILE": 0}, 0),
    "build-power": ({"BUILD_PROFILE": 13}, 0),
    "value-gems": ({"VALUE_PROFILE": 0}, 0),
    "fight": ({"FIGHT_V6": 7}, 0),
    "passive": ({"PASSIVE_ROLES": 1}, 0),
    "draft-best": ({"DRAFT_BEST": 1}, 8),
    "no-enroute": ({"EN_ROUTE": 0}, 0),
    "no-respawn": ({"TEAM_RESPAWN": 0}, 0),
}


def run_candidate(name: str, knobs: dict[str, int], out: Path) -> dict:
    bot = v6_trials.make("main", f"_mt-{name}", knobs)
    speed = speedrun.evaluate(bot, out / "speed", release=True)
    v6_trials.SANITY = True
    v6_trials.FAMILY[:] = v6_trials.SUBMISSION
    h2h = v6_trials.evaluate(bot, out / "h2h" / name, ["family"])
    return {"speed": speed["per_map"], "h2h": h2h["per_map_family"], "speed_mean": speed["mean"],
            "h2h_win": h2h["family"]["win"]}


def select(rows: dict[str, dict]) -> dict[str, tuple[str, float]]:
    base = rows["base"]
    choice = {}
    for m in base["speed"]:
        best, best_gain = "base", 0.0
        for name, r in rows.items():
            if name == "base" or m not in r["speed"] or m not in r["h2h"]:
                continue
            speed_gain = (base["speed"][m] - r["speed"][m]) / max(base["speed"][m], 1)
            h2h_gain = r["h2h"][m] - base["h2h"][m]
            gain = h2h_gain + SPEED_WEIGHT * speed_gain
            if h2h_gain >= -H2H_SLACK and gain >= MIN_GAIN and gain > best_gain:
                best, best_gain = name, gain
        choice[m] = (best, round(best_gain, 3))
    return choice


def knob_index() -> dict[str, int]:
    text = MEMORY_RS.read_text(encoding="utf-8")
    return {k: int(v) for k, v in re.findall(r"pub const ([A-Z_0-9]+): usize = (\d+);", text)}


def write_plans(choice: dict[str, tuple[str, float]]) -> None:
    fps = json.loads(FINGERPRINTS.read_text(encoding="utf-8"))
    idx = knob_index()
    entries = []
    for m, (name, gain) in sorted(choice.items()):
        if name == "base":
            continue
        knobs, draft = CANDIDATES[name]
        knobs = {k: v for k, v in knobs.items() if k != "DRAFT_BEST"}
        pairs = ", ".join(f"(knob::{k}, {v})" for k, v in knobs.items())
        entries.append(
            f"    // {m}: {name} (gain {gain})\n"
            f"    MapPlan {{ fingerprint: {fps[m]}, name: \"{m}\", knobs: &[{pairs}], draft: {draft} }},"
        )
        assert all(k in idx for k in knobs), knobs
    text = PLANS_RS.read_text(encoding="utf-8")
    body = "pub const PLANS: &[MapPlan] = &[\n" + "\n".join(entries) + "\n];" if entries else "pub const PLANS: &[MapPlan] = &[];"
    text = re.sub(r"pub const PLANS: &\[MapPlan\] = &\[.*?\];", lambda _: body, text, flags=re.S)
    if "use crate::bc_core::memory::knob;" not in text:
        text = text.replace("/// Draft preference", "use crate::bc_core::memory::knob;\n\n/// Draft preference", 1)
    PLANS_RS.write_text(text, encoding="utf-8", newline="\n")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--select", type=Path, help="re-select from a saved run directory")
    p.add_argument("--only", nargs="*", help="candidates to run (base is always run)")
    p.add_argument("--no-write", action="store_true")
    a = p.parse_args(argv)
    if a.select:
        out = a.select
        rows = json.loads((out / "rows.json").read_text(encoding="utf-8"))
    else:
        out = ROOT / "runs" / "map_tune" / time.strftime("%m%d-%H%M%S")
        out.mkdir(parents=True, exist_ok=True)
        rows = {}
        for name, (knobs, _) in CANDIDATES.items():
            if a.only and name != "base" and name not in a.only:
                continue
            t0 = time.time()
            rows[name] = run_candidate(name, knobs, out)
            print(f"{name:14s} speed {rows[name]['speed_mean']:7.1f}  h2h {rows[name]['h2h_win']:.3f}  "
                  f"({round(time.time() - t0)}s)", flush=True)
            (out / "rows.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    choice = select(rows)
    for m, (name, gain) in sorted(choice.items()):
        b = rows["base"]
        r = rows[name]
        print(f"  {m:12s} -> {name:13s} gain {gain:5.3f}  speed {b['speed'][m]}->{r['speed'][m]}  "
              f"h2h {b['h2h'][m]}->{r['h2h'][m]}")
    (out / "choice.json").write_text(json.dumps(choice, indent=1), encoding="utf-8")
    if not a.no_write:
        write_plans(choice)
        print(f"wrote {PLANS_RS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
