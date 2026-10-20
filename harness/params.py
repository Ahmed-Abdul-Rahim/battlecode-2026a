"""The registry of trainable parameters: one place that says what the trainers may change.

The bot's learned vector (bots/main/src/learned.rs, P_N floats) is, in order:

  W       18  multipliers on hand-set constants (camp scoring, fight margin, ...)  log scale
  TARGET  10  camp-target scorer weights (bot.rs target_bias)                       linear
  FIGHT    9  fight-advantage bias weights (bot.rs fight_bias)                      linear
  HOT      1  death-hotspot threat weight (off: tested harmful)                     linear
  U       60  utility arbiter, 6 options x 10 features (utility.rs)                 linear
  X        4  extras: Tavi re-spec flag, arbiter fuel floor (k fuel), 2 reserved    linear

Search space: W in log space (exp() on write), the rest linear, each with bounds and an
initial ES step. A policy-pack θ section holds the *bot-space* floats (after exp) and may
be shorter than P_N (older packs); the bot fills the rest with its compiled defaults.

Adding a parameter: append it to a block here and in learned.rs (bump that block's
length const), use it in Rust through learned::w / target / fight / utility / extra, then
run `python harness/params.py --check` (the trainers refuse a layout mismatch).

    python harness/params.py --check      # layout matches learned.rs
    python harness/params.py --list       # names, defaults, bounds
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEARNED = ROOT / "bots" / "main" / "src" / "learned.rs"

W_NAMES = [
    "gem",
    "gold",
    "dist",
    "clear",
    "strength_discount",
    "reward",
    "hysteresis",
    "shared_small",
    "opening_dist",
    "en_detour",
    "en_clear",
    "fight_margin",
    "retreat_hp",
    "aggression",
    "recall_gold",
    "tribute_min",
    "tribute_max",
    "assist_hp",
]
T_NAMES = ["gems", "gold", "dist", "clear", "eta", "claimants", "damage", "bag", "emblems", "round"]
F_NAMES = ["hp", "enemy_hp", "allies", "enemies", "ranged", "enemy_ranged", "bag", "spawn_dist", "bias"]
U_OPTIONS = ["keep", "step_back", "support", "fight", "walk_home", "recall"]
U_PHI = [
    "bias",
    "hp",
    "camp_dps",
    "enemy_dps",
    "enemies",
    "allies",
    "ally_trouble",
    "bag",
    "spawn_dist",
    "ranged",
]
X_NAMES = ["tavi_respec", "arbiter_fuel_floor_k", "reserved0", "reserved1"]

# (block, names, scale, lo, hi, sigma0) -- lo/hi in search space.
BLOCKS = [
    ("W", W_NAMES, "log", math.log(0.2), math.log(5.0), 0.25),
    ("TARGET", T_NAMES, "lin", -3.0, 3.0, 0.2),
    ("FIGHT", F_NAMES, "lin", -3.0, 3.0, 0.2),
    ("HOT", ["hotspot_w"], "lin", 0.0, 0.0, 0.0),
    ("U", [f"{o}.{p}" for o in U_OPTIONS for p in U_PHI], "lin", -4.0, 4.0, 0.3),
    # X (4 floats in Rust): split so each extra gets its own bounds and step.
    ("X.flag", X_NAMES[:1], "lin", 0.0, 1.0, 0.3),  # Tavi re-spec: > 0.5 = on
    ("X.fuel", X_NAMES[1:2], "lin", 25.0, 60.0, 3.0),  # arbiter fuel floor (k fuel)
    ("X.reserved", X_NAMES[2:], "lin", 0.0, 0.0, 0.0),
]
RUST_LEN = {"W": "N", "TARGET": "T_N", "FIGHT": "F_N", "U": "U_N"}
#: Blocks that together form one Rust array (name, length const).
RUST_SPLIT = {"X": (("X.flag", "X.fuel", "X.reserved"), "X_N")}


def names() -> list[str]:
    return [f"{b}.{n}" for b, ns, *_ in BLOCKS for n in ns]


def size() -> int:
    return sum(len(ns) for _, ns, *_ in BLOCKS)


def _arr(text: str, name: str) -> list[float] | None:
    m = re.search(rf"pub const {name}: \[f32; \w+\] = \[(.*?)\];", text, re.S)
    if not m:
        return None
    body = m.group(1)
    rep = re.fullmatch(r"\s*([-\d.]+)\s*;\s*\w+\s*", body)
    if rep:
        return None  # [x; N] form: caller fills the default
    return [float(x) for x in body.split(",") if x.strip()]


def defaults(learned: Path = LEARNED) -> list[float]:
    """Bot-space defaults from learned.rs (what a build without a pack θ plays)."""
    text = learned.read_text(encoding="utf-8")
    out = []
    x_vals = _arr(text, "X") or []
    x_at = 0
    for block, ns, scale, *_ in BLOCKS:
        if block.startswith("X."):
            vals = x_vals[x_at : x_at + len(ns)]
            out += vals + [0.0] * (len(ns) - len(vals))
            x_at += len(ns)
            continue
        name = {"HOT": "HOTSPOT_W"}.get(block, block)
        if block == "HOT":
            m = re.search(r"pub const HOTSPOT_W: f32 = ([-\d.]+);", text)
            out.append(float(m.group(1)) if m else 0.0)
            continue
        vals = _arr(text, name)
        if vals is None or len(vals) != len(ns):
            m = re.search(rf"pub const {name}: \[f32; \w+\] = \[\s*([-\d.]+)\s*;", text)
            fill = float(m.group(1)) if m else (1.0 if scale == "log" else 0.0)
            vals = [fill] * len(ns)
        out += vals
    return out


def to_search(bot: list[float]) -> list[float]:
    out, i = [], 0
    for _, ns, scale, *_ in BLOCKS:
        for _ in ns:
            out.append(math.log(max(bot[i], 1e-3)) if scale == "log" else bot[i])
            i += 1
    return out


def to_bot(search: list[float], base: list[float] | None = None) -> list[float]:
    """Search vector (possibly shorter: older states) -> clipped bot-space floats."""
    base_s = to_search(base or defaults())
    full = list(search) + base_s[len(search) :]
    out, i = [], 0
    for _, ns, scale, lo, hi, _ in BLOCKS:
        for _ in ns:
            x = full[i] if hi == lo else min(max(full[i], lo), hi)
            out.append(math.exp(x) if scale == "log" else x)
            i += 1
    return out


def sigma0() -> list[float]:
    return [s for _, ns, _, _, _, s in BLOCKS for _ in ns]


def check(learned: Path = LEARNED) -> list[str]:
    text = learned.read_text(encoding="utf-8")
    errs = []
    for arr, (blocks, const) in RUST_SPLIT.items():
        n = sum(len(ns) for b, ns, *_ in BLOCKS if b in blocks)
        m = re.search(rf"pub const {const}: usize = (\d+);", text)
        if not m or int(m.group(1)) != n:
            errs.append(f"{arr}: registry has {n}, learned.rs {const} = {m and m.group(1)}")
    for block, ns, *_ in BLOCKS:
        if block not in RUST_LEN:
            continue
        m = re.search(rf"pub const {RUST_LEN[block]}: usize = (\d+);", text)
        if not m or int(m.group(1)) != len(ns):
            errs.append(f"{block}: registry has {len(ns)}, learned.rs {RUST_LEN[block]} = {m and m.group(1)}")
    return errs


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--check", action="store_true")
    p.add_argument("--list", action="store_true")
    a = p.parse_args(argv)
    if a.check:
        errs = check()
        print("\n".join(errs) if errs else f"ok: {size()} parameters match learned.rs")
        return 1 if errs else 0
    if a.list:
        d = defaults()
        for n, v in zip(names(), d, strict=False):
            print(f"{n:28s} {v:8.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
