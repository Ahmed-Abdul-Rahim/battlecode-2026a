"""Layout of the teacher's observation and actions; mirrors bots/main/src/teacher.rs.

numpy only (the main venv and the GPU venv both import it).
"""

from __future__ import annotations

import numpy as np

SPEC = 0x4E4E_0001
S = 310  # scalars per turn
HIST = 3  # previous turns stacked
SIDE, CH = 15, 5  # local grid 15x15: blocked, bush, enemy, ally, camp
GRID_BITS = SIDE * SIDE * CH
GRID_BYTES = (GRID_BITS + 7) // 8
INPUT = S * (1 + HIST) + GRID_BITS
HEADS = (15, 10, 4, 2)  # goal, step, attack, cast
RULE_GOAL = 241  # scalar index of the rules' suggested goal (one-hot, HEADS[0] wide); teacher.rs
HEAD_NAMES = ("goal", "step", "attack", "cast")
GOALS = (
    "group0", "group1", "group2", "group3", "tribute", "shop", "retreat", "walk_home",
    "hold", "engage0", "engage1", "engage2", "kite0", "step_back", "sighting",
)
STEPS = ("auto", "stay", "N", "NE", "E", "SE", "S", "SW", "W", "NW")


def check_header(h: dict) -> None:
    got = (h.get("spec"), h.get("S"), h.get("hist"), h.get("side"), h.get("ch"), tuple(h.get("heads", ())))
    want = (SPEC, S, HIST, SIDE, CH, HEADS)
    if got != want:
        raise ValueError(f"teacher layout mismatch: bot {got} vs nn/spec.py {want}")


def build_inputs(s: np.ndarray, grid: np.ndarray, prev: np.ndarray) -> np.ndarray:
    """Network inputs [n, INPUT] float32 from scalars s [m, S] int8, grid [m, GRID_BYTES] uint8
    and prev [n, HIST] (row of each earlier turn in s, -1 = none). Rows of s/grid line up with
    the samples (n == m). Same order as teacher.rs `input`."""
    n = len(prev)
    x = np.zeros((n, INPUT), np.float32)
    x[:, :S] = s / 127.0
    for k in range(HIST):
        idx = prev[:, k]
        ok = idx >= 0
        x[ok, S * (k + 1) : S * (k + 2)] = s[idx[ok]] / 127.0
    bits = np.unpackbits(grid, axis=1, bitorder="little")[:, :GRID_BITS]
    x[:, S * (1 + HIST) :] = bits
    return x


def mask_bits(m: np.ndarray, head: int) -> np.ndarray:
    """Boolean [n, HEADS[head]] mask from the logged bit masks [n, 4]."""
    return ((m[:, head : head + 1] >> np.arange(HEADS[head])) & 1).astype(bool)
