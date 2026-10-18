"""Champion kit descriptors and initial champion vectors (Python mirror).

Mirrors ``bc_core::champion::builtin_kit`` / ``KitInfo::descriptor`` and
``bc_brain::champvec::from_descriptor`` bit-for-bit (float32 arithmetic, Rust
rounding). ``python -m ml.champions.descriptor`` rewrites
tests/fixtures/champion_vectors.json, which both test suites check.

The vector layout and the fine-tuning plan are in docs/champions.md.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

F = np.float32
DESCRIPTOR_LEN = 24
VEC_LEN = 16

# Ability kinds in descriptor order (dims 8..14).
KINDS = ["AttackReset", "Damage", "AoeDamage", "Cc", "Heal", "Dash", "Shield"]
PASSIVE_CLASS = {
    "BonusGold": 18,
    "RemoteShop": 18,
    "GemFocus": 19,
    "HealthStrength": 19,
    "SplashAttack": 20,
    "ResetCooldownsOnKill": 20,
    "Retaliation": 20,
    "StunImmunity": 21,
}

RESET = ("AttackReset", 0.0, 0.0, False)
DASH = ("Dash", 0.0, 2.0, False)

# name: (ranged, attack_range, hp, hp_growth, str, str_growth, focus, focus_growth,
#        abilities [(kind, modifier, range, excludes_walking)], passive)
KITS: dict[str, tuple] = {
    "Ryth": (False, 1, 1200, 100, 50, 5, 100, 10, [("AoeDamage", 1.0, 4.0, True), RESET], "HealthStrength"),
    "VesperVale": (True, 3, 900, 100, 45, 5, 100, 10, [RESET, DASH], "BonusGold"),
    "Vendetta": (False, 1, 800, 100, 70, 5, 100, 10, [RESET, DASH], "ResetCooldownsOnKill"),
    "Cassi": (
        True,
        3,
        700,
        71.25,
        30,
        3.5,
        70,
        15,
        [("Damage", 2.0, 4.0, True), ("Cc", 0.0, 4.0, False)],
        "None",
    ),
    "Issac": (False, 1, 900, 95, 40, 4.75, 100, 10, [RESET, DASH], "None"),
    "Stellar": (
        True,
        3,
        750,
        75,
        35,
        3.5,
        100,
        10,
        [("Damage", 2.0, 4.0, True), ("AoeDamage", 1.4, 4.0, True)],
        "None",
    ),
    "Prism": (
        True,
        3,
        700,
        75,
        35,
        3.5,
        80,
        10,
        [("Cc", 1.5, 4.0, True), ("AoeDamage", 1.0, 4.0, True)],
        "GemFocus",
    ),
    "Marcella": (False, 1, 1200, 100, 45, 5, 50, 10, [RESET, ("Damage", 2.0, 4.0, True)], "StunImmunity"),
    "Willow": (True, 3, 700, 50, 45, 3, 105, 5, [("Heal", 3.0, 4.0, False), RESET], "None"),
    "Ren": (False, 1, 900, 100, 40, 5, 100, 10, [("Cc", 1.25, 4.0, True), DASH], "Retaliation"),
    "Varg": (False, 1, 1000, 100, 50, 5, 100, 10, [RESET, ("Shield", 2.5, 0.0, False)], "SplashAttack"),
    "Tavi": (True, 3, 750, 75, 35, 5, 100, 10, [RESET, ("Cc", 2.2, 4.0, True)], "RemoteShop"),
}


def descriptor(name: str) -> np.ndarray:
    ranged, rng, h, hg, s, sg, f, fg, abilities, passive = KITS[name]
    d = np.zeros(DESCRIPTOR_LEN, dtype=np.float32)
    d[0] = F(h) / F(1200.0)
    d[1] = F(hg) / F(100.0)
    d[2] = F(s) / F(70.0)
    d[3] = F(sg) / F(5.0)
    d[4] = F(f) / F(105.0)
    d[5] = F(fg) / F(15.0)
    d[6] = F(1.0) if ranged else F(0.0)
    d[7] = F(rng) / F(3.0)
    burst, max_range, exclusive, count = F(0), F(0), F(0), F(0)
    for kind, modifier, a_range, excl in abilities:
        count += F(1)
        d[8 + KINDS.index(kind)] += F(1)
        if kind in ("Damage", "AoeDamage", "Cc"):
            burst += F(modifier) * F(f)
        max_range = max(max_range, F(a_range))
        if excl:
            exclusive += F(1)
    d[15] = burst / F(300.0)
    d[16] = max_range / F(4.0)
    d[17] = exclusive / count if count > 0 else F(0)
    if passive in PASSIVE_CLASS:
        d[PASSIVE_CLASS[passive]] = 1.0
    elif name in ("Cassi", "Issac"):
        d[22] = 1.0
    else:
        d[23] = 1.0
    return d


def stats_block(d: np.ndarray) -> list[np.float32]:
    return [
        F(0.6) * d[0] + F(0.4) * d[1],
        F(0.5) * d[2] + F(0.3) * d[3] + F(0.2) * d[8],
        F(0.4) * d[4] + F(0.3) * d[5] + F(0.3) * d[15],
        F(0.5) * d[6] + F(0.25) * d[7] + F(0.25) * d[16],
        min(d[13], F(1.0)),
        min(F(0.5) * d[11] + F(0.5) * d[12] + F(0.5) * d[14], F(1.0)),
    ]


def q32(v: np.float32) -> int:
    """Rust ``(v.clamp(-1, 1) * 127.0).round() as i8`` in float32."""
    x = F(min(max(F(v), F(-1.0)), F(1.0))) * F(127.0)
    return int(np.floor(x + F(0.5))) if x >= 0 else -int(np.floor(-x + F(0.5)))


def initial_vector(name: str) -> list[int]:
    v = [0] * VEC_LEN
    for i, x in enumerate(stats_block(descriptor(name))):
        v[i] = q32(F(2.0) * x - F(1.0))
    return v


def write_fixture(path: Path) -> dict[str, list[int]]:
    data = {name: initial_vector(name) for name in KITS}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    return data


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    for name, vec in write_fixture(root / "tests" / "fixtures" / "champion_vectors.json").items():
        print(f"{name:>11}: {vec[:6]}")
