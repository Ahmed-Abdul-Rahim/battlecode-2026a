"""Helpers shared by the ML tooling. Each mirrors a Rust function exactly."""

from __future__ import annotations


def crc16(data: bytes) -> int:
    """CRC-16/CCITT-FALSE, identical to ``bc_core::crc::crc16``."""
    crc = 0xFFFF
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) if crc & 0x8000 else (crc << 1)
            crc &= 0xFFFF
    return crc


def q(v: float) -> int:
    """int8 quantiser, identical to ``bc_brain::features::q``: round(clamp(v, -1, 1) * 127).

    Rust's ``f32::round`` rounds half away from zero, unlike Python's banker's rounding.
    """
    v = max(-1.0, min(1.0, float(v))) * 127.0
    return int(v + 0.5) if v >= 0 else -int(-v + 0.5)
