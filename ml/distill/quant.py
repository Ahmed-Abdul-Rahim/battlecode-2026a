"""int8 MLP blobs (BCW1) and a bit-exact reference of the Rust forward pass.

The Rust kernel lives in crates/bc-brain/src/int8.rs; keep the two in lockstep.
`write_reference_fixture` produces tests/fixtures/int8_reference.json, which
both the Rust and Python test suites check against.

Distillation (P-051) and quantisation-aware training (P-052) produce float
weights; `quantize_layer` turns them into int8 weights, int32 biases and a
power-of-two shift.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ml.util import crc16

MAGIC = b"BCW1"
NO_FILM = 0xFFFF


@dataclass
class Layer:
    weights: np.ndarray  # int8 [n_out, n_in]
    bias: np.ndarray  # int32 [n_out]
    shift: int = 0
    relu: bool = False
    film_offset: int | None = None

    @property
    def n_in(self) -> int:
        return int(self.weights.shape[1])

    @property
    def n_out(self) -> int:
        return int(self.weights.shape[0])


def encode(schema_hash: int, layers: list[Layer]) -> bytes:
    out = bytearray(MAGIC)
    out += struct.pack("<HBB", schema_hash, len(layers), 0)
    for layer in layers:
        film = NO_FILM if layer.film_offset is None else layer.film_offset
        out += struct.pack("<HHBBH", layer.n_in, layer.n_out, layer.shift, int(layer.relu), film)
        out += layer.weights.astype(np.int8).tobytes(order="C")
        out += layer.bias.astype("<i4").tobytes()
    out += struct.pack("<H", crc16(bytes(out)))
    return bytes(out)


def decode(blob: bytes) -> tuple[int, list[Layer]]:
    body, (crc,) = blob[:-2], struct.unpack("<H", blob[-2:])
    if body[:4] != MAGIC:
        raise ValueError("bad magic")
    if crc16(body) != crc:
        raise ValueError("bad checksum")
    schema_hash, n_layers, _ = struct.unpack_from("<HBB", body, 4)
    at = 8
    layers = []
    for _ in range(n_layers):
        n_in, n_out, shift, flags, film = struct.unpack_from("<HHBBH", body, at)
        at += 8
        w = np.frombuffer(body, dtype=np.int8, count=n_in * n_out, offset=at).reshape(n_out, n_in)
        at += n_in * n_out
        b = np.frombuffer(body, dtype="<i4", count=n_out, offset=at).astype(np.int32)
        at += 4 * n_out
        layers.append(Layer(w.copy(), b.copy(), shift, bool(flags & 1), None if film == NO_FILM else film))
    if at != len(body):
        raise ValueError("trailing bytes")
    return schema_hash, layers


def forward(layers: list[Layer], x, film: list[tuple[int, int]] | None = None) -> np.ndarray:
    """Bit-exact mirror of ``Mlp::forward``: returns int32 logits."""
    a = np.zeros(layers[0].n_in, dtype=np.int64)
    x = np.asarray(x, dtype=np.int64)[: layers[0].n_in]
    a[: len(x)] = x
    for li, layer in enumerate(layers):
        acc = layer.bias.astype(np.int64) + layer.weights.astype(np.int64) @ a
        if li == len(layers) - 1:
            return acc.astype(np.int32)
        v = acc >> layer.shift
        if film is not None and layer.film_offset is not None:
            for o in range(layer.n_out):
                k = layer.film_offset + o
                if k < len(film):
                    gamma, beta = film[k]
                    v[o] = ((v[o] * (64 + gamma)) >> 6) + beta
        if layer.relu:
            v = np.maximum(v, 0)
        a = np.clip(v, -127, 127)
    raise ValueError("empty network")


def quantize_layer(
    w: np.ndarray, b: np.ndarray, in_scale: float, out_scale: float
) -> tuple[np.ndarray, np.ndarray, int]:
    """Quantise a float layer y = W x + b.

    Inputs are int8 with real value x = x_q / in_scale; outputs should be int8
    with y = y_q / out_scale. Weights get a per-tensor scale; the requantisation
    factor is rounded to a power of two (the Rust kernel only shifts).
    """
    w_scale = 127.0 / max(float(np.abs(w).max()), 1e-8)
    w_q = np.clip(np.round(w * w_scale), -127, 127).astype(np.int8)
    acc_scale = w_scale * in_scale  # accumulator units per real unit
    shift = max(0, round(np.log2(acc_scale / out_scale)))
    b_q = np.round(b * acc_scale).astype(np.int32)
    return w_q, b_q, int(shift)


def write_reference_fixture(path: Path, schema_hash: int = 0xBDF9, seed: int = 7) -> None:
    rng = np.random.default_rng(seed)
    layers = [
        Layer(
            rng.integers(-60, 60, (12, 20), dtype=np.int8),
            rng.integers(-500, 500, 12).astype(np.int32),
            6,
            True,
            0,
        ),
        Layer(
            rng.integers(-60, 60, (8, 12), dtype=np.int8),
            rng.integers(-500, 500, 8).astype(np.int32),
            5,
            True,
            12,
        ),
        Layer(rng.integers(-60, 60, (9, 8), dtype=np.int8), rng.integers(-500, 500, 9).astype(np.int32)),
    ]
    blob = encode(schema_hash, layers)
    cases = []
    for _ in range(6):
        x = rng.integers(-127, 128, 20).tolist()
        film = [(int(g), int(b)) for g, b in rng.integers(-40, 40, (20, 2))]
        cases.append({"input": x, "film": film, "logits": forward(layers, x, film).tolist()})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"blob_hex": blob.hex(), "cases": cases}, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    write_reference_fixture(root / "tests" / "fixtures" / "int8_reference.json")
    print("wrote tests/fixtures/int8_reference.json")
