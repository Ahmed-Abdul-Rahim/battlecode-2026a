"""Read / write BCN1 teacher blobs in numpy (no torch): crates/bc-brain/src/nn.rs format."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Layer:
    w: np.ndarray  # float32 [n_out, n_in]
    b: np.ndarray  # float32 [n_out]
    relu: bool


@dataclass
class Blob:
    spec: int
    layers: list[Layer]
    heads: list[int]

    def forward_hidden(self, x: np.ndarray) -> np.ndarray:
        """Activations entering the last layer, for inputs x [n, n_in]."""
        h = x
        for l in self.layers[:-1]:
            h = h @ l.w.T + l.b
            if l.relu:
                h = np.maximum(h, 0)
        return h


def read(path: Path | str) -> Blob:
    b = Path(path).read_bytes()
    if b[:4] != b"BCN1":
        raise ValueError("not a BCN1 blob")
    spec, n = struct.unpack_from("<II", b, 4)
    at = 12
    layers = []
    for _ in range(n):
        n_in, n_out, relu = struct.unpack_from("<III", b, at)
        at += 12
        w = np.frombuffer(b, "<f4", n_in * n_out, at).reshape(n_out, n_in).copy()
        at += 4 * n_in * n_out
        bias = np.frombuffer(b, "<f4", n_out, at).copy()
        at += 4 * n_out
        layers.append(Layer(w, bias, bool(relu)))
    (k,) = struct.unpack_from("<I", b, at)
    heads = list(struct.unpack_from(f"<{k}I", b, at + 4))
    return Blob(spec, layers, heads)


def write(blob: Blob, path: Path | str) -> None:
    out = bytearray(b"BCN1")
    out += struct.pack("<II", blob.spec, len(blob.layers))
    for l in blob.layers:
        out += struct.pack("<III", l.w.shape[1], l.w.shape[0], int(l.relu))
        out += l.w.astype("<f4").tobytes() + l.b.astype("<f4").tobytes()
    out += struct.pack("<I", len(blob.heads)) + struct.pack(f"<{len(blob.heads)}I", *blob.heads)
    Path(path).write_bytes(bytes(out))


def with_goal_delta(base: Blob, dw: np.ndarray, db: np.ndarray) -> Blob:
    """A copy of `base` with dw / db added to the goal head's rows of the last layer."""
    g = base.heads[0]
    layers = [Layer(l.w, l.b, l.relu) for l in base.layers]
    last = base.layers[-1]
    w, b = last.w.copy(), last.b.copy()
    w[:g] += dw
    b[:g] += db
    layers[-1] = Layer(w, b, last.relu)
    return Blob(base.spec, layers, base.heads)
