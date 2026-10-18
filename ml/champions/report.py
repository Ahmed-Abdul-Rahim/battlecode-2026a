"""Champion-vector report: cosine-similarity heatmap (and synergy matrix if trained).

    python -m ml.champions.report [vectors.json] --out renders/champions.png

Without a vectors file it shows the initial kit projection. The MCP
`champion_report` tool returns this PNG so a multimodal model can review how
fine-tuning moved the vectors.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ml.champions.descriptor import KITS, initial_vector


def load_vectors(path: Path | None) -> dict[str, list[int]]:
    if path and path.exists():
        return json.loads(path.read_text(encoding="utf-8"))["vectors"]
    return {name: initial_vector(name) for name in KITS}


def cosine_matrix(vectors: dict[str, list[int]]) -> tuple[list[str], np.ndarray]:
    names = list(vectors)
    v = np.array([vectors[n] for n in names], dtype=np.float64)
    norm = np.linalg.norm(v, axis=1, keepdims=True)
    v = v / np.where(norm == 0, 1, norm)
    return names, v @ v.T


def heatmap(names: list[str], m: np.ndarray, title: str, cell: int = 34) -> Image.Image:
    pad = 110
    n = len(names)
    img = Image.new("RGB", (pad + n * cell + 10, pad + n * cell + 30), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.text((10, 8), title, fill=(0, 0, 0))
    for i, name in enumerate(names):
        d.text((5, pad + i * cell + cell // 3), name[:12], fill=(0, 0, 0))
        d.text((pad + i * cell + 2, pad - 18 - (i % 2) * 12), name[:5], fill=(0, 0, 0))
        for j in range(n):
            v = float(np.clip(m[i, j], -1, 1))
            # Blue (−1) → white (0) → red (+1).
            color = (
                (255, int(255 * (1 - v)), int(255 * (1 - v)))
                if v >= 0
                else (int(255 * (1 + v)), int(255 * (1 + v)), 255)
            )
            x0, y0 = pad + j * cell, pad + i * cell
            d.rectangle([x0, y0, x0 + cell - 1, y0 + cell - 1], fill=color)
            d.text((x0 + 4, y0 + cell // 3), f"{v:+.1f}", fill=(0, 0, 0))
    return img


def render(vectors_path: Path | None, out: Path) -> Path:
    vectors = load_vectors(vectors_path)
    names, sim = cosine_matrix(vectors)
    label = vectors_path.name if vectors_path and vectors_path.exists() else "initial kit projection"
    img = heatmap(names, sim, f"Champion vector cosine similarity ({label})")
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("vectors", nargs="?", type=Path)
    p.add_argument("--out", type=Path, default=Path("renders/champions.png"))
    a = p.parse_args(argv)
    print(render(a.vectors, a.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
