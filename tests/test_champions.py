"""Champion descriptor parity (Python mirror vs Rust fixture)."""

import json
from pathlib import Path

from ml.champions.descriptor import KITS, initial_vector

ROOT = Path(__file__).resolve().parents[1]


def test_descriptor_matches_rust_fixture():
    fixture = json.loads((ROOT / "tests" / "fixtures" / "champion_vectors.json").read_text())
    for name in KITS:
        assert initial_vector(name) == fixture[name], name

