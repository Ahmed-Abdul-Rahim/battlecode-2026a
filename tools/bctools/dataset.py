"""Export learned-head training data from match telemetry.

Every 4th turn our bot logs its full int8 feature vector (`x`, hex, schema in
ml/features/schema.yaml) next to the gate mode and expert the heuristics chose.
This turns a directory of replay bundles into arrays for distillation
(ml/distill) and feature selection (ml/features/select.py):

    X        int8 [n, N_FEATURES]
    expert   int  [n]   index into EXPERTS (router imitation target)
    complex  int  [n]   1 if the gate chose the MoE path (gate target)
    won      int  [n]   1 if our side won the match (outcome weighting)
    round    int  [n]
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from bctools import replay

EXPERTS = ["farm", "capture", "tribute", "fight", "kite", "retreat", "recall_shop", "scout", "altar_lock"]


def _hex_to_int8(s: str) -> np.ndarray:
    return np.frombuffer(bytes.fromhex(s), dtype=np.int8)


def from_bundles(paths: list[Path]) -> dict[str, np.ndarray]:
    xs, experts, complexes, won, rounds = [], [], [], [], []
    for p in paths:
        b = replay.load(p)
        side, log = b.ours()
        stats_path = p.with_suffix(".json")
        outcome = None
        if stats_path.exists():
            outcome = json.loads(stats_path.read_text(encoding="utf-8-sig")).get("match", {}).get("outcome")
        win = int(outcome == f"team{side.upper()}")
        for t in log.turns:
            if "x" not in t or t.get("e") not in EXPERTS:
                continue
            xs.append(_hex_to_int8(t["x"]))
            experts.append(EXPERTS.index(t["e"]))
            complexes.append(int(t.get("m") == "complex"))
            won.append(win)
            rounds.append(t["r"])
    if not xs:
        return {"X": np.zeros((0, 0), np.int8)}
    return {
        "X": np.stack(xs),
        "expert": np.array(experts),
        "complex": np.array(complexes),
        "won": np.array(won),
        "round": np.array(rounds),
    }


def export(run_dir: Path, out: Path) -> dict:
    bundles = sorted(Path(run_dir).glob("*.repl26a"))
    data = from_bundles(bundles)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **data)
    return {"bundles": len(bundles), "rows": int(data["X"].shape[0]), "out": str(out)}
