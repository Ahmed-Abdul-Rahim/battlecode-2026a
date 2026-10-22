"""Target decisions from logged games -> a training set (numpy only).

Each `S` line (knob2 TARGET_EXPLORE builds, debug runs) is one decision: every scored group's
inputs, the global inputs and the group picked. Its outcome is what the champion earned over
the next HORIZON rounds, per round: lifetime gold / 3 + gems picked up - DEATH per death.

    python -m ml.targeting.data <replay dir>... --out data.npz
"""

from __future__ import annotations

import argparse
import json
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

from ml.targeting.features import N_GLOBAL, N_GROUP

HORIZON = 60
MIN_WINDOW = 20
DEATH = 50.0
G_MAX = 64


def _hex(s: str) -> np.ndarray:
    return np.frombuffer(bytes.fromhex(s), dtype=np.int8)


def side_of(name: str, bot: str) -> list[str]:
    a, _, b = name.split("__", 1)[1].rpartition(".")[0].partition("-vs-")
    return [s for s, who in (("a", a), ("b", b)) if who == bot]


def game_decisions(bundle: Path, side: str) -> dict[str, list]:
    raw = zipfile.ZipFile(bundle).read(f"team-{side}.stdout").replace(b"\x00", b"").decode("utf-8", "replace")
    T: dict[str, dict[int, tuple[float, float]]] = defaultdict(dict)
    dead: dict[str, set[int]] = defaultdict(set)
    S = []
    last = 0
    for ln in raw.splitlines():
        if ln.startswith(("T ", "S ", "E ")):
            try:
                o = json.loads(ln[2:])
            except json.JSONDecodeError:
                continue
            c = o.get("c", "")
            if ln[0] == "T":
                T[c][o["r"]] = (float(o.get("lg", 0)), float(o.get("gm", 0)))
                last = max(last, o["r"])
            elif ln[0] == "S":
                S.append(o)
            elif o.get("k") == "dead":
                dead[c].add(o["r"])
    out: dict[str, list] = defaultdict(list)
    for s in S:
        c, r0 = s["c"], s["r"]
        r1 = min(r0 + HORIZON, last)
        if r1 - r0 < MIN_WINDOW or s["pick"] < 0:
            continue
        ids = [g[0] for g in s["g"]]
        if s["pick"] not in ids or len(ids) > G_MAX:
            continue
        rows = T[c]
        prev_lg, prev_gm = rows.get(r0, (None, None))
        if prev_lg is None:
            continue
        gold = gems = 0.0
        deaths = 0
        lg0 = prev_lg
        for r in range(r0 + 1, r1 + 1):
            if r in dead[c] and (r - 1) not in dead[c]:
                deaths += 1
            if r in rows:
                lg, gm = rows[r]
                gems += max(gm - prev_gm, 0.0)
                prev_lg, prev_gm = lg, gm
        gold = prev_lg - lg0
        g = np.zeros((G_MAX, N_GROUP), np.int8)
        for k, (_, h) in enumerate(s["g"]):
            v = _hex(h)
            g[k, : min(len(v), N_GROUP)] = v[:N_GROUP]
        x = np.zeros(N_GLOBAL, np.int8)
        xv = _hex(s["x"])
        x[: min(len(xv), N_GLOBAL)] = xv[:N_GLOBAL]
        out["G"].append(g)
        out["M"].append(np.arange(G_MAX) < len(ids))
        out["X"].append(x)
        out["A"].append(ids.index(s["pick"]))
        out["Y"].append((gold / 3.0 + gems - DEATH * deaths) / (r1 - r0))
        out["E"].append(bool(s.get("exp")))
        out["R"].append(r0)
    return out


def build(dirs: list[Path], bot: str) -> dict[str, np.ndarray]:
    parts: dict[str, list] = defaultdict(list)
    games = 0
    for d in dirs:
        for f in sorted(d.rglob("*.repl26a")):
            for side in side_of(f.name, bot):
                try:
                    dec = game_decisions(f, side)
                except (OSError, KeyError, zipfile.BadZipFile, ValueError):
                    continue
                if not dec:
                    continue
                n = len(dec["Y"])
                for k, v in dec.items():
                    parts[k] += v
                parts["GAME"] += [games] * n
                parts["MAP"] += [f.name.split("__")[0]] * n
                games += 1
    if not parts:
        return {}
    return {
        "G": np.stack(parts["G"]),
        "M": np.stack(parts["M"]),
        "X": np.stack(parts["X"]),
        "A": np.array(parts["A"], np.int16),
        "Y": np.array(parts["Y"], np.float32),
        "E": np.array(parts["E"], bool),
        "R": np.array(parts["R"], np.int32),
        "GAME": np.array(parts["GAME"], np.int32),
        "MAP": np.array(parts["MAP"]),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+", type=Path)
    ap.add_argument("--bot", default="_sw_data")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    d = build(a.dirs, a.bot)
    if not d:
        print("no decisions found")
        return 1
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, **d)
    print(
        f"{len(d['Y'])} decisions from {d['GAME'].max() + 1} games; explored {d['E'].mean():.0%}; "
        f"outcome mean {d['Y'].mean():.3f} sd {d['Y'].std():.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
