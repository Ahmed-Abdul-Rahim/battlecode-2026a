"""Games → training samples for the teacher (numpy only).

One sample per `N` line (one decision of one champion). Reward per decision covers what happened
to that champion until its next decision; the game result goes on its last one. Weighted towards
winning and gem margin; the shaped terms are small and only speed up early learning.
"""

from __future__ import annotations

import json
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

from nn import spec

WEIGHTS = {
    "win": 3.0,
    "margin": 2.0,
    "gold": 0.001,
    "deliver": 0.01,
    "kill": 0.2,
    "assist": 0.05,
    "death": 0.3,
    "fault": 1.0,
}
CLIP = 6.0
KEYS = ("S", "G", "PREV", "A", "M", "LP", "R", "DONE", "TRAJ", "T")


def _norm(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalpha())


def _lines(bundle: Path, side: str) -> list[str]:
    raw = zipfile.ZipFile(bundle).read(f"team-{side}.stdout")
    return raw.replace(b"\x00", b"").decode("utf-8", "replace").splitlines()


def _events(stats: Path, side: str) -> dict[tuple[str, int], dict[str, int]]:
    d = json.loads(stats.read_text(encoding="utf-8-sig"))
    team = f"team{side.upper()}"
    out: dict[tuple[str, int], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for e in d.get("events", []):
        owner = e.get("owner") or {}
        if owner.get("team") != team or owner.get("kind") != "champion":
            continue
        kind = {"botFault": "fault", "fuelExhausted": "fault"}.get(e["kind"], e["kind"])
        if kind not in ("kill", "assist", "death", "fault"):
            continue
        for x in e.get("entries", []):
            out[(_norm(owner.get("name", "")), int(x["timestamp"]["round"]))][kind] += 1
    return out


def game_samples(
    bundle: Path, stats: Path, side: str, won: float, margin: float, weights: dict | None = None
) -> dict[str, np.ndarray]:
    """Samples of our side of one game (empty dict if it logged no decisions)."""
    T: dict[tuple[str, int], dict] = {}
    dec: dict[str, list[dict]] = defaultdict(list)
    for ln in _lines(bundle, side):
        if ln.startswith("NH "):
            spec.check_header(json.loads(ln[3:]))
        elif ln.startswith(("N ", "T ")):
            try:
                o = json.loads(ln[2:])
            except json.JSONDecodeError:
                continue
            c = _norm(o.get("c", ""))
            if ln[0] == "N":
                dec[c].append(o)
            else:
                T[(c, o["r"])] = o
    if not dec:
        return {}
    ev = _events(stats, side)
    champs = max(len(dec), 1)
    last_round = max((o["r"] for ds in dec.values() for o in ds), default=0)
    w = weights or WEIGHTS

    def at(c: str, r: int, key: str) -> float:
        for rr in range(r, max(r - 40, 0), -1):
            if (c, rr) in T:
                return float(T[(c, rr)].get(key, 0))
        return 0.0

    S, G, PREV, A, M, LP, R, DONE, TRAJ, TM = ([] for _ in range(10))
    for ti, (c, ds) in enumerate(sorted(dec.items())):
        ds.sort(key=lambda o: o["r"])
        base = len(S)
        for k, o in enumerate(ds):
            r0 = o["r"]
            r1 = ds[k + 1]["r"] if k + 1 < len(ds) else last_round
            rew = w["gold"] * (at(c, r1, "lg") - at(c, r0, "lg"))
            rew += w["deliver"] * (at(c, r1, "at") - at(c, r0, "at")) / champs
            for r in range(r0 + 1, r1 + 1):
                e = ev.get((c, r))
                if e:
                    rew += w["kill"] * e.get("kill", 0) + w["assist"] * e.get("assist", 0)
                    rew -= w["death"] * e.get("death", 0) + w["fault"] * min(e.get("fault", 0), 3)
            last = k + 1 == len(ds)
            if last:
                rew += w["win"] * (2.0 * won - 1.0) + w["margin"] * margin
            S.append(np.frombuffer(bytes.fromhex(o["s"]), np.int8))
            G.append(np.frombuffer(bytes.fromhex(o["g"]), np.uint8))
            PREV.append([base + k - j - 1 if k - j - 1 >= 0 else -1 for j in range(spec.HIST)])
            A.append(o["a"])
            M.append(o["m"])
            LP.append(o["lp"] / 1000.0)
            R.append(min(max(rew, -CLIP), CLIP))
            DONE.append(last)
            TRAJ.append(ti)
            TM.append(o.get("t", 0))
    return {
        "S": np.stack(S),
        "G": np.stack(G),
        "PREV": np.array(PREV, np.int64),
        "A": np.array(A, np.int64),
        "M": np.array(M, np.int64),
        "LP": np.array(LP, np.float32),
        "R": np.array(R, np.float32),
        "DONE": np.array(DONE, bool),
        "TRAJ": np.array(TRAJ, np.int64),
        "T": np.array(TM, np.int64),
    }


def concat(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Join samples of several games; row indices (PREV) and trajectory ids stay valid."""
    parts = [p for p in parts if p]
    if not parts:
        return {}
    out: dict[str, list] = defaultdict(list)
    rows = traj = 0
    for p in parts:
        for k, v in p.items():
            if k == "PREV":
                v = np.where(v >= 0, v + rows, -1)
            elif k == "TRAJ":
                v = v + traj
            out[k].append(v)
        rows += len(p["S"])
        traj += int(p["TRAJ"].max()) + 1
    return {k: np.concatenate(v) for k, v in out.items()}


def save(path: Path, d: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    # Rollouts are read once and replaced next iteration: uncompressed is much faster.
    (np.savez if path.name == "rollout.npz" else np.savez_compressed)(tmp, **d)
    tmp.replace(path)


def load(paths: list[Path]) -> dict[str, np.ndarray]:
    parts = []
    for p in paths:
        try:
            with np.load(p) as z:
                parts.append({k: z[k] for k in KEYS})
        except (OSError, KeyError, ValueError, zipfile.BadZipFile, EOFError):
            continue
    return concat(parts)
