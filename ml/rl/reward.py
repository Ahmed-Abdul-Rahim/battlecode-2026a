"""Reward for the learned policies (ml/rl/ppo.py), from one logged game.

One sample per learned decision (an `R` line of the chosen arm). A decision's reward covers what
happened to that champion until its next decision; the game result is added to its last one:

    + WIN on a win, - WIN on a loss (0 for a tie / unfinished)
    + MARGIN * gem margin (ours - theirs) / (ours + theirs)           gems decide ties
    + GOLD * gold earned          + DELIVER * team gems delivered / champions
    + KILL * kills + ASSIST * assists
    - DEATH * deaths - DEAD * rounds spent dead
    - FAULT * (bot faults + fuel-exhausted turns)                     crashes / CPU overruns
    - WASTE * turns that should act but did nothing                   (farm/fight intent, no move,
                                                                       attack or cast)
    - FUEL * share of the turn budget used

At most FAULT_CAP fault events count per decision, and every stored reward is clipped to ±CLIP:
a handful of fault-heavy decisions used to carry half the reward of an iteration (down to -129
for one decision), which drowned every other signal after advantage normalisation.

Change the weights in WEIGHTS; everything else follows.
"""

from __future__ import annotations

import json
import re
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

WEIGHTS = {
    "win": 2.0,
    "margin": 1.0,
    "gold": 0.002,
    "deliver": 0.02,
    "kill": 0.3,
    "assist": 0.1,
    "death": 0.5,
    "dead": 0.005,
    "fault": 1.0,
    "waste": 0.01,
    "fuel": 0.002,
}
FAULT_CAP = 3  # fault events counted per decision
CLIP = 5.0  # |reward| of a stored decision (a win is 2 + margin)
#: Intents that should do something every turn (no move, attack or cast = a wasted turn).
ACTIVE_INTENTS = ("ClearCamp", "Engage", "Kite")
TURN_FUEL = 100_000


def norm(name: str) -> str:
    return re.sub(r"[^a-z]", "", name.lower())


def _lines(bundle: Path, side: str) -> list[str]:
    raw = zipfile.ZipFile(bundle).read(f"team-{side}.stdout")
    return raw.replace(b"\x00", b"").decode("utf-8", "replace").splitlines()


def _stats_events(stats: Path, side: str) -> dict[tuple[str, int], dict[str, int]]:
    """(champion, round) -> counts of kill / assist / death / fault from the match stats."""
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
            out[(norm(owner.get("name", "")), int(x["timestamp"]["round"]))][kind] += 1
    return out


def game_samples(
    bundle: Path,
    stats: Path,
    side: str,
    arm: int,
    won: float,
    margin: float,
    every: int = 1,
    terms: dict[str, float] | None = None,
) -> dict[str, np.ndarray]:
    """Samples of one arm for our side of one game: X int8 [n, d], A [n, heads], LP [n],
    M (masks, bits) [n, heads], R (reward) [n], DONE [n] (last decision of a champion), TRAJ [n].
    `every` > 1 keeps every n-th decision (and the last); the dropped ones' rewards are added to
    the decision kept before them, so nothing is lost from the returns (before the clip).
    `terms`, if given, gets each WEIGHTS term's |reward| added up before the cap and the clip
    (plus "faults_capped" and "clipped" counts), to see what drives the reward."""
    T: dict[tuple[str, int], dict] = {}
    dec: dict[str, list[dict]] = defaultdict(list)
    dead: set[tuple[str, int]] = set()
    did: dict[tuple[str, int], int] = {}
    for ln in _lines(bundle, side):
        if ln.startswith(("T ", "R ", "E ")):
            try:
                o = json.loads(ln[2:])
            except json.JSONDecodeError:
                continue
            c = norm(o.get("c", ""))
            if ln[0] == "T":
                T[(c, o["r"])] = o
            elif ln[0] == "R" and o.get("arm") == arm:
                dec[c].append(o)
            elif o.get("k") == "dead":
                dead.add((c, o["r"]))
            elif o.get("k") == "o":
                did[(c, o["r"])] = int(o["v"])
    if not dec:
        return {}
    ev = _stats_events(stats, side)
    champs = max(len({c for c, _ in T}), 1)
    last_round = max((r for _, r in T), default=0)
    w = WEIGHTS
    xs, acts, lps, masks, rews, done, traj = [], [], [], [], [], [], []

    def at(c: str, r: int, key: str) -> float:
        """A T-line value of champion c at round r, or the latest before it."""
        for rr in range(r, max(r - 40, 0), -1):
            if (c, rr) in T:
                return float(T[(c, rr)].get(key, 0))
        return 0.0

    for ti, (c, ds) in enumerate(sorted(dec.items())):
        ds.sort(key=lambda o: o["r"])
        for k, o in enumerate(ds):
            r0 = o["r"]
            r1 = ds[k + 1]["r"] if k + 1 < len(ds) else last_round
            part = defaultdict(float)  # this decision's reward by term
            part["gold"] = w["gold"] * (at(c, r1, "lg") - at(c, r0, "lg"))
            part["deliver"] = w["deliver"] * (at(c, r1, "at") - at(c, r0, "at")) / champs
            faults = 0
            for r in range(r0 + 1, r1 + 1):
                e = ev.get((c, r), {})
                part["kill"] += w["kill"] * e.get("kill", 0)
                part["assist"] += w["assist"] * e.get("assist", 0)
                part["death"] -= w["death"] * e.get("death", 0)
                faults += e.get("fault", 0)
                if (c, r) in dead:
                    part["dead"] -= w["dead"]
                t = T.get((c, r))
                if t is not None:
                    if did.get((c, r), 1) == 0 and str(t.get("i", "")).startswith(ACTIVE_INTENTS):
                        part["waste"] -= w["waste"]
                    part["fuel"] -= (
                        w["fuel"] * max(float(t.get("f0", 0)) - float(t.get("f1", 0)), 0.0) / TURN_FUEL
                    )
            part["fault"] = -w["fault"] * faults
            last = k + 1 == len(ds)
            if last:
                part["win"] = w["win"] * (2.0 * won - 1.0)
                part["margin"] = w["margin"] * margin
            if terms is not None:
                for name, v in part.items():
                    terms[name] = terms.get(name, 0.0) + abs(v)
                terms["faults_capped"] = terms.get("faults_capped", 0) + (faults > FAULT_CAP)
            part["fault"] = -w["fault"] * min(faults, FAULT_CAP)
            rew = sum(part.values())
            if k % every and not last and rews and traj[-1] == ti:
                rews[-1] += rew  # thinned out: its reward goes to the decision kept before it
                continue
            xs.append(np.frombuffer(bytes.fromhex(o["x"]), dtype=np.int8))
            acts.append(o["a"])
            lps.append(o["lp"] / 1000.0)
            masks.append(o["m"])
            rews.append(rew)
            done.append(last)
            traj.append(ti)
    if terms is not None:
        terms["clipped"] = terms.get("clipped", 0) + sum(abs(r) > CLIP for r in rews)
    rews = [min(max(r, -CLIP), CLIP) for r in rews]
    return {
        "X": np.stack(xs),
        "A": np.array(acts, dtype=np.int64),
        "LP": np.array(lps, dtype=np.float32),
        "M": np.array(masks, dtype=np.int64),
        "R": np.array(rews, dtype=np.float32),
        "DONE": np.array(done, dtype=bool),
        "TRAJ": np.array(traj, dtype=np.int64),
    }


def read_shard(p: Path, keys: tuple[str, ...]) -> dict[str, np.ndarray] | None:
    """A shard's arrays, or None if it is unreadable or incomplete (e.g. written on a full disk)."""
    try:
        d = np.load(p)
        return {k: d[k] for k in keys}
    except (OSError, KeyError, ValueError, zipfile.BadZipFile, EOFError):
        return None


def concat(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Join samples of several games; trajectory ids stay unique."""
    parts = [p for p in parts if p]
    if not parts:
        return {}
    out, offset = defaultdict(list), 0
    for p in parts:
        for k, v in p.items():
            out[k].append(v + offset if k == "TRAJ" else v)
        offset += int(p["TRAJ"].max()) + 1
    return {k: np.concatenate(v) for k, v in out.items()}
