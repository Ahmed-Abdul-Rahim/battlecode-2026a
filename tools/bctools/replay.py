"""Read cambc replay bundles (.repl26a zip files).

A bundle contains `replay.repl26a` (engine replay), `team-{a,b}.stdout`,
`team-{a,b}.debug` and `team-{a,b}.fuel`. Our bot's telemetry is in stdout as
lines prefixed `T ` (turn), `E ` (event) and `X ` (error), each followed by a
JSON object. Lines are padded with NUL bytes by the runner, which we strip.

Fuel log format (reverse-engineered, see docs/fuel-budget.md): 8-byte magic
`a62leuf\\0`, 5 header bytes, 8 u64 budget values, then one u64 of fuel used per
turn in turn order (draft picks first, then every champion turn).
"""

from __future__ import annotations

import json
import struct
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

FUEL_MAGIC = b"a62leuf\x00"
FUEL_HEADER = 8 + 5 + 8 * 8


@dataclass
class TeamLog:
    turns: list[dict] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    errors: list[dict] = field(default_factory=list)
    other: list[str] = field(default_factory=list)
    fuel: list[int] = field(default_factory=list)
    budgets: list[int] = field(default_factory=list)
    #: Terrain/group dump (`M ` line) written once by our bot in debug runs.
    map: dict | None = None


@dataclass
class Bundle:
    path: Path
    teams: dict[str, TeamLog]
    replay_bytes: int

    def team(self, side: str) -> TeamLog:
        return self.teams[side]

    def ours(self) -> tuple[str, TeamLog]:
        """The side running our bot: the one with telemetry turn lines."""
        side = max(self.teams, key=lambda s: len(self.teams[s].turns))
        return side, self.teams[side]


def parse_stdout(text: str) -> TeamLog:
    log = TeamLog()
    for raw in text.splitlines():
        line = raw.replace("\x00", "").strip()
        if len(line) < 2:
            continue
        tag, rest = line[:2], line[2:]
        if tag == "M " and log.map is None:
            try:
                log.map = json.loads(rest)
            except json.JSONDecodeError:
                log.other.append(line)
            continue
        target = {"T ": log.turns, "E ": log.events, "X ": log.errors}.get(tag)
        if target is None:
            log.other.append(line)
            continue
        try:
            target.append(json.loads(rest))
        except json.JSONDecodeError:
            log.other.append(line)
    return log


def parse_fuel(data: bytes) -> tuple[list[int], list[int]]:
    if not data.startswith(FUEL_MAGIC) or len(data) < FUEL_HEADER:
        return [], []
    budgets = list(struct.unpack_from("<8Q", data, 13))
    body = data[FUEL_HEADER:]
    n = len(body) // 8
    return budgets, list(struct.unpack_from(f"<{n}Q", body))


def load(path: str | Path) -> Bundle:
    path = Path(path)
    teams: dict[str, TeamLog] = {}
    with zipfile.ZipFile(path) as z:
        names = set(z.namelist())
        for side in ("a", "b"):
            stdout = f"team-{side}.stdout"
            log = parse_stdout(z.read(stdout).decode("utf-8", "replace")) if stdout in names else TeamLog()
            fuel = f"team-{side}.fuel"
            if fuel in names:
                log.budgets, log.fuel = parse_fuel(z.read(fuel))
            teams[side] = log
        replay_bytes = z.getinfo("replay.repl26a").file_size if "replay.repl26a" in names else 0
    return Bundle(path, teams, replay_bytes)


#: A suspended turn is recorded as having used exactly its whole budget.
EXHAUSTED = {100_000, 200_000, 1_000_000}


def exhausted_turns(log: TeamLog) -> int:
    """Turns that ran out of fuel (bot suspended mid-turn, losing the turn)."""
    return sum(1 for f in log.fuel if f in EXHAUSTED)


def summarize(log: TeamLog) -> dict:
    """Compact per-team summary used by the CLI and MCP tools."""
    used = [t["f0"] - t["f1"] for t in log.turns if "f0" in t and "f1" in t]
    return {
        "turn_lines": len(log.turns),
        "experts": dict(Counter(t.get("e") for t in log.turns).most_common()),
        "modes": dict(Counter(t.get("m") for t in log.turns).most_common()),
        "tiers": dict(Counter(t.get("tier") for t in log.turns).most_common()),
        "events": dict(Counter(e.get("k") for e in log.events).most_common()),
        "errors": dict(Counter(e.get("ctx") for e in log.errors).most_common(10)),
        "error_samples": [e.get("err") for e in log.errors[:5]],
        "fuel_max": max(used, default=0),
        "fuel_mean": round(sum(used) / len(used)) if used else 0,
        "fuel_p95": sorted(used)[int(len(used) * 0.95)] if used else 0,
        "fuel_exhausted_turns": exhausted_turns(log),
    }


def load_stats(path: str | Path) -> dict:
    with open(path, encoding="utf-8-sig") as f:
        return json.load(f)
