"""Run batches of local matches through `cambc run-batch`.

A full 2000-round match takes about a second, so a sweep of every map on both
sides finishes in well under a minute with parallel jobs. Paths in the manifest
must be absolute (cambc resolves them relative to the manifest otherwise).

Never calls `cambc submit` or any platform command.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from bctools import CAMBC, ROOT

DEFAULT_LOG_LIMIT = 20_000_000


@dataclass
class MatchResult:
    map: str
    bot_a: str
    bot_b: str
    winner: str | None  # bot name, or None if the match did not finish
    gems_a: int
    gems_b: int
    rounds: int
    deaths_a: int
    deaths_b: int
    replay: str
    stats: str

    def score_for(self, bot: str) -> float:
        """1 win, 0 loss, 0.5 if unfinished."""
        if self.winner is None:
            return 0.5
        return 1.0 if self.winner == bot else 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def list_maps(directory: Path | None = None) -> list[Path]:
    return sorted((directory or ROOT / "maps").glob("*.map26a"))


def bot_dir(bot: str) -> Path:
    p = Path(bot)
    return p if p.is_absolute() else ROOT / "bots" / bot


def _env() -> dict:
    env = dict(os.environ)
    cargo = Path.home() / ".cargo" / "bin"
    env["PATH"] = f"{cargo}{os.pathsep}{env.get('PATH', '')}"
    return env


def needs_build(bot: str, release: bool = False) -> bool:
    mode = "release" if release else "debug"
    wasm = bot_dir(bot) / ".cambc" / f"build-{mode}" / "wasm32-wasip1" / mode / "bot-injected.wasm"
    if not wasm.exists():
        return True
    src = bot_dir(bot) / "src"
    if not src.exists():
        return False
    w_mtime = wasm.stat().st_mtime
    return any(p.is_file() and p.stat().st_mtime > w_mtime for p in src.rglob("*"))


def build(bot: str, release: bool = False, force: bool = False) -> None:
    if not force and not needs_build(bot, release):
        return
    args = [str(CAMBC), "build", str(bot_dir(bot))]
    if release:
        args.append("--release")
    proc = subprocess.run(args, cwd=ROOT, env=_env(), capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"build of {bot} failed:\n{proc.stdout}\n{proc.stderr}")


def _parse(stats_path: Path, map_name: str, a: str, b: str, replay: Path) -> MatchResult:
    try:
        s = json.loads(stats_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return MatchResult(map_name, a, b, None, 0, 0, 0, 0, 0, str(replay), str(stats_path))
    teams = {t["team"]: int(t["submittedGems"]) for t in s.get("teams", [])}
    deaths = {"teamA": 0, "teamB": 0}
    for c in s.get("champions", []):
        deaths[c["team"]] = deaths.get(c["team"], 0) + int(c["metrics"].get("deaths", 0))
    outcome = s.get("match", {}).get("outcome")
    winner = {"teamA": a, "teamB": b}.get(outcome)
    return MatchResult(
        map_name,
        a,
        b,
        winner,
        teams.get("teamA", 0),
        teams.get("teamB", 0),
        int(s.get("match", {}).get("roundsCompleted", 0)),
        deaths["teamA"],
        deaths["teamB"],
        str(replay),
        str(stats_path),
    )


def run_matches(
    candidate: str,
    opponent: str,
    maps: list[Path] | None = None,
    out_dir: Path | None = None,
    swap: bool = True,
    jobs: int | None = None,
    rounds: int | None = None,
    release: bool = False,
    log_limit: int = DEFAULT_LOG_LIMIT,
) -> list[MatchResult]:
    """Play `candidate` vs `opponent` on every map (both sides if `swap`)."""
    pairs = [(candidate, opponent)] + ([(opponent, candidate)] if swap and candidate != opponent else [])
    return run_pairs(pairs, maps, out_dir, jobs, rounds, release, log_limit)


def run_pairs(
    pairs: list[tuple[str, str]],
    maps: list[Path] | None = None,
    out_dir: Path | None = None,
    jobs: int | None = None,
    rounds: int | None = None,
    release: bool = False,
    log_limit: int = DEFAULT_LOG_LIMIT,
) -> list[MatchResult]:
    """Play every (team A, team B) pair on every map, all in one parallel batch."""
    maps = maps or list_maps()
    out_dir = (out_dir or ROOT / "runs" / time.strftime("%Y%m%d-%H%M%S")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    for bot in dict.fromkeys(b for pair in pairs for b in pair):
        build(bot, release)
    rows, expected = [], []
    for m in maps:
        for a, b in dict.fromkeys(pairs):
            tag = f"{m.stem}__{Path(a).name}-vs-{Path(b).name}"
            replay, stats = out_dir / f"{tag}.repl26a", out_dir / f"{tag}.json"
            rows.append(f"{bot_dir(a)} {bot_dir(b)} {m.resolve()} {replay} --statistics {stats}")
            expected.append((stats, m.stem, a, b, replay))
    # Stale files from an earlier batch in the same directory would be read as results.
    for stats, _, _, _, replay in expected:
        Path(stats).unlink(missing_ok=True)
        Path(replay).unlink(missing_ok=True)
    manifest = out_dir / "manifest.txt"
    manifest.write_text("\n".join(rows) + "\n", encoding="utf-8")
    args = [str(CAMBC), "run-batch", str(manifest), "--jobs", str(jobs or max(1, (os.cpu_count() or 2) - 2))]
    args += ["--json", "--log-limit", str(log_limit)]
    if rounds:
        args += ["--rounds", str(rounds)]
    if release:
        args.append("--release")
    proc = subprocess.run(args, cwd=ROOT, env=_env(), capture_output=True, text=True)
    (out_dir / "run-batch.log").write_text(proc.stdout + proc.stderr, encoding="utf-8")
    results = [_parse(*e) for e in expected]
    (out_dir / "results.json").write_text(
        json.dumps([r.to_dict() for r in results], indent=1), encoding="utf-8"
    )
    return results


def summarize(results: list[MatchResult], bot: str) -> dict:
    from bctools.stats import wilson

    n = len(results)
    score = sum(r.score_for(bot) for r in results)
    lo, hi = wilson(score, n)
    per_map: dict[str, list[float]] = {}
    for r in results:
        per_map.setdefault(r.map, []).append(r.score_for(bot))
    return {
        "bot": bot,
        "matches": n,
        "score": score,
        "win_rate": round(score / n, 3) if n else 0.0,
        "wilson_95": [round(lo, 3), round(hi, 3)],
        "unfinished": sum(1 for r in results if r.winner is None),
        "per_map": {k: sum(v) / len(v) for k, v in sorted(per_map.items())},
        "losses": [
            {"map": r.map, "a": r.bot_a, "b": r.bot_b, "gems": [r.gems_a, r.gems_b], "rounds": r.rounds}
            for r in results
            if r.score_for(bot) < 1.0
        ],
    }
