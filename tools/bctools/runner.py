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
#: Wall-clock seconds per match (cambc --timeout); games take seconds, so this only stops hangs.
MATCH_TIMEOUT = 900
#: Seconds without a new statistics file before the whole batch is killed (2026-10-24: a batch
#: stopped starting games at 00:14 and blocked the experiment queue until a restart at 09:20).
STALL_TIMEOUT = 1800


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


_SDK_CHECKED = False


def _fresh_sdk() -> None:
    """`cambc init` writes sdk/ files dated 1970/1980, so Cargo keeps a stale compiled SDK in
    every bot's cache (2026-10-21: bots built before engine 1.0.0 failed on every Golem map
    with "invalid CampType value: 5"). Re-date such files once per process."""
    global _SDK_CHECKED
    if _SDK_CHECKED:
        return
    _SDK_CHECKED = True
    cutoff = 978307200  # 2001-01-01
    for f in (ROOT / "sdk" / "rust").rglob("*"):
        if f.is_file() and f.stat().st_mtime < cutoff:
            os.utime(f)


def build(bot: str, release: bool = False, force: bool = False) -> None:
    _fresh_sdk()
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


def _run_watched(args: list[str], log: Path, stats: list[Path], stall: int) -> bool:
    """Run cambc with its output in `log`; kill the process tree when no statistics file has
    appeared for `stall` seconds (returns False then). Games that never ran parse as unfinished."""
    with open(log, "w", encoding="utf-8") as f:
        proc = subprocess.Popen(args, cwd=ROOT, env=_env(), stdout=f, stderr=subprocess.STDOUT, text=True)
        done, last = 0, time.time()
        while proc.poll() is None:
            time.sleep(10)
            n = sum(1 for p in stats if p.exists())
            if n > done:
                done, last = n, time.time()
            elif time.time() - last > stall:
                if os.name == "nt":
                    subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True)
                proc.kill()
                proc.wait()
                f.write(f"\nbctools: killed after {stall}s without a finished game ({done}/{len(stats)} done)\n")
                print(f"run-batch stalled: killed after {done}/{len(stats)} games ({log})", flush=True)
                return False
    return True


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
    fuel: int | None = None,
    timeout: int | None = None,
) -> list[MatchResult]:
    """Play every (team A, team B) pair on every map, all in one parallel batch. `fuel` raises
    the per-turn fuel (local experiments only: the ladder gives 100k)."""
    maps = maps or list_maps()
    games = [(a, b, m) for m in maps for a, b in dict.fromkeys(pairs)]
    return run_games(games, out_dir, jobs, rounds, release, log_limit, fuel, timeout)


def run_games(
    games: list[tuple[str, str, Path]],
    out_dir: Path | None = None,
    jobs: int | None = None,
    rounds: int | None = None,
    release: bool = False,
    log_limit: int = DEFAULT_LOG_LIMIT,
    fuel: int | None = None,
    timeout: int | None = None,
) -> list[MatchResult]:
    """Play (team A, team B, map) games, all in one parallel batch. A repeated game gets its own
    files (deterministic, so only useful when a bot changed in between)."""
    out_dir = (out_dir or ROOT / "runs" / time.strftime("%Y%m%d-%H%M%S")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    for bot in dict.fromkeys(b for g in games for b in g[:2]):
        build(bot, release)
    rows, expected, seen = [], [], {}
    for a, b, m in games:
        tag = f"{m.stem}__{Path(a).name}-vs-{Path(b).name}"
        n = seen.get(tag, 0)
        seen[tag] = n + 1
        if n:
            tag = f"{tag}__{n}"
        replay, stats = out_dir / f"{tag}.repl26a", out_dir / f"{tag}.json"
        rows.append(f"{bot_dir(a)} {bot_dir(b)} {m.resolve()} {replay} --statistics {stats}")
        expected.append((stats, m.stem, a, b, replay))
    # Stale files from an earlier batch in the same directory would be read as results.
    for stats, _, _, _, replay in expected:
        Path(stats).unlink(missing_ok=True)
        Path(replay).unlink(missing_ok=True)
    manifest = out_dir / "manifest.txt"
    manifest.write_text("\n".join(rows) + "\n", encoding="utf-8")
    # BCTOOLS_JOBS: games at once (a dedicated cloud box can use every core; a desktop keeps 2 free).
    jobs = jobs or int(os.environ.get("BCTOOLS_JOBS") or 0) or max(1, (os.cpu_count() or 2) - 2)
    args = [str(CAMBC), "run-batch", str(manifest), "--jobs", str(jobs)]
    args += ["--json", "--log-limit", str(log_limit)]
    if rounds:
        args += ["--rounds", str(rounds)]
    if fuel:
        args += ["--fuel-quantum", str(fuel)]
    # A match without a wall-clock limit can hang the batch (and the league daemon) for hours.
    timeout = timeout or int(os.environ.get("BCTOOLS_TIMEOUT") or 0) or MATCH_TIMEOUT
    args += ["--timeout", str(timeout)]
    if release:
        args.append("--release")
    stall = max(STALL_TIMEOUT, 2 * timeout)
    if not _run_watched(args, out_dir / "run-batch.log", [e[0] for e in expected], stall):
        # One retry of the games that never finished, so a stall doesn't skew the results.
        left = [(r, e) for r, e in zip(rows, expected) if not Path(e[0]).exists()]
        retry = out_dir / "manifest-retry.txt"
        retry.write_text("\n".join(r for r, _ in left) + "\n", encoding="utf-8")
        args[2] = str(retry)
        _run_watched(args, out_dir / "run-batch-retry.log", [e[0] for _, e in left], stall)
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
