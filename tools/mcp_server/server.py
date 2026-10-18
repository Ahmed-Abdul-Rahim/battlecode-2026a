"""bctools MCP server (stdio).

Gives an MCP client (Claude Code, or the Qwen harness in harness/) the
self-improvement loop's senses and hands: run local matches, gate a
candidate, read telemetry, and *see* matches as rendered PNG frames
(multimodal review, cf. MLES P-083). Registered for the team in .mcp.json.

Safety: everything is local. No tool submits to or plays on the Battlecode
platform. Filing a GitHub issue requires `confirm=True`.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from mcp.server.mcpserver import Image, MCPServer

from bctools import ROOT, render, replay, runner
from bctools import gate as gate_mod

server = MCPServer(
    name="bctools",
    instructions=(
        "Local Battlecode tooling. Typical loop: gate(candidate) -> inspect losses with match_summary, "
        "render_timeline and render_frame -> change code on an exp/* branch -> gate again. "
        "Never submits to the platform."
    ),
)


def _inside_repo(path: str) -> Path:
    p = (ROOT / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()
    if not p.is_relative_to(ROOT):
        raise ValueError(f"path must be inside the repository: {path}")
    return p


def _png(path: Path) -> Image:
    return Image(data=path.read_bytes(), format="png")


@server.tool()
def run_matches(
    candidate: str = "main", opponent: str = "baseline", maps: list[str] | None = None, swap: bool = True
) -> dict:
    """Play candidate vs opponent on maps (default: all of maps/), both sides unless swap=False.
    Returns win rate with a Wilson interval, per-map scores, losses, and the run directory."""
    map_paths = [ROOT / "maps" / f"{m}.map26a" for m in maps] if maps else None
    results = runner.run_matches(candidate, opponent, map_paths, swap=swap)
    summary = runner.summarize(results, candidate)
    summary["run_dir"] = str(Path(results[0].replay).parent.relative_to(ROOT)) if results else None
    return summary


@server.tool()
def gate(candidate: str = "main", best: str = "baseline") -> dict:
    """Evaluation gate: candidate vs best on held-out maps, both sides.
    passed = win_rate >= 0.55 and Wilson lower bound > 0.5."""
    return gate_mod.gate(candidate, best)


@server.tool()
def list_runs(limit: int = 8) -> list[dict]:
    """Recent run directories with their summary (newest first)."""
    runs = sorted((ROOT / "runs").glob("*/results.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    out = []
    for p in runs[:limit]:
        results = [runner.MatchResult(**r) for r in json.loads(p.read_text(encoding="utf-8"))]
        bot = results[0].bot_a if results else "main"
        s = runner.summarize(results, bot)
        out.append(
            {
                "run_dir": str(p.parent.relative_to(ROOT)),
                "bot": bot,
                "win_rate": s["win_rate"],
                "matches": s["matches"],
                "losses": s["losses"][:5],
            }
        )
    return out


@server.tool()
def match_summary(bundle: str) -> dict:
    """Telemetry summary of one replay bundle (.repl26a): experts, modes, pathing tiers,
    events, errors and fuel statistics for our side, plus the engine's match result."""
    path = _inside_repo(bundle)
    side, log = replay.load(path).ours()
    summary = {"our_side": side, **replay.summarize(log)}
    stats = path.with_suffix(".json")
    if stats.exists():
        s = replay.load_stats(stats)
        summary["outcome"] = s.get("match", {}).get("outcome")
        summary["teams"] = s.get("teams")
        summary["rounds"] = s.get("match", {}).get("roundsCompleted")
    return summary


@server.tool()
def get_logs(
    bundle: str, champion: str | None = None, kind: str = "T", rounds: str | None = None, limit: int = 60
) -> list[dict]:
    """Telemetry lines from a bundle. kind: T (turns), E (events), X (errors).
    rounds: "100-200" to filter. champion: e.g. "Varg"."""
    _, log = replay.load(_inside_repo(bundle)).ours()
    rows = {"T": log.turns, "E": log.events, "X": log.errors}.get(kind.upper(), [])
    lo, hi = (int(x) for x in rounds.split("-")) if rounds else (0, 10**9)
    picked = [r for r in rows if lo <= r.get("r", 0) <= hi and (champion is None or r.get("c") == champion)]
    for r in picked:
        r.pop("x", None)  # feature hex is long; use the dataset export for it
    return picked[:limit]


@server.tool()
def render_frame(bundle: str, round: int, scale: int = 6) -> Image:
    """PNG of the map at a round: terrain, camp groups, our champions (HP bars, target lines),
    visible enemies and camps, plus a sidebar with each champion's mode/expert/intent/fuel."""
    out = ROOT / "renders" / f"{Path(bundle).stem}_r{round}.png"
    return _png(render.frame(_inside_repo(bundle), round, out, scale=scale))


@server.tool()
def render_timeline(bundle: str) -> Image:
    """PNG timeline: tributed gems, carried gems, mean fuel per turn and deaths over the match."""
    out = ROOT / "renders" / f"{Path(bundle).stem}_timeline.png"
    return _png(render.timeline(_inside_repo(bundle), out))


@server.tool()
def fuel_report(bundle: str) -> dict:
    """Fuel per stage (observe, gate, brain, path, act, precompute): mean and p95 over turns,
    plus turns that exhausted their budget (bot suspended)."""
    _, log = replay.load(_inside_repo(bundle)).ours()
    names = ["observe", "gate", "brain", "path", "act", "precompute"]
    stages: dict[str, dict] = {}
    for k, name in enumerate(names):
        vals = sorted(t["fs"][k] for t in log.turns if "fs" in t)
        if vals:
            stages[name] = {
                "mean": round(sum(vals) / len(vals)),
                "p95": vals[int(len(vals) * 0.95)],
                "max": vals[-1],
            }
    return {
        "stages": stages,
        "exhausted_turns": replay.exhausted_turns(log),
        **{k: v for k, v in replay.summarize(log).items() if k.startswith("fuel")},
    }


@server.tool()
def feature_selection_report(path: str = "ml/features/selected/report.json") -> dict:
    """Result of ml/features/select.py: selected feature indices and the per-level
    group size / objective / stability curve."""
    p = _inside_repo(path)
    if not p.exists():
        return {"error": f"{path} not found; run `python -m ml.features.select <dataset.npz>`"}
    return json.loads(p.read_text(encoding="utf-8"))


@server.tool()
def champion_report(vectors: str | None = None) -> Image:
    """Cosine-similarity heatmap of champion vectors (initial kit projection, or a
    fine-tuned vectors JSON from ml/champions/finetune.py)."""
    from ml.champions import report

    out = ROOT / "renders" / "champions.png"
    return _png(report.render(_inside_repo(vectors) if vectors else None, out))


def _gh(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["gh", *args], cwd=ROOT, capture_output=True, text=True)


@server.tool()
def list_backlog(label: str | None = None, state: str = "open", limit: int = 30) -> list[dict] | dict:
    """GitHub issues (the team backlog), optionally filtered by label (e.g. area:pathing)."""
    args = [
        "issue",
        "list",
        "--state",
        state,
        "--limit",
        str(limit),
        "--json",
        "number,title,labels,assignees",
    ]
    if label:
        args += ["--label", label]
    proc = _gh(*args)
    if proc.returncode != 0:
        return {"error": proc.stderr.strip() or "gh failed (run `gh auth login`)"}
    return json.loads(proc.stdout)


@server.tool()
def draft_issue(title: str, body: str, labels: list[str] | None = None, confirm: bool = False) -> dict:
    """Draft a backlog issue. With confirm=False (default) only returns the draft for review;
    confirm=True files it on GitHub."""
    draft = {"title": title, "body": body, "labels": labels or []}
    if not confirm:
        return {"draft": draft, "filed": False, "note": "call again with confirm=true to file it"}
    args = ["issue", "create", "--title", title, "--body", body]
    for label in labels or []:
        args += ["--label", label]
    proc = _gh(*args)
    return {"draft": draft, "filed": proc.returncode == 0, "output": (proc.stdout or proc.stderr).strip()}


def main() -> None:
    server.run("stdio")


if __name__ == "__main__":
    main()
