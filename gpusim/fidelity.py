"""How faithful is the GPU engine? Replays a fixed corpus of real games and reports, per game and
metric, the first round where the engine's statistics leave the official ones (Gate E: the
engine is good enough to train on when >= 18/20 games match every statistic to the end).

    python -m gpusim.fidelity build      # once: 20 release games on 20 maps (runs/gpusim/corpus)
    python -m gpusim.fidelity            # validate every corpus game, print the table
    python -m gpusim.fidelity --json out.json

The corpus is our bot against varied mimics so every champion kind appears; the games are
plain cambc games with full statistics files (official replays of other teams have no state).
Run `build` with the main .venv (it plays games) and the report with the CUDA venv (torch).
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "runs" / "gpusim" / "corpus"
GAMES = [
    # (team A, team B, map): 20 maps, every champion kind drafted by at least one side
    ("main", "mimic-procless", "00ambush"),
    ("mimic-3", "main", "00collision"),
    ("main", "mimic-policy-wonk", "00corners"),
    ("mimic-prism-best-girl", "main", "00eight"),
    ("main", "mimic-trzy-kwaterki", "00elkring"),
    ("mimic-sneaky-bush-campers", "main", "00expanse"),
    ("main", "mimic-emperor-tung", "00fragment"),
    ("mimic-seacaps", "main", "00gem-fields"),
    ("main", "mimic-intercontinental-fish", "00lanes"),
    ("mimic-new-team-4", "main", "00monolith"),
    ("main", "mimic-reddragon", "00packs"),
    ("mimic-sponge-albert-and-bob", "main", "00pilgrimage"),
    ("main", "baseline", "00reverse-spiral"),
    ("mimic-trust-me-bro", "main", "00rows"),
    ("main", "mimic-univardisty-of-shingtonwash", "00swirl"),
    ("mimic-wanderersofabyss", "main", "01big-world"),
    ("main", "mimic-gargantua", "01boss-battle"),
    ("mimic-ggez", "main", "01starvation"),
    ("main", "mimic-hungry-hungry-hungry", "01the-farm"),
    ("mimic-procless", "main", "01yellow-brick-road"),
]


def build() -> None:
    sys.path.insert(0, str(ROOT / "tools"))
    from bctools import runner

    games = [(a, b, ROOT / "maps" / f"{m}.map26a") for a, b, m in GAMES]
    for r in runner.run_games(games, CORPUS, release=True):
        print(f"{r.map}: {r.bot_a} {r.gems_a} - {r.gems_b} {r.bot_b}")


def report(max_rounds: int = 10**9) -> list[dict]:
    import torch

    from gpusim import validate

    torch.set_grad_enabled(False)
    rows = []
    for a, b, m in GAMES:
        tag = f"{m}__{a}-vs-{b}"
        bundle, stats = CORPUS / f"{tag}.repl26a", CORPUS / f"{tag}.json"
        if not bundle.exists():
            rows.append({"game": tag, "error": "missing (run build)"})
            continue
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                rep = validate.main(str(bundle), str(stats), m, max_rounds)
        except Exception as e:  # noqa: BLE001 - one broken game must not hide the others
            rows.append({"game": tag, "error": f"{type(e).__name__}: {e}"[:200]})
            continue
        first = {}
        for metric, champs in rep["metrics"].items():
            fds = [fd for *_, fd in champs if fd is not None]
            first[metric] = min(fds) if fds else None
        diverged = [v for v in first.values() if v is not None]
        rows.append(
            {
                "game": tag,
                "rounds": rep["rounds"],
                "first": first,
                "earliest": min(diverged) if diverged else None,
                "exact": not diverged,
                "issues": rep["issues"],
            }
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", nargs="?", default="report", choices=["build", "report"])
    p.add_argument("--rounds", type=int, default=10**9)
    p.add_argument("--json", type=Path)
    a = p.parse_args(argv)
    if a.cmd == "build":
        build()
        return 0
    rows = report(a.rounds)
    metrics = ["gems", "submittedGems", "earnedGold", "monsterDamageDone", "monsterDamageTaken", "deaths"]
    print(f"{'game':52s} {'rounds':>6s} " + " ".join(f"{m[:10]:>10s}" for m in metrics))
    for r in rows:
        if "error" in r:
            print(f"{r['game']:52s} ERROR {r['error']}")
            continue
        print(f"{r['game']:52s} {r['rounds']:6d} " + " ".join(f"{str(r['first'][m]) if r['first'][m] is not None else 'exact':>10s}" for m in metrics))
    ok = [r for r in rows if "error" not in r]
    exact = sum(r["exact"] for r in ok)
    med = sorted(r["earliest"] if r["earliest"] is not None else r["rounds"] for r in ok)
    print(f"\nGate E: {exact}/{len(rows)} games exact to the end (need 18/20); median first divergence round "
          f"{med[len(med) // 2] if med else '-'}")
    if a.json:
        a.json.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    raise SystemExit(main())
