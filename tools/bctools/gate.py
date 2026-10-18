"""Evaluation gate: may this candidate replace the current best bot?

Pass rule (docs/self-improvement.md), over every map on both sides:
  * win rate >= 55% and Wilson 95% lower bound > 50%, and
  * held-out win rate >= 50% (generalisation check, P-012 Procgen protocol).

Matches are deterministic per map and side, so the sample is maps x 2. Held-out
maps (copies in maps/heldout/) must never be used to tune parameters; they are
listed separately so overfitting to the tuning maps shows up.
"""

from __future__ import annotations

from pathlib import Path

from bctools import ROOT, runner

MIN_WIN_RATE = 0.55
MIN_LOWER_BOUND = 0.50
MIN_HELDOUT = 0.50


def heldout_names() -> set[str]:
    return {p.stem for p in runner.list_maps(ROOT / "maps" / "heldout")}


def gate(
    candidate: str, best: str = "baseline", maps: list[Path] | None = None, out_dir: Path | None = None
) -> dict:
    results = runner.run_matches(candidate, best, maps or runner.list_maps(), out_dir=out_dir, swap=True)
    summary = runner.summarize(results, candidate)
    held = [r for r in results if r.map in heldout_names()]
    held_rate = sum(r.score_for(candidate) for r in held) / len(held) if held else None
    lo = summary["wilson_95"][0]
    summary["heldout_win_rate"] = held_rate
    summary["passed"] = (
        summary["win_rate"] >= MIN_WIN_RATE
        and lo > MIN_LOWER_BOUND
        and (held_rate is None or held_rate >= MIN_HELDOUT)
    )
    summary["rule"] = (
        f"win_rate >= {MIN_WIN_RATE} and wilson_low > {MIN_LOWER_BOUND} and heldout >= {MIN_HELDOUT}"
    )
    summary["opponent"] = best
    return summary
