"""harness/experiments.py: registry and gate logic (no matches played)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))

import experiments as X


def test_add_needs_exactly_one_spec_and_numbers_ids():
    reg = {"experiments": []}
    a = X.add(reg, "a", "combat", knobs={"FIGHT_V6": 1})
    b = X.add(reg, "b", "combat", ref="feat/x")
    assert (a["id"], b["id"]) == (1, 2)
    assert a["spec"] == {"knobs": {"FIGHT_V6": 1}} and a["status"] == "queued"
    with pytest.raises(AssertionError):
        X.add(reg, "c", "x")
    both = X.add(reg, "c", "x", knobs={"PASSIVE_AWARE": 1}, ref="feat/p")
    assert both["spec"] == {"ref": "feat/p", "knobs": {"PASSIVE_AWARE": 1}}


def test_gates():
    live = {"win": 0.55, "fitness": 0.37}
    assert X.screen_passes(live, {"win": 0.545, "fitness": 0.38})  # within slack, fitter
    assert not X.screen_passes(live, {"win": 0.60, "fitness": 0.36})  # less fit
    assert X.full_passes(live, {"win": 0.56, "fitness": 0.38}, 0.5)
    assert not X.full_passes(live, {"win": 0.56, "fitness": 0.38}, 0.40)  # loses head-to-head
    assert not X.full_passes(live, {"win": 0.54, "fitness": 0.40}, 0.6)  # fewer wins


def test_next_queued_and_tex_escaping():
    reg = {"experiments": []}
    X.add(reg, "done", "x", knobs={"A": 1}, status="rejected")
    q = X.add(reg, "bags 6/42 & en_route", "economy", knobs={"EN_ROUTE": 2})
    assert X.next_queued(reg) is q
    out = X.tex(reg)
    assert r"bags 6/42 \& en\_route" in out and r"\begin{tabularx}" in out
