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
    assert X.screen_passes({"pairs": 182, "mean": -0.01, "se": 0.02})  # within noise
    assert not X.screen_passes({"pairs": 182, "mean": -0.05, "se": 0.02})  # clearly worse
    assert not X.screen_passes({"pairs": 0, "mean": 0.0, "se": 0.0})
    assert X.full_passes(live, {"win": 0.56, "fitness": 0.38}, 0.5)
    assert not X.full_passes(live, {"win": 0.56, "fitness": 0.38}, 0.40)  # loses head-to-head
    assert not X.full_passes(live, {"win": 0.54, "fitness": 0.40}, 0.6)  # fewer wins


def test_claims_and_stale_claims():
    reg = {"experiments": []}
    other = X.add(reg, "theirs", "x", knobs={"A": 1}, status="screening")
    other.update(host="other-pc", claimed="2026-10-21T10:00:00Z")
    mine = X.add(reg, "mine, interrupted", "x", knobs={"A": 2}, status="full")
    mine.update(host=X.L.HOST, claimed="2026-10-21T10:00:00Z")
    now = X.L.ts("2026-10-21T12:00:00Z")
    assert X.next_queued(reg, now) is mine  # resume our own interrupted claim
    mine["status"] = "rejected"
    assert X.next_queued(reg, now) is None  # the other PC's claim is fresh
    assert X.next_queued(reg, now + X.CLAIM_TTL) is other  # ... until it goes stale


def test_registry_merge_keeps_progress_and_first_claim():
    L = X.L
    base = {"experiments": []}
    a = X.add(base, "a", "x", knobs={"A": 1})
    b = X.add(base, "b", "x", knobs={"B": 1})
    import copy

    up, ours = copy.deepcopy(base), copy.deepcopy(base)
    up["experiments"][0].update(status="rejected", results={"screen": 1})
    ours["experiments"][0].update(status="screening", host="me", claimed="2026-10-21T11:00:00Z")
    up["experiments"][1].update(status="screening", host="pc2", claimed="2026-10-21T10:00:00Z")
    ours["experiments"][1].update(status="screening", host="me", claimed="2026-10-21T10:05:00Z")
    X.add(ours, "c (new here)", "x", knobs={"C": 1})
    m = L.merge_experiments(ours, up)["experiments"]
    assert [e["id"] for e in m] == [a["id"], b["id"], 3]
    assert m[0]["status"] == "rejected"  # finished beats claimed
    assert m[1]["host"] == "pc2"  # the earlier claim wins
    again = copy.deepcopy(up)
    X.requeue(again["experiments"][0], "re-run")
    m2 = L.merge_experiments(again, up)["experiments"]
    assert m2[0]["status"] == "queued" and m2[0]["run"] == 2  # a re-run beats the old result


def test_paired_difference_matches_games():
    R = X.runner.MatchResult

    def g(m, a, b, win_a, ga, gb):
        return R(m, a, b, a if win_a else b, ga, gb, 100, 0, 0, "", "")

    res = [
        g("m1", "main", "opp", True, 10, 0), g("m1", "opp", "main", False, 0, 10),
        g("m1", "v", "opp", True, 10, 0), g("m1", "opp", "v", True, 10, 0),  # v loses side b
    ]
    p = X.paired(res, "main", "v")
    assert p["pairs"] == 2 and p["mean"] < 0 and p["se"] > 0


def test_next_queued_and_tex_escaping():
    reg = {"experiments": []}
    X.add(reg, "done", "x", knobs={"A": 1}, status="rejected")
    q = X.add(reg, "bags 6/42 & en_route", "economy", knobs={"EN_ROUTE": 2})
    assert X.next_queued(reg) is q
    out = X.tex(reg)
    assert r"bags 6/42 \& en\_route" in out and r"\begin{tabularx}" in out
