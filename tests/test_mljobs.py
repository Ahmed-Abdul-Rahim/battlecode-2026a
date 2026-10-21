"""harness/mljobs.py: queue planning, claims and registry merges (no games played)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))

import experiments as X
import mljobs as M


def test_plan_trains_after_enough_new_shards():
    reg = {"jobs": []}
    assert M.plan_next(reg)[0] == "gen"
    for _ in range(M.TRAIN_EVERY):
        M.add(reg, "gen", {"explore": 10})["status"] = "done"
    kind, params = M.plan_next(reg)
    assert kind == "train" and params["hidden"] == 32
    M.add(reg, "train", params)["status"] = "done"
    kind, params = M.plan_next(reg)
    assert kind == "gen" and params["offset"] == M.TRAIN_EVERY * M.GEN_MAPS  # maps rotate


def test_claims_resume_ours_and_free_stale_ones():
    reg = {"jobs": []}
    a = M.add(reg, "gen")
    a.update(status="running", host="other", claimed="2026-10-21T10:00:00Z")
    now = M.L.ts("2026-10-21T11:00:00Z")
    assert M.next_job(reg, now) is None
    assert M.next_job(reg, now + M.CLAIM_TTL) is a
    a["host"] = M.L.HOST
    assert M.next_job(reg, now) is a


def test_job_registry_merge():
    ours, theirs = {"jobs": []}, {"jobs": []}
    M.add(theirs, "gen")["status"] = "done"
    j = M.add(ours, "gen")
    j.update(status="running", host="me", claimed="2026-10-21T10:00:00Z")
    M.add(ours, "train")
    m = M.L.merge_experiments(ours, theirs, key="jobs")["jobs"]
    assert [x["status"] for x in m] == ["done", "queued"]


def test_variant_files_swap_in(monkeypatch, tmp_path):
    monkeypatch.setattr(X, "ROOT", tmp_path)
    (tmp_path / "bots" / "_x7" / "src" / "weights").mkdir(parents=True)
    monkeypatch.setattr(X, "knob_variant", lambda name, knobs, copy=True: name)
    monkeypatch.setattr(X, "fetch_file", lambda src: b"BCW1blob" if src == "ml-data:candidates/r.bin" else b"")
    e = {
        "id": 7,
        "spec": {"knobs": {"ROUTER": 1}, "files": {"bots/main/src/weights/router.bin": "ml-data:candidates/r.bin"}},
    }
    assert X.build_variant(e) == "_x7"
    assert (tmp_path / "bots" / "_x7" / "src" / "weights" / "router.bin").read_bytes() == b"BCW1blob"
