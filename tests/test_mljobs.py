"""harness/mljobs.py: queue planning, claims and registry merges (no games played)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))

import experiments as X
import mljobs as M


def test_rl_plan_init_rollouts_update_per_arm():
    reg = {"jobs": []}
    assert M.plan_rl(reg) == ("rlinit", {"arm": 1})
    M.add(reg, "rlinit", {"arm": 1})["status"] = "done"
    M.add(reg, "rlinit", {"arm": 2})["status"] = "done"
    kind, params = M.plan_rl(reg)
    assert kind == "rollout" and params["it"] == 0 and params["part"] == 0
    for part in range(M.RL_PARTS):
        M.add(reg, "rollout", {"arm": 1, "it": 0, "part": part})["status"] = "done"
        M.add(reg, "rollout", {"arm": 2, "it": 0, "part": part})["status"] = "done"
    kind, params = M.plan_rl(reg)
    assert kind == "update" and params["it"] == 0
    M.add(reg, "update", {"arm": 1, "it": 0})["status"] = "done"
    assert M.rl_state(reg, 1)["it"] == 1 and M.rl_state(reg, 2)["it"] == 0
    running = M.add(reg, "rollout", {"arm": 1, "it": 1, "part": 0})
    running["status"] = "running"
    assert M.rl_state(reg, 1)["rolling"] == 1


def test_plan_trains_after_enough_new_shards():
    reg = {"jobs": []}
    assert M.plan_awr(reg)[0] == "gen"
    for _ in range(M.TRAIN_EVERY):
        M.add(reg, "gen", {"explore": 10})["status"] = "done"
    kind, params = M.plan_awr(reg)
    assert kind == "train" and params["hidden"] == 32
    M.add(reg, "train", params)["status"] = "done"
    kind, params = M.plan_awr(reg)
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


def test_status_report():
    reg = {"jobs": []}
    for _ in range(2):
        g = M.add(reg, "gen")
        g.update(status="done", result={"games": 78, "rows": 1000, "win": 0.3, "maps": ["a"]})
    t = M.add(reg, "train")
    t.update(status="done", result={"rows": 2000, "val_agreement": 0.81, "experiment": 50})
    xreg = {"experiments": [{"id": 50, "status": "rejected", "results": {}}]}
    out = M.status_report(reg, xreg)
    assert "156 data games, 2,000 training rows in 2 shards" in out
    assert "copies the rules 81% | experiment #50: rejected" in out
    assert f"after {M.TRAIN_EVERY} more data jobs" in out


def test_fit_is_fast_and_reports_held_out_agreement(tmp_path):
    import numpy as np

    rng = np.random.default_rng(1)
    x = rng.integers(-127, 127, (5000, 109)).astype(np.int8)
    y = (x[:, 0] > 0).astype(int)  # learnable
    rep = M.awr.fit({"X": x, "y": y, "R": rng.normal(size=5000)}, tmp_path / "r.bin", hidden=16, epochs=8)
    assert rep["val_agreement"] > 0.8 and (tmp_path / "r.bin").exists()


def test_variant_files_swap_in(monkeypatch, tmp_path):
    monkeypatch.setattr(X, "ROOT", tmp_path)
    (tmp_path / "bots" / "_x7" / "src" / "weights").mkdir(parents=True)
    monkeypatch.setattr(X, "knob_variant", lambda name, knobs, copy=True: name)
    monkeypatch.setattr(
        X, "fetch_file", lambda src: b"BCW1blob" if src == "ml-data:candidates/r.bin" else b""
    )
    e = {
        "id": 7,
        "spec": {
            "knobs": {"ROUTER": 1},
            "files": {"bots/main/src/weights/router.bin": "ml-data:candidates/r.bin"},
        },
    }
    assert X.build_variant(e) == "_x7"
    assert (tmp_path / "bots" / "_x7" / "src" / "weights" / "router.bin").read_bytes() == b"BCW1blob"


def test_status_shows_rl_arms():
    reg = {"jobs": []}
    M.add(reg, "rlinit", {"arm": 1})["status"] = "done"
    r = M.add(reg, "rollout", {"arm": 1, "it": 0, "part": 0})
    r.update(status="done", result={"games": 80, "win": 0.55})
    u = M.add(reg, "update", {"arm": 1, "it": 0})
    u.update(status="done", result={"it": 5, "mean_reward": 0.01, "experiment": 70})
    out = M.status_report(reg, {"experiments": [{"id": 70, "status": "queued", "results": {}}]})
    assert "arm 1 (experts + fight target): iteration 1, 80 games played" in out
    assert "iteration 5 -> experiment #70: queued" in out
    assert "arm 2 (micro control): not started" in out
