"""harness/mljobs.py: queue planning, claims, registry merges and status (no games played)."""

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


def test_claims_resume_ours_and_free_stale_ones():
    reg = {"jobs": []}
    a = M.add(reg, "rollout", {"arm": 1, "it": 0, "part": 0})
    a.update(status="running", host="other", claimed="2026-10-21T10:00:00Z")
    now = M.L.ts("2026-10-21T11:00:00Z")
    assert M.next_job(reg, now) is None
    assert M.next_job(reg, now + M.CLAIM_TTL) is a
    a["host"] = M.L.HOST
    assert M.next_job(reg, now) is a


def test_retired_job_kinds_are_ignored():
    reg = {
        "jobs": [
            {
                "id": 1,
                "kind": "gen",
                "params": {},
                "status": "running",
                "host": M.L.HOST,
                "claimed": "2026-10-21T10:00:00Z",
                "result": {},
            }
        ]
    }
    assert M.next_job(reg, M.L.ts("2026-10-21T11:00:00Z")) is None
    assert "gen" not in M.status_report(reg, {"experiments": []})


def test_job_registry_merge():
    ours, theirs = {"jobs": []}, {"jobs": []}
    M.add(theirs, "rlinit", {"arm": 1})["status"] = "done"
    j = M.add(ours, "rlinit", {"arm": 1})
    j.update(status="running", host="me", claimed="2026-10-21T10:00:00Z")
    M.add(ours, "rollout", {"arm": 1, "it": 0, "part": 0})
    m = M.L.merge_experiments(ours, theirs, key="jobs")["jobs"]
    assert [x["status"] for x in m] == ["done", "queued"]


def test_variant_files_swap_in(monkeypatch, tmp_path):
    monkeypatch.setattr(X, "ROOT", tmp_path)
    (tmp_path / "bots" / "_x7" / "src" / "weights").mkdir(parents=True)
    monkeypatch.setattr(X, "knob_variant", lambda name, knobs, copy=True: name)
    monkeypatch.setattr(X, "fetch_file", lambda src: b"BCW1blob" if src == "ml-data:rl/arm1/p.bin" else b"")
    e = {
        "id": 7,
        "spec": {
            "knobs": {"ROUTER": 1},
            "files": {"bots/main/src/weights/router.bin": "ml-data:rl/arm1/p.bin"},
        },
    }
    assert X.build_variant(e) == "_x7"
    assert (tmp_path / "bots" / "_x7" / "src" / "weights" / "router.bin").read_bytes() == b"BCW1blob"


def test_status_shows_rl_arms_and_only_recent_failures():
    reg = {"jobs": []}
    M.add(reg, "rlinit", {"arm": 1})["status"] = "done"
    r = M.add(reg, "rollout", {"arm": 1, "it": 0, "part": 0})
    r.update(status="done", result={"games": 80, "win": 0.55})
    u = M.add(reg, "update", {"arm": 1, "it": 0})
    u.update(status="done", result={"it": 5, "mean_reward": 0.01, "experiment": 70})
    old = M.add(reg, "rlinit", {"arm": 2})
    old.update(status="failed", finished="2026-10-20T00:00:00Z", result={"error": "old"})
    new = M.add(reg, "rollout", {"arm": 1, "it": 1, "part": 0})
    new.update(status="failed", finished="2026-10-22T10:00:00Z", result={"error": "new"})
    out = M.status_report(
        reg,
        {"experiments": [{"id": 70, "status": "queued", "results": {}}]},
        now=M.L.ts("2026-10-22T12:00:00Z"),
    )
    assert "arm 1 (experts + fight target): iteration 1, 80 games played" in out
    assert "iteration 5 -> experiment #70: queued" in out
    assert "arm 2 (micro control): not started" in out
    assert "new" in out and "old" not in out
    later = M.add(reg, "rollout", {"arm": 1, "it": 1, "part": 1})
    later.update(status="done", result={"games": 80, "win": 0.5})  # the same kind worked since
    out = M.status_report(reg, {"experiments": []}, now=M.L.ts("2026-10-22T12:00:00Z"))
    assert "UNRESOLVED" not in out
