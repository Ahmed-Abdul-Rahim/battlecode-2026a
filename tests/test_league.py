"""Pure logic of harness/league.py: champion merge, ledger merge, release rules, review notes."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))

import league

H = 3600


def promo(t: str, host: str, mp: str) -> dict:
    return {"version": 0, "time": t, "host": host, "map": mp, "gain": 0.1}


def champ(maps: dict, hist: list | None = None, tries: dict | None = None) -> dict:
    return {**league.empty_champion(), "maps": maps, "history": hist or [], "tries": tries or {}}


def test_merge_different_maps_takes_both_and_renumbers():
    base = champ({"lanes": [0.0]})
    up = champ(
        {"lanes": [0.0], "swirl": [1.0]},
        [promo("2026-10-20T10:00:00Z", "pc1", "swirl")],
        {"swirl": {"pc1": 3}},
    )
    me = champ(
        {"lanes": [2.0]}, [promo("2026-10-20T11:00:00Z", "pc2", "lanes")], {"swirl": {"pc1": 1, "pc2": 4}}
    )
    m = league.merge_champion(
        base, up, me, decide=lambda *_: (_ for _ in ()).throw(AssertionError("no conflict"))
    )
    assert m["maps"] == {"lanes": [2.0], "swirl": [1.0]}
    assert [(h["version"], h["host"]) for h in m["history"]] == [(1, "pc1"), (2, "pc2")]
    assert m["version"] == 2
    assert m["tries"] == {"swirl": {"pc1": 3, "pc2": 4}}


def test_merge_same_map_calls_decider():
    base = champ({"lanes": [0.0]})
    up = champ({"lanes": [1.0]}, [promo("2026-10-20T10:00:00Z", "pc1", "lanes")])
    me = champ({"lanes": [2.0]}, [promo("2026-10-20T10:05:00Z", "pc2", "lanes")])
    seen = []

    def decide(mp, a, b):
        seen.append((mp, a, b))
        return b

    m = league.merge_champion(base, up, me, decide)
    assert seen == [("lanes", [1.0], [2.0])]
    assert m["maps"]["lanes"] == [2.0] and m["version"] == 2


def test_merge_is_idempotent_on_duplicate_history():
    h = [promo("2026-10-20T10:00:00Z", "pc1", "lanes")]
    a = champ({"lanes": [1.0]}, h)
    m = league.merge_champion(champ({}), a, a, decide=None)
    assert m["version"] == 1 and len(m["history"]) == 1


def test_merge_ledger_keeps_resolved_losses():
    a = {"losses": [{"id": "x", "time": "t1", "status": "fixed"}], "ladder": {"x": {}}}
    b = {"losses": [{"id": "x", "time": "t1", "status": "open"}, {"id": "y", "time": "t2", "status": "open"}]}
    m = league.merge_ledger(a, b)
    assert [(x["id"], x["status"]) for x in m["losses"]] == [("x", "fixed"), ("y", "open")]
    assert "x" in m["ladder"]


def ledger(subs=(), losses=(), ladder=None) -> dict:
    return {
        **league.empty_ledger(),
        "submissions": list(subs),
        "losses": list(losses),
        "ladder": ladder or {},
    }


def iso(t: float) -> str:
    import datetime as dt

    return dt.datetime.fromtimestamp(t, dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def test_release_needs_a_reason_and_the_gap():
    now = 1_800_000_000.0
    ch = champ({}, [promo(iso(now - 100), "pc", "lanes")] * 3)
    ch["version"] = 3
    sub = {"version": 2, "time": iso(now - 7 * H), "ok": True}
    assert league.release_decision(ch, ledger([sub]), now) is None  # no loss, no benchmark jump
    loss = {"id": "l", "time": iso(now - 1 * H), "status": "open"}
    assert league.release_decision(ch, ledger([sub], [loss]), now)[0] == 3
    recent = {**sub, "time": iso(now - 2 * H)}
    assert league.release_decision(ch, ledger([recent], [loss]), now) is None  # rate limit
    ch["bench"] = {"2": {"overall": 0.50}, "3": {"overall": 0.56}}
    assert "benchmark" in league.release_decision(ch, ledger([sub]), now)[1]
    ch["version"] = 2
    assert league.release_decision(ch, ledger([sub], [loss]), now) is None  # nothing newer


def test_first_upload_waits_for_a_reason():
    now = 1_800_000_000.0
    ch = champ({})
    ch["version"] = 1
    assert league.release_decision(ch, ledger(), now) is None
    loss = {"id": "l", "time": iso(now - H), "status": "open"}
    assert league.release_decision(ch, ledger([], [loss]), now) == (1, "ladder loss since the last upload")


def test_revert_when_live_version_does_worse():
    subs = [{"version": 1, "time": "a", "ok": True}, {"version": 2, "time": "b", "ok": True}]
    games = {f"p{i}": {"live": 1, "kind": "ranked", "win": i < 7} for i in range(10)}
    games |= {f"l{i}": {"live": 2, "kind": "ranked", "win": i < 4} for i in range(10)}
    assert league.revert_decision(ledger(subs, ladder=games)) == 1
    few = {k: v for k, v in games.items() if not k.startswith("l") or int(k[1:]) < 5}
    assert league.revert_decision(ledger(subs, ladder=few)) is None
    reverted = [*subs, {"version": 1, "time": "c", "ok": True, "reason": "revert: v2 did worse"}]
    assert league.revert_decision(ledger(reverted, ladder=games)) is None


def test_review_keeps_reviewer_notes():
    old = "# old\nstuff\n\n## Reviewer notes\n\nNyx kites at range 4; add a kite feature.\n"
    new = league.splice_notes("# new\nbody\n\n## Reviewer notes\n\n_placeholder_\n", old)
    assert new.startswith("# new") and "kites at range 4" in new and "_placeholder_" not in new
    assert league.splice_notes("# new\n## Reviewer notes\n", None).startswith("# new")


def test_map_name():
    assert league.map_name("00contest") == "contest"
    assert league.map_name("03rushbait_v2") == "rushbait-v2"


def test_review_renders_from_empty_state():
    text = league.review_text(league.empty_champion(), league.empty_ledger())
    assert "## Top 10" in text and league.NOTES in text
