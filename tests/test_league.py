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


def _games(live: int, wins: list[bool], start: int) -> dict:
    return {
        f"g{live}-{i}": {"live": live, "kind": "ranked", "win": w, "time": iso(start + i * 60)}
        for i, w in enumerate(wins)
    }


def test_struggling_needs_more_than_noise():
    t0 = 1_800_000_000
    subs = [
        {"version": 1, "time": iso(t0), "ok": True},
        {"version": 2, "time": iso(t0 + 100_000), "ok": True},
    ]
    old = _games(1, [i % 2 == 0 for i in range(60)], t0 + 10)  # 50% before
    # 4 losses in a row is 6% likely at 50%: noise.
    assert (
        league.struggling(
            ledger(subs, ladder=old | _games(2, [True, False, False, False, False], t0 + 100_100))
        )
        is None
    )
    # 5 in a row is 3%: losing beyond chance.
    assert "5 ranked games in a row" in league.struggling(
        ledger(subs, ladder=old | _games(2, [False] * 5, t0 + 100_100))
    )
    # 2 of 14 wins (ending on a win) fails the binomial test.
    rec = [True, False, False, False, False, True] + [False, False, False] + [False] * 5
    rec[-1] = True
    assert "won 3/14" in league.struggling(ledger(subs, ladder=old | _games(2, rec, t0 + 100_100)))
    # When we usually lose (30%), a 5-loss streak is normal.
    weak = _games(1, [i % 10 < 3 for i in range(60)], t0 + 10)
    assert league.struggling(ledger(subs, ladder=weak | _games(2, [False] * 5, t0 + 100_100))) is None


def test_fast_upload_instead_of_revert():
    t0 = 1_800_000_000
    subs = [
        {"version": 1, "time": iso(t0), "ok": True},
        {"version": 2, "time": iso(t0 + 100_000), "ok": True},
    ]
    games = _games(1, [i % 2 == 0 for i in range(60)], t0 + 10) | _games(2, [False] * 6, t0 + 100_100)
    ch = champ({})
    ch["version"] = 3
    now = t0 + 100_000 + 45 * 60  # 45 min after the last upload: < 3 h, > 30 min
    dec = league.release_decision(ch, ledger(subs, ladder=games), now)
    assert dec and dec[0] == 3 and dec[1].startswith("fast upload")
    ch["version"] = 2  # nothing newer to upload: never goes back to v1
    assert league.release_decision(ch, ledger(subs, ladder=games), now) is None
    assert not hasattr(league, "revert_decision")


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


def test_map_name_new_pool():
    assert league.map_name("01big_world") == "big-world"
    assert league.map_name("01elk_arena") == "elk-arena"


def test_parse_fingerprint():
    lines = ["T {}", 'F {"fingerprint":716959943,"plan":"-"}', 'F {"fingerprint":1}']
    assert league.parse_fingerprint(lines) == 716959943
    assert league.parse_fingerprint(["F {broken", "X {}"]) is None


def test_active_maps_follow_the_pool(monkeypatch):
    files = [Path(f"maps/{n}.map26a") for n in ("contest", "01big_world", "swirl")]
    monkeypatch.setattr(league.runner, "list_maps", lambda directory=None: files)
    led = league.empty_ledger()
    assert sorted(league.active_maps(led)) == ["big-world", "contest", "swirl"]  # not synced: all
    led["maps"] = ["big-world", "contest", "islands"]  # swirl retired, islands not pulled yet
    assert sorted(league.active_maps(led)) == ["big-world", "contest"]


def test_locked_entries_are_the_team_cadence():
    names = league.params.names()
    assert {names[i] for i in league.LOCKED} == {"W.recall_gold", "W.tribute_min", "W.tribute_max"}


def test_static_state_tracks_unchanged_macro():
    first = league.static_state(None, "h1", "t1")
    assert first == {"macro_hash": "h1", "since": "t1", "static": False}
    same = league.static_state({"macro_hash": "h1", "since": "t1"}, "h1", "t2")
    assert same == {"macro_hash": "h1", "since": "t1", "static": True}
    moved = league.static_state({"macro_hash": "h1", "since": "t1"}, "h2", "t3")
    assert moved == {"macro_hash": "h2", "since": "t3", "static": False}


def test_macro_hash_ignores_openings_and_game_count():
    a = {"draft": {"5": [1, 2]}, "build": {}, "cadence": {"bag": 40}, "games": 10, "opening": {"x": 1}}
    b = {**a, "games": 30, "opening": {}}
    assert league.macro_hash(a) == league.macro_hash(b)
    assert league.macro_hash(a) != league.macro_hash({**a, "draft": {"5": [2, 1]}})


def test_evolved_mimic_merge_keeps_base():
    base = {**champ({}), "base": "h", "slug": "nyx"}
    up = {
        **champ({"lanes": [1.0]}, [promo("2026-10-20T10:00:00Z", "pc1", "lanes")]),
        "base": "h",
        "slug": "nyx",
    }
    me = {
        **champ({"eight": [2.0]}, [promo("2026-10-20T10:01:00Z", "pc2", "eight")]),
        "base": "h",
        "slug": "nyx",
    }
    m = league.merge_champion(base, up, me, lambda m, a, b: b)
    assert m["base"] == "h" and m["slug"] == "nyx" and m["version"] == 2
    assert sorted(m["maps"]) == ["eight", "lanes"]


def _maps(monkeypatch, names, fps=None):
    files = {n: Path(f"maps/00{n}.map26a") for n in names}
    monkeypatch.setattr(league, "active_maps", lambda led: dict(files))
    monkeypatch.setattr(
        league, "load_json", lambda p, d: {n: 1 for n in (fps or names)} if p == league.FINGERPRINTS else d
    )
    return files


def _local(**kw):
    return {"tries": {}, "rr": 0, "mlast": {}, "cycle": 1, **kw}


def test_new_maps_train_first_lost_ones_first(monkeypatch):
    _maps(monkeypatch, ["lanes", "islands", "hooks"])
    trained = league.full_theta(None)
    trained[0] += 0.5
    ch = champ({"lanes": trained})
    old_loss = {"id": "a", "time": "2026-10-20T10:00:00Z", "map": "lanes", "status": "open"}
    new_loss = {"id": "b", "time": "2026-10-20T11:00:00Z", "map": "hooks", "status": "open"}
    led = ledger(losses=[old_loss, new_loss])
    # lanes has the older loss, but hooks/islands were never trained: hooks (lost on) first.
    assert league.pick_map(ch, led, _local()) == ("hooks", new_loss)
    # once both new maps had their bootstrap rounds, the loss queue takes over
    done = {m: league.BOOTSTRAP_TRIES for m in ("hooks", "islands")}
    assert league.pick_map(ch, led, _local(tries=done)) == ("lanes", old_loss)


def test_stuck_loss_is_skipped_using_local_tries(monkeypatch):
    _maps(monkeypatch, ["lanes", "hooks"])
    th = league.full_theta(None)
    th[0] += 0.5
    ch = champ({"lanes": th, "hooks": th})
    loss = {"id": "a", "time": "t", "map": "lanes", "status": "open", "tries_at": 0}
    _mp, x = league.pick_map(ch, ledger(losses=[loss]), _local(tries={"lanes": league.STUCK_TRIES}))
    assert x is None  # 6 local rounds already: don't keep hammering it before the lead escalates


def test_maps_without_fingerprint_are_not_trained_and_reported(monkeypatch):
    _maps(monkeypatch, ["lanes", "islands"], fps=["lanes"])
    ch = champ({"gone": [0.0]})
    assert list(league.trainable_maps(ledger())) == ["lanes"]
    rep = league.map_report(ch, ledger())
    assert rep["active"] == 2 and rep["fingerprint"] == 1 and rep["dropped"] == ["gone"]


def test_seed_maps_makes_pack_match_pool(monkeypatch):
    _maps(monkeypatch, ["lanes", "islands"])
    ch = champ({})
    assert sorted(league.seed_maps(ch, ledger())) == ["islands", "lanes"]
    assert league.trained_maps(ch) == set()  # defaults are not "trained"
    assert league.seed_maps(ch, ledger()) == []


def test_mimic_target_rotates_teams_and_maps(monkeypatch):
    _maps(monkeypatch, ["lanes", "hooks"])
    monkeypatch.setattr(
        league,
        "MIMICS",
        type("P", (), {"__truediv__": lambda s, n: type("E", (), {"exists": lambda s: True})()})(),
    )
    pool = [{"slug": s, "static": True} for s in ("a", "b", "c", "d")]
    ch = champ({})
    ch["bench"] = {"0": {"per_team": {"a": 0.9, "b": 0.8, "c": 0.7, "d": 0.1}}}
    led = {**ledger(), "pool": pool}
    local = _local()
    picks = []
    for c in range(6):
        local["cycle"] = c
        team, mp = league.mimic_target(ch, led, local)
        picks.append((team["slug"], mp))
    assert {t for t, _ in picks[:3]} == {"a", "b", "c"}  # the 3 we beat most, in turn; never d
    assert picks[0][0] == picks[3][0] and picks[0][1] != picks[3][1]  # next visit: another map


def test_upload_supersedes_open_losses_only():
    losses = [
        {"id": "a", "time": "2026-10-21T01:00:00Z", "map": "lanes", "status": "open"},
        {"id": "b", "time": "2026-10-21T02:00:00Z", "map": "hooks", "status": "needs-human"},
        {"id": "c", "time": "2026-10-21T03:00:00Z", "map": "lanes", "status": "fixed"},
    ]
    led = ledger(losses=losses)
    assert league.supersede_losses(led, 9) == 1
    assert [x["status"] for x in led["losses"]] == ["superseded", "needs-human", "fixed"]
    assert led["losses"][0]["superseded_by"] == 9


def test_map_pressure_decays_and_weights_the_live_version():
    now = 1_800_000_000.0
    losses = [
        {"id": "old", "time": iso(now - 48 * H), "map": "lanes", "status": "superseded"},
        {"id": "new", "time": iso(now - 1 * H), "map": "hooks", "status": "open"},
        {"id": "stuck", "time": iso(now - 1 * H), "map": "swirl", "status": "needs-human"},
    ]
    subs = [{"version": 9, "time": iso(now - 2 * H), "ok": True}]
    led = ledger(subs, losses, ladder={"new": {"live": 9}, "old": {"live": 1}})
    p = league.map_pressure(led, now)
    assert "swirl" not in p  # humans have it
    assert p["hooks"] > 1.9  # fresh and against the live version: ~2
    assert p["lanes"] < 0.2  # two days old


def test_empty_queue_trains_where_pressure_is(monkeypatch):
    _maps(monkeypatch, ["lanes", "hooks", "islands"])
    th = league.full_theta(None)
    th[0] += 0.5
    ch = champ({"lanes": th, "hooks": th, "islands": th})
    now = league.time.time()
    losses = [
        {"id": f"l{i}", "time": iso(now - 600), "map": "hooks", "status": "superseded"} for i in range(5)
    ]
    led = ledger(losses=losses)
    picks = [league.pick_map(ch, led, _local(rr=r))[0] for r in range(60)]
    assert picks.count("hooks") > 40  # the pressured map dominates...
    assert set(picks) == {"lanes", "hooks", "islands"}  # ...but the others still get turns


def test_disk_guard_prunes_oldest_scratch_first(tmp_path):
    import os

    for i, d in enumerate(["cycles/01", "experiments/x20-a", "experiments/x21-b", "register/keep"]):
        (tmp_path / d).mkdir(parents=True)
        os.utime(tmp_path / d, (1000 + i, 1000 + i))
    gb = {"free": 3.0}

    def free():
        return gb["free"]

    orig = league.shutil.rmtree

    def rm(p, ignore_errors=False):
        orig(p, ignore_errors=ignore_errors)
        gb["free"] += 5.0

    league.shutil.rmtree, waits = rm, []
    try:
        league.ensure_space(free=free, sleep=lambda s: waits.append(s), scratch=tmp_path)
    finally:
        league.shutil.rmtree = orig
    left = sorted(str(p.relative_to(tmp_path)).replace("\\", "/") for p in tmp_path.glob("*/*"))
    assert left == ["register/keep"]  # 3 GB + 3 x 5 GB freed >= 15: the 3 oldest go, register stays
    assert waits == []


def test_disk_guard_waits_when_nothing_left_to_prune(tmp_path):
    calls = {"n": 0}

    def free():
        calls["n"] += 1
        return 1.0 if calls["n"] < 4 else 20.0

    waits = []
    league.ensure_space(free=free, sleep=lambda s: waits.append(s), scratch=tmp_path)
    assert waits and waits[0] == 600


def test_exploiter_acceptance():
    assert league.exploit_accepts(0.6, 0.5, None)  # first generation: just beat main
    assert not league.exploit_accepts(0.5, 0.5, None)
    assert league.exploit_accepts(0.6, 0.53, 0.5)
    assert not league.exploit_accepts(0.6, 0.51, 0.5)  # not better than the current exploiter


def test_past_versions_newest_first(tmp_path, monkeypatch):
    for n in (1, 9, 10, 15, 3):
        (tmp_path / "bots" / f"lg-v{n}").mkdir(parents=True)
    (tmp_path / "bots" / "lg-vx").mkdir()
    monkeypatch.setattr(league, "ROOT", tmp_path)
    assert league.past_versions(3) == ["lg-v15", "lg-v10", "lg-v9"]
