"""harness.search Store: streamed load and the per-candidate index agree with a full scan."""

import json
import zipfile

from harness.search import common as C


def test_store_index_matches_full_scan(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "RUNS", tmp_path)
    rows = [{"snap": "s", "cand": c, "map": m, "opp": "o", "side": sd, "res": 1.0}
            for c in ("base", "c1") for m in ("x", "y") for sd in "ab"]
    rows.append({"snap": "old", "cand": "c1", "map": "x", "opp": "o", "side": "a", "res": 0.0})
    d = tmp_path / "job"
    d.mkdir()
    (d / "games.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows) + '{"cut', encoding="utf-8")
    s = C.Store("job", "s")
    assert len(s.rows) == 8
    for cand in ("base", "c1"):
        scan = [r for r in s.rows.values() if r["cand"] == cand and r["map"] == "x"]
        assert s.select(cand, map="x") == scan
    # Re-adding a game replaces it in the index too.
    s.add([{"snap": "s", "cand": "c1", "map": "x", "opp": "o", "side": "a", "res": 0.0}])
    assert len(s.select("c1")) == 4
    assert [r["res"] for r in s.select("c1", map="x", side="a")] == [0.0]


def test_load_fuel_skips_stdout(tmp_path):
    from bctools import replay

    p = tmp_path / "g.repl26a"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("team-a.stdout", 'T {"r":1}\n')
        z.writestr("team-b.stdout", 'T {"r":1}\n')
    log = replay.load_fuel(p, "b")
    assert log.turns == [] and log.fuel == []
    assert replay.load(p).team("a").turns == [{"r": 1}]
