"""B ceiling test (plan Phase B): is there anything for an in-game net to learn?

At sampled contact decisions (an enemy champion within 6 tiles) of real games played with
A's searched plans active, replay the game with the choice forced to fight or retreat for a
few rounds (bots/main/src/branch.rs), play to the end, and compare with the base game.
Port of archive nn/expert.py's branch games: null branches (choice 0) must reproduce the
base game exactly, winners are verified with a second hold length.

    python -m harness.search branch setup
    python -m harness.search branch run [--hours H]
"""

from __future__ import annotations

import json
import random
import shutil
import time
import zipfile
from pathlib import Path

from harness.search import common as C
from harness.search import search as S

MAGIC = bytes([0x42, 0x52, 0x43, 0x48, 0x02, 0x5A, 0xA5, 0x3C])
BLOCK = 32
OFF = 0xFFFFFFFF
N_SLOTS = 28
HOLD, VERIFY_HOLD = 10, 6
DECISIONS_PER_GAME = 4
MIN_GAP = 30
MARGIN = 0.05  # fitness: a verified alternative must beat the rules by this much
NULL_EVERY = 6
OPPS = 8
LOG_LIMIT = 4_000_000
DIR = C.RUNS / "b"


def _use_b_pool() -> None:
    C.SLOT = "_pb{}"
    C.N_SLOTS = N_SLOTS
    C.RUNS = DIR


def block(frm: int = OFF, kind: int = 0, choice: int = 0, hold: int = 0, log: int = 1) -> bytes:
    words = b"".join(int(x).to_bytes(4, "little") for x in (frm, kind, choice, hold, log))
    return (MAGIC + words).ljust(BLOCK, b"\0")


def patch(bot: Path, blk: bytes) -> None:
    n = 0
    for f in C.wasm_files(bot):
        b = bytearray(f.read_bytes())
        i = b.find(MAGIC)
        if i < 0:
            continue
        if b[i : i + BLOCK] != blk:
            b[i : i + BLOCK] = blk
            f.write_bytes(b)
        n += 1
    if n == 0:
        raise RuntimeError(f"{bot}: branch block not found")


def setup() -> None:
    _use_b_pool()
    snap = C.build_slots(N_SLOTS)
    print(f"B pool {snap['wasm'][:12]} ({N_SLOTS} slots)")


# ---------------------------------------------------------------- games


def _stdout(replay: str, side: str) -> str:
    try:
        with zipfile.ZipFile(replay) as z:
            return z.read(f"team-{side}.stdout").decode("utf-8", "replace").replace("\x00", "")
    except (OSError, KeyError, zipfile.BadZipFile):
        return ""


def _play(jobs: list[dict], plans: dict[str, bytes], opps: dict[str, str]) -> list[dict]:
    """jobs: {map, opp, side, blk}; one slot per job (each job may carry its own branch)."""
    out: list[dict] = []
    maps = C.maps()
    for i in range(0, len(jobs), N_SLOTS):
        chunk = jobs[i : i + N_SLOTS]
        triples, meta = [], []
        for k, j in enumerate(chunk):
            slot = C.slot_dir(k)
            C.patch(slot, plans.get(j["map"], C.record()))
            patch(slot, j["blk"])
            us, them = str(slot), opps[j["opp"]]
            a, b = (us, them) if j["side"] == "a" else (them, us)
            triples.append((a, b, maps[j["map"]]))
            meta.append((j, us))
        tmp = DIR / "tmp" / f"b{int(time.time() * 1000)}"
        res = C.runner.run_games(triples, tmp, release=True, log_limit=LOG_LIMIT)
        for r, (j, us) in zip(res, meta):
            row = C._row(r, C.Game("x", j["map"], j["opp"], j["side"]), "b", us)
            text = _stdout(r.replay, "a" if r.bot_a == us else "b")
            row["D"] = [list(map(int, ln.split()[1:])) for ln in text.splitlines() if ln.startswith("D ")]
            row["fired"] = sum(1 for ln in text.splitlines() if ln.startswith("B "))
            out.append({**j, "blk": j["blk"].hex(), "row": row})
        shutil.rmtree(tmp, ignore_errors=True)
    return out


def _key(j: dict) -> str:
    return f"{j['map']}|{j['opp']}|{j['side']}|{j['blk'] if isinstance(j['blk'], str) else j['blk'].hex()}"


class Log:
    def __init__(self) -> None:
        DIR.mkdir(parents=True, exist_ok=True)
        self.path = DIR / "branch.jsonl"
        self.rows: dict[str, dict] = {}
        if self.path.exists():
            for ln in self.path.read_text(encoding="utf-8").splitlines():
                try:
                    r = json.loads(ln)
                except json.JSONDecodeError:
                    continue
                self.rows[_key(r)] = r

    def todo(self, jobs: list[dict]) -> list[dict]:
        return [j for j in jobs if _key(j) not in self.rows]

    def add(self, rows: list[dict]) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, separators=(",", ":")) + "\n")
                self.rows[_key(r)] = r

    def get(self, j: dict) -> dict | None:
        r = self.rows.get(_key(j))
        return r["row"] if r else None


def run(hours: float | None = None) -> dict:
    _use_b_pool()
    C.check_snapshot()
    deadline = C.deadline_from(hours)
    splits = json.loads((C.ROOT / "runs" / "search" / "splits.json").read_text(encoding="utf-8"))
    C.RUNS = C.ROOT / "runs" / "search"
    table = {m: r for m, r in S.table().items() if r["accepted"]}
    snap_a = C.load_json(C.RUNS / "snapshot.json", {})["wasm"]
    plans = {m: bytes.fromhex(C.Store(f"search/{m}", snap_a).cands[r["accepted"]]) for m, r in table.items()}
    C.RUNS = DIR
    rng = random.Random(2026)
    opps_all = C.opponents()
    opps = rng.sample(splits["opps_A"], OPPS)
    log = Log()

    def step(jobs: list[dict]) -> bool:
        todo = log.todo(jobs)
        for i in range(0, len(todo), N_SLOTS * 4):
            if deadline and time.time() > deadline:
                return False
            log.add(_play(todo[i : i + N_SLOTS * 4], plans, opps_all))
            print(f"  {len(log.rows)} games logged", flush=True)
        return True

    # 1. base games, A active, decisions logged
    base = [{"map": m, "opp": o, "side": s, "blk": block()} for m in plans for o in opps for s in "ab"]
    if not step(base):
        return {}
    # 2. sample decisions and branch them
    branches, decisions = [], []
    for j in base:
        row = log.get(j)
        ds = row["D"] if row else []
        rng2 = random.Random(f"{j['map']}{j['opp']}{j['side']}")
        rng2.shuffle(ds)
        picked: list[list[int]] = []
        for d in ds:
            if d[0] < 20 or d[0] > row["rounds"] - 50:
                continue
            if all(abs(d[0] - p[0]) >= MIN_GAP for p in picked):
                picked.append(d)
            if len(picked) >= DECISIONS_PER_GAME:
                break
        for d in picked:
            rnd, kind, cls = d[0], d[1], d[2]
            alts = [c for c in (1, 2) if c != cls]
            if len(decisions) % NULL_EVERY == 0:
                alts.append(0)
            decisions.append({"base": j, "round": rnd, "kind": kind, "cls": cls, "dist": d[3], "gems": d[4], "alts": alts})
            for c in alts:
                branches.append({**{k: j[k] for k in ("map", "opp", "side")}, "blk": block(rnd, kind, c, HOLD)})
    if not step(branches):
        return {}
    # 3. verify winners with a shorter hold
    def gain(j, c, hold) -> float | None:
        b = log.get({**j, "blk": block()})
        r = log.get({**{k: j[k] for k in ("map", "opp", "side")}, "blk": block(dd["round"], dd["kind"], c, hold)})
        if not b or not r or (c and not r["fired"]):
            return None
        return C.fitness(r) - C.fitness(b)

    verify = []
    for dd in decisions:
        for c in dd["alts"]:
            if c and (g := gain(dd["base"], c, HOLD)) is not None and g > MARGIN:
                verify.append({**{k: dd["base"][k] for k in ("map", "opp", "side")},
                               "blk": block(dd["round"], dd["kind"], c, VERIFY_HOLD)})
    if not step(verify):
        return {}
    return report(decisions, log)


def report(decisions: list[dict], log: Log) -> dict:
    def g(dd, c, hold):
        j = dd["base"]
        b = log.get({**j, "blk": block()})
        r = log.get({**{k: j[k] for k in ("map", "opp", "side")}, "blk": block(dd["round"], dd["kind"], c, hold)})
        if not b or not r:
            return None
        if c and not r["fired"]:
            return None
        return C.fitness(r) - C.fitness(b), r["res"] - b["res"]

    nulls_bad = sum(1 for dd in decisions if 0 in dd["alts"] and (x := g(dd, 0, HOLD)) and x[0] != 0)
    fired = overruled = 0
    gains, wins, by_cls = [], [], {0: [0, 0], 1: [0, 0], 2: [0, 0]}
    for dd in decisions:
        best = None
        any_fired = False
        for c in dd["alts"]:
            if not c:
                continue
            a = g(dd, c, HOLD)
            if a is None:
                continue
            any_fired = True
            v = g(dd, c, VERIFY_HOLD)
            ver = min(a[0], v[0]) if v else None
            if ver is not None and ver > MARGIN and (best is None or ver > best[0]):
                best = (ver, min(a[1], v[1]))
        if not any_fired:
            continue
        fired += 1
        by_cls[dd["cls"]][0] += 1
        if best:
            overruled += 1
            by_cls[dd["cls"]][1] += 1
            gains.append(best[0])
            wins.append(best[1])
    rate = overruled / max(fired, 1)
    out = {
        "decisions": len(decisions),
        "fired": fired,
        "null_mismatches": nulls_bad,
        "overrule_rate": round(rate, 4),
        "mean_gain_when_overruled": round(sum(gains) / max(len(gains), 1), 4),
        "win_flips_when_overruled": round(sum(wins) / max(len(wins), 1), 4),
        "by_rules_choice": {["other", "fight", "flee"][k]: {"n": v[0], "overruled": v[1]} for k, v in by_cls.items()},
        "verdict": "worth training B" if rate >= 0.02 else "no gain: stop B",
    }
    C.save_json(DIR / "branch_report.json", out)
    print(json.dumps(out, indent=1))
    return out
