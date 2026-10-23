"""Replay an official game's decoded actions through the GPU engine and compare timelines.

    python -m nn.gpusim.validate <bundle.repl26a> <stats.json> <map name>

Drives one game (CPU) in the official turn order with the recorded actions; at each round
compares per-champion timelines from the statistics file (earned gold, carried gems,
submitted gems, damage dealt to / taken from camps, deaths). Reports where and how much the
engine diverges, which is the to-do list for the engine (abilities, vision, items, ...).
"""

from __future__ import annotations

import collections
import json
import os
import struct
import sys
from pathlib import Path

import numpy as np
import torch

from nn.gpusim import maps, replay
from nn.gpusim.engine import KITS, Engine

KIND = {"Ryth": 0, "Vesper Vale": 1, "Vendetta": 2, "Cassi": 3, "Issac": 4, "Stellar": 5, "Prism": 6,
        "Marcella": 7, "Willow": 8, "Ren": 9, "Varg": 10, "Tavi": 11, "Heart of Ruin": 12}
METRICS = ("earnedGold", "gems", "submittedGems", "monsterDamageDone", "monsterDamageTaken", "deaths")


def timeline(runs: list[dict], n: int) -> np.ndarray:
    out = np.zeros(n)
    at = 0
    for r in runs:
        d = int(r["duration"])
        out[at : at + d] = float(r["value"])
        at += d
    if at < n:
        out[at:] = out[at - 1] if at else 0
    return out


def infer_camp_ids(ts, md, names) -> dict[int, int]:
    """Engine camp id -> dump camp index, from where basic attacks on each id were made."""
    camps = [(c.x, c.y) for g in md.groups for c in g.camps]
    pos, cons = {}, collections.defaultdict(lambda: set(range(len(camps))))
    for t in ts:
        for r in t.records:
            if r.name == "move":
                pos[t.champion] = (r.u32(0), r.u32(4))
            elif r.name == "attack" and r.payload[0] == 2 and t.champion in pos:
                rng = KITS[KIND[names[t.champion]]][1]
                p = pos[t.champion]
                cons[r.u32(1)] &= {i for i, (x, y) in enumerate(camps) if max(abs(x - p[0]), abs(y - p[1])) <= rng}
    out: dict[int, int] = {}
    changed = True
    while changed:
        changed = False
        for eid, c in cons.items():
            c -= set(out.values()) - ({out[eid]} if eid in out else set())
            if len(c) == 1 and eid not in out:
                out[eid] = next(iter(c))
                changed = True
    return out


def main(bundle: str, stats: str, map_name: str, max_rounds: int = 10**9) -> int:
    s = json.loads(Path(stats).read_text(encoding="utf-8-sig"))
    md = maps.load(map_name)
    map_bytes = next(Path("maps").glob(f"{map_name}.map26a")).read_bytes()
    all_turns = replay.turns(bundle, stats, map_bytes)
    ts = [t for t in all_turns if t.round < max_rounds]
    roster = s["roster"]["champions"]
    names = {c["id"]: c["name"] for c in roster}
    teams = [[c["id"] for c in roster if c["team"] == t] for t in ("teamA", "teamB")]
    T = max(len(x) for x in teams)
    kinds = [[KIND[names[cid]] for cid in tm] + [0] * (T - len(tm)) for tm in teams]
    slot = {cid: (ti, k) for ti, tm in enumerate(teams) for k, cid in enumerate(tm)}
    e = Engine(md, 1, kinds, "cpu")
    for ti, tm in enumerate(teams):  # padding slots never play
        for k in range(len(tm), T):
            e.alive[:, ti, k] = False
            e.respawn[:, ti, k] = 10**9
    camp_of = infer_camp_ids(all_turns, md, names)  # from the whole game, even when truncated
    n_rounds = ts[-1].round + 1
    sim = {m: np.zeros((len(roster), n_rounds)) for m in METRICS}
    issues = collections.Counter()
    placed = set()
    cur_round = -1
    for t in ts:
        if t.round != cur_round:
            if cur_round >= 0:
                snapshot(e, sim, slot, cur_round)
            cur_round = t.round
            e.camp_respawns()
            e.captures()
            hp0 = e.hp.clone()
            e.camp_actions()
            taken = (hp0 - e.hp).clamp(min=0)
            if os.environ.get("DEBUG_ROUND") and int(os.environ["DEBUG_ROUND"]) in (cur_round, cur_round - 1):
                print("round", cur_round, "taken", {names[c]: float(taken[0, ti, k]) for c, (ti, k) in slot.items() if taken[0, ti, k] > 0})
                print("  champs", {names[c]: e.pos[0, ti, k].tolist() for c, (ti, k) in slot.items()})
                for ci in range(e.camp_hp.shape[1]):
                    if e.camp_awake[0, ci] and e.camp_alive[0, ci]:
                        print("  camp", ci, md.groups and [(c.x, c.y) for g in md.groups for c in g.camps][ci],
                              "hp", float(e.camp_hp[0, ci]), "quiet", int(e.camp_quiet[0, ci]), "stun", int(e.camp_stun[0, ci]))
            for cid, (ti, k) in slot.items():
                e_taken[cid] = e_taken.get(cid, 0.0) + float(taken[0, ti, k])
        ti, k = slot[t.champion]
        watch = int(os.environ.get("DEBUG_CAMP", "-1"))
        hp_w = float(e.camp_hp[0, watch]) if watch >= 0 else 0.0
        act = bool(e.turn_start(ti, k)[0])
        if os.environ.get("DEBUG_CAST") and t.round < 6:
            print("turn", t.round, names[t.champion], "act", act, "alive", bool(e.alive[0, ti, k]), "stun", int(e.stun[0, ti, k]),
                  "recall", int(e.recall[0, ti, k]), "tribute", int(e.tribute[0, ti, k]), [r.name for r in t.records])
        for r in t.records:
            if r.name == "move":
                p = torch.tensor([r.u32(0), r.u32(4)])
                # the replay records attempted moves: one onto a living champion failed
                others = (e.pos[0] == p).all(-1) & e.alive[0]
                others[ti, k] = False
                if t.champion in placed and bool(others.any()):
                    issues["move onto an occupied tile (failed)"] += 1
                    continue
                if t.champion not in placed or e.cheb(p, e.pos[0, ti, k]).item() > 1:
                    issues["jump (first move, dash, teleport or respawn)"] += t.champion in placed
                placed.add(t.champion)
                e.pos[0, ti, k] = p
            elif r.name == "attack" and act:
                kind, tid = r.payload[0], r.u32(1)
                one = torch.ones(1, dtype=torch.bool)
                if kind == 2:
                    ci = camp_of.get(tid)
                    if ci is None or not e.camp_alive[0, ci]:
                        issues["attack on an unknown or dead camp"] += 1
                        continue
                    hp_before = float(e.camp_hp[0].clamp(min=0).sum())
                    hit = e.basic_attack(ti, k, torch.tensor([2]), torch.tensor([ci]), one)
                    if not bool(hit[0]):
                        issues["attack refused by the engine (range / allowance)"] += 1
                    e_done[t.champion] = e_done.get(t.champion, 0.0) + hp_before - float(e.camp_hp[0].clamp(min=0).sum())
                elif kind == 1 and tid in slot:
                    oti, ok_ = slot[tid]
                    if oti == ti:
                        issues["attack on an ally?"] += 1
                        continue
                    if not bool(e.basic_attack(ti, k, torch.tensor([1]), torch.tensor([ok_]), one)[0]):
                        issues["attack refused by the engine (range / allowance)"] += 1
            elif r.name == "cast" and act:
                sl, kind = r.payload[0], r.payload[4]
                tid = struct.unpack_from("<I", r.payload, 5)[0]
                x, y = struct.unpack_from("<ii", r.payload, 9)
                one = torch.ones(1, dtype=torch.bool)
                idx = 0
                if kind == 1 and tid in slot:
                    idx = slot[tid][1]
                elif kind == 2:
                    idx = camp_of.get(tid, -1)
                    if idx < 0:
                        issues["cast on an unknown camp"] += 1
                        continue
                hp_before = float(e.camp_hp[0].clamp(min=0).sum())
                went = e.cast(ti, k, int(sl), one, torch.tensor([int(kind)]), torch.tensor([idx]), torch.tensor([[x, y]]))
                if os.environ.get("DEBUG_CAST") and t.round < 6:
                    print("cast", t.round, names[t.champion], "slot", sl, "kind", kind, "idx", idx, "went", bool(went[0]),
                          "camp hp", [round(float(v)) for v in e.camp_hp[0]], "done", float(e.camp_dmg_done[0, ti, k]))
                if not bool(went[0]):
                    issues["cast refused by the engine"] += 1
                    if os.environ.get("DEBUG_CAST"):
                        print("refused", t.round, names[t.champion], "slot", sl, "kind", kind, "idx", idx, "xy", (x, y),
                              "pos", e.pos[0, ti, k].tolist(), "cd", e.cd[0, ti, k].tolist(), "walked", bool(e.walked[0, ti, k]),
                              "cast_done", bool(e.cast_done[0, ti, k]), "camp alive", bool(e.camp_alive[0, idx]) if kind == 2 else None)
                e_done[t.champion] = e_done.get(t.champion, 0.0) + hp_before - float(e.camp_hp[0].clamp(min=0).sum())
            elif r.name == "recall":
                e.recall[0, ti, k] = 5
            elif r.name == "tribute" and e.gems[0, ti, k] > 0:
                e.tribute[0, ti, k] = max(1, int(np.ceil(float(e.gems[0, ti, k]) / 10)))
            elif r.name == "buy":
                issues["buys (item effects not modelled yet)"] += 1
        e.turn_end(ti, k, auto_shop=False)
        if watch >= 0 and float(e.camp_hp[0, watch]) != hp_w and t.round < 40:
            print("camp", watch, "round", t.round, names[t.champion], [r.name for r in t.records],
                  f"{hp_w:.0f} -> {float(e.camp_hp[0, watch]):.0f}")
    snapshot(e, sim, slot, cur_round)
    # ---- compare ----
    off = {c["id"]: {} for c in roster}
    for tl in s["championTimelines"]:
        cid = tl["owner"]["id"]
        if tl["metric"] in METRICS:
            # official timelines hold the value at the start of each round: index r + 1 is the
            # state after our round r
            off[cid][tl["metric"]] = timeline(tl["runs"], n_rounds + 1)[1:]
    for nm in os.environ.get("DEBUG_CHAMPS", "").split(","):
        if not nm:
            continue
        cid = next(c for c in names if names[c] == nm)
        for m in os.environ.get("DEBUG_METRICS", "monsterDamageDone").split(","):
            a, b = sim[m][cid], off[cid][m]
            print(nm, m, "round: official/engine", [(r, int(b[r]), int(a[r])) for r in range(min(40, n_rounds)) if r == 0 or b[r] != b[r - 1] or a[r] != a[r - 1]])
    print(f"{map_name}: {n_rounds} rounds, {len(ts)} turns, camp ids mapped {len(camp_of)}")
    print("issues:", dict(issues))
    for m in METRICS:
        errs = []
        for cid in names:
            if m in off[cid]:
                a, b = sim[m][cid], off[cid][m]
                errs.append((cid, names[cid], b[-1], a[-1], first_divergence(a, b)))
        print(f"\n{m}: (champion: official final, engine final, first round they differ)")
        for cid, nm, b, a, fd in errs:
            print(f"   {nm:12s} {b:9.0f} {a:9.0f}   {fd}")
    return 0


e_done: dict[int, float] = {}
e_taken: dict[int, float] = {}


def snapshot(e: Engine, sim: dict, slot: dict, rnd: int) -> None:
    for cid, (ti, k) in slot.items():
        sim["gems"][cid, rnd] = float(e.gems[0, ti, k])
        sim["deaths"][cid, rnd] = float(e.deaths[0, ti, k])
        sim["monsterDamageDone"][cid, rnd] = float(e.camp_dmg_done[0, ti, k])
        sim["monsterDamageTaken"][cid, rnd] = float(e.camp_dmg_taken[0, ti, k])
        sim["earnedGold"][cid, rnd] = float(e.earned[0, ti, k])
        sim["submittedGems"][cid, rnd] = float(e.submitted[0, ti, k])


def first_divergence(a: np.ndarray, b: np.ndarray, tol: float = 0.5) -> int | None:
    d = np.flatnonzero(np.abs(a - b) > tol)
    return int(d[0]) if len(d) else None


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    raise SystemExit(main(*sys.argv[1:4], *(int(a) for a in sys.argv[4:5])))
