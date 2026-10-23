"""Does search win fights? Small fights in the GPU engine: greedy vs determinized search.

    python -m nn.gpusim.skirmish [--scenarios 512] [--rounds 20] [--depth 6] [--dets 8]

Scenario: a real map (terrain), camps removed, two teams of three random champions placed a few
tiles apart. Policies:
- greedy: approach the nearest enemy, attack, cast an offensive ability when ready (heal the
  weakest ally / shield when hurt);
- search: before each of its champions acts, every candidate action (9 moves x attack or not x
  3 spell choices) is played forward `depth` rounds with greedy for everyone, in `dets`
  determinizations of what it can't see (enemy cooldowns are sampled, not read), and the best
  mean outcome is taken (flat Monte Carlo over determinizations: the simple form of ISMCTS).
Outcome after `rounds`: team health fraction (dead = 0) + 0.5 per enemy death, A minus B.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import random
import time

import numpy as np
import torch

from nn.gpusim import maps
from nn.gpusim.engine import ABILITIES, DIRS, SPELL_RANGE, Engine

OFFENSIVE = {"dmg", "stun", "aoe", "aoe_stun", "hordash"}
CANDIDATES = torch.tensor([[m, a, s, 0] for m, a, s in itertools.product(range(9), (0, 1), (0, 1, 2))])


def clone(e: Engine, idx: torch.Tensor) -> Engine:
    """Engine over games `idx` of `e` (repeats allowed); static tensors shared."""
    n = copy.copy(e)
    for k, v in e.__dict__.items():
        if torch.is_tensor(v) and v.dim() > 0 and v.shape[0] == e.B and k not in ("kinds", "camp_max", "camp_str", "camp_rng"):
            setattr(n, k, v[idx].clone())
    n.B = len(idx)
    return n


def greedy(e: Engine, team: int) -> torch.Tensor:
    """long [B, T, 4] greedy fight actions for `team`."""
    B, T, dev = e.B, e.T, e.dev
    out = torch.zeros(B, T, 4, dtype=torch.long, device=dev)
    dirs = DIRS.to(dev)
    for k in range(T):
        here = e.pos[:, team, k]
        ep = e.pos[:, 1 - team]
        de = e.cheb(ep, here.unsqueeze(1))
        de = torch.where(e.alive[:, 1 - team], de, 10_000)
        tgt = de.argmin(1)
        d = de.gather(1, tgt.unsqueeze(1)).squeeze(1)
        tpos = ep[torch.arange(B, device=dev), tgt]
        in_rng = d <= e.rng[:, team, k]
        # best step towards the target
        cand = here.unsqueeze(1) + dirs.unsqueeze(0)  # [B, 9, 2]
        x = cand[..., 0].clamp(0, e.s.w - 1)
        y = cand[..., 1].clamp(0, e.s.h - 1)
        ok = e.s.walkable[y, x] & (cand[..., 0] == x) & (cand[..., 1] == y)
        dist = e.cheb(cand, tpos.unsqueeze(1))
        dist = torch.where(ok, dist, 10_000)
        dist[:, 0] = dist[:, 0] - 0  # staying is allowed
        mv = dist.argmin(1)
        ranged = e.ranged[:, team, k]
        move = torch.where(in_rng, 0, mv)
        out[:, k, 0] = move
        out[:, k, 1] = torch.where(in_rng | ~ranged, 1, 0)
        kind = int(e.kinds[team, k])
        hurt_self = e.hp[:, team, k] / e.max_hp[:, team, k] < 0.5
        allies = torch.where(e.alive[:, team], e.hp[:, team] / e.max_hp[:, team], 2.0)
        hurt_ally = allies.min(1).values < 0.6
        for slot in (0, 1):
            typ = ABILITIES[kind][slot][0]
            ready = e.cd[:, team, k, slot] <= 0
            if typ in OFFENSIVE:
                use = ready & (d <= SPELL_RANGE)
            elif typ == "heal":
                use = ready & hurt_ally
            elif typ == "shield":
                use = ready & hurt_self & (d <= 3)
            elif typ == "reset":
                use = ready & in_rng
            else:  # dash: not used by greedy
                use = torch.zeros_like(ready)
            out[:, k, 2] = torch.where((out[:, k, 2] == 0) & use, slot + 1, out[:, k, 2])
    return out


def outcome(e: Engine, deaths0: torch.Tensor) -> torch.Tensor:
    """[B] A minus B: health fraction (dead = 0) + 0.5 per enemy death since deaths0."""
    frac = torch.where(e.alive, e.hp / e.max_hp, 0.0).sum(-1)  # [B, 2]
    dd = (e.deaths - deaths0).sum(-1).float()  # [B, 2]
    return (frac[:, 0] - frac[:, 1]) + 0.5 * (dd[:, 1] - dd[:, 0])


def search(e: Engine, team: int, depth: int, dets: int, know_cd: bool) -> torch.Tensor:
    """long [B, T, 4]: per champion of `team`, the candidate with the best mean rollout."""
    B, T, dev = e.B, e.T, e.dev
    C = len(CANDIDATES)
    acts = greedy(e, team)
    cand = CANDIDATES.to(dev)
    sign = 1.0 if team == 0 else -1.0
    for k in range(T):
        idx = torch.arange(B, device=dev).repeat_interleave(C * dets)
        n = clone(e, idx)
        if not know_cd:  # the enemy's cooldowns are hidden: sample them
            n.cd[:, 1 - team] = torch.randint(0, 9, n.cd[:, 1 - team].shape, device=dev)
        d0 = n.deaths.clone()
        mine = acts[idx].clone()
        mine[:, k] = cand.repeat_interleave(dets, 0).repeat(B, 1)  # row = (game, candidate, det)
        for r in range(depth):
            a = torch.zeros(n.B, 2, T, 4, dtype=torch.long, device=dev)
            a[:, team] = mine if r == 0 else greedy(n, team)
            a[:, 1 - team] = greedy(n, 1 - team)
            n.step(a)
        v = (sign * outcome(n, d0)).view(B, C, dets).mean(-1)
        acts[:, k] = cand[v.argmax(1)]
    return acts


def scenarios(md, n: int, rng: random.Random) -> tuple[list[list[int]], list]:
    kinds = [rng.sample(range(13), 3), rng.sample(range(13), 3)]
    walk = ~(md.walls | md.holes | md.altars | md.spawn_a | md.spawn_b)
    free = [(x, y) for y in range(md.h) for x in range(md.w) if walk[y, x]]
    pos = []
    while len(pos) < n:
        cx, cy = rng.choice(free)
        a = [(cx - 3 + rng.randint(-1, 1), cy + rng.randint(-2, 2)) for _ in range(3)]
        b = [(cx + 3 + rng.randint(-1, 1), cy + rng.randint(-2, 2)) for _ in range(3)]
        tiles = a + b
        if len(set(tiles)) == 6 and all(0 <= x < md.w and 0 <= y < md.h and walk[y, x] for x, y in tiles):
            pos.append((a, b))
    return kinds, pos


def play(md, kinds, pos, pa: str, pb: str, rounds: int, depth: int, dets: int, know_cd: bool, dev) -> torch.Tensor:
    B = len(pos)
    e = Engine(md, B, kinds, dev)
    e.camp_alive[:] = False
    e.camp_awake[:] = False
    e.g_state[:] = 2
    e.g_timer[:] = 10**6
    for t in range(2):
        for k in range(3):
            e.pos[:, t, k] = torch.tensor([p[t][k] for p in pos], device=dev)
    d0 = e.deaths.clone()
    for _ in range(rounds):
        a = torch.zeros(B, 2, 3, 4, dtype=torch.long, device=dev)
        for t, p in ((0, pa), (1, pb)):
            a[:, t] = search(e, t, depth, dets, know_cd) if p == "search" else greedy(e, t)
        e.step(a)
    return outcome(e, d0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", type=int, default=256)
    ap.add_argument("--comps", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--dets", type=int, default=8)
    ap.add_argument("--know-cd", action="store_true")
    ap.add_argument("--map", default="00fragment")
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    md = maps.load(a.map)
    rng = random.Random(7)
    res = {"greedy-greedy": [], "search-greedy": [], "greedy-search": []}
    t0 = time.time()
    for c in range(a.comps):
        kinds, pos = scenarios(md, a.scenarios // a.comps, rng)
        for key in res:
            pa, pb = key.split("-")
            v = play(md, kinds, pos, pa, pb, a.rounds, a.depth, a.dets, a.know_cd, dev)
            res[key].append(v.cpu().numpy())
        print(f"comp {c} {kinds}: " + ", ".join(
            f"{k} A-wins {np.mean(res[k][-1] > 0):.2f}" for k in res), f"({time.time() - t0:.0f}s)", flush=True)
    for k, v in res.items():
        v = np.concatenate(v)
        print(f"{k:14s} A wins {np.mean(v > 0):.3f}  A margin {v.mean():+.3f}  ({len(v)} fights)")
    sg = np.concatenate(res["search-greedy"])
    gs = np.concatenate(res["greedy-search"])
    print(f"search's win rate over both sides: {(np.mean(sg > 0) + np.mean(gs < 0)) / 2:.3f}")


if __name__ == "__main__":
    main()
