"""Rule tests for the GPU engine (docs/GameRules.md). Run in the torch venv:

    python -m pytest nn/gpusim/test_engine.py -q      (or: python nn/gpusim/test_engine.py)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nn.gpusim.engine import Engine  # noqa: E402
from nn.gpusim.maps import Camp, Group, MapData  # noqa: E402

DEV = "cpu"


def tiny(w=12, h=12, walls=(), altars=(), groups=()) -> MapData:
    z = lambda: np.zeros((h, w), bool)  # noqa: E731
    wl, al, sa, sb = z(), z(), z(), z()
    for x, y in walls:
        wl[y, x] = True
    for x, y in altars:
        al[y, x] = True
    sa[0, 0] = sa[0, 1] = True
    sb[h - 1, w - 1] = sb[h - 1, w - 2] = True
    return MapData("tiny", w, h, wl, z(), z(), al, sa, sb, list(groups))


def eng(md, kinds=((0,), (0,)), B=2) -> Engine:
    return Engine(md, B, [list(kinds[0]), list(kinds[1])], DEV)


def act(e: Engine, team, k, move=0, attack=0, spell=0, special=0):
    a = torch.zeros(e.B, 4, dtype=torch.long)
    a[:] = torch.tensor([move, attack, spell, special])
    e.champion_turn(team, k, a)


def put(e: Engine, team, k, x, y):
    e.pos[:, team, k] = torch.tensor([x, y])


def test_move_legal_and_blocked():
    e = eng(tiny(walls=[(5, 6)]))
    put(e, 0, 0, 5, 5)
    act(e, 0, 0, move=1)  # +y
    assert e.pos[0, 0, 0].tolist() == [5, 5], "wall blocks"
    act(e, 0, 0, move=3)  # +x
    assert e.pos[0, 0, 0].tolist() == [6, 5]
    put(e, 1, 0, 7, 5)
    act(e, 0, 0, move=3)
    assert e.pos[0, 0, 0].tolist() == [6, 5], "occupied tile blocks"
    put(e, 0, 0, 10, 10)
    act(e, 0, 0, move=2)  # onto (11, 11): enemy spawn
    assert e.pos[0, 0, 0].tolist() == [10, 10], "enemy spawn blocks"


def test_attack_ranges_and_ranged_rule():
    e = eng(tiny(), kinds=((0,), (1,)))  # Ryth melee (45 str) vs Vesper ranged (50 str)
    put(e, 0, 0, 5, 5)
    put(e, 1, 0, 7, 5)
    hp = e.hp[0, 1, 0].item()
    act(e, 0, 0, attack=1)
    assert e.hp[0, 1, 0].item() == hp, "melee range 1"
    act(e, 0, 0, move=3, attack=1)  # melee may move then attack
    assert e.hp[0, 1, 0].item() == hp - 45
    put(e, 1, 0, 8, 5)
    hp0 = e.hp[0, 0, 0].item()
    act(e, 1, 0, move=7, attack=1)  # ranged: attacking forbids the move this turn
    assert e.pos[0, 1, 0].tolist() == [8, 5]
    assert e.hp[0, 0, 0].item() == hp0 - 50, "ranged range 3"


def test_camp_wakes_hits_nearest_then_sleeps():
    g = Group([Camp(0, 6, 5)], [(6, 5)], 4, 40, 40)
    e = eng(tiny(groups=[g]))
    put(e, 0, 0, 5, 5)
    put(e, 1, 0, 10, 10)
    act(e, 0, 0, attack=2)
    assert e.camp_awake[0, 0] and e.camp_hp[0, 0].item() == 1350 - 45
    hp = e.hp[0, 0, 0].item()
    e.camp_actions()
    assert e.hp[0, 0, 0].item() == hp - 20, "wolf hits the nearest champion for 20"
    put(e, 0, 0, 0, 11)  # far away: no champion in range
    e.camp_actions()
    e.camp_actions()
    assert not e.camp_awake[0, 0], "asleep after two quiet rounds"


def test_clear_capture_split_and_respawn():
    g = Group([Camp(2, 6, 5)], [(6, 5)], 7, 0, 50)  # gem cluster: 120 hp, 7 gems
    e = eng(tiny(groups=[g]), kinds=((0, 1), (0,)) if False else ((0,), (0,)))
    put(e, 0, 0, 5, 5)
    for _ in range(3):
        act(e, 0, 0, attack=2)
    assert not e.camp_alive[0, 0] and e.g_state[0, 0] == 1
    act(e, 0, 0, move=3)  # onto the spawn tile
    for _ in range(4):
        e.captures()
    assert e.gems[0, 0, 0].item() == 0, "five capture phases needed"
    e.captures()
    assert e.gems[0, 0, 0].item() == 7 and e.g_state[0, 0] == 2
    act(e, 0, 0, move=7)
    for _ in range(49):
        e.camp_respawns()
    assert not e.camp_alive[0, 0]
    e.camp_respawns()
    assert e.camp_alive[0, 0], "respawns after its timer"


def test_contested_capture_resets():
    g = Group([Camp(2, 6, 5)], [(6, 5), (6, 6)], 7, 0, 50)
    e = eng(tiny(groups=[g]))
    e.camp_alive[:] = False
    e.g_state[:] = 1
    put(e, 0, 0, 6, 5)
    e.captures()
    e.captures()
    put(e, 1, 0, 6, 6)
    e.captures()
    assert e.g_timer[0, 0] == 0, "an enemy on the region resets it"


def test_tribute_timing():
    e = eng(tiny(altars=[(5, 6)]))
    put(e, 0, 0, 5, 5)
    e.gems[:, 0, 0] = 15.0  # 2 ticks
    act(e, 0, 0, special=2)  # start; tick at this turn's end -> 1
    assert e.tribute[0, 0, 0] == 1 and e.tributed[0, 0] == 0
    act(e, 0, 0)  # turn end -> 0
    assert e.tributed[0, 0] == 0
    act(e, 0, 0)  # turn start with 0: completes
    assert e.tributed[0, 0].item() == 15 and e.gems[0, 0, 0].item() == 0


def test_kill_loot_and_respawn():
    e = eng(tiny(), kinds=((2,), (3,)))  # Vendetta (70 str) vs Cassi (600 hp)
    put(e, 0, 0, 5, 5)
    put(e, 1, 0, 6, 5)
    e.gems[:, 1, 0] = 12.0
    for _ in range(9):
        act(e, 0, 0, attack=1)
    assert not e.alive[0, 1, 0]
    assert e.gems[0, 0, 0].item() == 12 and e.gold[0, 0, 0].item() == 100
    for _ in range(59):
        act(e, 1, 0)
    assert not e.alive[0, 1, 0]
    act(e, 1, 0)
    act(e, 1, 0)
    assert e.alive[0, 1, 0], "respawns after 60 owner turn ends"


def test_win_at_1000():
    e = eng(tiny())
    e.tributed[:, 1] = 1000.0
    e.step(torch.zeros(e.B, 2, e.T, 4, dtype=torch.long))
    assert e.done.all() and (e.winner == 1).all()


def test_turn_order_interleaves():
    e = eng(tiny(), kinds=((0, 0), (0, 0)))
    seen = []
    orig = e.champion_turn
    e.champion_turn = lambda team, k, a: (seen.append((team, k)), orig(team, k, a))
    z = torch.zeros(e.B, 2, e.T, 4, dtype=torch.long)
    e.step(z)
    e.step(z)
    assert seen == [(0, 0), (1, 0), (0, 1), (1, 1), (1, 0), (0, 0), (1, 1), (0, 1)]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
