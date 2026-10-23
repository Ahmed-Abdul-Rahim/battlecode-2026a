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
    assert e.hp[0, 0, 0].item() == hp0 - 55, "ranged range 3 (Vesper 55 strength, live)"


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


def cast(e, team, k, slot, kind=1, idx=0, xy=(0, 0)):
    m = torch.ones(e.B, dtype=torch.bool)
    return e.cast(team, k, slot, m, torch.full((e.B,), kind), torch.full((e.B,), idx), torch.tensor([xy] * e.B))


def test_stellar_damage_and_aoe():
    g = Group([Camp(0, 7, 5), Camp(0, 8, 5)], [(7, 5), (8, 5)], 4, 40, 40)
    e = eng(tiny(groups=[g]), kinds=((5,), (0,)))  # Stellar: 2 x 70 single, 1.4 x 70 area (live)
    put(e, 0, 0, 5, 5)
    e.turn_start(0, 0)
    assert cast(e, 0, 0, 0, kind=2, idx=0).all()
    assert e.camp_hp[0, 0].item() == 1350 - 140
    assert not cast(e, 0, 0, 1, kind=0, xy=(8, 5)).any(), "one cast per turn"
    e.turn_end(0, 0)
    e.turn_start(0, 0)
    assert cast(e, 0, 0, 1, kind=0, xy=(8, 5)).all()
    assert abs(e.camp_hp[0, 1].item() - (1350 - 98)) < 1e-3 and abs(e.camp_hp[0, 0].item() - (1210 - 98)) < 1e-3


def test_stun_stops_camp_and_marcella_is_immune():
    g = Group([Camp(0, 6, 5)], [(6, 5)], 4, 40, 40)
    e = eng(tiny(groups=[g]), kinds=((6,), (7,)))  # Prism vs Marcella
    put(e, 0, 0, 5, 5)
    put(e, 1, 0, 6, 6)
    e.turn_start(0, 0)
    cast(e, 0, 0, 0, kind=2, idx=0)  # damage + stun the wolf
    hp = e.hp[0, 0, 0].item()
    e.camp_actions()
    assert e.hp[0, 0, 0].item() == hp, "stunned camp does not attack"
    e.turn_end(0, 0)
    e.turn_start(0, 0)
    cast(e, 0, 0, 0, kind=1, idx=0)
    assert e.stun[0, 1, 0] == 0, "Marcella is immune to stuns"


def test_shield_heal_splash_and_vendetta_reset():
    e = eng(tiny(), kinds=((10, 8), (2, 0)))  # Varg, Willow vs Vendetta, Ryth
    put(e, 0, 0, 5, 5)
    put(e, 0, 1, 4, 4)
    put(e, 1, 0, 6, 5)
    put(e, 1, 1, 7, 5)
    e.turn_start(0, 0)
    cast(e, 0, 0, 1)  # Varg shield 2.5 x 100
    assert e.shield[0, 0, 0].item() == 250
    e.basic_attack(0, 0, torch.ones(e.B, dtype=torch.long), torch.zeros(e.B, dtype=torch.long), torch.ones(e.B, dtype=torch.bool))
    assert e.hp[0, 1, 0].item() == 1000 - 40 and e.hp[0, 1, 1].item() == 1200 - 40, "Varg splash"
    hp = e.hp[0, 0, 0].item()
    e.apply_damage(torch.tensor([[[100.0, 0.0], [0.0, 0.0]]] * e.B), (1, 0))
    assert e.hp[0, 0, 0].item() == hp and e.shield[0, 0, 0].item() == 150, "shield absorbs"
    e.hp[:, 0, 0] = 500.0
    e.turn_start(0, 1)
    cast(e, 0, 1, 0, kind=1, idx=0)  # Willow heal 3 x 120
    assert e.hp[0, 0, 0].item() == 860
    e.hp[:, 0, 0] = 30.0
    e.shield[:, 0, 0] = 0.0
    e.cd[:, 1, 0] = 5
    e.turn_start(1, 0)
    e.basic_attack(1, 0, torch.ones(e.B, dtype=torch.long), torch.zeros(e.B, dtype=torch.long), torch.ones(e.B, dtype=torch.bool))
    assert not e.alive[0, 0, 0] and (e.cd[0, 1, 0] == 0).all(), "Vendetta kill resets cooldowns"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
