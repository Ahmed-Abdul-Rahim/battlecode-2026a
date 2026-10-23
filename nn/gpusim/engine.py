"""GPU engine v0: N games of one map in lockstep, every rule a batched tensor op (torch).

Faithful to docs/GameRules.md where implemented; simplifications are listed in SIMPLIFIED.
Self-play only: both teams are driven by actions given to `step` (networks on the GPU), no
rule-bot code runs here. Results must be re-checked on the official engine.

State is a dict of tensors with a leading batch dimension B. Champions are [B, 2, T] (team,
slot), camps [B, C], groups [B, G].

Per round (rules: "The exact turn order"):
  round start: camp respawns -> captures -> camp actions (awake camps hit the nearest champion)
  champion turns: the two teams' champions interleave (A1 B1 A2 B2 ...), the starting team
  swaps every round
  each champion turn: respawn / channel completion at turn start, then its action, then
  turn-end ticks (cooldowns, channels), spawn heal, shop.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from nn.gpusim.maps import MapData

SIMPLIFIED = [
    "vision: Chebyshev radius 8, walls/bushes ignored (no shadowcasting yet)",
    "no wards, consumables, altar locks, teleport crystals, emblem passives",

    "shop: on spawn, gold buys alternating strength/health shards (30 gold each)",
    "camps keep their health while asleep",
]

# engine CampType: wolf, elk, gem cluster, gold cluster, monolith, golem
CAMP_HP = [1350.0, 2400.0, 120.0, 120.0, 20000.0, 5000.0]
CAMP_STR = [20.0, 60.0, 0.0, 0.0, 200.0, 120.0]
CAMP_RANGE = [3, 4, 0, 0, 5, 4]
# Champion kits: (ranged, range, health, health growth, strength, strength growth, focus, focus
# growth) by kind index (bc_core::champion::ChampionKind order). LIVE engine values (debug "K"
# dump of champion_designs(), cambc 3.0.5, 2026-10-23): several differ from the handbook
# (Stellar 650 hp / 70 focus, Vesper 1000 hp / 55 str, Vendetta 1000 hp, Tavi 900 hp / 110
# focus, Heart of Ruin 30 str). Camp-damage multipliers are all 1.
KITS = [
    (False, 1, 1200, 100, 45, 5, 100, 10),  # Ryth
    (True, 3, 1000, 100, 55, 5, 100, 10),  # Vesper Vale
    (False, 1, 1000, 100, 70, 6, 100, 10),  # Vendetta
    (True, 3, 600, 65, 30, 3, 70, 13),  # Cassi
    (False, 1, 900, 80, 40, 3, 100, 10),  # Issac
    (True, 3, 650, 65, 30, 3.5, 70, 7),  # Stellar
    (True, 3, 700, 60, 35, 2.5, 80, 8),  # Prism
    (False, 1, 1200, 100, 45, 5, 60, 15),  # Marcella
    (True, 3, 700, 50, 55, 3, 120, 5),  # Willow
    (False, 1, 900, 100, 35, 5, 100, 9),  # Ren
    (False, 1, 1000, 100, 40, 5, 100, 10),  # Varg
    (True, 3, 900, 75, 45, 5, 110, 10),  # Tavi
    (False, 1, 1000, 100, 30, 3.5, 100, 12),  # Heart of Ruin
]
# Abilities per kind: two slots of (type, focus modifier, range, radius). LIVE values (debug "A"
# dump, runs/nn/gpusim/live_kits.txt): Ren 0.75 (handbook 1.25), Heart of Ruin dash 1.5 (1.75),
# all cooldowns 8.
# Types: dmg (unit), stun (unit: damage + one-round stun), aoe (ground radius), aoe_stun,
# heal (ally), shield (self, 5 turns), reset (attack reset), mreset (Marcella: reset + the next
# basic attack stuns), dash (ground 2), hordash (dash + 1.5 x focus around the landing + stun).
ABILITIES = [
    [("aoe_stun", 1.0, 4, 1), ("reset", 0, 0, 0)],  # Ryth
    [("reset", 0, 0, 0), ("dash", 0, 2, 0)],  # Vesper Vale
    [("reset", 0, 0, 0), ("dash", 0, 2, 0)],  # Vendetta
    [("dmg", 2.0, 4, 0), ("stun", 0.0, 4, 0)],  # Cassi
    [("reset", 0, 0, 0), ("dash", 0, 2, 0)],  # Issac
    [("dmg", 2.0, 4, 0), ("aoe", 1.4, 4, 1)],  # Stellar
    [("stun", 1.5, 4, 0), ("aoe", 1.0, 4, 1)],  # Prism
    [("mreset", 0, 0, 0), ("dmg", 2.0, 4, 0)],  # Marcella
    [("heal", 3.0, 4, 0), ("reset", 0, 0, 0)],  # Willow
    [("stun", 0.75, 4, 0), ("dash", 0, 2, 0)],  # Ren
    [("reset", 0, 0, 0), ("shield", 2.5, 0, 0)],  # Varg
    [("reset", 0, 0, 0), ("stun", 1.4, 4, 0)],  # Tavi
    [("shield", 1.5, 0, 0), ("hordash", 1.5, 2, 1)],  # Heart of Ruin
]
# Abilities that can't share a turn with a walk.
EXCLUDES_WALK = {"dmg", "stun", "aoe", "aoe_stun"}
# ... except these (kind, slot): Cassi's stun may share a turn with a walk (live).
WALK_OK = {(3, 1)}
RYTH, VESPER, VENDETTA, PRISM, MARCELLA, REN, VARG, HOR = 0, 1, 2, 6, 7, 9, 10, 12
DIRS = torch.tensor([[0, 0], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0], [-1, 1]])
WIN_GEMS, MAX_ROUNDS = 1000, 2000
RESPAWN_ROUNDS, CAPTURE_ROUNDS, KILL_GOLD, SHARD_COST = 60, 5, 100, 30
VISION, SPELL_RANGE, SPELL_CD, RECALL_TURNS = 8, 4, 8, 5
# Item families (cambc::ItemType = 6 x family + [str shard, str emblem, focus shard, focus emblem,
# health shard, health emblem]); stats are the champion's growth per shard.
CAMP_SLEEP, CAMP_SLEEP_ROUNDS = "no_target", 1  # validation experiments
CAMP_METRIC = "cheb"
CAMP_TARGET = "nearest"  # or "aggro_then_nearest" / "aggro_only" (validation experiments)
MONSTER, POWER, DURABILITY, TREASURE, JUNGLE, HOME, BRUISER = range(7)
ITEM_STAT = (0, 0, 1, 1, 2, 2)  # strength, focus, health
# Action layout per champion: move (9) x attack target (3: none, nearest enemy, nearest camp)
# x spell (3: none, slot 0, slot 1; aimed at the attack target, or self/ally/ground as the
# ability needs) x special (3: none, recall, tribute) -> four integer heads.
ACT_MOVE, ACT_ATTACK, ACT_SPELL, ACT_SPECIAL = 9, 3, 3, 3


@dataclass
class Static:
    """Per-map tensors shared by every game of the batch."""

    h: int
    w: int
    walkable: torch.Tensor  # bool [H, W]: not wall / hole / altar
    spawn: torch.Tensor  # bool [2, H, W]
    near_altar: torch.Tensor  # bool [H, W]: Chebyshev <= 1 of an altar
    camp_xy: torch.Tensor  # long [C, 2]
    camp_kind: torch.Tensor  # long [C]
    camp_group: torch.Tensor  # long [C]
    group_gems: torch.Tensor  # float [G]
    group_gold: torch.Tensor  # float [G]
    group_respawn: torch.Tensor  # long [G]
    group_tile: torch.Tensor  # long [H, W]: group whose spawn tile this is, -1 none
    spawn_tiles: list[torch.Tensor]  # per team: long [S, 2]


def make_static(md: MapData, dev: torch.device) -> Static:
    walk = ~(md.walls | md.holes | md.altars)
    alt = md.altars
    near = np.zeros_like(alt)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            near |= np.roll(np.roll(alt, dy, 0), dx, 1)
    camps = [(c, gi) for gi, g in enumerate(md.groups) for c in g.camps]
    gt = -np.ones((md.h, md.w), np.int64)
    for gi, g in enumerate(md.groups):
        for x, y in g.tiles:
            gt[y, x] = gi
    t = lambda a, dt=torch.long: torch.as_tensor(np.asarray(a), dtype=dt, device=dev)  # noqa: E731
    spawns = [md.spawn_a, md.spawn_b]
    return Static(
        md.h, md.w, t(walk, torch.bool), t(np.stack(spawns), torch.bool), t(near, torch.bool),
        t(np.array([[c.x, c.y] for c, _ in camps], np.int64).reshape(-1, 2)), t(np.array([c.kind for c, _ in camps], np.int64)),
        t(np.array([gi for _, gi in camps], np.int64)),
        t(np.array([g.gems for g in md.groups], np.float32), torch.float32),
        t(np.array([g.gold for g in md.groups], np.float32), torch.float32),
        t(np.array([max(g.respawn, 1) for g in md.groups], np.int64)), t(gt),
        [t(np.argwhere(s)[:, ::-1].copy()) for s in spawns],
    )


class Engine:
    def __init__(self, md: MapData, batch: int, kinds: list[list[int]], dev: str | torch.device = "cuda"):
        self.dev = torch.device(dev)
        self.s = make_static(md, self.dev)
        self.B, self.T = batch, len(kinds[0])
        self.kinds = torch.as_tensor(kinds, device=self.dev)  # [2, T]
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self) -> None:
        B, T, dev, s = self.B, self.T, self.dev, self.s
        k = torch.as_tensor(KITS, dtype=torch.float32, device=dev)[self.kinds]  # [2, T, 8]
        ex = lambda v: v.unsqueeze(0).expand(B, *v.shape).clone()  # noqa: E731
        self.ranged = ex(k[..., 0] > 0)
        self.rng = ex(k[..., 1].long())
        self.max_hp = ex(k[..., 2])
        self.hp_growth = ex(k[..., 3])
        self.strength = ex(k[..., 4])
        self.str_growth = ex(k[..., 5])
        self.focus = ex(k[..., 6])
        self.focus_growth = ex(k[..., 7])
        self.base_hp = self.max_hp.clone()
        self.hp = self.max_hp.clone()
        self.pos = torch.zeros(B, 2, T, 2, dtype=torch.long, device=dev)
        for team in range(2):
            tiles = s.spawn_tiles[team]
            self.pos[:, team] = tiles[torch.arange(T, device=dev) % len(tiles)]
        z = lambda dt=torch.float32: torch.zeros(B, 2, T, dtype=dt, device=dev)  # noqa: E731
        self.alive = torch.ones(B, 2, T, dtype=torch.bool, device=dev)
        self.respawn = z(torch.long)
        self.gems, self.gold = z(), z()
        self.cd = torch.zeros(B, 2, T, 2, dtype=torch.long, device=dev)  # per ability slot
        self.stun = z(torch.long)  # turns this champion still can't act
        self.shield, self.shield_t = z(), z(torch.long)
        self.can_attack = torch.ones(B, 2, T, dtype=torch.bool, device=dev)
        self.can_move = torch.ones(B, 2, T, dtype=torch.bool, device=dev)
        self.walked = torch.zeros(B, 2, T, dtype=torch.bool, device=dev)
        self.cast_done = torch.zeros(B, 2, T, dtype=torch.bool, device=dev)
        self.charged = torch.zeros(B, 2, T, dtype=torch.bool, device=dev)  # Marcella's stun attack
        self.recall = z(torch.long) - 1  # -1: not channelling
        self.tribute = z(torch.long) - 1
        self.shards = z(torch.long)
        self.fam_shards = torch.zeros(B, 2, T, 7, 3, dtype=torch.long, device=dev)
        self.emb = torch.zeros(B, 2, T, 7, device=dev)  # 1 = this family's emblem is complete
        self.kills, self.deaths = z(torch.long), z(torch.long)
        self.camp_dmg_done, self.camp_dmg_taken = z(), z()  # statistics (validation)
        C, G = len(s.camp_kind), len(s.group_gems)
        self.camp_max = torch.as_tensor(CAMP_HP, device=dev)[s.camp_kind]
        self.camp_str = torch.as_tensor(CAMP_STR, device=dev)[s.camp_kind]
        self.camp_rng = torch.as_tensor(CAMP_RANGE, device=dev)[s.camp_kind]
        self.camp_hp = self.camp_max.unsqueeze(0).expand(B, C).clone()
        self.camp_alive = torch.ones(B, C, dtype=torch.bool, device=dev)
        self.camp_awake = torch.zeros(B, C, dtype=torch.bool, device=dev)
        self.camp_quiet = torch.zeros(B, C, dtype=torch.long, device=dev)
        self.camp_aggro = torch.full((B, C), -1, dtype=torch.long, device=dev)  # last damager (team * T + k)
        self.camp_stun = torch.zeros(B, C, dtype=torch.long, device=dev)
        # group: 0 alive, 1 cleared (capturable), 2 respawning
        self.g_state = torch.zeros(B, G, dtype=torch.long, device=dev)
        self.g_timer = torch.zeros(B, G, dtype=torch.long, device=dev)
        self.g_team = torch.full((B, G), -1, dtype=torch.long, device=dev)
        self.tributed = torch.zeros(B, 2, device=dev)
        self.submitted = torch.zeros(B, 2, self.T, device=dev)  # per champion (statistics)
        self.lifetime_gold = torch.zeros(B, 2, device=dev)
        self.earned = torch.zeros(B, 2, self.T, device=dev)  # per champion (statistics)
        self.round = 0
        self.done = torch.zeros(B, dtype=torch.bool, device=dev)
        self.winner = torch.full((B,), -1, dtype=torch.long, device=dev)
        self.damaged_this_round = torch.zeros(B, 2, T, dtype=torch.bool, device=dev)

    # ------------------------------------------------------------------ helpers
    def occupied(self) -> torch.Tensor:
        """bool [B, H, W]: tiles with a living champion or camp."""
        s, B = self.s, self.B
        occ = torch.zeros(B, s.h, s.w, dtype=torch.bool, device=self.dev)
        bi = torch.arange(B, device=self.dev)
        p = self.pos.view(B, -1, 2)
        a = self.alive.view(B, -1)
        bb = bi[:, None].expand_as(a)
        occ[bb[a], p[..., 1][a], p[..., 0][a]] = True
        cx, cy = s.camp_xy[:, 0], s.camp_xy[:, 1]
        bc = bi[:, None].expand_as(self.camp_alive)
        occ[bc[self.camp_alive], cy.expand_as(self.camp_alive)[self.camp_alive], cx.expand_as(self.camp_alive)[self.camp_alive]] = True
        return occ

    @staticmethod
    def cheb(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return (a - b).abs().amax(-1)

    def kill(self, team: int, slot: int, killer: tuple[int, int] | None, mask: torch.Tensor) -> None:
        """Champion (team, slot) dies in games `mask`; a champion killer gets its gems + 100 gold
        (a camp kill destroys the gems)."""
        if not mask.any():
            return
        if killer is not None:
            kt, ks = killer
            self.gems[:, kt, ks] += torch.where(mask, self.gems[:, team, slot], 0.0)
            kg = torch.where(mask, float(KILL_GOLD) * self.gold_mult(kt, ks), 0.0)
            self.gold[:, kt, ks] += kg
            self.lifetime_gold[:, kt] += kg
            self.earned[:, kt, ks] += kg
            self.kills[:, kt, ks] += mask.long()
        self.gems[:, team, slot] = torch.where(mask, 0.0, self.gems[:, team, slot])
        self.alive[:, team, slot] &= ~mask
        self.deaths[:, team, slot] += mask.long()
        self.respawn[:, team, slot] = torch.where(mask, RESPAWN_ROUNDS, self.respawn[:, team, slot])
        self.recall[:, team, slot] = torch.where(mask, -1, self.recall[:, team, slot])
        self.tribute[:, team, slot] = torch.where(mask, -1, self.tribute[:, team, slot])

    # ------------------------------------------------------------------ round phases
    def camp_respawns(self) -> None:
        resp = self.g_state == 2
        self.g_timer = torch.where(resp, self.g_timer - 1, self.g_timer)
        ready = resp & (self.g_timer <= 0)
        if ready.any():
            back = ready[:, self.s.camp_group]  # [B, C]
            self.camp_alive |= back
            self.camp_hp = torch.where(back, self.camp_max, self.camp_hp)
            self.camp_awake &= ~back
            self.g_state = torch.where(ready, 0, self.g_state)

    def captures(self) -> None:
        s = self.s
        cleared = self.g_state == 1
        if not cleared.any():
            return
        # group of the tile each living champion stands on
        gt = s.group_tile[self.pos[..., 1], self.pos[..., 0]]  # [B, 2, T]
        gt = torch.where(self.alive, gt, -1)
        G = s.group_gems.shape[0]
        on = torch.zeros(self.B, 2, G + 1, dtype=torch.long, device=self.dev)
        on.scatter_add_(2, gt + 1, torch.ones_like(gt))
        on = on[..., 1:] > 0  # [B, 2, G]
        uncontested = on[:, 0] ^ on[:, 1]
        team = torch.where(on[:, 0], 0, 1)
        progress = cleared & uncontested & ((self.g_team == team) | (self.g_team < 0))
        self.g_team = torch.where(cleared & uncontested, team, torch.where(cleared, -1, self.g_team))
        self.g_timer = torch.where(progress, self.g_timer + 1, torch.where(cleared, 0, self.g_timer))
        # the first phase starts the five-round timer, the next five count it down (validated)
        done = progress & (self.g_timer > CAPTURE_ROUNDS)
        if done.any():
            for t in range(2):
                win = done & (team == t)
                if not win.any():
                    continue
                present = (gt.unsqueeze(-1) == torch.arange(G, device=self.dev)) & win.unsqueeze(1).unsqueeze(1)
                present = present[:, t]  # [B, T, G]
                n = present.sum(1).clamp(min=1).float()  # [B, G]
                share_g = torch.where(win, s.group_gems / n, 0.0)
                share_o = torch.where(win, s.group_gold / n, 0.0)
                self.gems[:, t] += (present.float() * share_g.unsqueeze(1)).sum(-1).floor()
                # Vesper +33%, Treasure emblem +20%
                mult = 1.0 + 0.33 * (self.kinds[t] == VESPER).float().view(1, -1) + 0.2 * self.emb[:, t, :, TREASURE]
                gold = ((present.float() * share_o.unsqueeze(1)).sum(-1) * mult).floor()
                self.gold[:, t] += gold
                self.lifetime_gold[:, t] += gold.sum(-1)
                self.earned[:, t] += gold
            self.g_state = torch.where(done, 2, self.g_state)
            self.g_timer = torch.where(done, s.group_respawn, self.g_timer)
            self.g_team = torch.where(done, -1, self.g_team)

    def camp_actions(self) -> None:
        """Awake, unstunned camps hit the nearest living champion in range (lowest index on ties)."""
        s, B = self.s, self.B
        stunned = self.camp_stun > 0
        self.camp_stun = (self.camp_stun - 1).clamp(min=0)
        act = self.camp_alive & self.camp_awake & (self.camp_str > 0) & ~stunned
        if not act.any():
            return
        cp = s.camp_xy.view(1, -1, 1, 2)  # [1, C, 1, 2]
        chp = self.pos.view(B, 1, -1, 2)  # [B, 1, 2T, 2]
        if CAMP_METRIC == "euclid":  # squared distance against squared range
            dd = (cp - chp).pow(2).sum(-1)
            ok = self.alive.view(B, 1, -1) & (dd <= self.camp_rng.view(1, -1, 1).pow(2)) & act.unsqueeze(-1)
            d = dd
        else:
            d = self.cheb(cp, chp)  # [B, C, 2T]
            ok = self.alive.view(B, 1, -1) & (d <= self.camp_rng.view(1, -1, 1)) & act.unsqueeze(-1)
        d = torch.where(ok, d, 10_000)
        tgt = d.argmin(-1)  # [B, C]
        hit = ok.any(-1)
        if CAMP_TARGET != "nearest":
            ag = self.camp_aggro.clamp(min=0)
            ag_ok = (self.camp_aggro >= 0) & ok.gather(2, ag.unsqueeze(-1)).squeeze(-1)
            tgt = torch.where(ag_ok, ag, tgt)
            hit = ag_ok | (hit & (CAMP_TARGET == "aggro_then_nearest"))
        dmg = torch.zeros(B, 2 * self.T, device=self.dev)
        dmg.scatter_add_(1, tgt, torch.where(hit, self.camp_str.expand_as(hit), 0.0))
        dmg = dmg.view(B, 2, self.T) * (1 - 0.3 * self.emb[..., JUNGLE])
        self.camp_dmg_taken += dmg * (1 - 0.1 * self.emb[..., DURABILITY])
        # Ren retaliates against the camps that hit him (0.2 x focus each)
        ren = (self.kinds == REN).view(-1)
        if ren.any():
            ren_hit = hit & ren[tgt]  # [B, C]
            if ren_hit.any():
                back = torch.where(ren_hit, 0.2 * self.focus.view(B, -1).gather(1, tgt), 0.0)
                eff = torch.minimum(back, self.camp_hp.clamp(min=0)) * self.camp_alive
                self.camp_dmg_done.view(B, -1).scatter_add_(1, tgt, eff)
                self.damage_camps(back)
        self.apply_damage(dmg, None)
        # a camp with no champion in range for 2 rounds (and no damage taken) goes back to sleep
        if CAMP_SLEEP == "no_target":
            self.camp_quiet = torch.where(act & ~hit, self.camp_quiet + 1, torch.where(hit, 0, self.camp_quiet))
        else:  # rounds without taking damage (reset in damage_camps)
            self.camp_quiet = torch.where(self.camp_awake, self.camp_quiet + 1, self.camp_quiet)
        self.camp_awake &= ~(self.camp_quiet > CAMP_SLEEP_ROUNDS)

    def apply_damage(self, dmg: torch.Tensor, killer: tuple[int, int] | None) -> None:
        """dmg [B, 2, T] to champions: shields absorb first, damage cancels recalls, Ren
        retaliates against a champion attacker (0.2 x focus), deaths resolved."""
        hit = dmg > 0
        dmg = dmg * (1 - 0.1 * self.emb[..., DURABILITY])
        absorbed = torch.minimum(self.shield, dmg)
        self.shield = self.shield - absorbed
        dmg = dmg - absorbed
        self.hp = self.hp - dmg
        self.recall = torch.where(hit, -1, self.recall)
        if killer is not None:
            kt, ks = killer
            ren = hit & (self.kinds == REN).unsqueeze(0)
            if ren.any():
                back = (ren.float() * 0.2 * self.focus).sum((1, 2))  # [B]
                rd = torch.zeros_like(self.hp)
                rd[:, kt, ks] = back
                absorbed = torch.minimum(self.shield, rd)
                self.shield = self.shield - absorbed
                self.hp = self.hp - (rd - absorbed)
        dead = self.alive & (self.hp <= 0)
        if not dead.any():
            return
        for t in range(2):
            for k in range(self.T):
                self.kill(t, k, killer if killer is not None and killer[0] != t else None, dead[:, t, k])

    # ------------------------------------------------------------------ passives
    def eff_focus(self, team: int, k: int) -> torch.Tensor:
        """Focus with Prism's +1 per carried gem."""
        f = self.focus[:, team, k]
        if int(self.kinds[team, k]) == PRISM:
            f = f + self.gems[:, team, k]
        return f

    def eff_strength(self, team: int, k: int) -> torch.Tensor:
        """Strength with Ryth's +1% of bonus health."""
        st = self.strength[:, team, k]
        if int(self.kinds[team, k]) == RYTH:
            st = st + 0.01 * (self.max_hp[:, team, k] - self.base_hp[:, team, k])
        return st

    def dmg_mult(self, team: int, k: int) -> torch.Tensor:
        """Heart of Ruin: up to +50% damage, linear in missing health."""
        if int(self.kinds[team, k]) != HOR:
            return torch.ones(self.B, device=self.dev)
        miss = 1 - (self.hp[:, team, k] / self.max_hp[:, team, k]).clamp(0, 1)
        return 1 + 0.5 * miss

    def gold_mult(self, team: int, k: int) -> torch.Tensor:
        """Vesper +33%, Treasure emblem +20% (they add)."""
        return 1.0 + 0.33 * float(int(self.kinds[team, k]) == VESPER) + 0.2 * self.emb[:, team, k, TREASURE]

    def recall_turns(self, team: int, k: int) -> torch.Tensor:
        """Home emblem: five turn ends become three."""
        return torch.where(self.emb[:, team, k, HOME] > 0, 3, RECALL_TURNS)

    def buy(self, team: int, k: int, item: int, count: int, mask: torch.Tensor) -> torch.Tensor:
        """Buy `count` of cambc ItemType `item` (an emblem = its five shards) on spawn or while
        dead. Five matching shards complete the family's emblem: Power adds one more shard's
        worth, Bruiser one of each other stat. Health bought on spawn heals what it adds."""
        fam, stat = item // 6, ITEM_STAT[item % 6]
        n = count * (5 if item % 2 else 1)
        here = self.pos[:, team, k]
        on_spawn = self.alive[:, team, k] & self.s.spawn[team][here[:, 1], here[:, 0]]
        ok = mask & (on_spawn | ~self.alive[:, team, k]) & (self.gold[:, team, k] >= SHARD_COST * n)
        if not ok.any():
            return ok
        self.gold[:, team, k] -= torch.where(ok, float(SHARD_COST * n), 0.0)
        before = self.fam_shards[:, team, k, fam, stat].clone()
        self.fam_shards[:, team, k, fam, stat] += torch.where(ok, n, 0)
        done = ok & (before < 5) & (self.fam_shards[:, team, k, fam, stat] >= 5)
        add = torch.zeros(self.B, 3, device=self.dev)
        add[:, stat] = torch.where(ok, float(n), 0.0)
        if fam == POWER:
            add[:, stat] += done.float()
        if fam == BRUISER:
            for o in range(3):
                if o != stat:
                    add[:, o] += done.float()
        self.emb[:, team, k, fam] = torch.where(done, 1.0, self.emb[:, team, k, fam])
        self.shards[:, team, k] += torch.where(ok, n, 0)
        self.strength[:, team, k] += add[:, 0] * self.str_growth[:, team, k]
        self.focus[:, team, k] += add[:, 1] * self.focus_growth[:, team, k]
        dh = add[:, 2] * self.hp_growth[:, team, k]
        self.max_hp[:, team, k] += dh
        self.hp[:, team, k] += torch.where(on_spawn, dh, 0.0)
        return ok

    def on_kill(self, team: int, k: int, mask: torch.Tensor) -> None:
        """Vendetta: any kill (champion or camp) resets her ability cooldowns."""
        if int(self.kinds[team, k]) == VENDETTA and mask.any():
            self.cd[:, team, k] = torch.where(mask.unsqueeze(-1), 0, self.cd[:, team, k])

    # ------------------------------------------------------------------ actions
    def move_to(self, team: int, k: int, dest: torch.Tensor, mask: torch.Tensor, dash: bool = False) -> torch.Tensor:
        """Move (or dash, up to 2 tiles) to dest [B, 2] where legal; returns games that moved."""
        s, B = self.s, self.B
        cur = self.pos[:, team, k]
        reach = 2 if dash else 1
        x, y = dest[:, 0].clamp(0, s.w - 1), dest[:, 1].clamp(0, s.h - 1)
        inb = (dest[:, 0] >= 0) & (dest[:, 0] < s.w) & (dest[:, 1] >= 0) & (dest[:, 1] < s.h)
        occ = self.occupied()
        bi = torch.arange(B, device=self.dev)
        dist = self.cheb(dest, cur)
        legal = inb & (dist <= reach) & (dist > 0) & s.walkable[y, x] & ~s.spawn[1 - team][y, x] & ~occ[bi, y, x]
        go = mask & legal
        self.pos[:, team, k] = torch.where(go.unsqueeze(-1), torch.stack([x, y], -1), cur)
        return go

    def stun_targets(self, team: int, e_idx: torch.Tensor, c_idx: torch.Tensor) -> None:
        """One-round stun on enemy slot e_idx / camp c_idx (-1 = none); Marcella is immune.
        A stun cancels recalls and tributes."""
        bi = torch.arange(self.B, device=self.dev)
        for kk in range(self.T):
            if int(self.kinds[1 - team, kk]) == MARCELLA:
                continue
            hit = e_idx == kk
            self.stun[:, 1 - team, kk] = torch.where(hit, 1, self.stun[:, 1 - team, kk])
            self.recall[:, 1 - team, kk] = torch.where(hit, -1, self.recall[:, 1 - team, kk])
            self.tribute[:, 1 - team, kk] = torch.where(hit, -1, self.tribute[:, 1 - team, kk])
        if self.camp_stun.shape[1]:
            c = c_idx >= 0
            self.camp_stun[bi[c], c_idx[c]] = 1

    def resolve(self, team: int, k: int, dmg_e: torch.Tensor, dmg_c: torch.Tensor) -> None:
        """Apply damage dealt by (team, k) to enemy champions and camps."""
        if dmg_e.any():
            before = self.alive.clone()
            self.apply_damage(dmg_e, (team, k))
            self.on_kill(team, k, (before & ~self.alive).any(-1).any(-1))
        if dmg_c.shape[1] and dmg_c.any():
            dmg_c = dmg_c * (1 + 0.1 * self.emb[:, team, k, MONSTER]).unsqueeze(1)
            self.camp_aggro = torch.where(dmg_c > 0, team * self.T + k, self.camp_aggro)
            before = self.camp_alive.clone()
            eff = torch.minimum(dmg_c, self.camp_hp.clamp(min=0)) * self.camp_alive
            self.camp_dmg_done[:, team, k] += eff.sum(-1)
            self.damage_camps(dmg_c)
            self.on_kill(team, k, (before & ~self.camp_alive).any(-1))

    def basic_attack(self, team: int, k: int, kind: torch.Tensor, idx: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Basic attack on a champion (kind 1, idx = enemy slot) or a camp (kind 2, idx = camp
        index), in range, using the attack allowance. Varg splashes; Marcella's charged attack
        stuns. Returns the games where it hit."""
        s, B, dev = self.s, self.B, self.dev
        here = self.pos[:, team, k]
        bi = torch.arange(B, device=dev)
        st = self.eff_strength(team, k) * self.dmg_mult(team, k)
        ok = mask & self.can_attack[:, team, k] & self.alive[:, team, k]
        ec = idx.clamp(0, self.T - 1)
        epos = self.pos[bi, 1 - team, ec]
        is_e = ok & (kind == 1) & self.alive[bi, 1 - team, ec] & (self.cheb(epos, here) <= self.rng[:, team, k])
        C = self.camp_hp.shape[1]
        cc = idx.clamp(0, max(C - 1, 0))
        is_c = ok & (kind == 2) & (C > 0)
        if C:
            is_c &= self.camp_alive[bi, cc] & (self.cheb(s.camp_xy[cc], here) <= self.rng[:, team, k])
        done = is_e | is_c
        if not done.any():
            return done
        dmg_e = torch.zeros(B, 2, self.T, device=dev)
        dmg_c = torch.zeros(B, C, device=dev)
        dmg_e[bi, 1 - team, ec] += torch.where(is_e, st, 0.0)
        if C:
            dmg_c[bi, cc] += torch.where(is_c, st, 0.0)
        if int(self.kinds[team, k]) == VARG:  # splash onto other enemies and camps within 1 of the target
            tpos = torch.where(is_e.unsqueeze(-1), epos, s.camp_xy[cc] if C else epos)
            near_e = self.alive[:, 1 - team] & (self.cheb(self.pos[:, 1 - team], tpos.unsqueeze(1)) <= 1)
            near_e[bi, ec] &= ~is_e
            dmg_e[:, 1 - team] += torch.where(done.unsqueeze(-1) & near_e, st.unsqueeze(-1), 0.0)
            if C:
                near_c = self.camp_alive & (self.cheb(s.camp_xy.unsqueeze(0), tpos.unsqueeze(1)) <= 1)
                near_c[bi, cc] &= ~is_c
                dmg_c += torch.where(done.unsqueeze(-1) & near_c, st.unsqueeze(-1), 0.0)
        if int(self.kinds[team, k]) == MARCELLA:
            stun = done & self.charged[:, team, k]
            self.stun_targets(team, torch.where(is_e & stun, ec, -1), torch.where(is_c & stun, cc, -1))
            self.charged[:, team, k] &= ~done
        self.can_attack[:, team, k] &= ~done
        if bool(self.ranged[0, team, k]):
            self.can_move[:, team, k] &= ~done
        self.resolve(team, k, dmg_e, dmg_c)
        return done

    def cast(self, team: int, k: int, slot: int, mask: torch.Tensor, kind: torch.Tensor, idx: torch.Tensor,
             xy: torch.Tensor) -> torch.Tensor:
        """Cast ability `slot` where `mask`: target kind (0 ground, 1 champion, 2 camp; a heal
        uses kind 1 with an ally slot), idx (enemy / ally slot or camp index), xy [B, 2] for
        ground targets. Returns the games where it went off."""
        s, B, dev = self.s, self.B, self.dev
        typ, mod, rng_, rad = ABILITIES[int(self.kinds[team, k])][slot]
        bi = torch.arange(B, device=dev)
        here = self.pos[:, team, k]
        ok = mask & self.alive[:, team, k] & (self.cd[:, team, k, slot] <= 0) & ~self.cast_done[:, team, k]
        excl = typ in EXCLUDES_WALK and (int(self.kinds[team, k]), slot) not in WALK_OK
        if excl:
            ok &= ~self.walked[:, team, k]
        power = self.eff_focus(team, k) * mod * self.dmg_mult(team, k)
        C = self.camp_hp.shape[1]
        dmg_e = torch.zeros(B, 2, self.T, device=dev)
        dmg_c = torch.zeros(B, C, device=dev)
        ec = idx.clamp(0, self.T - 1)
        cc = idx.clamp(0, max(C - 1, 0))
        if typ in ("dmg", "stun"):
            epos = self.pos[bi, 1 - team, ec]
            on_e = ok & (kind == 1) & self.alive[bi, 1 - team, ec] & (self.cheb(epos, here) <= rng_)
            on_c = ok & (kind == 2) & (C > 0)
            if C:
                on_c &= self.camp_alive[bi, cc] & (self.cheb(s.camp_xy[cc], here) <= rng_)
                dmg_c[bi, cc] += torch.where(on_c, power, 0.0)
            dmg_e[bi, 1 - team, ec] += torch.where(on_e, power, 0.0)
            went = on_e | on_c
            if typ == "stun":
                self.stun_targets(team, torch.where(on_e, ec, -1), torch.where(on_c, cc, -1))
        elif typ in ("aoe", "aoe_stun", "hordash"):
            if typ == "hordash":
                went = self.move_to(team, k, xy, ok, dash=True)
                centre = self.pos[:, team, k]
            else:
                went = ok & (self.cheb(xy, here) <= rng_)
                centre = xy
            near_e = self.alive[:, 1 - team] & (self.cheb(self.pos[:, 1 - team], centre.unsqueeze(1)) <= rad)
            dmg_e[:, 1 - team] += torch.where(went.unsqueeze(-1) & near_e, power.unsqueeze(-1), 0.0)
            near_c = torch.zeros(B, C, dtype=torch.bool, device=dev)
            if C:
                near_c = self.camp_alive & (self.cheb(s.camp_xy.unsqueeze(0), centre.unsqueeze(1)) <= rad)
                dmg_c += torch.where(went.unsqueeze(-1) & near_c, power.unsqueeze(-1), 0.0)
            if typ != "aoe":
                for kk in range(self.T):
                    hit = went & near_e[:, kk]
                    if int(self.kinds[1 - team, kk]) != MARCELLA:
                        self.stun[:, 1 - team, kk] = torch.where(hit, 1, self.stun[:, 1 - team, kk])
                        self.recall[:, 1 - team, kk] = torch.where(hit, -1, self.recall[:, 1 - team, kk])
                        self.tribute[:, 1 - team, kk] = torch.where(hit, -1, self.tribute[:, 1 - team, kk])
                if C:
                    self.camp_stun = torch.where(went.unsqueeze(-1) & near_c, 1, self.camp_stun)
        elif typ == "heal":
            ac = idx.clamp(0, self.T - 1)
            apos = self.pos[bi, team, ac]
            went = ok & (kind == 1) & (ac != k) & self.alive[bi, team, ac] & (self.cheb(apos, here) <= rng_)
            healed = torch.minimum(self.hp[bi, team, ac] + power, self.max_hp[bi, team, ac])
            self.hp[bi, team, ac] = torch.where(went, healed, self.hp[bi, team, ac])
        elif typ == "shield":
            went = ok
            self.shield[:, team, k] = torch.where(went, power, self.shield[:, team, k])
            self.shield_t[:, team, k] = torch.where(went, 5, self.shield_t[:, team, k])
        elif typ in ("reset", "mreset"):
            went = ok
            self.can_attack[:, team, k] |= went
            if typ == "mreset":
                self.charged[:, team, k] |= went
        elif typ == "dash":
            went = self.move_to(team, k, xy, ok, dash=True)
        else:
            raise ValueError(typ)
        self.cd[:, team, k, slot] = torch.where(went, SPELL_CD, self.cd[:, team, k, slot])
        self.cast_done[:, team, k] |= went
        self.resolve(team, k, dmg_e, dmg_c)
        return went

    # ------------------------------------------------------------------ champion turn
    def turn_start(self, team: int, k: int) -> torch.Tensor:
        """Respawn, channel completions, stun; returns which games let this champion act."""
        s = self.s
        live = ~self.done
        dead = ~self.alive[:, team, k]
        ready = dead & (self.respawn[:, team, k] <= 0) & live
        if ready.any():
            tiles = s.spawn_tiles[team]
            occ = self.occupied()
            free = ~occ[:, tiles[:, 1], tiles[:, 0]]  # [B, S]
            first = free.float().argmax(1)
            can = ready & free.any(1)
            self.pos[:, team, k] = torch.where(can.unsqueeze(-1), tiles[first], self.pos[:, team, k])
            self.alive[:, team, k] |= can
            self.hp[:, team, k] = torch.where(can, self.max_hp[:, team, k], self.hp[:, team, k])
            self.cd[:, team, k] = torch.where(can.unsqueeze(-1), 0, self.cd[:, team, k])
        rc = self.recall[:, team, k]
        arrive = (rc == 0) & self.alive[:, team, k]
        if arrive.any():
            tiles = s.spawn_tiles[team]
            occ = self.occupied()
            free = ~occ[:, tiles[:, 1], tiles[:, 0]]
            self.pos[:, team, k] = torch.where((arrive & free.any(1)).unsqueeze(-1), tiles[free.float().argmax(1)],
                                               self.pos[:, team, k])
            self.recall[:, team, k] = torch.where(arrive, -1, rc)
        tr = self.tribute[:, team, k]
        fin = (tr == 0) & self.alive[:, team, k]
        if fin.any():
            g = torch.where(fin, self.gems[:, team, k], 0.0)
            self.tributed[:, team] += g
            self.submitted[:, team, k] += g
            if int(self.kinds[team, k]) == MARCELLA:  # 2 gold per gem tributed
                self.gold[:, team, k] += 2 * g
                self.lifetime_gold[:, team] += 2 * g
                self.earned[:, team, k] += 2 * g
            self.gems[:, team, k] = torch.where(fin, 0.0, self.gems[:, team, k])
            self.tribute[:, team, k] = torch.where(fin, -1, tr)
        stunned = self.stun[:, team, k] > 0
        self.can_attack[:, team, k] = True
        self.can_move[:, team, k] = True
        self.walked[:, team, k] = False
        self.cast_done[:, team, k] = False
        me = self.alive[:, team, k] & live & ~stunned
        busy = (self.recall[:, team, k] >= 0) | (self.tribute[:, team, k] >= 0)
        return me & ~busy

    def damage_camps(self, dmg_c: torch.Tensor) -> None:
        """dmg_c [B, C]: damage camps, wake their whole group, mark cleared groups."""
        s, B, dev = self.s, self.B, self.dev
        hitc = dmg_c > 0
        self.camp_hp = self.camp_hp - dmg_c
        g_hit = torch.zeros(B, self.g_state.shape[1], device=dev).scatter_add_(1, s.camp_group.expand_as(hitc), hitc.float()) > 0
        self.camp_awake |= g_hit[:, s.camp_group] & self.camp_alive
        reset = hitc if CAMP_SLEEP == "no_damage_camp" else g_hit[:, s.camp_group]
        self.camp_quiet = torch.where(reset, 0, self.camp_quiet)
        died = self.camp_alive & (self.camp_hp <= 0)
        self.camp_alive &= ~died
        self.camp_awake &= self.camp_alive
        alive_in_g = torch.zeros(B, self.g_state.shape[1], device=dev).scatter_add_(
            1, s.camp_group.expand_as(self.camp_alive), self.camp_alive.float())
        cleared = (self.g_state == 0) & (alive_in_g == 0)
        self.g_state = torch.where(cleared, 1, self.g_state)
        self.g_timer = torch.where(cleared, 0, self.g_timer)

    def turn_end(self, team: int, k: int, auto_shop: bool = True) -> None:
        """Turn-end ticks (cooldowns, channels, stun, shield, respawn), spawn heal, shop."""
        s, live = self.s, ~self.done
        self.cd[:, team, k] = (self.cd[:, team, k] - 1).clamp(min=0)
        for ch in (self.recall, self.tribute):
            ch[:, team, k] = torch.where(ch[:, team, k] > 0, ch[:, team, k] - 1, ch[:, team, k])
        self.stun[:, team, k] = (self.stun[:, team, k] - 1).clamp(min=0)
        self.shield_t[:, team, k] = (self.shield_t[:, team, k] - 1).clamp(min=0)
        self.shield[:, team, k] = torch.where(self.shield_t[:, team, k] > 0, self.shield[:, team, k], 0.0)
        self.respawn[:, team, k] = torch.where(~self.alive[:, team, k], (self.respawn[:, team, k] - 1).clamp(min=0),
                                               self.respawn[:, team, k])
        on_spawn = self.alive[:, team, k] & s.spawn[team][self.pos[:, team, k, 1], self.pos[:, team, k, 0]]
        self.hp[:, team, k] = torch.where(on_spawn, self.max_hp[:, team, k], self.hp[:, team, k])
        shop = (on_spawn | ~self.alive[:, team, k]) & (self.gold[:, team, k] >= SHARD_COST) & live & auto_shop
        if shop.any():
            n = torch.where(shop, torch.div(self.gold[:, team, k], SHARD_COST, rounding_mode="floor"), 0.0)
            self.gold[:, team, k] -= n * SHARD_COST
            # alternate strength / health shards
            even = (self.shards[:, team, k] % 2 == 0)
            s_n = torch.where(even, torch.ceil(n / 2), torch.floor(n / 2))
            h_n = n - s_n
            self.strength[:, team, k] += s_n * self.str_growth[:, team, k]
            self.max_hp[:, team, k] += h_n * self.hp_growth[:, team, k]
            self.hp[:, team, k] = torch.where(on_spawn, self.max_hp[:, team, k], self.hp[:, team, k])
            self.shards[:, team, k] += n.long()

    def champion_turn(self, team: int, k: int, a: torch.Tensor) -> None:
        """a: long [B, 4] = (move, attack target, spell, special) for champion (team, k).
        attack target: 0 none, 1 nearest enemy champion, 2 nearest camp. spell: 0 none, 1/2 =
        slot 0/1, aimed at the attack target (or the nearest enemy), the lowest-health ally
        (heals), self, or 2 tiles along the move direction (dashes)."""
        s, B, dev = self.s, self.B, self.dev
        act = self.turn_start(team, k)
        mv, at, sp, spc = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
        bi = torch.arange(B, device=dev)
        # move first (a ranged champion that attacks gives up its move)
        wants_move = act & (mv > 0) & ~(self.ranged[:, team, k] & (at > 0))
        d = DIRS.to(dev)[mv]
        if wants_move.any():
            moved = self.move_to(team, k, self.pos[:, team, k] + d, wants_move)
            self.walked[:, team, k] |= moved
            self.can_move[:, team, k] &= ~moved
            if bool(self.ranged[0, team, k]):
                self.can_attack[:, team, k] &= ~moved
        here = self.pos[:, team, k]
        ep = self.pos[:, 1 - team]
        de = self.cheb(ep, here.unsqueeze(1))
        de = torch.where(self.alive[:, 1 - team] & ~s.spawn[1 - team][ep[..., 1], ep[..., 0]], de, 10_000)
        e_idx = de.argmin(1)
        C = self.camp_hp.shape[1]
        if C:
            dc = torch.where(self.camp_alive, self.cheb(s.camp_xy.unsqueeze(0), here.unsqueeze(1)), 10_000)
            c_idx = dc.argmin(1)
        else:
            c_idx = torch.zeros(B, dtype=torch.long, device=dev)
        tkind = torch.where(at == 2, 2, 1)
        tidx = torch.where(at == 2, c_idx, e_idx)
        for slot in (0, 1):
            m = act & (sp == slot + 1)
            if not m.any():
                continue
            typ = ABILITIES[int(self.kinds[team, k])][slot][0]
            target_xy = s.camp_xy[c_idx] if C else here
            xy = torch.where((tkind == 2).unsqueeze(-1), target_xy, ep[bi, e_idx])
            kind_, idx_ = tkind, tidx
            if typ in ("dash", "hordash"):
                xy = here + 2 * d
            if typ == "heal":
                ally = torch.where(self.alive[:, team], self.hp[:, team] / self.max_hp[:, team], 2.0)
                ally[:, k] = 2.0
                kind_, idx_ = torch.ones_like(tkind), ally.argmin(1)
            self.cast(team, k, slot, m, kind_, idx_, xy)
        self.basic_attack(team, k, tkind, tidx, act & (at > 0))
        here = self.pos[:, team, k]
        rec = act & (spc == 1) & ~s.spawn[team][here[:, 1], here[:, 0]]
        self.recall[:, team, k] = torch.where(rec, self.recall_turns(team, k), self.recall[:, team, k])
        trib = act & (spc == 2) & (self.gems[:, team, k] > 0) & s.near_altar[here[:, 1], here[:, 0]]
        dur = torch.clamp(torch.ceil(self.gems[:, team, k] / 10).long(), min=1)
        self.tribute[:, team, k] = torch.where(trib, dur, self.tribute[:, team, k])
        self.turn_end(team, k)

    def step(self, actions: torch.Tensor) -> None:
        """One round. actions: long [B, 2, T, 4]."""
        self.camp_respawns()
        self.captures()
        self.camp_actions()
        # Champions of the two teams interleave one by one (A1 B1 A2 B2 ...), the starting
        # team swaps every round (confirmed on official replays: nn/gpusim/replay.py).
        first = self.round % 2
        for k in range(self.T):
            for team in (first, 1 - first):
                self.champion_turn(team, k, actions[:, team, k])
        self.round += 1
        won = (self.tributed >= WIN_GEMS) & ~self.done.unsqueeze(1)
        w = torch.where(won[:, 0], 0, torch.where(won[:, 1], 1, -1))
        self.winner = torch.where((w >= 0) & ~self.done, w, self.winner)
        self.done |= won.any(1)
        if self.round >= MAX_ROUNDS:
            undecided = ~self.done
            a_more = (self.tributed[:, 0] > self.tributed[:, 1]) | (
                (self.tributed[:, 0] == self.tributed[:, 1]) & (self.lifetime_gold[:, 0] > self.lifetime_gold[:, 1]))
            self.winner = torch.where(undecided, torch.where(a_more, 0, 1), self.winner)
            self.done |= True
