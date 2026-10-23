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
    "abilities: one generic spell per champion (2 x focus damage, range 4, 8-turn cooldown)",
    "no wards, consumables, altar locks, dashes, teleport crystals, emblem passives",
    "shop: on spawn, gold buys alternating strength/health shards (30 gold each)",
    "camps keep their health while asleep",
]

# engine CampType: wolf, elk, gem cluster, gold cluster, monolith, golem
CAMP_HP = [1350.0, 2400.0, 120.0, 120.0, 20000.0, 5000.0]
CAMP_STR = [20.0, 60.0, 0.0, 0.0, 200.0, 120.0]
CAMP_RANGE = [3, 4, 0, 0, 5, 4]
# Champion kits: (ranged, range, health, health growth, strength, strength growth, focus) by kind
# index (bc_core::champion::ChampionKind order).
KITS = [
    (False, 1, 1200, 100, 45, 5, 100),  # Ryth
    (True, 3, 900, 100, 50, 5, 100),  # Vesper Vale
    (False, 1, 800, 100, 70, 6, 100),  # Vendetta
    (True, 3, 600, 65, 30, 3, 70),  # Cassi
    (False, 1, 900, 80, 40, 3, 100),  # Issac
    (True, 3, 750, 75, 30, 3.5, 85),  # Stellar
    (True, 3, 700, 60, 35, 2.5, 80),  # Prism
    (False, 1, 1200, 100, 45, 5, 60),  # Marcella
    (True, 3, 700, 50, 55, 3, 120),  # Willow
    (False, 1, 900, 100, 35, 5, 100),  # Ren
    (False, 1, 1000, 100, 40, 5, 100),  # Varg
    (True, 3, 750, 75, 45, 5, 100),  # Tavi
    (False, 1, 1000, 100, 40, 4, 100),  # Heart of Ruin
]
DIRS = torch.tensor([[0, 0], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0], [-1, 1]])
WIN_GEMS, MAX_ROUNDS = 1000, 2000
RESPAWN_ROUNDS, CAPTURE_ROUNDS, KILL_GOLD, SHARD_COST = 60, 5, 100, 30
VISION, SPELL_RANGE, SPELL_CD, RECALL_TURNS = 8, 4, 8, 5
# Action layout per champion: move (9) x attack target (3: none, nearest enemy, nearest camp)
# x spell (2) x special (3: none, recall, tribute) -> four integer heads.
ACT_MOVE, ACT_ATTACK, ACT_SPELL, ACT_SPECIAL = 9, 3, 2, 3


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
        k = torch.as_tensor(KITS, dtype=torch.float32, device=dev)[self.kinds]  # [2, T, 7]
        ex = lambda v: v.unsqueeze(0).expand(B, *v.shape).clone()  # noqa: E731
        self.ranged = ex(k[..., 0] > 0)
        self.rng = ex(k[..., 1].long())
        self.max_hp = ex(k[..., 2])
        self.hp_growth = ex(k[..., 3])
        self.strength = ex(k[..., 4])
        self.str_growth = ex(k[..., 5])
        self.focus = ex(k[..., 6])
        self.hp = self.max_hp.clone()
        self.pos = torch.zeros(B, 2, T, 2, dtype=torch.long, device=dev)
        for team in range(2):
            tiles = s.spawn_tiles[team]
            self.pos[:, team] = tiles[torch.arange(T, device=dev) % len(tiles)]
        z = lambda dt=torch.float32: torch.zeros(B, 2, T, dtype=dt, device=dev)  # noqa: E731
        self.alive = torch.ones(B, 2, T, dtype=torch.bool, device=dev)
        self.respawn = z(torch.long)
        self.gems, self.gold, self.cd = z(), z(), z(torch.long)
        self.recall = z(torch.long) - 1  # -1: not channelling
        self.tribute = z(torch.long) - 1
        self.shards = z(torch.long)
        self.kills, self.deaths = z(torch.long), z(torch.long)
        C, G = len(s.camp_kind), len(s.group_gems)
        self.camp_max = torch.as_tensor(CAMP_HP, device=dev)[s.camp_kind]
        self.camp_str = torch.as_tensor(CAMP_STR, device=dev)[s.camp_kind]
        self.camp_rng = torch.as_tensor(CAMP_RANGE, device=dev)[s.camp_kind]
        self.camp_hp = self.camp_max.unsqueeze(0).expand(B, C).clone()
        self.camp_alive = torch.ones(B, C, dtype=torch.bool, device=dev)
        self.camp_awake = torch.zeros(B, C, dtype=torch.bool, device=dev)
        self.camp_quiet = torch.zeros(B, C, dtype=torch.long, device=dev)
        # group: 0 alive, 1 cleared (capturable), 2 respawning
        self.g_state = torch.zeros(B, G, dtype=torch.long, device=dev)
        self.g_timer = torch.zeros(B, G, dtype=torch.long, device=dev)
        self.g_team = torch.full((B, G), -1, dtype=torch.long, device=dev)
        self.tributed = torch.zeros(B, 2, device=dev)
        self.lifetime_gold = torch.zeros(B, 2, device=dev)
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
            self.gold[:, kt, ks] += torch.where(mask, float(KILL_GOLD), 0.0)
            self.lifetime_gold[:, kt] += torch.where(mask, float(KILL_GOLD), 0.0)
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
        done = progress & (self.g_timer >= CAPTURE_ROUNDS)
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
                gold = (present.float() * share_o.unsqueeze(1)).sum(-1).floor()
                self.gold[:, t] += gold
                self.lifetime_gold[:, t] += gold.sum(-1)
            self.g_state = torch.where(done, 2, self.g_state)
            self.g_timer = torch.where(done, s.group_respawn, self.g_timer)
            self.g_team = torch.where(done, -1, self.g_team)

    def camp_actions(self) -> None:
        """Awake camps hit the nearest living champion in range (lowest index on ties)."""
        s, B = self.s, self.B
        act = self.camp_alive & self.camp_awake & (self.camp_str > 0)
        if not act.any():
            return
        cp = s.camp_xy.view(1, -1, 1, 2)  # [1, C, 1, 2]
        chp = self.pos.view(B, 1, -1, 2)  # [B, 1, 2T, 2]
        d = self.cheb(cp, chp)  # [B, C, 2T]
        ok = self.alive.view(B, 1, -1) & (d <= self.camp_rng.view(1, -1, 1)) & act.unsqueeze(-1)
        d = torch.where(ok, d, 10_000)
        tgt = d.argmin(-1)  # [B, C]
        hit = ok.any(-1)
        dmg = torch.zeros(B, 2 * self.T, device=self.dev)
        dmg.scatter_add_(1, tgt, torch.where(hit, self.camp_str.expand_as(hit), 0.0))
        dmg = dmg.view(B, 2, self.T)
        self.apply_damage(dmg, None)
        # a camp with no champion in range for 2 rounds (and no damage taken) goes back to sleep
        self.camp_quiet = torch.where(act & ~hit, self.camp_quiet + 1, torch.where(hit, 0, self.camp_quiet))
        self.camp_awake &= ~(self.camp_quiet >= 2)

    def apply_damage(self, dmg: torch.Tensor, killer: tuple[int, int] | None) -> None:
        """dmg [B, 2, T] to champions; recalls are cancelled by damage; deaths resolved."""
        hit = dmg > 0
        self.hp = self.hp - dmg
        self.recall = torch.where(hit, -1, self.recall)
        dead = self.alive & (self.hp <= 0)
        if not dead.any():
            return
        for t in range(2):
            for k in range(self.T):
                self.kill(t, k, killer if killer is not None and killer[0] != t else None, dead[:, t, k])

    # ------------------------------------------------------------------ champion turn
    def turn_start(self, team: int, k: int) -> torch.Tensor:
        """Respawn and channel completions; returns which games let this champion act."""
        s, B, dev = self.s, self.B, self.dev
        live = ~self.done
        # --- turn start: respawn, channel completions
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
            self.cd[:, team, k] = torch.where(can, 0, self.cd[:, team, k])
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
            self.tributed[:, team] += torch.where(fin, self.gems[:, team, k], 0.0)
            self.gems[:, team, k] = torch.where(fin, 0.0, self.gems[:, team, k])
            self.tribute[:, team, k] = torch.where(fin, -1, tr)
        me = self.alive[:, team, k] & live
        busy = (self.recall[:, team, k] >= 0) | (self.tribute[:, team, k] >= 0)
        act = me & ~busy
        return act

    def damage_camps(self, dmg_c: torch.Tensor) -> None:
        """dmg_c [B, C]: damage camps, wake their whole group, mark cleared groups."""
        s, B, dev = self.s, self.B, self.dev
        hitc = dmg_c > 0
        self.camp_hp = self.camp_hp - dmg_c
        g_hit = torch.zeros(B, self.g_state.shape[1], device=dev).scatter_add_(1, s.camp_group.expand_as(hitc), hitc.float()) > 0
        self.camp_awake |= g_hit[:, s.camp_group] & self.camp_alive
        self.camp_quiet = torch.where(g_hit[:, s.camp_group], 0, self.camp_quiet)
        died = self.camp_alive & (self.camp_hp <= 0)
        self.camp_alive &= ~died
        self.camp_awake &= self.camp_alive
        alive_in_g = torch.zeros(B, self.g_state.shape[1], device=dev).scatter_add_(
            1, s.camp_group.expand_as(self.camp_alive), self.camp_alive.float())
        cleared = (self.g_state == 0) & (alive_in_g == 0)
        self.g_state = torch.where(cleared, 1, self.g_state)
        self.g_timer = torch.where(cleared, 0, self.g_timer)

    def turn_end(self, team: int, k: int, auto_shop: bool = True) -> None:
        """Turn-end ticks (cooldowns, channels, respawn), spawn heal, shop."""
        s, live = self.s, ~self.done
        self.cd[:, team, k] = (self.cd[:, team, k] - 1).clamp(min=0)
        for ch in (self.recall, self.tribute):
            ch[:, team, k] = torch.where(ch[:, team, k] > 0, ch[:, team, k] - 1, ch[:, team, k])
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
        """a: long [B, 4] = (move, attack, spell, special) for champion (team, k) in every game."""
        s, B, dev = self.s, self.B, self.dev
        act = self.turn_start(team, k)
        mv, at, sp, spc = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
        ranged = self.ranged[:, team, k]
        # --- move (ranged champions that attack don't move)
        wants_move = act & (mv > 0) & ~(ranged & (at > 0))
        if wants_move.any():
            d = DIRS.to(dev)[mv]
            np_ = self.pos[:, team, k] + d
            inb = (np_[:, 0] >= 0) & (np_[:, 0] < s.w) & (np_[:, 1] >= 0) & (np_[:, 1] < s.h)
            npc = torch.stack([np_[:, 0].clamp(0, s.w - 1), np_[:, 1].clamp(0, s.h - 1)], -1)
            occ = self.occupied()
            bi = torch.arange(B, device=dev)
            legal = inb & s.walkable[npc[:, 1], npc[:, 0]] & ~s.spawn[1 - team][npc[:, 1], npc[:, 0]] \
                & ~occ[bi, npc[:, 1], npc[:, 0]]
            go = wants_move & legal
            self.pos[:, team, k] = torch.where(go.unsqueeze(-1), npc, self.pos[:, team, k])
        here = self.pos[:, team, k]
        # --- targets in range: nearest enemy champion / nearest living camp
        ep = self.pos[:, 1 - team]  # [B, T, 2]
        de = self.cheb(ep, here.unsqueeze(1))
        de = torch.where(self.alive[:, 1 - team] & ~s.spawn[1 - team][ep[..., 1], ep[..., 0]], de, 10_000)
        e_idx, e_d = de.argmin(1), de.amin(1)
        dc = self.cheb(s.camp_xy.unsqueeze(0), here.unsqueeze(1))
        dc = torch.where(self.camp_alive, dc, 10_000)
        if dc.shape[1]:
            c_idx, c_d = dc.argmin(1), dc.amin(1)
        else:
            c_idx = torch.zeros(B, dtype=torch.long, device=dev)
            c_d = torch.full((B,), 10_000, dtype=torch.long, device=dev)
        rng_ = self.rng[:, team, k]
        dmg_e = torch.zeros(B, 2, self.T, device=dev)
        dmg_c = torch.zeros(B, self.camp_hp.shape[1], device=dev)
        bi = torch.arange(B, device=dev)
        strength = self.strength[:, team, k]
        hit_e = act & (at == 1) & (e_d <= rng_)
        hit_c = act & (at == 2) & (c_d <= rng_)
        dmg_e[bi, 1 - team, e_idx] += torch.where(hit_e, strength, 0.0)
        if dmg_c.shape[1]:
            dmg_c[bi, c_idx] += torch.where(hit_c, strength, 0.0)
        # --- generic spell: 2 x focus at the attack's target type, range 4, 8-turn cooldown
        cast = act & (sp == 1) & (self.cd[:, team, k] <= 0)
        se = cast & (at == 1) & (e_d <= SPELL_RANGE)
        sc = cast & (at == 2) & (c_d <= SPELL_RANGE)
        dmg_e[bi, 1 - team, e_idx] += torch.where(se, 2 * self.focus[:, team, k], 0.0)
        if dmg_c.shape[1]:
            dmg_c[bi, c_idx] += torch.where(sc, 2 * self.focus[:, team, k], 0.0)
        self.cd[:, team, k] = torch.where(se | sc, SPELL_CD, self.cd[:, team, k])
        # --- resolve damage: champions, then camps (wake their whole group)
        if dmg_e.any():
            self.apply_damage(dmg_e, (team, k))
        if dmg_c.any():
            self.damage_camps(dmg_c)
        # --- specials: recall, tribute (next to an altar with gems)
        rec = act & (spc == 1) & ~s.spawn[team][here[:, 1], here[:, 0]]
        self.recall[:, team, k] = torch.where(rec, RECALL_TURNS, self.recall[:, team, k])
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
