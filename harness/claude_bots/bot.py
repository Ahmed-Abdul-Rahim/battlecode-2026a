"""Hand-written exploiter bots (harness/claude_bots): one file, three strategies.

harness/claude_bots/make.py copies bots/starter-python into bots/claude-<strategy>/
and writes this file as bot.py with STRATEGY set. Strategies:

  ambusher  wait in a bush next to a camp group; strike weak or gem-carrying
            champions that walk past, then go back into the bush.
  pillager  farm lightly, then guard altars: kill carriers and channelling
            tributers, and drop altar locks to cancel their tributes.
  focus     farm; when enemies show up, the whole team focus-fires the weakest
            one, called through the 8-byte team shared memory.
  swarm     two squads + a scout. The scout roams the enemy half and calls the
            enemies it sees; each squad sticks to its rally point (shared memory
            bytes 4..8), farms together and only fights when it outnumbers.

Only API names from sdk/python/cambc are used (harness check_api passes).
"""

import struct

import cambc
from cambc import game_map
from randomness import below

STRATEGY = "ambusher"

Team, Item = cambc.Team, cambc.ItemType

# Shared memory layout (8 bytes): target x, target y (u8), round (u16), 4 spare.
CALL_FMT = "<BBH"
CALL_MAX_AGE = 3


def distance(a, b):
    return a.chebyshev_distance(b)


def tiles_around(position, radius):
    tiles = cambc.Bitboard()
    tiles.draw_chebyshev(position, radius)
    return tiles


def blocked_tiles():
    blocked = (
        game_map.walls()
        | game_map.holes()
        | game_map.altars()
        | game_map.spawn_region(Team.ENEMY)
        | cambc.vision.champion_tiles()
        | cambc.vision.camp_tiles()
        | cambc.vision.teleport_tiles()
    )
    blocked.set(cambc.me().position(), False)
    return blocked


def move_towards_closest(goals):
    me = cambc.me()
    origin = me.position()
    if not me.can_move() or goals.is_set(origin):
        return False
    blocked = blocked_tiles()
    goals &= ~blocked | cambc.vision.camp_tiles()
    if goals.is_empty():
        return False
    if goals.count() == 1 and not blocked.is_set(goals.positions()[0]):
        return me.move_towards(goals.positions()[0])
    field = cambc.DistanceField()
    field.compute_to(blocked & ~goals, goals, tiles_around(origin, 0), cambc.Bitboard())
    reachable = blocked.reachable_tiles(origin, me.base_stats().base_move_speed)
    step = min([origin, *reachable.positions()], key=field.get)
    return step != origin and me.move_absolute(step)


def attack_first_in_range(targets):
    me = cambc.me()
    if not targets or not me.can_attack():
        return False
    attack_tiles = me.attack_tiles()
    for target in targets:
        if target.is_visible() and attack_tiles.is_set(target.position()) and target.health() > 0:
            me.attack(target)
            return True
    return False


def visible_enemies():
    return [e for e in cambc.vision.champions(Team.ENEMY) if e.is_visible() and e.is_alive()]


def gems_of(champion):
    try:
        return champion.gem_count()
    except cambc.Error:
        return 0


def hp_frac(champion):
    return champion.health() / max(champion.max_health(), 1.0)


def bank_gems():
    """Walk to the nearest unlocked altar and tribute. True if this turn was used."""
    me = cambc.me()
    if me.gem_count() == 0:
        return False
    goals = cambc.Bitboard()
    for altar in game_map.altars().positions():
        if cambc.vision.has_altar_lock(altar) and cambc.vision.altar_lock_at(altar).is_enemy():
            continue
        if distance(me.position(), altar) <= cambc.constants.SUBMIT_GEMS_RANGE:
            me.submit_gems(altar)
            return True
        goals |= tiles_around(altar, cambc.constants.SUBMIT_GEMS_RANGE)
    move_towards_closest(goals)
    return True


def farm_camps(bot):
    if attack_first_in_range(cambc.vision.camps()):
        return
    if bot.camp_layout is None:
        spawns = cambc.Bitboard()
        groups = [(group, group.spawn_tiles()) for group in game_map.camp_groups()]
        for _, tiles in groups:
            spawns |= tiles
        bot.camp_layout = spawns, groups
    spawns, groups = bot.camp_layout
    goals = spawns.copy()
    visible = cambc.vision.visible_tiles()
    for group, tiles in groups:
        if not (tiles & visible).is_empty() and isinstance(group.state(), game_map.CampGroupRespawning):
            goals &= ~tiles
    if move_towards_closest(goals):
        attack_first_in_range(cambc.vision.camps())


# ----- team call (shared memory) -------------------------------------------


def read_call():
    data = bytearray(4)
    cambc.shared_memory.read(0, data)
    x, y, rnd = struct.unpack(CALL_FMT, bytes(data))
    if rnd == 0 or (cambc.round_number() & 0xFFFF) - rnd > CALL_MAX_AGE:
        return None
    return cambc.Vec2(x, y)


def write_call(position):
    rnd = cambc.round_number() & 0xFFFF
    cambc.shared_memory.write(0, struct.pack(CALL_FMT, position.x & 0xFF, position.y & 0xFF, max(rnd, 1)))


def strike(target, chase):
    """Attack target if in range, else step towards it when within `chase` tiles."""
    me = cambc.me()
    if attack_first_in_range([target]):
        write_call(target.position())
        return True
    if distance(me.position(), target.position()) <= chase and move_towards_closest(
        tiles_around(target.position(), 1)
    ):
        if attack_first_in_range([target]):
            write_call(target.position())
        return True
    return False


class Bot:
    def __init__(self):
        self.camp_layout = None
        self.post = None
        self.last_seen = 0

    # ----- shared pieces ---------------------------------------------------

    def shop(self, item):
        me = cambc.me()
        count = me.gold_count() // item.cost()
        if count and me.can_buy(item, count):
            me.buy(item, count)

    def pick_post(self, near_altars):
        """A bush tile near a camp group (or an altar), spread across champions by start tile."""
        me = cambc.me()
        anchors = (
            game_map.altars().positions()
            if near_altars
            else [
                g.spawn_tiles().positions()[0]
                for g in game_map.camp_groups()
                if not g.spawn_tiles().is_empty()
            ]
        )
        if not anchors:
            return None
        anchor = anchors[(me.position().x * 7 + me.position().y * 13) % len(anchors)]
        bushes = (game_map.bushes() & tiles_around(anchor, 5)).positions()
        if not bushes:
            return anchor
        return min(bushes, key=lambda p: distance(p, anchor))

    def go_post(self):
        me = cambc.me()
        if self.post is None or distance(me.position(), self.post) == 0:
            return False
        goal = tiles_around(self.post, 0)
        if not move_towards_closest(goal):
            move_towards_closest(tiles_around(self.post, 1))
        return True

    # ----- strategies --------------------------------------------------------

    def ambusher(self, enemies):
        if self.post is None:
            self.post = self.pick_post(near_altars=False)
        called = read_call()
        prey = [
            e
            for e in enemies
            if hp_frac(e) < 0.5 or gems_of(e) > 0 or (called is not None and e.position() == called)
        ]
        if prey:
            target = min(prey, key=lambda e: e.health())
            if strike(target, chase=3):
                return
        if attack_first_in_range(enemies):
            return
        if bank_gems():
            return
        if cambc.round_number() - self.last_seen > 150:
            farm_camps(self)
            return
        self.go_post()

    def pillager(self, enemies):
        me = cambc.me()
        if self.post is None:
            self.post = self.pick_post(near_altars=True)
        carriers = [e for e in enemies if gems_of(e) > 0 or e.is_submitting_gems()]
        for e in carriers:
            if e.is_submitting_gems() and self.drop_lock(e.position()):
                return
        if carriers:
            target = max(carriers, key=lambda e: (e.is_submitting_gems(), gems_of(e)))
            if strike(target, chase=6):
                return
        if attack_first_in_range(enemies):
            return
        if bank_gems():
            return
        if cambc.round_number() < 300:
            farm_camps(self)
            return
        if not self.go_post() and me.can_attack():
            attack_first_in_range(cambc.vision.camps())

    def drop_lock(self, near):
        """Place an altar lock on the altar next to `near` (cancels the tribute there)."""
        me = cambc.me()
        slots = [c for c in me.consumables() if c.kind() == cambc.ConsumableType.ALTAR_LOCK and c.count() > 0]
        if not slots:
            return False
        for altar in game_map.altars().positions():
            if distance(altar, near) <= 1 and not cambc.vision.has_altar_lock(altar):
                if distance(me.position(), altar) <= cambc.constants.SUBMIT_GEMS_RANGE + 1:
                    try:
                        return slots[0].use(altar)
                    except cambc.Error:
                        return False
                return move_towards_closest(tiles_around(altar, 1))
        return False

    def focus(self, enemies):
        called = read_call()
        if enemies:
            target = min(enemies, key=lambda e: (e.health(), distance(cambc.me().position(), e.position())))
            if called is not None:
                at = [e for e in enemies if e.position() == called]
                target = at[0] if at else target
            write_call(target.position())
            if strike(target, chase=6):
                return
        elif called is not None and distance(cambc.me().position(), called) <= 10:
            if move_towards_closest(tiles_around(called, 1)):
                return
        if bank_gems():
            return
        farm_camps(self)

    def swarm(self, enemies):
        me = cambc.me()
        role = int(me.champion_name()) % 3  # 0 scout, 1 squad A, 2 squad B
        here = me.position()
        if role == 0:
            if enemies:
                write_call(min(enemies, key=lambda e: distance(here, e.position())).position())
                near = [e for e in enemies if distance(here, e.position()) <= 4]
                if near and move_towards_closest(
                    tiles_around(game_map.spawn_region(Team.ALLY).positions()[0], 2)
                ):
                    return
            if bank_gems():
                return
            if self.post is None or distance(here, self.post) <= 1:
                spawn = game_map.spawn_region(Team.ALLY).positions()[0]
                far = [
                    g.spawn_tiles().positions()[0]
                    for g in game_map.camp_groups()
                    if not g.spawn_tiles().is_empty()
                ]
                far.sort(key=lambda p: -distance(p, spawn))
                self.post = far[(cambc.round_number() // 150) % max(1, min(4, len(far)))] if far else spawn
            if attack_first_in_range(cambc.vision.camps()):
                return
            move_towards_closest(tiles_around(self.post, 1))
            return
        # Squad member: rally = last position written by the squad (bytes 4..6 or 6..8).
        slot = 4 if role == 1 else 6
        data = bytearray(2)
        cambc.shared_memory.read(slot, data)
        rally = cambc.Vec2(data[0], data[1]) if data[0] or data[1] else None
        cambc.shared_memory.write(slot, bytes([here.x & 0xFF, here.y & 0xFF]))
        allies = [
            a for a in cambc.vision.champions(Team.ALLY) if a.is_alive() and distance(here, a.position()) <= 3
        ]
        if enemies:
            if len(allies) >= len(enemies):  # allies includes us
                target = min(enemies, key=lambda e: e.health())
                write_call(target.position())
                if strike(target, chase=4):
                    return
            elif move_towards_closest(
                tiles_around(rally or game_map.spawn_region(Team.ALLY).positions()[0], 1)
            ):
                return
        called = read_call()
        if (
            called is not None
            and len(allies) >= 2
            and distance(here, called) <= 12
            and move_towards_closest(tiles_around(called, 1))
        ):
            return
        if bank_gems():
            return
        if rally is not None and distance(here, rally) > 3 and move_towards_closest(tiles_around(rally, 1)):
            return
        farm_camps(self)

    # ----- turn ----------------------------------------------------------------

    def play_turn(self):
        me = cambc.me()
        if (
            not me.is_alive()
            or me.is_recalling()
            or me.is_submitting_gems()
            or me.is_teleporting()
            or (not me.can_move() and not me.can_attack())
        ):
            return
        at_spawn = game_map.spawn_region(Team.ALLY).is_set(me.position())
        if at_spawn:
            if STRATEGY == "pillager" and me.can_buy(cambc.ConsumableType.ALTAR_LOCK):
                me.buy(cambc.ConsumableType.ALTAR_LOCK)
            self.shop(Item.POWER_SHARD_OF_STRENGTH if STRATEGY != "ambusher" else Item.POWER_SHARD_OF_HEALTH)
        enemies = visible_enemies()
        if enemies:
            self.last_seen = cambc.round_number()
        if (
            not at_spawn
            and hp_frac(me) < 0.3
            and all(distance(me.position(), e.position()) > 4 for e in enemies)
        ):
            me.recall()
            return
        {"ambusher": self.ambusher, "pillager": self.pillager, "focus": self.focus, "swarm": self.swarm}[
            STRATEGY
        ](enemies)
        me.set_priority(1 + below(100))
