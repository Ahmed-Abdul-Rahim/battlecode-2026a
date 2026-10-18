"""Champion decisions. main.py dispatches here after the draft."""
import struct
from randomness import below

import cambc
from cambc import game_map

Team, Item = cambc.Team, cambc.ItemType


def distance(a, b):
    return a.chebyshev_distance(b)


def tiles_around(position, radius):
    tiles = cambc.Bitboard()
    tiles.draw_chebyshev(position, radius)
    return tiles


def blocked_tiles():
    # Combine terrain and visible occupants using native bitboard operations.
    blocked = (game_map.walls() | game_map.holes() | game_map.altars()
               | game_map.spawn_region(Team.ENEMY) | cambc.vision.champion_tiles()
               | cambc.vision.camp_tiles() | cambc.vision.teleport_tiles())
    blocked.set(cambc.me().position(), False)
    return blocked


def move_towards_closest(goals):
    me = cambc.me()
    origin = me.position()
    if not me.can_move() or goals.is_set(origin):
        return False

    # Only free goals or occupied camp tiles can seed the field.
    blocked = blocked_tiles()
    goals &= ~blocked | cambc.vision.camp_tiles()
    if goals.is_empty():
        return False
    if goals.count() == 1 and not blocked.is_set(goals.positions()[0]):
        return me.move_towards(goals.positions()[0])

    # Expand from the goals until our current tile is reached.
    field = cambc.DistanceField()
    field.compute_to(blocked & ~goals, goals, tiles_around(origin, 0), cambc.Bitboard())

    # Choose a reachable step with a smaller distance to the goal.
    reachable = blocked.reachable_tiles(origin, me.base_stats().base_move_speed)
    step = min([origin, *reachable.positions()], key=field.get)
    return step != origin and me.move_absolute(step)


def attack_first_in_range(targets):
    # Compute the attack mask once, rather than once per visible target.
    me = cambc.me()
    if not targets or not me.can_attack():
        return False
    attack_tiles = me.attack_tiles()
    for target in targets:
        if target.is_visible() and attack_tiles.is_set(target.position()) and target.health() > 0:
            me.attack(target)
            return True
    return False


def farm_camps(bot):
    # Attack a camp already in range, otherwise move toward the nearest camp tiles.
    if attack_first_in_range(cambc.vision.camps()):
        return

    # Spawn locations never change. Cache their union, then remove only groups
    # we can currently see respawning. Keep unseen camps as exploration goals.
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

    # Fresh vision after movement may reveal a camp already in range.
    if move_towards_closest(goals):
        attack_first_in_range(cambc.vision.camps())


class Bot:
    def __init__(self):
        # Per-champion state survives between turns. Pick one stat and keep buying it.
        self.chosen_stat = None
        self.camp_layout = None

    def buy_items(self):
        # Spend only on power shards of the randomly chosen stat.
        me = cambc.me()
        item = (Item.POWER_SHARD_OF_STRENGTH, Item.POWER_SHARD_OF_FOCUS,
                Item.POWER_SHARD_OF_HEALTH)[self.chosen_stat]
        count = me.gold_count() // item.cost()
        if count and me.can_buy(item, count):
            me.buy(item, count)

    def play_turn(self):
        # Initialise this champion's build and read the draft message once.
        if self.chosen_stat is None:
            self.chosen_stat = below(3)
            print("draft:", bytes(cambc.DraftMemoryBuffer())[:11].decode())

        # Read before writing: little-endian u32 at shared-memory byte offset zero.
        data = bytearray(4)
        cambc.shared_memory.read(0, data)
        print("shared round:", struct.unpack("<I", data)[0])
        cambc.shared_memory.write(0, struct.pack("<I", cambc.round_number()))

        # Priorities: shop and recall, attack enemies in range, bank gems, then farm.
        me = cambc.me()

        # Leave channels running and skip turns where we cannot act.
        if (not me.is_alive() or me.is_recalling() or me.is_submitting_gems() or me.is_teleporting()
                or (not me.can_move() and not me.can_attack())):
            return

        # Shop before leaving spawn.
        at_spawn = game_map.spawn_region(Team.ALLY).is_set(me.position())
        if at_spawn:
            self.buy_items()

        # Recall immediately when badly hurt away from spawn.
        if not at_spawn and me.health() < me.max_health() * 0.35:
            me.recall()
            return

        # Attack the first enemy champion already in range.
        if attack_first_in_range(cambc.vision.champions(Team.ENEMY)):
            return

        # Bank any carried gems before farming.
        if me.gem_count() > 0:
            # Union all altar approach tiles: the field chooses the nearest reachable altar.
            goals = cambc.Bitboard()
            for altar in game_map.altars().positions():
                if cambc.vision.has_altar_lock(altar):
                    continue
                if distance(me.position(), altar) <= cambc.constants.SUBMIT_GEMS_RANGE:
                    me.submit_gems(altar)
                    return
                goals |= tiles_around(altar, cambc.constants.SUBMIT_GEMS_RANGE)
            move_towards_closest(goals)
            return

        farm_camps(self)

        me.set_priority(1 + below(100))

