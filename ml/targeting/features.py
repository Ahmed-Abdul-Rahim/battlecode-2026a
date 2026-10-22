"""Inputs of a target decision, in the order bot.rs writes them (`S` lines, sweep_row and
sweep_global). Every value is int8 = round(clamp(v / scale, -1, 1) * 127).

FUEL is a rough per-feature cost in the bot (fuel per group, or per decision for global
inputs) beyond what the rule scoring already computes; backward elimination weighs it.
"""

from __future__ import annotations

# (name, fuel): 0 = the rule scoring has it anyway.
GROUP = [
    ("gems", 0),
    ("gold", 0),
    ("total_hp", 0),
    ("max_strength", 0),
    ("total_strength", 0),
    ("tiles", 0),
    ("monolith", 0),
    ("dist", 0),
    ("clear", 0),
    ("respawn_eta", 0),
    ("state_alive", 0),
    ("state_unknown", 0),
    ("state_reward", 0),
    ("state_respawning", 0),
    ("ally_claimants", 0),
    ("claimable", 0),
    ("buddy_on_it", 0),
    ("live_hp", 0),
    ("camps_seen", 60),
    ("value_raw", 0),
    ("value_adjusted", 0),
    ("learned_bias", 0),
    ("rule_score", 0),
    ("infeasible", 0),
    ("solo_cap", 0),
    ("early_ramp", 0),
    ("strength_factor", 0),
    ("is_target", 0),
    ("same_cluster_as_target", 20),
    ("cluster_live_value", 40),
    ("cluster_size", 40),
    ("cluster_home_dist", 20),
    ("in_route_plan", 30),
    ("route_plan_pos", 30),
    ("enemy_sighting_dist", 80),
    ("enemy_sightings_near", 80),
    ("death_hotspots", 120),
    ("altar_dist", 60),
    ("straight_dist", 10),
]
GLOBAL = [
    ("round", 0),
    ("hp_frac", 0),
    ("dps", 0),
    ("max_hp", 0),
    ("gold", 0),
    ("gems_carried", 0),
    ("emblems", 0),
    ("team_size", 0),
    ("ally_tributed", 0),
    ("enemy_tributed", 0),
    ("at_home", 0),
    ("slot", 0),
    ("kind", 0),
    ("opening", 0),
    ("n_groups", 0),
    ("team_aggression", 0),
]
N_GROUP, N_GLOBAL = len(GROUP), len(GLOBAL)
NAMES = [f"g.{n}" for n, _ in GROUP] + [f"x.{n}" for n, _ in GLOBAL]
#: Fuel per decision: group features are paid once per scored group (~20 on average).
FUEL = [f * 20 for _, f in GROUP] + [f for _, f in GLOBAL]
