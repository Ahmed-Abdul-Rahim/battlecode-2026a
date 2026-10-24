# Rule audit (harness/search/audit.py)

Each plan-settable decision probed at two values on 12 train maps × 8 search (A) opponents × 2 sides,
paired against the base settings (same snapshot wasm, byte-patched: games differ only through the setting).
**changed** = share of games with any different outcome or fuel; **hetero** = Cochran's Q / df of the per-map
fitness differences (> 2: the best value differs by map); **verdict** plan = changes games and is map-heterogeneous
or clearly better/worse everywhere.

| item | slot | scope (code) | value | changed | Δfit ± SE | Δwin ± SE | hetero | best map (Δfit) | Δfuel mean | ΔOOF | verdict | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AGGRESSION | 0 | turn | 220 | 88% | +0.011 ± 0.042 | +0.016 ± 0.033 | 1.76 | great-wall (+0.43) | -137 | -1 | leave | fight margin scale, team params |
| AGGRESSION | 0 | turn | 40 | 88% | +0.005 ± 0.043 | -0.005 ± 0.034 | 0.94 | great-wall (+0.25) | +379 | +1 | leave | fight margin scale, team params |
| ALTAR_LOCK | 18 | turn | 0 | 100% | -0.031 ± 0.038 | -0.016 ± 0.032 | 1.23 | corridors (+0.16) | -173 | -1 | leave | leave a lock on our altars |
| BUILD_PROFILE | 8 | map | 0 | 92% | +0.088 ± 0.042 | +0.016 ± 0.035 | 1.88 | shortcut (+0.39) | +1560 | -1 | plan | emblem order (shop.rs profiles) |
| BUILD_PROFILE | 8 | map | 4 | 99% | -0.078 ± 0.046 | -0.062 ± 0.037 | 3.13 | great-wall (+0.34) | -159 | +0 | plan | emblem order (shop.rs profiles) |
| CAMP_FEASIBLE | 23 | map | 0 | 100% | -0.165 ± 0.049 | -0.115 ± 0.040 | 5.27 | shortcut (+0.31) | -230 | +0 | plan | infeasible-group discount, early ramp |
| CAMP_FEASIBLE | 23 | map | 25 | 75% | +0.114 ± 0.045 | +0.047 ± 0.036 | 3.21 | great-wall (+0.50) | +1226 | +0 | plan | infeasible-group discount, early ramp |
| CAMP_SAFE | 17 | turn | 1 | 92% | +0.033 ± 0.048 | +0.016 ± 0.037 | 0.78 | great-wall (+0.18) | +664 | -1 | leave | leave awake camps / don't wake unsurvivable |
| CAMP_SAFE | 17 | turn | 3 | 100% | +0.057 ± 0.046 | +0.031 ± 0.037 | 1.43 | great-wall (+0.41) | +1092 | +0 | leave | leave awake camps / don't wake unsurvivable |
| COMBAT_FSM | 31 | turn | 0 | 100% | +0.011 ± 0.044 | +0.026 ± 0.034 | 0.73 | great-wall (+0.30) | -292 | -1 | leave | fight hysteresis / help bits |
| COMBAT_FSM | 31 | turn | 193 | 100% | -0.009 ± 0.044 | -0.010 ± 0.035 | 0.88 | gold-rush (+0.18) | +88 | -1 | leave | fight hysteresis / help bits |
| COMBAT_SEARCH | 16 | turn | 1 | 100% | -0.092 ± 0.048 | -0.068 ± 0.039 | 1.68 | great-wall (+0.31) | +2575 | +0 | leave | skirmish portfolio search (fuel) |
| DENY_TRIBUTE | 13 | turn | 0 | 100% | -0.039 ± 0.023 | -0.036 ± 0.019 | 1.23 | corridors (+0.09) | +19 | +0 | leave | stun/lock enemy tributes |
| ENDGAME_BANK | 14 | turn | 0 | 42% | -0.005 ± 0.012 | -0.005 ± 0.012 | 0.56 | corridors (+0.07) | +19 | +0 | leave | bank all from round 1850 |
| EN_ROUTE | 29 | map | 0 | 100% | -0.182 ± 0.047 | -0.130 ± 0.037 | 3.51 | shortcut (+0.16) | +269 | -1 | plan | small camps on the way |
| EN_ROUTE | 29 | map | 2 | 92% | -0.018 ± 0.044 | -0.005 ± 0.034 | 1.58 | shortcut (+0.22) | +124 | -1 | leave | small camps on the way |
| FIGHT_MARGIN_PCT | 7 | turn | 5 | 88% | +0.021 ± 0.048 | +0.026 ± 0.039 | 2.64 | gold-rush (+0.46) | -24 | +0 | plan | router fight threshold |
| FIGHT_MARGIN_PCT | 7 | turn | 60 | 91% | -0.039 ± 0.047 | -0.036 ± 0.037 | 1.45 | great-wall (+0.22) | +213 | -1 | leave | router fight threshold |
| FIGHT_V6 | 21 | turn | 1 | 100% | -0.043 ± 0.035 | -0.036 ± 0.028 | 1.38 | shortcut (+0.16) | +177 | -1 | leave | focus / join / finish bits |
| FIGHT_V6 | 21 | turn | 7 | 100% | -0.060 ± 0.046 | -0.047 ± 0.037 | 1.89 | gold-rush (+0.35) | +327 | -1 | leave | focus / join / finish bits |
| HUNT_MODE | 12 | turn | 1 | 100% | -0.020 ± 0.048 | +0.000 ± 0.039 | 1.35 | great-wall (+0.29) | -264 | -1 | leave | whole-team hunt |
| HUNT_MODE | 12 | turn | 2 | 100% | +0.015 ± 0.017 | +0.010 ± 0.015 | 0.28 | stampede (+0.10) | +18 | +0 | leave | whole-team hunt |
| K2_CLUSTER | 56 | map | 10 | 100% | -0.127 ± 0.044 | -0.120 ± 0.035 | 1.81 | corners (+0.19) | +1728 | +498 | plan | cluster value (per-turn compute: fuel) |
| K2_DRAFT_PLAN | 61 | map | 1 | 91% | +0.002 ± 0.026 | +0.000 ± 0.021 | 0.47 | great-wall (+0.19) | +22 | -1 | leave | v3 plan readers (bits) |
| K2_DRAFT_PLAN | 61 | map | 16 | 100% | -0.024 ± 0.031 | -0.021 ± 0.024 | 0.47 | gold-rush (+0.22) | +60 | +288 | leave | v3 plan readers (bits) |
| K2_DRAFT_PLAN | 61 | map | 29 | 100% | -0.019 ± 0.044 | -0.005 ± 0.036 | 1.29 | gold-rush (+0.23) | -194 | +287 | leave | v3 plan readers (bits) |
| K2_DRAFT_PLAN | 61 | map | 4 | 74% | +0.011 ± 0.034 | +0.010 ± 0.028 | 0.6 | shortcut (+0.26) | +34 | +0 | leave | v3 plan readers (bits) |
| K2_DRAFT_PLAN | 61 | map | 8 | 92% | -0.024 ± 0.043 | -0.021 ± 0.035 | 0.73 | gold-rush (+0.23) | +115 | -1 | leave | v3 plan readers (bits) |
| K2_EARLY_GOLD | 51 | map | 1 | 84% | -0.044 ± 0.047 | -0.036 ± 0.038 | 2.15 | great-wall (+0.37) | +928 | +0 | plan | early gold bonus |
| K2_EASE_FIRST | 49 | map | 1 | 100% | +0.015 ± 0.049 | -0.021 ± 0.039 | 4.74 | stampede (+0.45) | +808 | -1 | plan | easier groups first alone |
| K2_ENDGAME | 34 | turn | 0 | 100% | -0.010 ± 0.019 | -0.010 ± 0.018 | 0.51 | rows (+0.07) | +1 | +0 | leave | endgame mode |
| K2_FOCUS | 33 | turn | 0 | 100% | +0.081 ± 0.044 | +0.068 ± 0.034 | 1.66 | shortcut (+0.36) | +133 | +0 | leave | team focus target |
| K2_GOLD_ORDER | 53 | turn | 1 | 100% | -0.038 ± 0.030 | -0.031 ± 0.025 | 0.55 | fracture (+0.04) | +37 | +0 | leave | who takes gold rewards |
| K2_GOLD_ORDER | 53 | turn | 2 | 100% | -0.027 ± 0.028 | -0.021 ± 0.023 | 0.54 | corners (+0.07) | +36 | -1 | leave | who takes gold rewards |
| K2_KEEP_BUYING | 57 | map | 0 | 100% | -0.040 ± 0.026 | -0.031 ± 0.023 | 1.49 | great-wall (+0.15) | +50 | +0 | leave | keep buying the last emblem |
| K2_LOS_RETREAT | 36 | turn | 1 | 93% | -0.054 ± 0.046 | -0.021 ± 0.036 | 1.45 | shortcut (+0.19) | +34 | +1 | leave | retreat out of line of sight |
| K2_PASSIVE_AWARE | 32 | map | 1 | 100% | -0.072 ± 0.043 | -0.062 ± 0.035 | 1.56 | gold-rush (+0.17) | +596 | +1 | leave | generic passive rules |
| K2_PUNISH_RECALL | 45 | turn | 6 | 100% | -0.036 ± 0.030 | -0.026 ± 0.024 | 1.01 | shortcut (+0.14) | -88 | +0 | leave | attack lone recallers within N |
| K2_SOLO_CAP | 48 | map | 3 | 100% | +0.003 ± 0.044 | -0.021 ± 0.035 | 2.48 | great-wall (+0.36) | +426 | +0 | plan | don't solo big groups |
| K2_STUN_LOCK | 40 | turn | 1 | 100% | +0.020 ± 0.051 | +0.010 ± 0.041 | 1.72 | gold-rush (+0.43) | +379 | +0 | leave | chain stuns |
| K2_TEAM_UP | 42 | turn | 0 | 100% | -0.073 ± 0.046 | -0.047 ± 0.038 | 2.06 | great-wall (+0.23) | +6 | +0 | plan | join ally fights within N tiles |
| K2_TEAM_UP | 42 | turn | 12 | 99% | -0.047 ± 0.022 | -0.036 ± 0.019 | 1.1 | corridors (+0.07) | -30 | -1 | plan | join ally fights within N tiles |
| K2_TP_CRYSTAL | 52 | map | 1 | 100% | -0.009 ± 0.030 | -0.010 ± 0.026 | 1.18 | shortcut (+0.18) | +517 | -1 | leave | teleport crystal |
| K2_UNSTICK | 46 | turn | 1 | 100% | -0.013 ± 0.044 | -0.005 ± 0.037 | 1.34 | shortcut (+0.22) | +181 | +0 | leave | recall when oscillating |
| K2_UPGRADE_FIRST | 41 | turn | 10 | 100% | -0.031 ± 0.045 | -0.026 ± 0.036 | 0.61 | shortcut (+0.21) | +293 | -1 | leave | recall before hard fights |
| K2_WALK_HOME | 47 | turn | 0 | 100% | +0.036 ± 0.037 | +0.026 ± 0.030 | 1.29 | gold-rush (+0.47) | +298 | +0 | leave | walk instead of recall near spawn |
| K2_WALK_HOME | 47 | turn | 12 | 95% | +0.014 ± 0.035 | +0.010 ± 0.029 | 0.59 | great-wall (+0.26) | +334 | +0 | leave | walk instead of recall near spawn |
| K2_WEAK_CAMP_FIRST | 50 | turn | 1 | 96% | -0.012 ± 0.050 | -0.005 ± 0.040 | 1.23 | great-wall (+0.33) | -111 | -1 | leave | weakest monster first |
| MONOLITH_LIVE | 27 | map | 0 | 100% | -0.016 ± 0.037 | +0.005 ± 0.026 | 5.7 | fracture (+0.48) | +38 | +1 | plan | monolith gate on live damage |
| PASSIVE_ROLES | 24 | map | 1 | 100% | -0.029 ± 0.040 | -0.026 ± 0.032 | 1.05 | great-wall (+0.25) | +301 | +0 | leave | Vesper / Prism passive roles |
| RECALL_GOLD | 6 | map | 40 | 91% | -0.020 ± 0.040 | -0.010 ± 0.034 | 1.14 | great-wall (+0.23) | +144 | +0 | leave | shop trip gold x10 |
| RECALL_GOLD | 6 | map | 6 | 96% | -0.051 ± 0.047 | -0.062 ± 0.038 | 1.8 | great-wall (+0.25) | +631 | +1 | leave | shop trip gold x10 |
| RECALL_SAFE | 20 | turn | 0 | 100% | +0.021 ± 0.038 | +0.016 ± 0.032 | 1.71 | gold-rush (+0.21) | +314 | -1 | leave | recall cancel / start rule |
| RECALL_SAFE | 20 | turn | 1 | 90% | -0.082 ± 0.045 | -0.062 ± 0.035 | 1.25 | collision (+0.12) | -149 | -1 | leave | recall cancel / start rule |
| REMOTE_FARMER | 11 | map | 1 | 98% | +0.068 ± 0.040 | +0.042 ± 0.033 | 1.63 | stampede (+0.40) | +373 | -1 | leave | last slot farms far groups |
| RETREAT_HP_PCT | 1 | turn | 15 | 100% | -0.320 ± 0.054 | -0.135 ± 0.041 | 5.63 | collision (+0.06) | -2584 | -1 | plan | low_hp threshold |
| RETREAT_HP_PCT | 1 | turn | 60 | 100% | -0.034 ± 0.053 | -0.109 ± 0.044 | 3.78 | great-wall (+0.64) | +2241 | -1 | plan | low_hp threshold |
| SAFE_ALTAR | 26 | map | 0 | 95% | +0.022 ± 0.023 | +0.021 ± 0.018 | 0.26 | gold-rush (+0.30) | +60 | +0 | leave | avoid altars by the enemy spawn |
| SHOP_FIRST | 28 | map | 1 | 91% | -0.067 ± 0.043 | -0.057 ± 0.036 | 1.48 | collision (+0.17) | +132 | +0 | leave | recall to shop while carrying |
| SQUAD_JOIN_X10 | 9 | map | 15 | 100% | -0.021 ± 0.049 | -0.021 ± 0.040 | 1.19 | gold-rush (+0.27) | +701 | +0 | leave | pairs join each other's group |
| STUN_THREAT | 22 | turn | 1 | 91% | -0.047 ± 0.045 | -0.047 ± 0.036 | 1.45 | gold-rush (+0.23) | +313 | -1 | leave | stun kits on a 6-round horizon |
| TAVI_META | 19 | map | 0 | 100% | -0.119 ± 0.048 | -0.083 ± 0.039 | 0.81 | great-wall (+0.14) | -51 | +0 | plan | Tavi field shop / build / hunt (draft list is compiled) |
| TEAM_FIGHT | 15 | turn | 1 | 100% | -0.001 ± 0.043 | -0.021 ± 0.035 | 0.75 | collision (+0.14) | +1474 | +1 | leave | Lanchester skirmish rule |
| TEAM_RESPAWN | 30 | turn | 0 | 100% | -0.092 ± 0.048 | -0.073 ± 0.039 | 1.3 | gold-rush (+0.27) | -7 | -1 | leave | respawn broadcast in the channel |
| TRIBUTE_DISTANCE_SCALE | 4 | map | 25 | 80% | -0.073 ± 0.039 | -0.062 ± 0.033 | 1.54 | collision (+0.19) | +60 | -1 | leave | gems per altar tile x0.1 |
| TRIBUTE_DISTANCE_SCALE | 4 | map | 3 | 83% | +0.069 ± 0.038 | +0.047 ± 0.032 | 1.49 | gold-rush (+0.37) | +228 | -1 | leave | gems per altar tile x0.1 |
| TRIBUTE_MAX_GEMS | 3 | map | 220 | 49% | -0.023 ± 0.025 | -0.021 ± 0.022 | 0.85 | eight (+0.08) | +95 | -1 | leave | bag cap |
| TRIBUTE_MAX_GEMS | 3 | map | 50 | 79% | +0.013 ± 0.035 | +0.010 ± 0.029 | 0.82 | great-wall (+0.16) | -82 | -1 | leave | bag cap |
| TRIBUTE_MIN_GEMS | 2 | map | 10 | 92% | -0.028 ± 0.044 | -0.026 ± 0.036 | 1.42 | great-wall (+0.22) | +291 | +0 | leave | draft writes 25+0.4 walk; this overrides it |
| TRIBUTE_MIN_GEMS | 2 | map | 60 | 78% | -0.128 ± 0.032 | -0.104 ± 0.027 | 2.04 | collision (+0.02) | -53 | -1 | plan | draft writes 25+0.4 walk; this overrides it |
| VALUE_PROFILE | 10 | map | 0 | 100% | -0.069 ± 0.046 | -0.062 ± 0.037 | 1.69 | shortcut (+0.28) | +422 | -1 | leave | camp value formula |
| VALUE_PROFILE | 10 | map | 5 | 100% | +0.016 ± 0.047 | +0.005 ± 0.039 | 1.17 | gold-rush (+0.27) | +555 | -1 | leave | camp value formula |
| draft | draft | map | 1 | 100% | -0.026 ± 0.050 | -0.078 ± 0.038 | 4.79 | shortcut (+0.57) | +1535 | +0 | plan | roster (draft_prefs profile) |
| draft | draft | map | 10 | 100% | -0.091 ± 0.050 | -0.078 ± 0.040 | 1.57 | rows (+0.14) | +468 | -1 | leave | roster (draft_prefs profile) |
| draft | draft | map | 11 | 100% | -0.088 ± 0.050 | -0.073 ± 0.040 | 2.7 | shortcut (+0.38) | +650 | +0 | plan | roster (draft_prefs profile) |
| draft | draft | map | 12 | 100% | -0.158 ± 0.055 | -0.109 ± 0.044 | 2.78 | great-wall (+0.21) | +85 | -1 | plan | roster (draft_prefs profile) |
| draft | draft | map | 13 | 100% | -0.011 ± 0.054 | -0.026 ± 0.043 | 3.17 | great-wall (+0.39) | +645 | -1 | plan | roster (draft_prefs profile) |
| draft | draft | map | 14 | 100% | -0.106 ± 0.050 | -0.099 ± 0.041 | 3.08 | corners (+0.35) | +698 | -1 | plan | roster (draft_prefs profile) |
| draft | draft | map | 2 | 100% | -0.209 ± 0.060 | -0.213 ± 0.049 | 32.19 | great-wall (+0.39) | +576 | -1 | plan | roster (draft_prefs profile) |
| draft | draft | map | 3 | 100% | -0.045 ± 0.055 | -0.031 ± 0.043 | 4.32 | corners (+0.35) | +27 | +63 | plan | roster (draft_prefs profile) |
| draft | draft | map | 5 | 100% | -0.013 ± 0.052 | -0.088 ± 0.041 | 8.84 | corridors (+0.46) | +1609 | +0 | plan | roster (draft_prefs profile) |
| draft | draft | map | 6 | 100% | -0.036 ± 0.051 | -0.078 ± 0.041 | 2.34 | shortcut (+0.33) | +933 | -1 | plan | roster (draft_prefs profile) |
| draft | draft | map | 7 | 100% | -0.221 ± 0.057 | -0.172 ± 0.045 | 3.64 | shortcut (+0.17) | +552 | -1 | plan | roster (draft_prefs profile) |
| draft | draft | map | 8 | 100% | -0.229 ± 0.054 | -0.172 ± 0.044 | 5.31 | gold-rush (+0.23) | -307 | -1 | plan | roster (draft_prefs profile) |
| style | style | map | aggro | 90% | -0.040 ± 0.053 | -0.005 ± 0.041 | 3.48 | gold-rush (+0.44) | -404 | -1 | plan | data/styles.json bundles |
| style | style | map | bank | 84% | +0.060 ± 0.034 | +0.057 ± 0.029 | 1.55 | shortcut (+0.38) | -151 | -1 | leave | data/styles.json bundles |
| style | style | map | careful | 100% | -0.045 ± 0.051 | -0.083 ± 0.041 | 1.06 | gold-rush (+0.23) | +1601 | +0 | leave | data/styles.json bundles |
| style | style | map | contest | 100% | -0.073 ± 0.045 | -0.052 ± 0.036 | 1.77 | gold-rush (+0.28) | +123 | +1 | leave | data/styles.json bundles |
| style | style | map | farmer | 100% | -0.138 ± 0.055 | -0.125 ± 0.045 | 1.59 | gold-rush (+0.21) | +732 | -1 | plan | data/styles.json bundles |
| style | style | map | hunt | 100% | +0.015 ± 0.017 | +0.010 ± 0.015 | 0.28 | stampede (+0.10) | +18 | +0 | leave | data/styles.json bundles |
| style | style | map | safe | 100% | -0.079 ± 0.053 | -0.078 ± 0.042 | 1.86 | gold-rush (+0.32) | +1054 | +0 | leave | data/styles.json bundles |
| style | style | map | search | 100% | -0.092 ± 0.048 | -0.068 ± 0.039 | 1.68 | great-wall (+0.31) | +2575 | +0 | leave | data/styles.json bundles |
| style | style | map | teamfight | 100% | -0.001 ± 0.043 | -0.021 ± 0.035 | 0.75 | collision (+0.14) | +1474 | +1 | leave | data/styles.json bundles |

Dead (never read in play, slots free): K2_SPACING, K2_CAMP_DIVERT, K2_PREDICT, K2_WARDS, DRAFT_BEST, MONOLITH_MIN_POWER
