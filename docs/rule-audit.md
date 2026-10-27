# Rule audit (harness/search/audit.py)

Each plan-settable decision probed at two values on 12 train maps × 8 search (A) opponents × 2 sides,
paired against the base settings (same snapshot wasm, byte-patched: games differ only through the setting).
**changed** = share of games with any different outcome or fuel; **hetero** = Cochran's Q / df of the per-map
fitness differences (> 2: the best value differs by map); **verdict** plan = changes games and is map-heterogeneous
or clearly better/worse everywhere.

| item | slot | scope (code) | value | changed | Δfit ± SE | Δwin ± SE | hetero | best map (Δfit) | Δfuel mean | ΔOOF | verdict | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AGGRESSION | 0 | turn | 220 | 92% | -0.199 ± 0.049 | -0.167 ± 0.039 | 2.94 | adgato-gemcombat1 (+0.08) | -169 | +0 | plan | fight margin scale, team params |
| AGGRESSION | 0 | turn | 40 | 92% | -0.116 ± 0.051 | -0.109 ± 0.043 | 11.2 | adgato-gemcombat1 (+0.94) | +483 | +0 | plan | fight margin scale, team params |
| ALTAR_LOCK | 18 | turn | 0 | 100% | -0.109 ± 0.041 | -0.094 ± 0.034 | 4.03 | expanse (+0.34) | -147 | +0 | plan | leave a lock on our altars |
| BUILD_PROFILE | 8 | map | 0 | 92% | +0.223 ± 0.047 | +0.130 ± 0.038 | 7.84 | adgato-gemcombat1 (+0.97) | +1402 | +0 | plan | emblem order (shop.rs profiles; 14-17 engine 2.0.0 families) |
| BUILD_PROFILE | 8 | map | 14 | 100% | -0.219 ± 0.042 | -0.162 ± 0.035 | 2.93 | adgato-gemcombat1 (+0.05) | -414 | +0 | plan | emblem order (shop.rs profiles; 14-17 engine 2.0.0 families) |
| BUILD_PROFILE | 8 | map | 15 | 100% | -0.169 ± 0.038 | -0.125 ± 0.031 | 4.69 | galaxy (+0.17) | -331 | +0 | plan | emblem order (shop.rs profiles; 14-17 engine 2.0.0 families) |
| BUILD_PROFILE | 8 | map | 16 | 100% | -0.270 ± 0.045 | -0.213 ± 0.037 | 4.37 | rushbait-v2 (+0.01) | -454 | +0 | plan | emblem order (shop.rs profiles; 14-17 engine 2.0.0 families) |
| BUILD_PROFILE | 8 | map | 17 | 100% | -0.015 ± 0.038 | -0.047 ± 0.032 | 3.36 | expanse (+0.27) | +710 | +1 | plan | emblem order (shop.rs profiles; 14-17 engine 2.0.0 families) |
| BUILD_PROFILE | 8 | map | 4 | 100% | -0.131 ± 0.048 | -0.104 ± 0.041 | 3.83 | adgato-gemcombat1 (+0.41) | -7 | +0 | plan | emblem order (shop.rs profiles; 14-17 engine 2.0.0 families) |
| CAMP_FEASIBLE | 23 | map | 0 | 100% | -0.161 ± 0.047 | -0.130 ± 0.038 | 5.53 | galaxy (+0.39) | -171 | +0 | plan | infeasible-group discount, early ramp |
| CAMP_FEASIBLE | 23 | map | 25 | 92% | -0.086 ± 0.041 | -0.099 ± 0.034 | 2.03 | guardians (+0.37) | +1502 | +0 | plan | infeasible-group discount, early ramp |
| CAMP_SAFE | 17 | turn | 1 | 92% | -0.040 ± 0.041 | -0.047 ± 0.036 | 3.02 | galaxy (+0.43) | +763 | +0 | plan | leave awake camps / don't wake unsurvivable |
| CAMP_SAFE | 17 | turn | 3 | 100% | +0.021 ± 0.038 | -0.010 ± 0.032 | 2.81 | expanse (+0.35) | +1438 | +0 | plan | leave awake camps / don't wake unsurvivable |
| COMBAT_FSM | 31 | turn | 0 | 100% | +0.011 ± 0.048 | +0.000 ± 0.039 | 9.78 | adgato-gemcombat1 (+1.04) | -168 | +0 | plan | fight hysteresis / help bits |
| COMBAT_FSM | 31 | turn | 193 | 100% | -0.027 ± 0.050 | -0.016 ± 0.040 | 10.52 | adgato-gemcombat1 (+0.99) | +234 | +0 | plan | fight hysteresis / help bits |
| COMBAT_SEARCH | 16 | turn | 1 | 100% | -0.045 ± 0.049 | -0.031 ± 0.038 | 4.85 | galaxy (+0.74) | +2777 | +0 | plan | skirmish portfolio search (fuel) |
| DENY_TRIBUTE | 13 | turn | 0 | 100% | -0.034 ± 0.040 | -0.036 ± 0.033 | 8.65 | adgato-gemcombat1 (+1.05) | +38 | +0 | plan | stun/lock enemy tributes |
| DENY_TRIBUTE | 13 | turn | 4 | 100% | -0.251 ± 0.056 | -0.193 ± 0.044 | 5.11 | adgato-gemcombat1 (+0.48) | -676 | +0 | plan | stun/lock enemy tributes |
| ENDGAME_BANK | 14 | turn | 0 | 27% | +0.003 ± 0.010 | +0.005 ± 0.009 | 0.47 | eight (+0.07) | +9 | +0 | leave | bank all from round 1850 |
| EN_ROUTE | 29 | map | 0 | 100% | -0.163 ± 0.054 | -0.130 ± 0.045 | 11.87 | adgato-gemcombat1 (+1.02) | +870 | +0 | plan | small camps on the way |
| EN_ROUTE | 29 | map | 2 | 92% | -0.191 ± 0.052 | -0.146 ± 0.043 | 6.66 | expanse (+0.26) | -647 | +0 | plan | small camps on the way |
| FIGHT_MARGIN_PCT | 7 | turn | 5 | 93% | -0.138 ± 0.045 | -0.125 ± 0.036 | 5.55 | galaxy (+0.26) | +152 | +0 | plan | router fight threshold |
| FIGHT_MARGIN_PCT | 7 | turn | 60 | 92% | -0.036 ± 0.048 | -0.036 ± 0.039 | 2.66 | adgato-gemcombat1 (+0.57) | +678 | +0 | plan | router fight threshold |
| FIGHT_V6 | 21 | turn | 1 | 100% | -0.062 ± 0.037 | -0.047 ± 0.031 | 3.2 | adgato-gemcombat1 (+0.48) | -38 | +0 | plan | focus / join / finish bits |
| FIGHT_V6 | 21 | turn | 7 | 100% | -0.044 ± 0.043 | -0.042 ± 0.037 | 2.86 | adgato-gemcombat1 (+0.40) | +303 | +0 | plan | focus / join / finish bits |
| HUNT_MODE | 12 | turn | 1 | 100% | -0.123 ± 0.045 | -0.099 ± 0.037 | 2.12 | galaxy (+0.30) | -432 | +0 | plan | whole-team hunt |
| HUNT_MODE | 12 | turn | 2 | 100% | +0.007 ± 0.006 | +0.005 ± 0.005 | 0.55 | split (+0.07) | +58 | +0 | leave | whole-team hunt |
| K2_CLUSTER | 56 | map | 10 | 100% | -0.120 ± 0.046 | -0.099 ± 0.037 | 1.69 | shortcut (+0.10) | +1415 | +352 | plan | cluster value (per-turn compute: fuel) |
| K2_DRAFT_PLAN | 61 | map | 1 | 90% | -0.097 ± 0.043 | -0.078 ± 0.035 | 7.17 | adgato-gemcombat1 (+0.58) | -140 | +0 | plan | v3 plan readers (bits) |
| K2_DRAFT_PLAN | 61 | map | 16 | 100% | -0.075 ± 0.037 | -0.068 ± 0.029 | 2.87 | expanse (+0.41) | +156 | +192 | plan | v3 plan readers (bits) |
| K2_DRAFT_PLAN | 61 | map | 29 | 100% | -0.130 ± 0.055 | -0.125 ± 0.045 | 6.06 | galaxy (+0.59) | +7 | +192 | plan | v3 plan readers (bits) |
| K2_DRAFT_PLAN | 61 | map | 4 | 95% | -0.186 ± 0.045 | -0.167 ± 0.036 | 2.54 | rushbait-v2 (+0.02) | +338 | +0 | plan | v3 plan readers (bits) |
| K2_DRAFT_PLAN | 61 | map | 8 | 92% | -0.116 ± 0.047 | -0.104 ± 0.042 | 6.97 | adgato-gemcombat1 (+0.52) | -175 | +0 | plan | v3 plan readers (bits) |
| K2_EARLY_GOLD | 51 | map | 1 | 84% | -0.094 ± 0.040 | -0.078 ± 0.034 | 1.87 | galaxy (+0.34) | +353 | +0 | plan | early gold bonus |
| K2_EASE_FIRST | 49 | map | 1 | 100% | -0.032 ± 0.046 | -0.021 ± 0.038 | 2.72 | adgato-gemcombat1 (+0.51) | +414 | +0 | plan | easier groups first alone |
| K2_ENDGAME | 34 | turn | 0 | 100% | -0.044 ± 0.018 | -0.042 ± 0.016 | 1.83 | eight (+0.07) | +19 | +0 | plan | endgame mode |
| K2_FOCUS | 33 | turn | 0 | 100% | -0.176 ± 0.050 | -0.156 ± 0.040 | 6.84 | adgato-gemcombat1 (+0.58) | +326 | +0 | plan | team focus target |
| K2_GOLD_ORDER | 53 | turn | 1 | 100% | -0.004 ± 0.017 | -0.005 ± 0.016 | 0.29 | split (+0.08) | +52 | +0 | leave | who takes gold rewards |
| K2_GOLD_ORDER | 53 | turn | 2 | 100% | -0.008 ± 0.012 | -0.010 ± 0.010 | 0.24 | split (+0.04) | +115 | +0 | leave | who takes gold rewards |
| K2_JUMP_HOLES | 62 | map | 1 | 98% | +0.036 ± 0.048 | +0.042 ± 0.038 | 8.84 | adgato-gemcombat1 (+1.06) | -18 | +0 | plan | dash champions path over 1-wide hole gaps |
| K2_KEEP_BUYING | 57 | map | 0 | 100% | -0.138 ± 0.025 | -0.115 ± 0.023 | 4.14 | anvil (+0.01) | -41 | +0 | plan | keep buying the last emblem |
| K2_LOS_RETREAT | 36 | turn | 1 | 94% | -0.231 ± 0.055 | -0.172 ± 0.043 | 10.55 | adgato-gemcombat1 (+0.38) | -343 | +0 | plan | retreat out of line of sight |
| K2_PASSIVE_AWARE | 32 | map | 1 | 100% | -0.121 ± 0.047 | -0.104 ± 0.038 | 2.98 | guardians (+0.20) | +314 | +0 | plan | generic passive rules |
| K2_PUNISH_RECALL | 45 | turn | 6 | 100% | +0.016 ± 0.018 | +0.010 ± 0.015 | 1.15 | split (+0.16) | +53 | +0 | leave | attack lone recallers within N |
| K2_SOLO_CAP | 48 | map | 3 | 100% | -0.111 ± 0.038 | -0.109 ± 0.033 | 2.26 | ambush (+0.19) | +858 | +0 | plan | don't solo big groups |
| K2_STUN_LOCK | 40 | turn | 1 | 100% | +0.022 ± 0.044 | +0.016 ± 0.036 | 3.15 | galaxy (+0.53) | +105 | +0 | plan | chain stuns |
| K2_TEAM_UP | 42 | turn | 0 | 100% | -0.281 ± 0.044 | -0.219 ± 0.038 | 6.78 | expanse (-0.00) | -92 | +0 | plan | join ally fights within N tiles |
| K2_TEAM_UP | 42 | turn | 12 | 98% | -0.057 ± 0.036 | -0.047 ± 0.029 | 4.22 | adgato-gemcombat1 (+0.53) | -56 | +0 | plan | join ally fights within N tiles |
| K2_TP_CRYSTAL | 52 | map | 1 | 100% | -0.016 ± 0.028 | -0.016 ± 0.024 | 1.53 | expanse (+0.13) | +475 | +0 | leave | teleport crystal |
| K2_TRIBUTE_DETOUR | 63 | map | 0 | 100% | +0.035 ± 0.031 | +0.036 ± 0.027 | 3.79 | galaxy (+0.55) | -21 | +0 | plan | rewards / weak camps on the way to an altar |
| K2_TRIBUTE_DETOUR | 63 | map | 1 | 100% | +0.082 ± 0.043 | +0.062 ± 0.034 | 7.04 | adgato-gemcombat1 (+1.06) | +181 | +0 | plan | rewards / weak camps on the way to an altar |
| K2_UNSTICK | 46 | turn | 1 | 100% | -0.071 ± 0.045 | -0.057 ± 0.036 | 3.99 | adgato-gemcombat1 (+0.57) | +383 | +0 | plan | recall when oscillating |
| K2_UPGRADE_FIRST | 41 | turn | 10 | 100% | -0.090 ± 0.046 | -0.083 ± 0.038 | 3.62 | adgato-gemcombat1 (+0.52) | +238 | +0 | plan | recall before hard fights |
| K2_WALK_HOME | 47 | turn | 0 | 100% | -0.118 ± 0.042 | -0.104 ± 0.035 | 3.63 | adgato-gemcombat1 (+0.54) | +135 | +0 | plan | walk instead of recall near spawn |
| K2_WALK_HOME | 47 | turn | 12 | 74% | -0.056 ± 0.036 | -0.042 ± 0.030 | 3.34 | adgato-gemcombat1 (+0.46) | +20 | +0 | plan | walk instead of recall near spawn |
| K2_WEAK_CAMP_FIRST | 50 | turn | 1 | 100% | -0.135 ± 0.045 | -0.120 ± 0.038 | 3.9 | expanse (+0.33) | +106 | +0 | plan | weakest monster first |
| MONOLITH_LIVE | 27 | map | 0 | 100% | -0.006 ± 0.028 | -0.016 ± 0.025 | 1.91 | expanse (+0.41) | +305 | +0 | leave | monolith gate on live damage |
| PASSIVE_ROLES | 24 | map | 1 | 100% | +0.008 ± 0.028 | +0.005 ± 0.024 | 0.97 | galaxy (+0.20) | +58 | +0 | leave | Vesper / Prism passive roles |
| RECALL_GOLD | 6 | map | 40 | 88% | +0.027 ± 0.044 | +0.021 ± 0.038 | 4.9 | adgato-gemcombat1 (+0.44) | +106 | +0 | plan | shop trip gold x10 |
| RECALL_GOLD | 6 | map | 6 | 100% | -0.059 ± 0.041 | -0.052 ± 0.034 | 3.18 | adgato-gemcombat1 (+0.52) | +129 | +0 | plan | shop trip gold x10 |
| RECALL_SAFE | 20 | turn | 0 | 100% | +0.036 ± 0.036 | +0.026 ± 0.028 | 2.21 | adgato-gemcombat1 (+0.58) | +136 | +0 | plan | recall cancel / start rule |
| RECALL_SAFE | 20 | turn | 1 | 95% | -0.072 ± 0.051 | -0.057 ± 0.040 | 8.76 | adgato-gemcombat1 (+1.07) | +178 | +0 | plan | recall cancel / start rule |
| REMOTE_FARMER | 11 | map | 1 | 97% | -0.108 ± 0.053 | -0.094 ± 0.041 | 5.87 | adgato-gemcombat1 (+0.63) | +133 | +0 | plan | last slot farms far groups |
| RETREAT_HP_PCT | 1 | turn | 15 | 100% | -0.309 ± 0.052 | -0.146 ± 0.042 | 6.46 | adgato-gemcombat1 (+0.80) | -2613 | +0 | plan | low_hp threshold |
| RETREAT_HP_PCT | 1 | turn | 60 | 100% | -0.225 ± 0.052 | -0.229 ± 0.044 | 5.02 | adgato-gemcombat1 (+0.51) | +1664 | +0 | plan | low_hp threshold |
| SAFE_ALTAR | 26 | map | 0 | 94% | +0.013 ± 0.019 | +0.010 ± 0.015 | 1.09 | galaxy (+0.42) | -32 | +0 | leave | avoid altars by the enemy spawn |
| SHOP_FIRST | 28 | map | 1 | 92% | -0.151 ± 0.040 | -0.151 ± 0.033 | 1.44 | adgato-gemcombat1 (+0.01) | +419 | +0 | plan | recall to shop while carrying |
| SQUAD_JOIN_X10 | 9 | map | 15 | 100% | -0.073 ± 0.055 | -0.047 ± 0.044 | 12.27 | adgato-gemcombat1 (+0.94) | +397 | +0 | plan | pairs join each other's group |
| STUN_THREAT | 22 | turn | 1 | 92% | -0.058 ± 0.047 | -0.062 ± 0.037 | 3.08 | adgato-gemcombat1 (+0.65) | +603 | +0 | plan | stun kits on a 6-round horizon |
| TAVI_META | 19 | map | 0 | 100% | -0.129 ± 0.055 | -0.109 ± 0.044 | 12.01 | adgato-gemcombat1 (+1.01) | -17 | +0 | plan | Tavi field shop / build / hunt (draft list is compiled) |
| TEAM_FIGHT | 15 | turn | 1 | 100% | +0.045 ± 0.043 | +0.010 ± 0.037 | 8.58 | adgato-gemcombat1 (+1.02) | +1212 | +0 | plan | Lanchester skirmish rule |
| TEAM_RESPAWN | 30 | turn | 0 | 100% | -0.135 ± 0.051 | -0.120 ± 0.043 | 5.6 | adgato-gemcombat1 (+0.53) | +39 | +0 | plan | respawn broadcast in the channel |
| TRIBUTE_DISTANCE_SCALE | 4 | map | 25 | 99% | -0.023 ± 0.048 | -0.042 ± 0.038 | 3.83 | adgato-gemcombat1 (+0.64) | +450 | +0 | plan | gems per altar tile x0.1 |
| TRIBUTE_DISTANCE_SCALE | 4 | map | 3 | 98% | -0.057 ± 0.049 | -0.057 ± 0.041 | 4.17 | adgato-gemcombat1 (+0.57) | +285 | +0 | plan | gems per altar tile x0.1 |
| TRIBUTE_MAX_GEMS | 3 | map | 220 | 28% | -0.058 ± 0.024 | -0.052 ± 0.022 | 1.25 | split (+0.07) | +21 | +0 | plan | bag cap |
| TRIBUTE_MAX_GEMS | 3 | map | 50 | 95% | +0.099 ± 0.033 | +0.073 ± 0.028 | 2.78 | adgato-gemcombat1 (+0.55) | +334 | +0 | plan | bag cap |
| TRIBUTE_MIN_GEMS | 2 | map | 10 | 100% | +0.038 ± 0.050 | +0.021 ± 0.040 | 8.92 | adgato-gemcombat1 (+1.03) | +305 | +0 | plan | draft writes 25+0.4 walk; this overrides it |
| TRIBUTE_MIN_GEMS | 2 | map | 60 | 93% | -0.002 ± 0.046 | -0.005 ± 0.039 | 8.38 | adgato-gemcombat1 (+0.95) | +350 | +0 | plan | draft writes 25+0.4 walk; this overrides it |
| VALUE_PROFILE | 10 | map | 0 | 100% | +0.011 ± 0.044 | -0.005 ± 0.037 | 7.99 | adgato-gemcombat1 (+0.97) | +545 | +0 | plan | camp value formula |
| VALUE_PROFILE | 10 | map | 5 | 100% | -0.153 ± 0.043 | -0.130 ± 0.036 | 4.63 | expanse (+0.36) | +157 | +0 | plan | camp value formula |
| VALUE_PROFILE | 10 | map | 6 | 92% | -0.086 ± 0.044 | -0.068 ± 0.038 | 2.85 | expanse (+0.32) | +281 | +0 | plan | camp value formula |
| draft | draft | map | 1 | 100% | -0.211 ± 0.047 | -0.224 ± 0.038 | 11.28 | guardians (+0.49) | +744 | +0 | plan | roster (draft_prefs profile) |
| draft | draft | map | 10 | 100% | +0.040 ± 0.043 | +0.021 ± 0.038 | 3.99 | galaxy (+0.52) | +527 | +0 | plan | roster (draft_prefs profile) |
| draft | draft | map | 11 | 100% | -0.014 ± 0.049 | -0.036 ± 0.041 | 5.86 | adgato-gemcombat1 (+0.59) | +815 | +0 | plan | roster (draft_prefs profile) |
| draft | draft | map | 12 | 100% | -0.105 ± 0.045 | -0.109 ± 0.039 | 4.16 | expanse (+0.25) | +751 | +0 | plan | roster (draft_prefs profile) |
| draft | draft | map | 13 | 100% | +0.022 ± 0.048 | -0.026 ± 0.040 | 3.88 | galaxy (+0.48) | +1360 | +0 | plan | roster (draft_prefs profile) |
| draft | draft | map | 14 | 100% | -0.062 ± 0.050 | -0.088 ± 0.041 | 9.52 | galaxy (+0.69) | +916 | +0 | plan | roster (draft_prefs profile) |
| draft | draft | map | 2 | 100% | -0.199 ± 0.060 | -0.208 ± 0.050 | 18.72 | adgato-gemcombat1 (+0.51) | +990 | +0 | plan | roster (draft_prefs profile) |
| draft | draft | map | 3 | 100% | -0.133 ± 0.045 | -0.104 ± 0.039 | 6.83 | expanse (+0.25) | +455 | +32 | plan | roster (draft_prefs profile) |
| draft | draft | map | 5 | 100% | +0.089 ± 0.041 | -0.010 ± 0.034 | 3.22 | expanse (+0.48) | +2046 | +5 | plan | roster (draft_prefs profile) |
| draft | draft | map | 6 | 100% | -0.010 ± 0.046 | -0.073 ± 0.039 | 4.05 | adgato-gemcombat1 (+0.55) | +1270 | +0 | plan | roster (draft_prefs profile) |
| draft | draft | map | 7 | 100% | -0.102 ± 0.051 | -0.135 ± 0.043 | 7.41 | expanse (+0.46) | +1356 | +0 | plan | roster (draft_prefs profile) |
| draft | draft | map | 8 | 100% | -0.229 ± 0.050 | -0.188 ± 0.042 | 4.71 | expanse (+0.47) | +190 | +0 | plan | roster (draft_prefs profile) |
| style | style | map | aggro | 93% | -0.120 ± 0.043 | -0.099 ± 0.035 | 2.41 | galaxy (+0.33) | -37 | +0 | plan | data/styles.json bundles |
| style | style | map | bank | 100% | -0.025 ± 0.050 | -0.031 ± 0.041 | 4.05 | adgato-gemcombat1 (+0.49) | +159 | +0 | plan | data/styles.json bundles |
| style | style | map | careful | 100% | -0.115 ± 0.049 | -0.141 ± 0.042 | 3.99 | adgato-gemcombat1 (+0.59) | +1324 | +0 | plan | data/styles.json bundles |
| style | style | map | contest | 100% | -0.120 ± 0.052 | -0.104 ± 0.042 | 3.32 | adgato-gemcombat1 (+0.58) | +240 | +0 | plan | data/styles.json bundles |
| style | style | map | farmer | 100% | -0.257 ± 0.049 | -0.213 ± 0.041 | 4.36 | expanse (+0.35) | +773 | +0 | plan | data/styles.json bundles |
| style | style | map | hunt | 100% | +0.007 ± 0.006 | +0.005 ± 0.005 | 0.55 | split (+0.07) | +58 | +0 | leave | data/styles.json bundles |
| style | style | map | safe | 100% | +0.008 ± 0.051 | -0.026 ± 0.041 | 1.96 | adgato-gemcombat1 (+0.62) | +1604 | +0 | leave | data/styles.json bundles |
| style | style | map | search | 100% | -0.045 ± 0.049 | -0.031 ± 0.038 | 4.85 | galaxy (+0.74) | +2777 | +0 | plan | data/styles.json bundles |
| style | style | map | teamfight | 100% | +0.045 ± 0.043 | +0.010 ± 0.037 | 8.58 | adgato-gemcombat1 (+1.02) | +1212 | +0 | plan | data/styles.json bundles |

Dead (never read in play, slots free): K2_SPACING, K2_CAMP_DIVERT, K2_PREDICT, K2_WARDS, DRAFT_BEST, MONOLITH_MIN_POWER
