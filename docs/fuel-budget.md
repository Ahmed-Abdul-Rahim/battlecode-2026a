# Fuel budget

Every bot runs in Wasmtime with a fuel limit: roughly one unit per wasm instruction. A Rust bot gets **100,000 per turn**, **1,000,000** on its team's first draft turn and **200,000** on each champion's first game turn. Unused fuel doesn't carry over. Running out suspends the bot mid-code: it resumes next round and has effectively **lost a turn**.

Numbers below were measured on `ambush` (100×100) with `bots/fuelbench` (`cambc run fuelbench baseline --rounds 20`), cambc 2.2.5. Re-run it after SDK updates.

## Measured costs

| Operation | Fuel | Notes |
| --- | ---: | --- |
| `fuel_remaining()` | 5 | |
| 1,000 iterations of a trivial loop | 2,031 | about 2 per iteration |
| `println!` (short) | 362 | only inside `debug_block!` |
| `Bitboard::new` / `clone` / `union_assign` | 457 / 359 / 224 | |
| `vision::champion_tiles(None)` | 1,608 | |
| `vision::camp_tiles()` / `visible_tiles()` | 422 / 422 | |
| **`champions(None)`** (full roster) | **12,643** | avoid per turn; use `vision::champions(Some(team))` |
| `vision::champions(Some(Enemy))` | 1,067 | |
| `vision::camps()` | 1,043 | |
| **`map::camp_groups()`** | **7,987** | cache once |
| Self queries (`position`, `health`, …) | 80–170 each | `state()` gives all flags in one call |
| `item_count()` | 438 | |
| `attack_tiles()` | 680 | |
| `DistanceField::new` | 304 | |
| **`DistanceField::compute` (full map)** | **19,383** | precompute on spare fuel, cache |
| `DistanceField::compute_to` (radius 12) | 4,851 | |
| `next_moves_from` | 1,531 | 8 × `get` is cheaper (~600) |
| `DistanceField::get` | ~74 | per call |
| `Bitboard::is_set` | ~50 | per call |
| `reachable_tiles` (r1 / r8) | 505 / 542 | |
| `shadow_cast` (r8) | 759 | |
| `Bitboard::positions()` | 629 | |
| `Bitboard::export_linear_buffer` | 1,635–1,759 | once at start-up |
| **`DistanceField::export_linear_buffer`** | **22,812** | avoid |
| Guest BFS over a 15×15 window | ~1,300 | |
| **Guest scan of every tile of a 100×100 map** | **>100,000** | never per turn |
| int8 layer 64×32 (2,048 MACs) | 5,915 | about 3 per MAC |
| int8 layer 32×16 | 4,123 | |

## Lessons (each one cost us turns)

- **`BitGrid::positions()` scans every tile in guest code:** about 150k fuel, and it ran the first turn out. Use the SDK's `positions()`.
- **Bitwise CRC-16 over the 1 KB draft memory:** about 50k fuel, and it ran the draft bot out of fuel on pick 2. The CRC is now table-driven.
- **Rust's `HashMap` hashes with SipHash:** about 150 fuel per lookup. Use dense arrays indexed by tile (`world::TileIndex`).
- **Windowed A\* costs about 3k fuel per expansion:** heap operations over 8 neighbours. Tier 3 is now a cheap greedy step by default, and A\* runs only with spare fuel and a hard cap (`DEEP_SEARCH_MIN_FUEL`, 8 expansions).
- **The draft's second and later picks get only 100k:** cache everything from the first pick and write θ_match only on the first and last picks.
- **Telemetry arguments outside `debug_block!` still cost fuel in release:** format strings, feature extraction. Keep all of it inside the block.

## Stage budgets

`bc_core::fuel::FuelGovernor` defaults, for a normal 100k turn:

| Stage | Budget | What runs |
| --- | ---: | --- |
| Observe | 15,000 | self, vision, camp-group state, team channel |
| Gate | 1,000 | meta-gate |
| Brain | 15,000 | target choice, context, router, learned heads |
| Path | 40,000 | tiers 1–4 |
| Act | 5,000 | attacks, casts, buys |
| Precompute | 10,000 | distance fields, from spare fuel |
| Reserve | 10,000 | never spent |

Current v0 bot, across the gate run: mean about 27k per turn, p95 about 40–68k depending on the map, **0 exhausted turns**. Check with `bctools summary <bundle>` or the MCP `fuel_report` tool.
