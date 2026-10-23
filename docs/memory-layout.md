# Memory layouts (both memories are model parameters)

Codecs live in `crates/bc-core/src/memory.rs`, with round-trip and corruption tests. Change the code and this file together, and bump the version byte whenever a layout changes.

## Draft memory: θ_match (1024 bytes, written in draft, read-only in game)

The draft bot sees the whole map with 10× fuel. It writes a per-match parameter vector that every champion bot reads once on its first turn. This is where map-specific adaptation happens (the UDA angle: one model, conditioned per map).

| Offset | Size | Field |
| ---: | ---: | --- |
| 0 | 2 | magic `BC` |
| 2 | 1 | layout version (1) |
| 3 | 1 | team size |
| 4 | 2 | feature schema hash (u16 LE): must match `features_gen::SCHEMA_HASH` |
| 6 | 2 | weights hash (u16 LE) |
| 8 | 8 | role per draft pick: 0 flex, 1 frontline, 2 ranged, 3 support |
| 16 | 1 | number of camp-priority entries (≤ 15) |
| 18 | 30 | camp priority: (group index, score 1–255) × 15, best first |
| 48 | 96 | **FiLM** (γ, β) int8 × 48 hidden units; scale = 1 + γ/64, identity = (0, 0) (P-032) |
| 144 | 32 | expert knobs (`memory::knob`): aggression, retreat HP %, tribute min/max/distance scale, monolith power, recall gold, fight margin |
| 176 | 160 | **champion vectors** 10 × 16 int8: allies in slots 0–4, enemies in slots 5–9 |
| 336 | 10 | champion kind per slot (0xFF = empty) |
| 352 | 192 | map-descriptor cache (normalised map features × 255) |
| 544 | 478 | reserved (zero) |
| 1022 | 2 | CRC-16/CCITT-FALSE of bytes 0..1022 |

All-zero memory, which happens if the draft bot crashed or never wrote, decodes as `Empty`. Bots then fall back to `ThetaMatch::default()`.

### Draft memory v3: the match plan (`crates/bc-core/src/plan.rs`, 2026-10-23)

The draft analyses the map once (first pick, 1M fuel) and writes a **match plan** into the
regions champions never read in play: the old FiLM block (48–143), the champion vectors
(176–335), the map cache (352–543) and the free tail (580–1021), 890 bytes used as one buffer
(`plan::SEGMENTS`, in that order: 580, 352, 176, 48). Version byte 0 = no plan; champions then
behave as before. The knobs, champion kinds (now including the enemy's picks), knobs2,
fingerprint and CRC stay where they were. The dormant ML router would read the plan bytes as FiLM
/ vectors: decoding zeroes those fields when a plan is present.

| Plan offset | Size | Content |
| ---: | ---: | --- |
| 0 | 12 | version 3, group / cluster / altar / field-order counts, symmetry, seed, counter profile, monolith round / 8, map class |
| 12 | 51 × 8 | per camp group (engine order): walk / 2 from our spawn, from theirs, to the nearest safe altar; that altar's index; class bits (gems, gold, elk, golem, monolith, bush within 3, cold, frontier); clear rounds / 4 and bodies needed for our roster; cluster id |
| 420 | 51 × 3 × 2 | 3 nearest groups by walking distance (index, walk / 2), from one bounded field per group |
| 726 | 12 × 2 | clusters (value class, entry group) |
| 750 | 8 × 2 | altars (safety = 128 + their walk − ours, walk from our spawn) |
| 766 | 6 × 4 | opening: first groups per body slot (rank of champion kind), split so no two share one |
| 790 | 6 | roles per body |
| 796 | 16 | field order: groups whose distance fields champions compute first |

**Fuel.** The map analysis costs ~550k (release, 51 groups) on the first pick; the whole draft memory
for the *predicted* final roster (from the preference list) is built one pick early, so the last pick
(already at ~80k of 100k before) only adds the enemy picks: ~44k. If the roster differs from the
prediction the last pick builds it the old way.

**Readers (knob2 DRAFT_PLAN bits).** 1 frontier discount near enemies; 4 plan distances in
`group_distance` until a field is ready; 8 the draft's opening split when the pack has none; 16 field
order. The 8-byte channel stays live state (leader parity, respawns, claims, sighting); the static
opening split now lives here.

## Shared memory: team channel (8 bytes, read/write all match)

Allies act one after another within a round, in priority order, so later allies see what earlier ones wrote this round.

Layout version **3** (2026-10-21). Bytes 3–7 are one 40-bit little-endian field.

| Byte | Bits | Field |
| ---: | --- | --- |
| 0 | 0–1 | version (0 = never written; 3 = this layout) |
| 0 | 2 | parity of the round the leader last wrote |
| 0 | 3–5 | leader slot |
| 0 | 6 | flag: the sighting is an enemy tributing |
| 0 | 7 | flag: an ally asked for help (cleared by each round's leader) |
| 1 | – | **team aggression parameter** (0–255) |
| 2 | – | **team tribute threshold** (gems) |
| 3–7 | 0–20 | camp claims, 6 slots packed base 11: θ_match priority index + 1 (0 = none; the 10 best camps) |
| 3–7 | 21–36 | latest enemy sighting x:7 y:7 r:2; r = round mod 4; (0,0) = none |
| 3–7 | 37–38 | **sighted enemy health** in quarters; allies within 4 tiles finish an enemy under 50 % (COMBAT_FSM bit 2) |
| 3–7 | 39 | spare |

Version 2 carried help calls with intent and time left; they were measured worse and removed.

**Leader protocol.** The first ally to act in a round sees a stale parity bit and becomes the leader. It clears the per-round flags, ages out sightings older than 2 rounds, and writes bytes 1–2 from `bc_brain::policy::team_params`. When we're ahead it raises aggression less and banks more often; when we're behind late in the match it pushes harder. Every bot reads all 8 bytes as features (`team_params` group) and as inputs to the gate and router.

**TEAM_RESPAWN (knob 30, on since v12).** Bytes 1–2 instead carry a *respawn broadcast*: byte 1 = camp group + 1 that someone just saw go down (0 = none), byte 2 = its respawn round mod 256 (unwrapped to the window [round − 100, round + 155]). Allies load it into their respawn belief, so they stop walking to camps a teammate cleared. Aggression and tribute threshold are then computed locally by every bot (`Bot::team_view`); they are a pure function of public state.

Bytes 3–7 go through `TeamChannel`. A learned discrete protocol (DIAL P-030, with an IMAC entropy budget P-031) can replace this hand-written codec later.
