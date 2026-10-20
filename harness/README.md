# harness/

The league loop and the helpers it uses. Full guide: [docs/league.pdf](../docs/league.pdf).

| Script | What it does |
| --- | --- |
| `league.py` | The loop: pull → (lead) map pool + ladder sync + watch → per-map ES vs mimics (or, every 3rd cycle, ES on an evolved mimic vs our champion) → promote → push → (lead) release + review. Subcommands `run`, `status`, `sync`, `maps`, `bench`, `review`, `submit` |
| `mimic.py` | Clone a ladder team's macro (draft, builds, openings, cadence) from match summaries into a pack spec. Micro (fights) is not cloned: the faithful mimic plays our engine's micro, and the evolved mimic (`league/mimics/`) learns its own. Also `map_name`, `geometry_from_dump` |
| `params.py` | Registry of the 102 trainable parameters (layout, scale, bounds, ES step). `--check` vs `bots/main/src/learned.rs` |
| `pack.py` | Policy-pack writer (`pack.bin`: θ, draft, builds, openings, hotspots per map) |
| `ladder_study.py` | Per-team tempo / survival / comp / build table from downloaded match summaries (for human review) |

Adding a trainable parameter: append it to a block in `params.py` and in `learned.rs`, use it from Rust, then run `python harness/params.py --check`. The daemons refuse to start on a layout mismatch.
