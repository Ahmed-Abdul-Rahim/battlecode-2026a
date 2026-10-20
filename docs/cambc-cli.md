# cambc CLI notes (3.0.1, engine 1.0.0)

Install or update with `python -m pip install --upgrade cambc`. It lives in `.venv`; `scripts/setup.ps1` does this.

> **Platform commands:** only the league's lead PC reads the platform (`ladder`, `matches`), and uploads go only through `harness/league.py` (see CONTRIBUTING.md and docs/league.pdf). Never run `cambc submit` or `match-groups` by hand.

## Commands we use

| Command | What it does |
| --- | --- |
| `cambc init --yes .` | Create or refresh the project: `sdk/`, `maps/`, `cambc.toml`, starter bots. Downloads wasi-sdk/cmake once into `%LOCALAPPDATA%\cambc`. |
| `cambc new <name> --lang rust` | New bot from `sdk/templates/rust`. |
| `cambc build <bot> [--release]` | Build the WASI module. `--release` matches the platform: `debug_block!` is skipped and fuel-pause imports aren't linked. |
| `cambc run <a> <b> [--map M] [--rounds N] [--statistics out.json --json] [-o dir] [--log-limit BYTES]` | One local match, about 1 s for 2000 rounds. |
| `cambc run-batch manifest.txt --jobs N --json --log-limit 20000000` | Many matches in parallel. Manifest rows: `botA botB map.map26a out.repl26a --statistics out.json`. **Use absolute paths**: relative ones resolve against the manifest's directory. |
| `cambc view [bundle]` | Open a replay in the visualiser. |
| `cambc maps pull` | Refresh `maps/` from the platform. |

Day to day, use `bctools run` / `bctools gate` instead. They build once, write the manifest, play both sides and summarise.

## Quirks we hit

- **`cambc init` dates sdk/ files 1980, so Cargo keeps a stale compiled SDK** in every bot's `.cambc` cache. After the 3.0.1 refresh, all local bots still had the old SDK and stood still on every map with a Golem ("invalid CampType value: 5"). `bctools.runner` now re-dates such files before building; by hand: `find sdk/rust -exec touch {} +`.
- **3.0.1 (2026-10-20):** the ladder runs engine 1.0.0; cambc 2.2.5 ran engine 0.0.3 and can't `maps pull` (HTTP 303). After upgrading, run `cambc init --yes .` (refreshes `sdk/`, `cambc.toml`, maps). `--statistics` is **binary** unless `--json` is given (bctools passes `--json`). Pulled maps are named `00contest`, `01islands`; `harness/league.py maps` tidies `maps/`.

- **Bot `Cargo.toml` must match the template exactly**, or the build fails with "managed by cambc". So the bot can't depend on our crates by path. `bctools sync` vendors `crates/*` into `bots/main/src/bc_*/` instead, and CI checks the copy is current.
- **Statistics are only written for finished matches.** With `--rounds` you get no `--statistics` file.
- **Bot stdout lines are padded with NUL bytes.** `bctools.replay` strips them.
- **Replay bundle** (`.repl26a`) is a zip containing `replay.repl26a`, `team-{a,b}.stdout`, `team-{a,b}.debug` and `team-{a,b}.fuel`.
- **Fuel log** (`team-x.fuel`): magic `a62leuf\0`, 5 header bytes, 8 × u64 budgets, then one u64 of fuel used per turn: draft picks first, then every champion turn. A turn that used exactly its budget was suspended.
- **Cassi & Issac** are one pick but two bots, so a 5-pick team runs 6 champion bots.
- **"up to date" without rebuilding:** once, cambc skipped a rebuild after an edit. `bctools` always runs `cambc build` explicitly before matches.
- **WASI randomness is always zero**, so seed from map state if you need randomness (see the starter's `random_seed`).
- **Bitboard linear export** is `ceil(w*h/8)` bytes, row-major `index = y*width + x`, LSB-first. `y` grows north.
- **`CampGroup::state()`** needs one of the group's original spawn tiles in vision, otherwise `ApiNotInVision`. `Respawning(0)` means respawn is blocked by an occupied spawn tile.
