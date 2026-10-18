"""Team tooling for the Cambridge Battlecode 2026a bot.

Modules:
    sync     vendor crates/* into bots/main/src (cambc bots must be self-contained)
    replay   read .repl26a bundles: stdout telemetry, fuel logs, statistics
    runner   run batches of local matches through `cambc run-batch`
    stats    win rates with Wilson intervals
    gate     candidate-vs-baseline evaluation gate
    render   PNG frames of a match from telemetry
    dataset  export learned-head training data from telemetry
    cli      `bctools` command line
"""

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _find_cambc() -> Path:
    for candidate in (ROOT / ".venv" / "Scripts" / "cambc.exe", ROOT / ".venv" / "bin" / "cambc"):
        if candidate.exists():
            return candidate
    found = shutil.which("cambc")
    return Path(found) if found else Path("cambc")


CAMBC = _find_cambc()
