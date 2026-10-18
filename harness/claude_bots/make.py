"""Create bots/claude-<strategy>/ from bots/starter-python and harness/claude_bots/bot.py.

python harness/claude_bots/make.py                # all strategies
python harness/claude_bots/make.py ambusher
"""

from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(__file__).with_name("bot.py")
STRATEGIES = ["ambusher", "pillager", "focus", "swarm"]
# Draft preference lists (ChampionDraft values): ranged-only / melee-only compositions.
DRAFTS = {
    "ranged": [4, 5, 1, 10, 7],  # Stellar, Prism, Vesper Vale, Tavi, Willow
    "melee": [9, 2, 8, 0, 6],  # Varg, Vendetta, Ren, Ryth, Marcella
}
DRAFT_PY = """import cambc
from randomness import below

PREFS = {prefs}


def play():
    chosen = cambc.draft.current_draft(cambc.Team.ALLY)
    for pick in PREFS:
        if pick not in chosen:
            cambc.draft.pick(pick)
            return
    available = [pick for pick in range(11) if pick not in chosen]
    cambc.draft.pick(available[below(len(available))])
"""


def make(strategy: str, draft: str | None = None) -> Path:
    d = ROOT / "bots" / (f"claude-{strategy}-{draft}" if draft else f"claude-{strategy}")
    if d.exists():
        shutil.rmtree(d)
    shutil.copytree(ROOT / "bots" / "starter-python", d)
    code = re.sub(
        r'^STRATEGY = ".*"$', f'STRATEGY = "{strategy}"', SOURCE.read_text(encoding="utf-8"), flags=re.M
    )
    (d / "src" / "bot.py").write_text(code, encoding="utf-8")
    if draft:
        (d / "src" / "draft.py").write_text(DRAFT_PY.format(prefs=DRAFTS[draft]), encoding="utf-8")
    return d


if __name__ == "__main__":
    for arg in sys.argv[1:] or STRATEGIES:
        strategy, _, draft = arg.partition(":")
        print(make(strategy, draft or None))
