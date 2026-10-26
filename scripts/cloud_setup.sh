#!/usr/bin/env bash
# Set up the plan search (harness/search) on a fresh Linux checkout, e.g. a Claude Code cloud
# session. Run from the repository root on branch feat/plan-search:
#   bash scripts/cloud_setup.sh
# It installs cambc + numpy + cmaes, unpacks the opponents, map cache, live pack and search
# states from branch `search-data`, and builds the slot pool. See docs/plan-search.md.
set -euo pipefail
cd "$(dirname "$0")/.."

python3 -m pip install --quiet --upgrade cambc numpy cmaes
git fetch --quiet origin search-data
get() { git show "origin/search-data:$1"; }

mkdir -p bots runs/gpusim runs/search
get opponents.tar.gz | tar -xz -C bots               # 75 mimics / exploiters / baseline / lg-v23-fuel
get opponents-fresh.tar.gz | tar -xz -C bots         # our rule bot (_rules) + the fresh top 10
get mapcache.tar.gz | tar -xz -C runs/gpusim         # map features for the generalisation study
get search-states.tar.gz | tar -xz -C runs/search    # per-map search states + result summaries
get live-pack.bin > runs/search/live-pack.bin        # the pack the search's slots were built with

# The slots must use the live pack (bots/main's pack.bin on this branch holds the plan table).
export BC_MAIN="$PWD"
python3 - <<'EOF'
import sys
sys.path.insert(0, ".")
from pathlib import Path
from harness.search import common as C
C.runner.build(str(Path("bots/_rules")), release=True)  # archived league rule bot, as the search used it
snap = C.build_slots(pack=Path("runs/search/live-pack.bin"))
print("slots built:", snap["wasm"][:12])
EOF
echo "done. Next: export BC_MAIN=\$PWD; python3 -m harness.search status   (docs/plan-search.md)"
