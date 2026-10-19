"""Hand the training setup to another PC: the gitignored files git doesn't carry.

    python scripts/handoff.py pack                  # -> handoff.zip (~3 MB)
    python scripts/handoff.py pack --with-ladder    # + ladder match summaries (~150 MB)
    python scripts/handoff.py restore handoff.zip   # on the other PC, inside the clone
    python scripts/handoff.py refs                  # only rebuild bots/main-v* from submission/*.zip

What goes in (all gitignored on purpose):
  submission/*.zip                 every packaged version; restore unzips them into
                                   bots/main-vN (frozen opponents) and bots/co*-vN (champions)
  runs/coevo/state.json            co-evolution lineages, halls of fame, Elo (continue a run)
  runs/league/state.json           league trainer champion, tracks, ladder history
  runs/mimic/                      extracted top-team clones (also committed in data/mimic)
  bots/lin-b, bots/lin-c           source of co-evolution lineages B and C
  best_bot_dump/                   the 5 games vs the best bot (analysis)
  --with-ladder: runs/ladder_v8, ladder_v10, ladder_top, ladder_live, ranked_v5
Without this zip a fresh clone still trains: delete nothing, just run scripts/train.ps1
start; the daemons re-seed from the committed champion (learned.rs, pack.bin).
"""

from __future__ import annotations

import argparse
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP = {".cambc", "target"}
BASE = ["submission", "runs/coevo/state.json", "runs/league/state.json", "runs/mimic",
        "bots/lin-b", "bots/lin-c", "best_bot_dump"]
LADDER = ["runs/ladder_v8", "runs/ladder_v10", "runs/ladder_top", "runs/ladder_live", "runs/ranked_v5"]


def files(rel: str):
    p = ROOT / rel
    if p.is_file():
        yield p
    elif p.is_dir():
        for f in p.rglob("*"):
            if f.is_file() and not SKIP & set(f.relative_to(ROOT).parts):
                if f.suffix in (".repl26a",) and "ladder" not in rel and "ranked" not in rel:
                    continue
                yield f


def pack(with_ladder: bool, out: Path) -> None:
    items = BASE + (LADDER if with_ladder else [])
    n = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for rel in items:
            for f in files(rel):
                z.write(f, f.relative_to(ROOT).as_posix())
                n += 1
    print(f"wrote {out} ({n} files, {out.stat().st_size / 1e6:.1f} MB)")


def bot_name(zip_name: str) -> str:
    """main-v5-bag30.zip -> main-v5; coA-v2.zip -> coA-v2; main-v13b-es.zip -> main-v13b."""
    stem = zip_name[:-4]
    m = re.match(r"(main-v\d+[a-z]?|co[A-Z]-v\d+|auto-v\d+)", stem)
    return m.group(1) if m else stem


def refs() -> None:
    for z in sorted((ROOT / "submission").glob("*.zip")):
        name = bot_name(z.name)
        dst = ROOT / "bots" / name
        if dst.exists() or name.startswith("main-sprint"):
            continue
        dst.mkdir(parents=True)
        with zipfile.ZipFile(z) as zz:
            zz.extractall(dst)
        print(f"restored bots/{name} from {z.name}")


def restore(path: Path) -> None:
    with zipfile.ZipFile(path) as z:
        z.extractall(ROOT)
    print(f"extracted {path}")
    refs()


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("pack")
    a.add_argument("--with-ladder", action="store_true")
    a.add_argument("--out", type=Path, default=ROOT / "handoff.zip")
    r = sub.add_parser("restore")
    r.add_argument("zip", type=Path)
    sub.add_parser("refs")
    args = p.parse_args(argv)
    if args.cmd == "pack":
        pack(args.with_ladder, args.out)
    elif args.cmd == "restore":
        restore(args.zip)
    else:
        refs()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
