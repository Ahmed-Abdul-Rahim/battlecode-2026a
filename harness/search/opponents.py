"""Opponent pool for a self-contained checkout (engine 2.0.0 worktree on the PC).

`opponents()` lists only bots that are built under BC_MAIN/bots, so a fresh worktree needs its
pool put together once:

    python -m harness.search opponents [--from D:\\Code\\battlecode\\bots]

1. copies the sources of the old pool (mimic-*, xpl-*, lg-v23-fuel) from another checkout, if
   given (they are gitignored; the old copies are the faithful-mimic and exploiter bots);
2. builds the refreshed top 10 from league/ledger.json + data/mimic specs into bots/top-<slug>
   (this branch's bot source with each team's pack; final test only, never a search opponent);
3. builds `_rules`: this branch's bot with the reference pack (no per-map table);
4. builds everything missing in release, four at a time, and lists what failed.

Set BC_MAIN to the repository root for every harness command (scripts/search.ps1 does).
"""

from __future__ import annotations

import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from harness.search import common as C

COPY = ("Cargo.toml", "Cargo.lock", "rust-toolchain.toml", ".gitignore")


def copy_sources(src_root: Path) -> list[str]:
    done = []
    for d in sorted(src_root.iterdir()):
        n = d.name
        if not (n.startswith(("mimic-", "xpl-")) or n == "lg-v23-fuel") or not (d / "src").is_dir():
            continue
        dst = C.ROOT / "bots" / n
        if (dst / "src").exists():
            continue
        dst.mkdir(parents=True, exist_ok=True)
        for f in COPY:
            if (d / f).exists():
                shutil.copy2(d / f, dst / f)
        shutil.copytree(d / "src", dst / "src")
        if (d / ".cargo").is_dir():
            shutil.copytree(d / ".cargo", dst / ".cargo", dirs_exist_ok=True)
        done.append(n)
    return done


def build_top_pool() -> list[str]:
    """bots/top-<slug> for every team in league/ledger.json's pool that has a spec in data/mimic:
    this branch's bot source with the team's pack (the refreshed ladder top 10, final test only)."""
    import json

    from harness.search.mvp import _league

    lg = _league()
    built = []
    for p in lg.load_ledger()["pool"]:
        spec_p = lg.MIMICS / f"{p['slug']}.spec.json"
        if not spec_p.exists():
            continue
        spec = json.loads(spec_p.read_bytes())
        lg.make_bot(f"top-{p['slug']}", lg.pack_bytes_for({}, spec))
        built.append(p["slug"])
    return built


def build_all(jobs: int = 4) -> tuple[int, list[str]]:
    todo = []
    for d in sorted((C.ROOT / "bots").iterdir()):
        n = d.name
        if n.startswith(("mimic-", "xpl-", "top-")) or n in ("baseline", "lg-v23-fuel", "_rules"):
            if (d / "src").is_dir() and not C.injected(d).exists():
                todo.append(d)
    failed = []

    def one(d: Path) -> None:
        try:
            C.runner.build(str(d), release=True)
        except RuntimeError as e:
            failed.append(f"{d.name}: {str(e).splitlines()[-1][:120] if str(e) else 'build failed'}")

    with ThreadPoolExecutor(max_workers=jobs) as ex:
        list(ex.map(one, todo))
    return len(todo), failed


def run(src: str | None) -> None:
    if src:
        print(f"copied {len(copy_sources(Path(src)))} old opponents from {src}")
    top = build_top_pool()
    print(f"fresh top pool from specs: {len(top)} teams (bots/top-*)")
    C.build_rules()
    n, failed = build_all()
    print(f"built {n} bots; failed {len(failed)}")
    for f in failed:
        print("  FAILED", f)
    print(f"opponents available: {len(C.opponents())}")
