"""`bctools` command line. Run `bctools <command> --help` for options.

Local only: nothing here submits to or plays on the Battlecode platform.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from bctools import ROOT, dataset, gate, render, replay, runner, sync


def _print(obj) -> None:
    print(json.dumps(obj, indent=1, default=str))


def cmd_run(a: argparse.Namespace) -> int:
    maps = [Path(m) if Path(m).suffix else ROOT / "maps" / f"{m}.map26a" for m in a.maps] if a.maps else None
    results = runner.run_matches(
        a.candidate,
        a.opponent,
        maps,
        Path(a.out) if a.out else None,
        swap=not a.no_swap,
        jobs=a.jobs,
        rounds=a.rounds,
        release=a.release,
    )
    _print(runner.summarize(results, a.candidate))
    return 0


def cmd_gate(a: argparse.Namespace) -> int:
    summary = gate.gate(a.candidate, a.best, out_dir=Path(a.out) if a.out else None)
    _print(summary)
    return 0 if summary["passed"] else 1


def cmd_summary(a: argparse.Namespace) -> int:
    b = replay.load(a.bundle)
    side, log = b.ours()
    _print({"our_side": side, **replay.summarize(log)})
    return 0


def cmd_render(a: argparse.Namespace) -> int:
    print(render.frame(a.bundle, a.round, a.out, scale=a.scale))
    return 0


def cmd_timeline(a: argparse.Namespace) -> int:
    print(render.timeline(a.bundle, a.out))
    return 0


def cmd_dataset(a: argparse.Namespace) -> int:
    _print(dataset.export(Path(a.runs), Path(a.out)))
    return 0


def cmd_sync(a: argparse.Namespace) -> int:
    return sync.main(["--bot", a.bot] + (["--check"] if a.check else []))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="bctools", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="play candidate vs opponent on maps (both sides by default)")
    r.add_argument("candidate")
    r.add_argument("opponent")
    r.add_argument("--maps", nargs="*", help="map names or paths (default: maps/*.map26a)")
    r.add_argument("--out")
    r.add_argument("--jobs", type=int)
    r.add_argument("--rounds", type=int)
    r.add_argument("--no-swap", action="store_true")
    r.add_argument("--release", action="store_true", help="platform build: no debug telemetry")
    r.set_defaults(fn=cmd_run)

    g = sub.add_parser("gate", help="evaluation gate: candidate vs best on held-out maps")
    g.add_argument("candidate")
    g.add_argument("--best", default="baseline")
    g.add_argument("--out")
    g.set_defaults(fn=cmd_gate)

    s = sub.add_parser("summary", help="telemetry summary of a replay bundle")
    s.add_argument("bundle")
    s.set_defaults(fn=cmd_summary)

    f = sub.add_parser("render", help="PNG frame of a round")
    f.add_argument("bundle")
    f.add_argument("round", type=int)
    f.add_argument("out")
    f.add_argument("--scale", type=int, default=6)
    f.set_defaults(fn=cmd_render)

    t = sub.add_parser("timeline", help="PNG timeline of a match")
    t.add_argument("bundle")
    t.add_argument("out")
    t.set_defaults(fn=cmd_timeline)

    d = sub.add_parser("dataset", help="export features + labels from a run directory to .npz")
    d.add_argument("runs")
    d.add_argument("out")
    d.set_defaults(fn=cmd_dataset)

    y = sub.add_parser("sync", help="vendor crates/* into the bot")
    y.add_argument("--bot", default="main")
    y.add_argument("--check", action="store_true")
    y.set_defaults(fn=cmd_sync)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
