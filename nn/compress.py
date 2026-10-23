"""Compression study: how small can the teacher / student get before it loses strength or the
ability to learn (plan: C:\\Users\\USMAN\\.claude\\plans\\make-a-new-branch-dynamic-fog.md).

    python nn/compress.py short --teacher <ckpt> [--data <shards...>]

`short` = measure A (offline fidelity + plasticity diagnostics, GPU) for a ladder of sizes, then
measure B (games, CPU) for the candidates with goal agreement >= AGREE_MIN: vs the rule bot (80
map x side games), head-to-head vs the teacher (80), and a noise band from a near-identical
copy of the teacher. Results: runs/nn/compress.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import nn.run as R  # noqa: E402

OUT = R.OUT / "compress"
REPORT = R.OUT / "compress.json"
AGREE_MIN = 0.995
# (width, first-layer rank; 0 = full). Width 16 is the deliberately-too-small sanity case.
LADDER = [(512, 0), (512, 256), (512, 128), (512, 64), (512, 32), (512, 16),
          (256, 0), (128, 0), (64, 0), (32, 0), (16, 0)]


def head_to_head(a: Path, b: Path, width: int, tag: str) -> dict:
    """Two teachers (argmax) on every train + held-out map, both sides; score for `a`."""
    train, held = R.maps()
    ba = R.variant("_nn_h2h_a", 2, 0, a)
    bb = R.variant("_nn_h2h_b", 2, 0, b)
    out = R.OUT / "games" / f"h2h-{tag}"
    res = R.play([(ba, bb), (bb, ba)], train + held, out, width)
    import shutil

    shutil.rmtree(out, ignore_errors=True)
    win = sum(r.score_for(ba) for r in res) / max(len(res), 1)
    mg = sum(((r.gems_a - r.gems_b) if r.bot_a == ba else (r.gems_b - r.gems_a)) for r in res) / max(len(res), 1)
    return {"h2h_win": round(win, 3), "h2h_margin": round(mg, 1), "h2h_games": len(res)}


def jitter_copy(src: Path, dst: Path, scale: float = 1e-3) -> None:
    """A near-identical teacher (tiny weight noise): its results set the noise band."""
    code = (
        "import sys,torch;sys.path.insert(0,r'%s');from nn import model,spec;import importlib.util as u;"
        "s=u.spec_from_file_location('t',r'%s');t=u.module_from_spec(s);s.loader.exec_module(t);"
        "from pathlib import Path;p,v,m=t.load_nets(Path(r'%s'),'mlp',512);torch.manual_seed(7)\n"
        "with torch.no_grad():\n [w.add_(torch.randn_like(w)*%g*w.abs().mean()) for w in p.parameters()]\n"
        "t.save(Path(r'%s'),p,v,m)" % (ROOT, ROOT / "nn" / "train.py", src, scale, dst)
    )
    import subprocess

    subprocess.run([str(R.GPU_PY), "-c", code], check=True, cwd=ROOT)


def short(a) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    teacher = Path(a.teacher)
    shards = a.data or sorted(str(p) for p in (R.OUT / "data" / "imitation").glob("*.npz"))[:3]
    rep = {"teacher": str(teacher), "time": time.strftime("%Y-%m-%d %H:%M"), "offline": [], "games": []}
    # ---- A: offline fidelity + plasticity diagnostics ----
    R.log("compress A: int8 teacher")
    rep["offline"].append(R.gpu(["distill", "--teacher", str(teacher), "--data", *shards, "--int8-only"]))
    for w, r in LADDER:
        R.log(f"compress A: width {w} rank {r}")
        m = R.gpu(["distill", "--teacher", str(teacher), "--data", *shards, "--width", str(w),
                   "--rank", str(r), "--out", str(OUT / f"w{w}r{r}")])
        rep["offline"].append(m)
        R.log(f"  {m}")
        REPORT.write_text(json.dumps(rep, indent=1))
    # ---- B: games for the faithful candidates ----
    jit = OUT / "jitter"
    jitter_copy(teacher, jit)
    base = R.evaluate(teacher / "teacher.bin", 512, "cmp-teacher")
    rep["games"].append({"candidate": "teacher", **base})
    noise = {**R.evaluate(jit / "teacher.bin", 512, "cmp-jitter"),
             **head_to_head(jit / "teacher.bin", teacher / "teacher.bin", 512, "jitter")}
    rep["games"].append({"candidate": "teacher + tiny noise (noise band)", **noise})
    R.log(f"compress B: teacher {base}; noise band {noise}")
    REPORT.write_text(json.dumps(rep, indent=1))
    for m in rep["offline"]:
        if m["candidate"] == "teacher int8" or (m["goal"] or 0) < AGREE_MIN:
            continue
        if a.games and m["candidate"].replace(" ", "") not in a.games:
            continue
        ck = OUT / f"w{m['candidate'].split()[0][1:]}r{m['candidate'].split()[1][1:]}"
        g = {"candidate": m["candidate"], **R.evaluate(ck / "teacher.bin", 512, f"cmp-{ck.name}"),
             **head_to_head(ck / "teacher.bin", teacher / "teacher.bin", 512, ck.name)}
        rep["games"].append(g)
        R.log(f"compress B: {g}")
        REPORT.write_text(json.dumps(rep, indent=1))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("short")
    s.add_argument("--teacher", required=True)
    s.add_argument("--data", nargs="*")
    s.add_argument("--games", nargs="*", help="only these candidates in measure B, e.g. w512r64 w16r0")
    s.set_defaults(fn=short)
    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
