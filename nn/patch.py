"""Patch watcher: never train on stale rules (docs/architecture.md, loop 3, Phase 5).

    python nn/patch.py check      # what changed (nothing is applied)
    python nn/patch.py apply      # apply: SDK files, map pool; record the new patch

A patch is identified by a hash of the SDK sources, the rules document and the map pool. On a
new patch the nn loop starts a new baseline folder (nn/baselines/<patch>/) and fine-tunes its
current best teacher there instead of starting over. Changed champions and maps are listed in
runs/nn/patches.json (with a local-Qwen summary when its server is already running).

A newer cambc on PyPI is reported; with NN_AUTO_UPGRADE=1 it is installed, and rolled back if
the bot no longer builds or plays.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "runs" / "nn" / "patches.json"
QWEN_URL = f"http://127.0.0.1:{os.environ.get('QWEN_PORT', '11435')}/v1/chat/completions"
CAMBC = ROOT / ".venv" / "Scripts" / "cambc.exe"


def sh(*args: str, cwd: Path = ROOT, timeout: int = 600) -> subprocess.CompletedProcess:
    return subprocess.run(list(args), cwd=cwd, capture_output=True, text=True, timeout=timeout)


def installed_version() -> str:
    return sh(str(CAMBC), "--version").stdout.strip().split()[-1]


def latest_version() -> str | None:
    try:
        with urllib.request.urlopen("https://pypi.org/pypi/cambc/json", timeout=20) as r:
            return json.load(r)["info"]["version"]
    except (OSError, ValueError, KeyError):
        return None


def file_hash(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def patch_id(sdk: Path, maps: Path, rules: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(sdk.rglob("*")):
        if p.is_file():
            h.update(p.relative_to(sdk).as_posix().encode())
            h.update(p.read_bytes())
    for p in sorted(maps.glob("*.map26a")):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    if rules.exists():
        h.update(rules.read_bytes())
    return h.hexdigest()[:12]


def current_id() -> str:
    return patch_id(ROOT / "sdk", ROOT / "maps", ROOT / "docs" / "GameRules.md")


def fresh_project() -> Path:
    """A throwaway `cambc init` project (latest SDK of the installed cambc) with the live map pool."""
    tmp = Path(tempfile.mkdtemp(prefix="cambc-patch-"))
    p = sh(str(CAMBC), "init", "--yes", str(tmp), timeout=1800)
    if p.returncode != 0:
        raise RuntimeError(f"cambc init failed: {p.stderr[-500:]}")
    p = sh(str(CAMBC), "maps", "pull", cwd=tmp, timeout=900)
    if p.returncode != 0:
        print(f"warning: cambc maps pull failed (login?): {p.stderr[-300:]}", file=sys.stderr)
    return tmp


def constant_changes(old: Path, new: Path) -> list[str]:
    """Changed `const` lines of the SDK's Rust constants and ability tables (old -> new)."""
    out = []
    for rel in ("rust/src/constants.rs", "rust/src/ability.rs"):
        a = (old / rel).read_text(encoding="utf-8").splitlines() if (old / rel).exists() else []
        b = (new / rel).read_text(encoding="utf-8").splitlines() if (new / rel).exists() else []
        for ln in difflib.unified_diff(a, b, lineterm="", n=0):
            if ln[:1] in "+-" and not ln.startswith(("+++", "---")) and re.search(r"\d", ln):
                out.append(f"{rel}: {ln.strip()}")
    return out


def map_changes(old: Path, new: Path) -> dict[str, list[str]]:
    o = {p.name: file_hash(p) for p in old.glob("*.map26a")}
    n = {p.name: file_hash(p) for p in new.glob("*.map26a")} if new.exists() else {}
    if not n:  # pull failed: no information about the pool
        return {"added": [], "changed": [], "removed": []}
    return {
        "added": sorted(set(n) - set(o)),
        "changed": sorted(k for k in set(n) & set(o) if n[k] != o[k]),
        "removed": sorted(set(o) - set(n)),
    }


def qwen_summary(lines: list[str]) -> str | None:
    """One short summary from the local Qwen server, only if it is already up (it shares the GPU
    with training, so it is never started from here). Checked against the raw diff by the reader."""
    if not lines:
        return None
    body = {
        "messages": [{"role": "user", "content": (
            "Game balance patch, SDK constant diff (- old, + new). List each change as "
            "'champion or system: stat old -> new', one per line, nothing else.\n\n" + "\n".join(lines[:200])
        )}],
        "max_tokens": 600,
        "temperature": 0,
    }
    try:
        req = urllib.request.Request(QWEN_URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.load(r)["choices"][0]["message"]["content"].strip()
    except (OSError, ValueError, KeyError):
        return None


def check() -> dict:
    report = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "installed": installed_version(), "latest": latest_version()}
    tmp = fresh_project()
    try:
        report["constants"] = constant_changes(ROOT / "sdk", tmp / "sdk")
        sdk_files = []
        for p in (tmp / "sdk").rglob("*"):
            if p.is_file():
                q = ROOT / "sdk" / p.relative_to(tmp / "sdk")
                if not q.exists() or file_hash(q) != file_hash(p):
                    sdk_files.append(p.relative_to(tmp / "sdk").as_posix())
        report["sdk_files"] = sorted(sdk_files)
        report["maps"] = map_changes(ROOT / "maps", tmp / "maps")
        report["new_id"] = patch_id(tmp / "sdk", tmp / "maps" if any((tmp / "maps").glob("*.map26a")) else ROOT / "maps",
                                    ROOT / "docs" / "GameRules.md")
    finally:
        report["_tmp"] = str(tmp)
    report["current_id"] = current_id()
    report["changed"] = bool(report["sdk_files"] or any(report["maps"].values()))
    return report


def apply(report: dict) -> dict:
    """Copy the new SDK and map pool into the repo (removed maps go to maps/retired/), then
    verify the bot still builds and wins against the starter bot; undo on failure."""
    tmp = Path(report["_tmp"])
    backup = Path(tempfile.mkdtemp(prefix="cambc-backup-"))
    shutil.copytree(ROOT / "sdk", backup / "sdk")
    shutil.copytree(ROOT / "maps", backup / "maps")
    try:
        if report["sdk_files"]:
            shutil.rmtree(ROOT / "sdk")
            shutil.copytree(tmp / "sdk", ROOT / "sdk")
            toml = ROOT / "cambc.toml"
            t = re.sub(r'sdk_version = "[^"]+"', f'sdk_version = "{report["installed"]}"', toml.read_text())
            toml.write_text(t)
        m = report["maps"]
        for name in m["added"] + m["changed"]:
            shutil.copyfile(tmp / "maps" / name, ROOT / "maps" / name)
        (ROOT / "maps" / "retired").mkdir(exist_ok=True)
        for name in m["removed"]:
            shutil.move(str(ROOT / "maps" / name), str(ROOT / "maps" / "retired" / name))
        if not smoke_ok():
            raise RuntimeError("bot failed the smoke match after the patch")
    except Exception:
        shutil.rmtree(ROOT / "sdk", ignore_errors=True)
        shutil.copytree(backup / "sdk", ROOT / "sdk")
        for p in (ROOT / "maps").glob("*.map26a"):
            p.unlink()
        for p in (backup / "maps").glob("*.map26a"):
            shutil.copyfile(p, ROOT / "maps" / p.name)
        raise
    report["applied_id"] = current_id()
    return report


def smoke_ok() -> bool:
    maps = sorted((ROOT / "maps").glob("*.map26a"))
    if not maps:
        return False
    out = Path(tempfile.mkdtemp(prefix="patch-smoke-"))
    p = sh(str(CAMBC), "run", "main", "starter-python", "--map", str(maps[0]), "-o", str(out), "--release", timeout=1800)
    ok = p.returncode == 0 and "main" in p.stdout and "wins" in p.stdout
    shutil.rmtree(out, ignore_errors=True)
    return ok


def upgrade(version: str) -> bool:
    old = installed_version()
    uv = shutil.which("uv")
    if not uv:
        return False
    if sh(uv, "pip", "install", "--python", str(ROOT / ".venv"), f"cambc=={version}", timeout=1800).returncode != 0:
        return False
    if smoke_ok():
        return True
    sh(uv, "pip", "install", "--python", str(ROOT / ".venv"), f"cambc=={old}", timeout=1800)
    return False


def record(report: dict) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    hist = json.loads(LOG.read_text()) if LOG.exists() else []
    hist.append({k: v for k, v in report.items() if not k.startswith("_")})
    LOG.write_text(json.dumps(hist, indent=1))


def run_check(do_apply: bool) -> dict:
    r = {}
    latest = latest_version()
    if latest and latest != installed_version() and os.environ.get("NN_AUTO_UPGRADE") == "1":
        r["upgraded"] = upgrade(latest)
    rep = check()
    rep.update(r)
    rep["summary"] = qwen_summary(rep["constants"])
    if do_apply and rep["changed"]:
        apply(rep)
    shutil.rmtree(rep.pop("_tmp"), ignore_errors=True)
    if rep["changed"] or rep.get("upgraded") is not None or (rep["latest"] and rep["latest"] != rep["installed"]):
        record(rep)
    return rep


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["check", "apply", "id"])
    a = p.parse_args(argv)
    if a.cmd == "id":
        print(current_id())
        return 0
    rep = run_check(a.cmd == "apply")
    print(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
