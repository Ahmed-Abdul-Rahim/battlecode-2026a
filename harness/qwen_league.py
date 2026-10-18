"""Qwen league: the local Qwen model writes exploiter bots for `main` to train against.

Bots can't call an LLM (no network in the sandbox), so Qwen plays by *authoring*
Python opponents, one per strategy (AlphaStar-style league exploiters, P-006).
Each bot starts as a copy of bots/starter-python with Qwen's bot.py. It is built,
smoke-tested against `main`, and repaired from the errors it logs, up to
--repairs times. Bots that run cleanly become sparring partners for
harness/tune_knobs.py.

    python harness/qwen_league.py                      # every strategy
    python harness/qwen_league.py --only rush hunter   # some of them
    python harness/qwen_league.py --refine rush        # stronger version, given main's results

Needs the qwen llama-server on 127.0.0.1:11435 (started by the qwen MCP server's
first `ask`). Local only: nothing here touches the Battlecode platform.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import shutil
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from bctools import ROOT, replay, runner

BASE_URL = os.environ.get("QWEN_URL", "http://127.0.0.1:11435")
MODEL = "qwen3-27b"
LOG = ROOT / "runs" / "qwen-league" / "log.jsonl"
SMOKE_MAPS = ["corners", "lanes"]
#: Error lines a bot may log in a smoke match before it counts as broken.
MAX_ERRORS = 20

# What `main` does, so Qwen can aim at it (from crates/bc-brain router + policy).
MAIN_SUMMARY = """\
How the opponent (`main`) plays:
- Drafts strong farmers (Varg, Ryth, Cassi & Issac, Vesper Vale...) and ranks camp groups by value / walking distance.
- Each champion farms the best camp group, captures it, then walks to an altar to tribute once it carries ~20+ gems
  (always at 80). It recalls to shop when gold >= 150, and recalls when HP is low (~35%).
- It fights an enemy champion only when it expects to win the duel by ~25% (Lanchester estimate incl. nearby allies)
  and prefers chasing enemies that carry >= 20 gems. It avoids enemies standing on their own spawn.
- It skips monolith camps unless the team is strong, and skips camp groups walled off from its spawn.
"""

STRATEGIES = {
    "rush": "Rush: from round 1, walk all champions together toward the enemy spawn region and hunt enemy champions. "
    "Focus the lowest-health visible enemy. Never farm unless no enemy has been seen for 100 rounds.",
    "hunter": "Carrier hunter: farm normally, but whenever an enemy champion carrying gems is visible, attack it "
    "(the killer takes the gems). Stand near the altars closest to the enemy and attack champions that "
    "come to tribute (submitting gems is a channel that takes several turns).",
    "stealer": "Camp stealer: farm the camp groups closest to the ENEMY spawn first (contest their best camps), "
    "flee from fights you are losing, bank gems as soon as you carry 10 or more.",
    "banker": "Greedy banker: never attack champions unless attacked first; farm the nearest camps and tribute "
    "every time you carry 8 or more gems, using the nearest unlocked altar. Buy health shards.",
    "pack": "Pack: keep all champions within 3 tiles of each other, moving as a group from camp to camp, so every "
    "fight is many-vs-one. Attack any enemy champion in range of 2+ allies.",
    "ambusher": "Bush ambusher: walk to a bush tile (game_map.bushes()) next to a camp group or on the path to an "
    "altar and wait there. Attack an enemy champion that comes in range if its health is below 50% of max or it "
    "carries gems; after attacking, step back into a bush. Farm only when no enemy was seen for 150 rounds.",
    "pillager": "Pillager: for the first 300 rounds farm lightly. After that, stand near the altars and attack any "
    "enemy champion that carries gems or is submitting gems (is_submitting_gems()); chase carriers up to 6 tiles. "
    "Bank your own gems at once when you carry any.",
    "altar_locker": "Altar locker: buy an altar lock (consumable) at spawn whenever you can afford it, walk to the "
    "altar closest to the ENEMY spawn and place it, then guard that altar, attacking enemies that come near. "
    "Use other altars for your own tributes.",
    "spawn_camper": "Spawn camper: wait 4-6 tiles outside the enemy spawn region (not on it) and attack enemy "
    "champions that leave it or arrive there, focusing low-health ones. Recall when below 40% health.",
    "picker": "Picker: find the visible enemy champion with the fewest other enemies within 4 tiles of it (the most "
    "isolated) and attack it with every champion that can see it. Otherwise farm the nearest camp.",
    "third_party": "Third party: follow enemy champions at distance 5-7. When one is attacking a camp (it is near "
    "camp tiles) or its health drops below 60%, move in and attack it. Otherwise farm.",
    "late_rush": "Late rush: farm and bank gems normally until round 1000. From round 1000, every champion walks "
    "to and attacks the nearest enemy champion, never retreating.",
}

RULE_SLICES = [(1, 44), (88, 134), (169, 273), (355, 378)]


def rules_excerpt() -> str:
    lines = (ROOT / "docs" / "GameRules.md").read_text(encoding="utf-8").splitlines()
    return "\n".join("\n".join(lines[a - 1 : b]) for a, b in RULE_SLICES)


SDK = ROOT / "sdk" / "python" / "cambc"
#: (file, prefix the bot uses to reach it). `from cambc import game_map` makes game_map.X work too.
API_MODULES = [
    ("__init__.py", "cambc."),
    ("game_map.py", "cambc.game_map."),
    ("vision.py", "cambc.vision."),
    ("draft.py", "cambc.draft."),
    ("constants.py", "cambc.constants."),
]


def _sig(f: ast.FunctionDef, prefix: str = "") -> str:
    ret = f" -> {ast.unparse(f.returns)}" if f.returns else ""
    doc = (ast.get_docstring(f) or "").split("\n")[0]
    return f"def {prefix}{f.name}({ast.unparse(f.args)}){ret}" + (f"  # {doc}" if doc else "")


def _top_level(tree: ast.Module) -> list[ast.stmt]:
    """Module statements, looking inside `if`/`else` blocks (the SDK defines enums under version checks)."""
    out: list[ast.stmt] = []
    for node in tree.body:
        if isinstance(node, ast.If):
            out += node.body if node.body else []
            out += [n for n in node.orelse if not isinstance(n, ast.If)]
        else:
            out.append(node)
    return out


def _assigned(nodes: list[ast.stmt]) -> list[str]:
    names = []
    for n in nodes:
        targets = (
            n.targets if isinstance(n, ast.Assign) else [n.target] if isinstance(n, ast.AnnAssign) else []
        )
        names += [t.id for t in targets if isinstance(t, ast.Name) and not t.id.startswith("_")]
    return names


def api_names() -> dict[str, set[str]]:
    """Public names per prefix, for the static API check."""
    out: dict[str, set[str]] = {}
    for fname, prefix in API_MODULES:
        nodes = _top_level(ast.parse((SDK / fname).read_text(encoding="utf-8")))
        names = {n.name for n in nodes if isinstance(n, ast.FunctionDef | ast.ClassDef)}
        out[prefix] = names | set(_assigned(nodes))
    out["cambc."] |= {
        "constants",
        "debug",
        "draft",
        "game_map",
        "vision",
        "Vec2",
        "Bitboard",
        "DistanceField",
    }
    return out


SKIP_CLASSES = {"ErrorCode", "Error"}


def api_digest() -> str:
    """Public signatures + first docstring line of the `cambc` package (the full source is ~80 KB)."""
    lines, seen = [], set()
    for fname, prefix in API_MODULES:
        for node in _top_level(ast.parse((SDK / fname).read_text(encoding="utf-8"))):
            if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
                lines.append(_sig(node, prefix))
            elif isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
                # Enums are defined twice (per Python version); error types only bloat the prompt.
                if prefix + node.name in seen or node.name in SKIP_CLASSES:
                    continue
                seen.add(prefix + node.name)
                doc = (ast.get_docstring(node) or "").split("\n")[0]
                lines.append(f"class {prefix}{node.name}:" + (f"  # {doc}" if doc else ""))
                members = _assigned(node.body)
                if members:
                    lines.append("    members: " + ", ".join(members))
                lines += [
                    "    " + _sig(m)
                    for m in node.body
                    if isinstance(m, ast.FunctionDef) and not m.name.startswith("_")
                ]
            elif fname == "constants.py":
                lines += [f"{prefix}{n}" for n in _assigned([node])]
    return "\n".join(lines)


def check_api(code: str) -> list[str]:
    """Names reached through cambc / game_map / cambc.vision ... that the SDK doesn't define."""
    known = api_names()
    aliases = {"cambc": "cambc.", "game_map": "cambc.game_map.", "vision": "cambc.vision."}
    bad = []
    for node in ast.walk(ast.parse(code)):
        if not isinstance(node, ast.Attribute):
            continue
        chain = ast.unparse(node.value)
        prefix = aliases.get(chain) or (chain + "." if chain + "." in known else None)
        if prefix and node.attr not in known.get(prefix, set()):
            bad.append(f"{chain}.{node.attr} does not exist in the API")
    return sorted(set(bad))


def native_types() -> str:
    """Vec2 / Bitboard / DistanceField from the native stub (re-exported as cambc.Vec2 etc.)."""
    lines = (ROOT / "sdk" / "python" / "_cambc.pyi").read_text(encoding="utf-8").splitlines()
    end = next(i for i, line in enumerate(lines) if line.startswith("class _Record"))
    return "\n".join(line for line in lines[:end] if line.strip() not in ('"""', "...") and '"""' not in line)


def context() -> str:
    starter = (ROOT / "bots" / "starter-python" / "src" / "bot.py").read_text(encoding="utf-8")
    return (
        "# Game rules (excerpt)\n"
        + rules_excerpt()
        + "\n\n# cambc API (signatures; champions are cambc.Champion, `cambc.me()` is ours)\n```python\n"
        + api_digest()
        + "\n```\n\n# Native types re-exported by cambc (cambc.Vec2, cambc.Bitboard, cambc.DistanceField)\n```python\n"
        + native_types()
        + "\n```\n"
        + "\n# Starter bot.py (works; main.py calls Bot().play_turn() every turn and catches exceptions)\n"
        + "```python\n"
        + starter
        + "\n```\n\n"
        + MAIN_SUMMARY
    )


SYSTEM = (
    "You write Python bots for a turn-based game engine (cambc). Use ONLY functions that exist in the API listing "
    "or the starter bot, and don't hide API calls in try/except (main.py already catches errors). Output exactly one complete Python file in a single ```python code block: the new "
    "bot.py. Keep the `Bot` class with `play_turn(self)`, the imports the starter uses (`cambc`, `game_map`, "
    "`randomness.below`), and its helper functions. No explanation outside the code block."
)


def chat(messages: list[dict], max_tokens: int = 7000) -> str:
    payload = {
        "model": MODEL,
        "stream": False,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.6,
        "top_k": 20,
        "top_p": 0.95,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        BASE_URL + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.load(r)["choices"][0]["message"]["content"]


STARTER = ROOT / "bots" / "starter-python" / "src" / "bot.py"


def _norm(code: str) -> str:
    return "\n".join(line.rstrip() for line in code.replace("\r\n", "\n").strip().splitlines())


def extract_code(reply: str) -> str | None:
    blocks = re.findall(r"```(?:python)?\n(.*?)```", reply, re.S)
    code = max(blocks, key=len) if blocks else None
    if code and "class Bot" in code and "def play_turn" in code:
        return code
    return None


def make_bot(name: str, code: str) -> Path:
    d = ROOT / "bots" / name
    if d.exists():
        shutil.rmtree(d)
    shutil.copytree(ROOT / "bots" / "starter-python", d)
    (d / "src" / "bot.py").write_text(code, encoding="utf-8")
    return d


def smoke(name: str) -> dict:
    """Play the bot against main on a few maps; collect its logged errors."""
    out = ROOT / "runs" / "qwen-league" / f"{name}-{time.strftime('%H%M%S')}"
    maps = [ROOT / "maps" / f"{m}.map26a" for m in SMOKE_MAPS]
    try:
        results = runner.run_matches(name, "main", maps, out, swap=False)
    except RuntimeError as e:  # build failure
        return {"ok": False, "errors": [str(e)[-3000:]], "results": []}
    errors: list[str] = []
    for r in results:
        try:
            log = replay.load(r.replay).team("a")
        except (OSError, KeyError, ValueError) as e:
            errors.append(f"match did not produce a replay: {e}")
            continue
        errors += [line for line in log.other if "error" in line.lower() or "Traceback" in line]
    unfinished = sum(r.winner is None for r in results)
    uniq = list(dict.fromkeys(errors))
    return {
        "ok": len(errors) <= MAX_ERRORS and unfinished == 0,
        "errors": uniq[:15],
        "n_errors": len(errors),
        "results": [
            {"map": r.map, "winner": r.winner, "gems": [r.gems_a, r.gems_b], "rounds": r.rounds}
            for r in results
        ],
    }


def log(entry: dict) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def author(strategy: str, brief: str, repairs: int, prior_code: str | None = None) -> dict:
    name = f"exploit-qwen-{strategy}"
    ask = f"Write bot.py implementing this strategy:\n{brief}"
    if prior_code:
        ask += (
            "\n\nImprove this previous version (it lost; make it beat the opponent):\n```python\n"
            + prior_code
            + "\n```"
        )
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": context() + "\n\n" + ask}]
    status: dict = {"ok": False}
    for attempt in range(repairs + 1):
        t0 = time.time()
        reply = chat(messages)
        code = extract_code(reply)
        bad = check_api(code) if code else []
        if code is None:
            status = {"ok": False, "errors": ["no complete bot.py code block in reply"]}
            feedback = "Your reply had no complete ```python block with class Bot and play_turn. Output the full file."
        elif _norm(code) == _norm(STARTER.read_text(encoding="utf-8")):
            status = {"ok": False, "errors": ["identical to the starter bot"]}
            feedback = (
                "That is the unchanged starter bot. Implement the strategy: change play_turn and helpers."
            )
        elif bad:
            status = {"ok": False, "errors": bad}
            feedback = (
                "These names don't exist; use only the API listed above (no try/except around API calls):\n"
                + "\n".join(bad)
            )
        else:
            make_bot(name, code)
            status = smoke(name)
            feedback = (
                "The bot logged these errors while playing (fix them; keep the strategy):\n"
                + "\n".join(status["errors"])
            )
        log({"strategy": strategy, "attempt": attempt, "secs": round(time.time() - t0), **status})
        print(
            f"[{strategy}] attempt {attempt}: ok={status['ok']} errors={status.get('n_errors', '?')}",
            flush=True,
        )
        if status["ok"]:
            return {"bot": name, **status}
        # Only the latest attempt goes back: the 32K window holds one context + one bot.
        messages = [
            *messages[:2],
            {"role": "assistant", "content": reply},
            {"role": "user", "content": feedback},
        ]
    if prior_code:  # keep the working version we were refining
        make_bot(name, prior_code)
        return {"bot": name, "refined": False, **status}
    shutil.rmtree(ROOT / "bots" / name, ignore_errors=True)
    return {"bot": None, **status}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", nargs="*", choices=list(STRATEGIES))
    p.add_argument("--refine", nargs="*", choices=list(STRATEGIES), help="improve existing exploit bots")
    p.add_argument("--repairs", type=int, default=3)
    a = p.parse_args(argv)
    done = {}
    if a.refine:
        for s in a.refine:
            path = ROOT / "bots" / f"exploit-qwen-{s}" / "src" / "bot.py"
            prior = path.read_text(encoding="utf-8") if path.exists() else None
            done[s] = author(s, STRATEGIES[s], a.repairs, prior)
    else:
        for s in a.only or list(STRATEGIES):
            done[s] = author(s, STRATEGIES[s], a.repairs)
    print(json.dumps({s: {"bot": r["bot"], "results": r.get("results")} for s, r in done.items()}, indent=1))
    return 0 if any(r["bot"] for r in done.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
