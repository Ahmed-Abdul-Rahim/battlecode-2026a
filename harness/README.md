# Qwen harness (owner: teammate with the RTX 5060 Ti)

This is a placeholder and interface contract for the local-LLM harness. The harness itself is the teammate's to build.

## Hard constraints

- **Bots can't call an LLM.** Every cambc bot (Rust or Python) runs in a Wasmtime sandbox with **no network**, about 100k fuel per turn, and 8 bytes of shared team memory. An LLM can't play live from inside a bot.
- **Speed.** A 2,000-round match with 5–6 champions is about 10k decisions. A 27B model on a 16 GB card at roughly Q4 (partly offloaded) would take hours per match.
- **Platform.** Nothing in the harness may submit to or play on the Battlecode platform.

## Roles that fit those constraints

1. **Exploiter author** (AlphaStar-style league exploiters, P-006): the model reads our replays through MCP and writes rival bots under `bots/exploit-*/` that target our weaknesses. Python bots get 25× fuel, which is handy for quick experiments. We then gate `main` against them.
2. **Labeller:** the model labels logged game states (`bctools dataset` rows) with yes/no or expert choices. Those labels train the distilled heads (`ml/distill/train.py`).
3. **Reviewer:** the model looks at `render_frame` / `render_timeline` PNGs (if it's multimodal) and drafts experiment issues.

## Interface: the bctools MCP server

Start the server: `.venv/Scripts/python.exe -m mcp_server.server` (stdio). Any MCP client works, for example the `mcp` Python SDK's `ClientSession`, or a tool-calling wrapper around an OpenAI-compatible local server (llama.cpp, vLLM). Tools are listed in docs/self-improvement.md. `draft_issue` needs `confirm=true` to file.

## Deferred (decided in planning, 2026-10-18)

- OpenJev / small-model yes/no tool (licence: CC BY-NC 4.0, so check before any non-research use).
- Mamba / SSM belief model.
