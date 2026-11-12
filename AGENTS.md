# AGENTS.md — Lomin Agent Voice Gateway

## What this is

Lomin Agent Voice Gateway is a voice gateway that gives agents a voice. It is a single Python 3.13
stdlib HTTP service (`gateway.py`) that fronts speech-to-text (Whisper), a
brain (an agent that answers one framed prompt) and text-to-speech (a chain of
backends). It handles speech gating, speaker identification, addressee
detection in shared rooms, sentence cutting, gap-fill audio while the brain
thinks, and echo filtering. It is part of the Karakos stack.

Files:

- `gateway.py` — the HTTP service and the turn pipeline.
- `tts_backends.py` — the swappable TTS chain (`MVG_TTS_CHAIN`).
- `desktop_tts.py`, `desktop_embed.py` — optional GPU-host servers (TTS, speaker embedding), reached over an ssh reverse tunnel.
- `talker_backend.py` — the Talker contract for live voice (`TalkerBackend`, `MockBackend`, `CascadedTextBackend` (`cascaded`), `load_backend`, `requires`); plugin backends via `MVG_TALKER_BACKENDS=name=module:Class`. Stdlib only.
- `live_harness.py` — `LiveSession`: drives one Talker backend and owns the coordination tools `call_thinker`, `cancel_thinker`, `answer_prompt` (stub), `set_progress_updates`. Progress lines and thinker results go through `send_context`, which holds them while the Talker is speaking and releases them at `turn_end`. Stdlib only; callbacks come from the gateway.
- `live_ws.py` — stdlib RFC 6455 server framing and the `/voice/live` bridge (`run_live_session`); the route is `_handle_voice_live` in `gateway.py`.
- `gemini_live.py` — built-in `gemini` Talker backend: stdlib asyncio WebSocket client for the Gemini Live API; the four coordination tools are declared as function declarations.
- `sentence_segmenter.py`, `text_utils.py`, `multipart.py`, `audio_utils.py`, `t3_fast.py` — helpers.
- `examples/reference_plugin.py`, `examples/README.md` — stdlib-only reference plugin (in-memory or configurable HTTP via `REF_PLUGIN_BRAIN_URL`) demonstrating every plugin hook.
- `tools/gemini_live_smoke.py` — live WebSocket smoke harness for `/voice/live`.
- `specs/` — design specifications for live voice and custom TTS paths.
- `tests/` — the suite. `intents/`, household nodes and household services are not here.

## Running the tests

```
python3 -m pytest -q tests/
```

No network. Tests that need the plugin's `classify` module skip when it is not
importable. `test_render_gap_fill_sfx.py` and `test_voice_latency_report.py`
skip unless their scripts (outside this repo) are present. `tests/conftest.py`
sets the agent-name env vars before `gateway` is imported; `test_standalone.py`
and `test_plugin.py` run the gateway in subprocesses with a clean env.

## The plugin

`MVG_PLUGIN` names an importable module that supplies the site-specific parts.
Every hook is optional:

- `apply_defaults()` — fill env defaults with `setdefault` before config reads them.
- `brain_call(target, prompt, timeout) -> str` — the agent answers one framed prompt.
- `brain_cancel(target, task_id) -> bool` — abort a running brain call; `cancel_active_turn(node_id)` in `gateway.py` uses it.
- `brain_call_stream(target, prompt, timeout, on_partial, stages_out, *, task_id=None, audio_clips=None) -> str` — streaming brain. Optional keywords `task_id` and `audio_clips` (list of WAV bytes when `MVG_SPLICE_RAW_AUDIO=1`) are passed only if the signature accepts them.
- `talker_instructions(target, node_id) -> str | None` — system instructions for `/voice/live` Talker sessions, checked before `<MVG_PERSONA_DIR>/<target>/voice_talker.md` and `voice.md`.
- `classify` — a module with `classify_choice(...)` (the addressee classifier).
- `try_intent(text) -> dict | None` — deterministic fast path.
- `AMBIENT_ADDRESSED` — a string: the option key the addressee classifier answers with (default `addressed_to_agent`).

The gateway reads these as module attributes at import (`_plugin_brain_call`,
`_plugin_brain_call_stream`, `_jev_classifier`, `_try_intent`). Tests replace
those attributes with monkeypatch, so new behaviour belongs behind a hook and
a test patches the hook.

## The rule

Household code lives in a plugin and never in this repo. That includes
household nodes, device names, people's names, addresses, paths, intents,
brain targets and any service that only one household runs. If a change needs
something household-specific, add a hook or an env var here, and put the
household value in the plugin or in the host's env file.

Do not add `nodes.json`, `intents/`, household `.service` units, household
wheels, or household names to tests or docs. `nodes.example.json`,
`lomin.service.example`, and `examples/` (`reference_plugin.py`, `README.md`) are the only examples. Before a PR, check the tree for host paths, LAN addresses, personal names and
domains. The PR checklist lists the exact commands.

## Conventions

- Stdlib only at runtime. The optional dependencies (`onnxruntime`, `numpy`, `requests`) are guarded and fail open.
- Config is env-driven. Every new env var goes in the README config table and in `.env.example`.
- Log lines are JSON, one per request, with no audio bytes and no keys.
- A failing plugin is loud (stderr), never silent.
- `MVG_SPOKEN_CANCEL=1` / `MVG_CANCEL_PHRASES`: spoken cancel on `/voice/turn` (`is_cancel_phrase`, `cancel_active_turn`); tests in `tests/test_spoken_cancel.py`.
- `MVG_SPOKEN_PROGRESS=1` (+ `MVG_PROGRESS_*`): spoken progress from `stages_out["events"]` (`ProgressNarrator`); tests in `tests/test_spoken_progress.py`.
- `MVG_CONTEXT_OPENER=1`: opener from the first stage event (`_maybe_context_opener`); tests in `tests/test_context_opener.py`.
- `MVG_PERSONA_DIR`: directory of per-target persona markdown files (`<MVG_PERSONA_DIR>/<target>/voice_talker.md` or `voice.md`) used as fallback Talker instructions on `/voice/live`.
