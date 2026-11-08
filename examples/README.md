# Lomin Plugin Examples & Transport Guide

This directory contains a self-contained reference plugin (`examples/reference_plugin.py`) and documentation for implementing custom agent transports with the Lomin plugin contract.

## 1. Reference Plugin (`examples/reference_plugin.py`)

Run Lomin with the reference plugin:

```bash
MVG_PLUGIN=examples.reference_plugin python3 gateway.py
```

By default (with `REF_PLUGIN_BRAIN_URL` unset), the reference plugin runs entirely in-memory with zero external dependencies.

### Configuration via Environment Variables

The reference plugin can be configured via environment variables:

| Variable | Description | Default |
|---|---|---|
| `REF_PLUGIN_BRAIN_URL` | HTTP endpoint URL for brain calls (POST). When empty, uses deterministic in-memory responses. | `""` |
| `REF_PLUGIN_BRAIN_TOKEN` | Optional Bearer token sent in `Authorization: Bearer <token>` header. | `""` |
| `REF_PLUGIN_TALKER_INSTRUCTIONS` | Global fallback system instructions for live voice Talker sessions. | Built-in default |
| `REF_PLUGIN_TALKER_INSTRUCTIONS_<TARGET>` | Per-target system instructions (e.g. `REF_PLUGIN_TALKER_INSTRUCTIONS_RESEARCHER`). | Built-in default |

### HTTP Mode Contract

When `REF_PLUGIN_BRAIN_URL` is configured, `brain_call` and `brain_call_stream` forward requests over HTTP.

#### Request JSON Payload
```json
{
  "target": "agent",
  "prompt": "[VOICE via node desk] Check system status",
  "task_id": "task-stream-1",
  "stream": true,
  "audio_clips_b64": ["..."]
}
```

- `target`: destination agent or persona name.
- `prompt`: framed gateway prompt string.
- `task_id`: 12-character task identifier for cancellation tracking.
- `stream`: `true` for streaming (`brain_call_stream`), `false` for single-shot (`brain_call`).
- `audio_clips_b64`: array of base64-encoded WAV audio chunks (if provided and `MVG_SPLICE_RAW_AUDIO=1`).

#### Streaming Response (Server-Sent Events)
When `stream: true`, the endpoint should return `Content-Type: text/event-stream`:

```
data: {"stage": "tool_start", "detail": "routing request to agent"}

data: {"delta": "System status "}
data: {"delta": "is nominal."}

data: [DONE]
```

Event types supported:
- `stage`: reports task progress; appended to `stages_out["events"]` with timestamp `t`.
- `delta` / `text`: incremental text token or phrase appended to cumulative text and pushed to `on_partial`.
- `reply`: replacement text for cumulative response pushed to `on_partial`.
- `[DONE]`: stream completion sentinel.

For non-streaming responses or non-SSE responses, the endpoint returns JSON:
```json
{
  "reply": "System status is nominal.",
  "events": [
    {"stage": "tool_start", "detail": "checked metrics", "t": 0.15}
  ]
}
```

### Supported Plugin Hooks

The reference plugin demonstrates every optional hook in `gateway.py`:

- `apply_defaults() -> None` — sets default environment variables with `os.environ.setdefault`.
- `brain_call(target, prompt, timeout) -> str` — synchronous single-reply brain call.
- `brain_call_stream(target, prompt, timeout, on_partial, stages_out, *, task_id=None, audio_clips=None) -> str` — streaming brain call that invokes `on_partial(text_so_far)`, records progress dicts in `stages_out["events"]`, forwards `audio_clips`, and supports cancellation via `task_id`.
- `brain_cancel(target, task_id) -> bool` — tracks in-flight requests by `task_id` and aborts running streams / closes active connections when a turn is cancelled.
- `talker_instructions(target, node_id) -> str | None` — supplies dynamic system instructions for `/voice/live` Talker sessions.
- `try_intent(text) -> dict | None` — deterministic fast-path intent handler.

Run the test suite:

```bash
timeout 120 python3 -m pytest -q tests/test_reference_plugin.py
```

---

## 2. Writing Your Own Plugin

Lomin is transport-agnostic: `gateway.py` interacts with your agent architecture exclusively through the plugin hooks. When writing a custom plugin for your site:

### Streaming & Partial Output (`on_partial`)
In `brain_call_stream`, invoke `on_partial(text_so_far)` each time new assistant text is available. The gateway feeds partial text into sentence cutting and TTS synthesis, minimizing time-to-first-audio.

### Progress Stage Events (`stages_out["events"]`)
Long-running agent tools or operations can report milestone events by appending dictionaries to `stages_out.setdefault("events", [])`:

```python
stages_out.setdefault("events", []).append({
    "stage": "tool_start",
    "detail": "searching documentation",
    "t": round(time.monotonic() - t0, 3),
})
```

- `stage`: machine name (e.g., `"tool_start"`, `"milestone"`).
- `detail`: human-readable description for spoken progress or UI updates.
- `t`: seconds since the start of the brain call.

### Cancellation (`task_id` and `brain_cancel`)
When a user interrupts or cancels a turn (via `/voice/turn` spoken cancel or `/voice/live` `cancel_thinker`), the gateway passes the turn's `task_id` to `brain_cancel(target, task_id)`. Maintain an in-memory registry or send an abort signal to your backend so cancelled work stops immediately.

### Multimodal Audio Clips (`audio_clips`)
When `MVG_SPLICE_RAW_AUDIO=1` is set, `brain_call_stream` receives `audio_clips: list[bytes] | None` containing the raw 16 kHz WAV audio clips for the turn. This allows custom plugins to pass audio directly to multimodal LLMs.
