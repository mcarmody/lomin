# Lomin Plugin Examples & Transport Patterns

This directory contains a self-contained, zero-dependency reference plugin (`examples/reference_plugin.py`) and an example of how a site-specific queue or broker transport can implement the Lomin plugin contract.

## 1. Quick Start with the Reference Plugin

Run Lomin with the reference plugin:

```bash
MVG_PLUGIN=examples.reference_plugin python3 gateway.py
```

The reference plugin implements every optional hook in `gateway.py`:

- `apply_defaults() -> None` — sets default environment variables with `os.environ.setdefault`.
- `brain_call(target, prompt, timeout) -> str` — synchronous single-reply brain call.
- `brain_call_stream(target, prompt, timeout, on_partial, stages_out, *, task_id=None, audio_clips=None) -> str` — streaming brain call that invokes `on_partial(text_so_far)`, records progress dicts in `stages_out["events"]`, inspects `audio_clips` when `MVG_SPLICE_RAW_AUDIO=1`, and honours `task_id` cancellation.
- `brain_cancel(target, task_id) -> bool` — tracks in-flight requests by `task_id` and signals their cancellation event when `/voice/turn` (spoken cancel) or `/voice/live` (`cancel_thinker`) aborts a turn.
- `talker_instructions(target, node_id) -> str | None` — supplies dynamic system instructions for `/voice/live` Talker sessions before falling back to `<MVG_PERSONA_DIR>/<target>/voice_talker.md` or `voice.md`.
- `try_intent(text) -> dict | None` — deterministic fast-path intent handler before calling the brain.

Run the reference plugin test suite:

```bash
python3 -m pytest -q tests/test_reference_plugin.py
```

---

## 2. Site-Specific Transport Example (SQLite / Broker Queue Pattern)

Lomin itself is transport-agnostic: `gateway.py` only calls `brain_call` and `brain_call_stream` (or POSTs to `MVG_BRAIN_URL` when `MVG_BRAIN=http`). If your site routes agent turns through a local SQLite message queue, a PTY supervisor, or a broker daemon, keep those details inside your plugin module rather than in the gateway.

### Caller Identity (`from_agent`)
When inserting a voice prompt into a multi-agent message queue, set `from_agent="gateway"` rather than impersonating an agent leg:
- `reply_to_agent` / `author` is displayed truthfully as `gateway` in supervisor logs and chat mirrors (`#brain`), avoiding self-collision checks if an agent leg refuses messages from its own name.
- Every voice prompt is mirrored by the queue transport's own mirror policy; the gateway neither adds nor suppresses mirroring.

### Stage Timestamps & Side Table (`brain_call_stages`)
To break down hand-off latency (`brain_first_text_ms`) across multiple hosts in `voice-latency.jsonl`, a queue-backed plugin can populate `stages_out` with raw epoch-millisecond timestamps stamped on the observing host's clock:

| Stage | Clock | Stamped by |
|---|---|---|
| `call_start_ms`, `inserted_ms` | gateway host | `brain_call_stream` start / queue insert returned |
| `claimed_pi_ms` | gateway host | broker: the claim request took the row |
| `claim_recv_ms` | worker host | supervisor: claim reply in hand |
| `dispatch_start_ms` | worker host | supervisor: about to type into tmux / PTY |
| `send_begin_ms`, `typed_ms` | worker host | `send()`: text `send-keys` returned |
| `enter_sent_ms` | worker host | `send()`: Enter `send-keys` returned (submit) |
| `send_returned_ms` | worker host | `send()` returned (after drain check) |
| `first_text_seen_ms` | worker host | text pump saw the first assistant text |
| `partial_write_start_ms` | worker host | first partial write begins |
| `partial_stored_pi_ms` | gateway host | broker/local write: first partial landed in the DB |
| `first_partial_seen_ms`, `final_seen_ms` | gateway host | plugin poll saw the first partial / final answer |

In a queue-backed deployment:
- The worker supervisor notes stages in memory and ships unsent stages along with partial responses (`POST /v1/queue/set-partial` into `message_queue.partial_response` and `brain_call_stages(call_id, stages_json)`).
- Same-host deltas (`insert_to_claim_ms`, `claim_to_dispatch_ms`, `typed_to_enter_ms`, `enter_to_first_text_ms`, `partial_to_gateway_seen_ms`, `call_to_first_partial_ms`) are exact; cross-host deltas (`*_xclock_ms`) include clock skew plus one-way network latency.

### Event-Driven Claim & `PRAGMA data_version` Wake Polling
- **Long-poll claim (`CLAIM_LONGPOLL=1`):** `POST /v1/queue/claim` can accept `wait_ms` (e.g. capped at 1.5 s–25 s), holding the request until a row is claimable or woken by an in-process condition / 20 ms `PRAGMA data_version` probe.
- **Plugin wake polling (`MVG_HIVE_POLL_SEC`):** Instead of opening a new SQLite connection on every poll tick, `brain_call_stream` can poll a persistent SQLite WAL connection gated on `PRAGMA data_version` every `MVG_HIVE_POLL_SEC` (e.g. `0.03` s), running `SELECT` only when the database state changes.
