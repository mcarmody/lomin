# Lomin Spec: Talker + Thinker Live Architecture

- **Status:** Draft, revised after review of PR #12 (2026-11-05)
- **Target repository:** `mcarmody/lomin`
- **Rule:** nothing household-specific lives in this repo (`AGENTS.md`, "The rule"). Examples use `nodes.example.json` only.

---

## 1. Summary and goals

Lomin's pipeline (`POST /voice/turn`) is built for discrete, utterance-bounded turns. When the brain is a full agent, replies take seconds, and the user hears silence or canned gap-fill clips (`_maybe_gap_fill`). The user cannot steer, cancel or ask for status during that time.

This spec adds a **Talker + Thinker** split. A fast **Talker** holds the conversation. The existing brain, reached through `MVG_PLUGIN` hooks, is the **Thinker**. The Talker:

- speaks a short, specific acknowledgement while the Thinker works;
- narrates progress when asked, or at milestones if enabled;
- accepts corrections and cancel while the Thinker runs;
- relays approval prompts, under the safety rules in section 6.5.

The design goals, in order:

- **Configurable Talker.** Any backend that meets one small contract can be the Talker. Gemini Live is the first, not the design.
- **Custom voices keep working.** An agent's cloned or local voice is a first-class mode.
- **Cheap wins first.** Contextual openers, spoken progress and spoken cancel land in `/voice/turn` with no new transport and no new dependency.
- **Stdlib core.** The default install stays stdlib-only. Everything else is a guarded optional dependency, as `onnxruntime` already is.
- **Off by default.** Cloud audio, metered keys and verbal approvals are all opt-in per node.

## 2. Talker backends

A backend implements `TalkerBackend`. The harness owns the coordination tools (section 4). A backend only turns audio or text into a stream of events.

```python
class TalkerBackend:
    capabilities: set[str]        # see below
    async def open(self, config: dict) -> None: ...
    async def send_audio(self, pcm16_16k: bytes) -> None: ...
    async def send_text(self, text: str, *, silent: bool = False) -> None: ...
    async def send_tool_result(self, call_id: str, result: dict) -> None: ...
    def events(self) -> AsyncIterator[dict]: ...   # audio, text, tool_call, interrupted, turn_end
    async def cancel(self) -> None: ...
    async def close(self) -> None: ...
```

Capabilities a backend declares: `audio_in`, `audio_out`, `text_out`, `full_duplex`, `tool_calls`, `silent_context` (accepts context without triggering speech). The harness degrades by capability. A backend without `silent_context` gets progress updates only on request.

### 2.1 Backend classes

| Class | Hears raw audio | Speaks | Turn-taking | Example |
|---|---|---|---|---|
| Native live | yes | its own voice (`voice_mode: native`) | full duplex | Gemini Live, other realtime speech models |
| Text-out, audio-aware | yes | Lomin TTS (`voice_mode: custom_tts`) | Lomin VAD | an audio-in, text-out model called per utterance |
| Text-out, cascaded | no (uses the Whisper transcript) | Lomin TTS (`voice_mode: custom_tts`) | Lomin VAD | a small, fast text model after Whisper |

The text-out classes are **Mode 2**. They keep every existing custom voice, because the speaking path is unchanged: Talker text goes through `SentenceSegmenter` and `tts_backends.synthesize()`.

### 2.2 What each class gains

- **Custom voice unchanged:** voice quality, synthesis latency and the TTS chain. First audio is the Talker's first sentence plus first-chunk synthesis, so it will not match native audio.
- **Gained by the split itself (all classes):** contextual openers, spoken progress, mid-task corrections, spoken cancel, status answers without interrupting the Thinker.
- **Gained only with an audio-aware backend:** fewer speech-to-text errors on technical terms.
- **Gained only with a full-duplex backend:** prosody-aware turn-taking and instant barge-in.

### 2.3 Candidate backends (facts as of 2026-11-05, to be re-checked before building)

Sourced from Google's docs, read 2026-11-05:

- **Gemini Live, `gemini-3.8-live`:** listed as stable, latest update September 2026 ([model page](https://aistudio.google.com/docs/models/gemini-3.8-live)). That page states "Audio is the only supported response modality (`response_modalities=["AUDIO"]`)", with `output_audio_transcription` for a text transcript. A text-only Live session is **not** available.
- **Audio in, text out:** `gemini-3.8-flash` accepts audio and returns text through the Interactions API (`client.interactions.create(...)`), request and response, not a stream. Inline limit 20 MB per request ([audio docs](https://aistudio.google.com/docs/audio)).
- **Voices:** 30 prebuilt voices plus an "Extended Voice Library" of "hundreds", not listed on the page ([speech generation](https://aistudio.google.com/docs/speech-generation)). The earlier draft's "70 voices" and the "Daikon" and "Constellation" family names were not found there. The implementation takes the voice name as an opaque string and lists nothing.
- **Latency:** no first-audio figure is published on the pages read. No latency number appears in this spec until it is measured (section 8).

From secondary sources, provisional (confirm on official docs): OpenAI `gpt-realtime` family, Amazon Nova Sonic, xAI Grok Voice, NVIDIA NemotronLabs VoiceChat (open weights, native tool calling, weak argument accuracy in NVIDIA's own benchmark), NVIDIA PersonaPlex-7B (open weights, tool calling only in community forks).

## 3. System topology

```
Client (watch, desktop mic, browser)
   |  16 kHz PCM or ADPCM up; 24 kHz PCM / WAV down; flush on barge-in
   v
Lomin harness (gateway.py, optional live_harness.py)
   |  VAD, session state, coordination tools, progress relay
   +---- TalkerBackend (configurable)
   +---- Thinker: MVG_PLUGIN brain_call_stream (existing hook)
```

The Thinker is whatever the plugin routes to. This repo names no agent.

## 4. Coordination tools (harness-owned)

Four tools, all with simple arguments, because tool-call argument accuracy varies by backend:

1. `call_thinker(request: str)`: start a Thinker task. The Talker speaks one short specific sentence in the same turn ("let me check the build logs"), replacing generic gap-fill.
2. `cancel_thinker()`: abort the active task.
3. `answer_prompt(prompt_id: str, decision: "yes" | "no")`: answer a pending approval, under section 6.5.
4. `set_progress_updates(enabled: bool)`.

Today the brain hook is a one-shot call with a timeout. `cancel_thinker()` needs a brain-side abort, defined in section 5.

## 5. Required hook extensions (PR A)

All optional; a plugin that lacks one behaves as it does today.

- `brain_call_stream(..., audio_clips=None)`: raw 16 kHz WAV clips alongside the transcript. The gateway passes it only if the plugin's signature accepts it.
- `brain_cancel(task_id)`: abort a running brain call. Without it, cancel stops speech and drops the result but cannot stop the Thinker, and the Talker says so.
- `stages_out` events get a stable schema: `{"stage": str, "detail": str, "t": float}`.
- Approval prompts from the Thinker: `{"prompt_id", "summary", "command", "risk"}`. Defined here; wired in section 6.5.

Each hook gets a test that patches the hook, per `AGENTS.md`.

## 6. Mechanisms

### 6.1 Turn-taking and ambient gating

- Default: unchanged. Open-room gating (Silero, Whisper, pregate, classifier) stays local.
- Full-duplex backends may use control tokens (`👂` wait, `❌` ignore) to replace the Lomin VAD for that node. This sends continuous room audio to the provider, so it is opt-in per node (`live_gate: "backend"`), documented in the README privacy section, and keeps speaker ID and echo filtering active on the live path.
- The harness strips control tokens before audio or TTS and logs a compact `live_gate` event.

### 6.2 Pre-roll and barge-in splice

- `/voice/turn` receives a finished upload, so the server has no audio before speech start. Pre-roll is a **client capture contract** (recommended lead-in 600 ms, to be tuned) and is documented, not implemented server-side.
- On `/voice/live`, the harness keeps a rolling PCM ring buffer, and may splice an utterance that the user restarts within a short window (default 3 s, configurable).
- With `audio_clips` available, the Thinker receives the user's audio alongside the transcript.

### 6.3 Progress and status

- The harness maps `stages_out` events to one-line updates.
- Backends with `silent_context` receive them silently and answer "what step is it on?" from context.
- Other backends get the same lines only when the user asks, or at milestones if `set_progress_updates(true)`, rate-limited so the Talker is not talking over itself.
- Works in `/voice/turn` too: a status readout replaces the timed gap-fill clip when the stage text is available.

### 6.4 Steering

- v1 queues a follow-up behind the running task (`handoff_id` increments). Mid-turn injection into a running Thinker needs a brain API that does not exist; it is out of scope until a plugin supplies one.
- When a task resolves, the harness returns the result matched to its `handoff_id`.

### 6.5 Verbal approvals (default off)

- A new subsystem, not an extension of `pending_and_auth.py` (which handles pending messages and device auth).
- **Allowlist, not denylist.** Only prompts the Thinker marks `risk: "low"` may be approved by voice. Any other prompt, or a prompt with no `risk` field, requires an out-of-band confirmation.
- Each approval binds to a `prompt_id`, and the Talker reads back the summary before the decision counts.
- Requires speaker verification for the approving utterance, and ignores audio flagged as ambient or echo.
- A pattern check for destructive commands is a second layer only, never the gate.

### 6.6 Output pacing and barge-in flush

- The harness paces output in fixed frames and tracks how much audio the client has been sent.
- On interruption the server sends `{"event": "clear_audio_buffer", "interrupted": true}` and stops queued synthesis.
- The time until playback actually stops depends on the client buffer and the network. The spec states a **client contract** (flush on the event) and makes no server latency guarantee.

## 7. Wire protocol, config and fallback

- `GET /voice/live`: WebSocket upgrade, JSON control frames and binary PCM. The WebSocket layer is a guarded optional dependency; the default install has no `/voice/live`.
- Per-node keys in `nodes.example.json`: `talker_backend`, `voice_mode` (`native` | `custom_tts`), `live_voice`, `progress_updates`, `live_gate`, `splice_raw_audio`.
- Auth: the existing per-node bearer tokens (`MVG_DEVICE_TOKENS`) apply to `/voice/live`.
- Fallback: if the live session or Talker backend fails, the node answers through `/voice/turn`, and the failure is logged.
- Cost and keys: backends that call a metered API take their key from an env var only. They are off unless a node enables them. The README states that these calls are billed per use.

## 8. Phased plan

Prerequisite: record a baseline of response times (first audio, total) from `voice-latency.jsonl`, so every later claim is measured. Do not publish latency targets before that.

- **PR A, hooks:** section 5 hook extensions, with tests. No new endpoint.
- **PR B, `/voice/turn` improvements:** contextual openers and status readouts from `stages_out`, spoken cancel with `brain_cancel`, client pre-roll contract in the docs. No new transport or dependency.
- **PR C, `TalkerBackend` and `/voice/live`:** the contract, the harness, the WebSocket endpoint as an optional module, and a first backend (Gemini Live for `native`, an audio-in text-out backend for `custom_tts`). Tests use a mock backend.
- **Later:** verbal approvals (6.5), as a separate spec and PR once speaker verification is in place.

## 9. Tests

- Mock `TalkerBackend` driving: control-token filtering, handoff matching, silent progress, cancel with and without `brain_cancel`, fallback to `/voice/turn`.
- Hook tests patch the hook, per `AGENTS.md`.
- The household-name grep from the PR checklist runs on every PR.
- No test needs the network or a key.

## 10. Open questions

- Which WebSocket implementation: stdlib handshake, or an optional `websockets` dependency?
- Does any audio-in text-out backend stream tokens, or is each call whole-response? (Docs read show request and response only.)
- Where does speaker verification live for approvals?
