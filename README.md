# Lomin Agent Voice Gateway

A voice gateway that gives agents a voice. Part of the [Karakos](https://github.com/mcarmody/karakos-package) stack.

The gateway fronts three backends over plain HTTP: speech-to-text (Whisper), a brain (an agent that answers a framed prompt) and text-to-speech (a chain of backends). It does the voice work: speech gating, speaker identification, addressee detection in shared rooms, sentence cutting, gap-fill audio while the brain thinks, and echo filtering. Site-specific parts (which brain answers which node, the addressee classifier, the fast-path intents, default env values) come from a plugin; see "Plugin" below.

Python 3.13 stdlib only (`http.server.ThreadingHTTPServer`), with optional, guarded dependencies. The plugin's addressee classifier is optional: without it `/ambient` engages on the agent's name alone (classifier `name-only`). `/ambient`'s pre-STT speech gate needs `onnxruntime` + `numpy` and Silero's ONNX model; if any is missing the gate fails open and the clip goes to Whisper as before.

## Quick start

```
git clone https://github.com/mcarmody/lomin && cd lomin
python3 -m pytest -q tests/             # classifier-backed tests skip without a plugin
cp .env.example .env                    # edit; then: set -a; . ./.env; set +a
MVG_BRAIN=http MVG_BRAIN_URL=http://127.0.0.1:8000/brain python3 gateway.py
```

`GET /health` on port 9098 reports which upstreams answer.

## Plugin

`MVG_PLUGIN` names an importable module that supplies the site-specific parts. Every hook is optional:

| Hook | Signature | Used for |
|---|---|---|
| `apply_defaults()` | `-> None` | fill env defaults (`setdefault` only) before the config reads them |
| `brain_call(target, prompt, timeout)` | `-> str` | the agent answers one framed prompt |
| `brain_call_stream(target, prompt, timeout, on_partial, stages_out)` | `-> str` | streaming brain: call `on_partial(text_so_far)` as the reply grows. If its signature also takes `task_id` (or `**kwargs`) it gets the turn's task id, the same one `brain_cancel` is called with |
| `brain_cancel(target, task_id)` | `-> bool` | optional: abort the running brain call with that task id; truthy = it was stopped. Without it, or if it returns falsy or raises, a cancel stops speech and drops the result but cannot stop the brain |
| `brain_call_stream(target, prompt, timeout, on_partial, stages_out)` | `-> str` | streaming brain: call `on_partial(text_so_far)` as the reply grows. If the signature also takes `audio_clips` (or `**kwargs`) and `MVG_SPLICE_RAW_AUDIO=1`, it gets `audio_clips=[wav_bytes, ...]`: the turn's 16 kHz WAV, so a multimodal brain can hear the original audio. A hook without it is called exactly as before |
| `talker_instructions(target, node_id)` | `-> str \| None` | optional: return the system instructions for a `/voice/live` Talker session on `node_id` targeting `target`. Checked before the `<MVG_PERSONA_DIR>/<target>/voice_talker.md` and `voice.md` persona files and `MVG_GEMINI_INSTRUCTIONS` / `MVG_TALKER_INSTRUCTIONS` |
| `classify` | module with `classify_choice(...)` | addressee classifier for `/ambient` and open rooms |
| `try_intent(text)` | `-> dict \| None` | deterministic fast path; see "Fast path" below |
| `AMBIENT_LAYA_QUESTION`, `VOICE_ADDRESSED_INSTRUCTIONS` | `str` | replace the addressee classifier's wording, for a site whose classifier was tuned on its own text |
| `AMBIENT_ADDRESSED` | `str` | the option key the addressee classifier answers with; default `addressed_to_agent` |

See [`examples/reference_plugin.py`](examples/reference_plugin.py) and [`examples/README.md`](examples/README.md) for a complete, zero-dependency reference plugin implementing `brain_call_stream`, `brain_cancel`, and `talker_instructions`, plus site-specific queue/broker transport patterns.

**Cancel.** Each `/voice/turn`, `/ask/stream`, and `/voice/live` (`call_thinker`) brain call gets a 12-character `task_id` and is registered as its node's active task until it ends (success, error or cancel). `gateway.cancel_active_turn(node_id)` returns `"cancelled"` (the `brain_cancel` hook ran and returned truthy), `"stopped_speech_only"` (no hook, or it returned falsy or raised: the turn stops synthesising, drops the result and ends with outcome `cancelled`) or `"no_active_turn"` (nothing running, or already cancelled). With `MVG_SPOKEN_CANCEL=1` a spoken cancel on `/voice/turn` calls it (see below); on `/voice/live` the `cancel_thinker` tool calls it.

**Spoken cancel.** `MVG_SPOKEN_CANCEL=1` (default off): after STT on `/voice/turn`, a transcript that is exactly one of the cancel phrases (`stop`, `cancel`, `cancel that`, `never mind`, `nevermind`, `stop that`; override with the comma list `MVG_CANCEL_PHRASES`) or at most 3 words starting with `stop` or `cancel` calls `cancel_active_turn(node_id)`. The reply is a fixed, normally synthesised acknowledgement: `Cancelled.` or `Stopped talking, but the task may still be running.`. With no active turn the transcript is an ordinary turn. Matching is whole-utterance, never a substring, so "stop the deploy script from restarting" goes to the brain. The turn record carries `intent: "cancel"`.

**Stage events.** What a plugin puts in `stages_out` for progress consumers has this shape, one dict per event:

```json
{"stage": "tool_start", "detail": "checking the build logs", "t": 2.4}
```

`stage` is a short machine name, `detail` a one-line human-readable note (at most 120 characters), `t` seconds since the brain call started. `gateway.normalize_stage_event(raw)` returns this shape from looser input (alternate keys `name`/`event`, `message`/`text`, `t_s`, `t_ms`; missing `detail` becomes `""`, missing `t` becomes `0.0`) and `None` for a non-dict or a missing stage name. The turn record's `stages` field is unchanged.

**Context opener.** `MVG_CONTEXT_OPENER=1` (default off, pcm stream mode only): when the first entry in `stages_out["events"]` arrives before anything has been spoken, the gateway speaks `One moment, <detail>` through the normal TTS path and counts the generic opener clip as satisfied, so it does not also play. The later gap-fill slots are unchanged. No opener when `detail` is empty, or when the real reply started first. The turn record gains `opener_kind` (`stage`, `gap_fill` or `none`) and `opener_ms`, so `voice-latency.jsonl` compares before and after.

**Spoken progress.** `MVG_SPOKEN_PROGRESS=1` (default off, pcm stream mode only): a plugin that appends event dicts to `stages_out["events"]` can have a one-line update spoken during a long task. The `detail` of the newest unseen event is spoken only when nothing is playing, the real reply has not started, its `stage` is in `MVG_PROGRESS_STAGES` (default `tool_start,milestone`), at least `MVG_PROGRESS_MIN_GAP_SEC` (default 12) have passed since the last line, and fewer than `MVG_PROGRESS_MAX_PER_TURN` (default 3) lines have been spoken. Lines use the normal TTS path. The turn record counts them as `progress_lines`; the `events` key is left out of the record's `stages`. It runs on the stream's drain loop, so there is no extra thread.

Without `brain_call`, `MVG_BRAIN` defaults to `http`. A plugin that fails to import prints `lomin: plugin '<name>' failed to load (<error>); running without it` on stderr and the gateway runs without it. Site code lives in a plugin, never in this repo.

## Brain: `MVG_BRAIN=http` contract

`MVG_BRAIN` is `plugin` (the default when the plugin has `brain_call`) or `http` (the default otherwise). Under `http`, the gateway sends one POST per turn to `MVG_BRAIN_URL`:

```
POST $MVG_BRAIN_URL
Content-Type: application/json
Accept: text/event-stream, application/json  (only when MVG_HTTP_STREAM=1)
Authorization: Bearer <MVG_BRAIN_TOKEN>      (only if set)

{"prompt": "[VOICE via node <node> · ...] <text>...",
 "session_id": "<session>",
 "node_id": "<node>",
 "target": "<brain target for the node>",
 "image_b64": "<base64 JPEG frame>",        (when vision enabled and present)
 "burst_images_b64": ["<frame1>", ...],      (when vision enabled and present)
 "audio_b64": "<base64 WAV audio>"}          (when MVG_BRAIN_DIRECT_AUDIO=1 and present)
```

### Standard JSON Mode (`MVG_HTTP_STREAM=0`, default)
When `MVG_HTTP_STREAM=0` (or unset), `http_brain_call` strictly expects a standard JSON response with a `"reply"` string:

```json
{"reply": "<the answer, spoken as-is>"}
```

In standard JSON mode, `http_brain_call` only accepts `application/json` and rejects or ignores streaming SSE responses. The streaming hook fires once with the full reply. A missing `MVG_BRAIN_URL` raises `MVG_BRAIN_URL not set`.

### Streaming SSE Mode (`MVG_HTTP_STREAM=1`)
When `MVG_HTTP_STREAM=1`, the gateway requests `Accept: text/event-stream, application/json` and incrementally consumes Server-Sent Events (SSE) lines from `MVG_BRAIN_URL`:
- `event: sentence` with `{"text": "..."}` or `{"reply": "..."}`
- `event: delta` with `{"text": "..."}` or `{"delta": "..."}` or `{"reply": "..."}`
- `event: reply` with `{"reply": "..."}` or `{"text": "..."}`
- OpenAI-compatible `data: {"choices": [{"delta": {"content": "..."}}]}` chunks (terminated by `data: [DONE]`)

Partial text is streamed to caller listeners and sentence segmenters as each chunk arrives. If the brain answers with `application/json`, the gateway falls back gracefully to single-reply parsing.

Streams are bounded by two safety gates:
- `MAX_SSE_CHARS` (default `256 * 1024` chars, override with `MVG_MAX_SSE_CHARS`): halts accumulation if total text exceeds this cap to avoid memory bloat.
- Wall-clock deadline (`max_stream_seconds`, defaulting to `MVG_HIVE_TIMEOUT`): halts consumption if the connection remains open past the timeout.

### Direct-Audio Mode (`MVG_BRAIN_DIRECT_AUDIO=1`)
When `MVG_BRAIN_DIRECT_AUDIO=1`, voice audio captured on a turn is base64-encoded and passed directly to the brain via the `"audio_b64"` field in the request payload.

## Backends

Each external piece is swappable over HTTP or by env. Nothing here needs a
plugin. Startup prints one status line per backend (see the end of
this section).

**STT.** `POST WHISPER_URL` with the raw WAV as the body
(`Content-Type: audio/wav`). The response is JSON `{"text": "..."}`. Any
server with that contract works. `WHISPER_HEALTH_URL` defaults to
`WHISPER_URL` with `/health` in place of the last path segment.

**TTS chain.** `MVG_TTS_CHAIN` is a comma list, tried in order. The first
backend that returns a WAV wins and its name is the engine. Default:
`desktop,piper` (the chain this gateway has always used). Every backend
returns a complete WAV (starting `RIFF`) or raises; a non-2xx status or a
non-WAV body is a failure. Each failure is recorded as `<name>:<error>`. When
an entry after the first one speaks, the failures are logged to
`logs/voice_fallback.log`. If every entry fails, the request fails with those
reasons.

| Name | What it calls | Env |
|---|---|---|
| `desktop` | the desktop TTS server | `MVG_DESKTOP_TTS_URL`, `MVG_DESKTOP_TTS_TIMEOUT` |
| `piper` | the local Piper binary and voice | `MVG_PIPER_CONFIG` (binary and voice come from that file) |
| `elevenlabs` | ElevenLabs text-to-speech, WAV at 24 kHz, 10 s timeout | `ELEVENLABS_API_KEY`, `MVG_ELEVENLABS_VOICE_ID`, `MVG_ELEVENLABS_MODEL` |
| `openai` | any OpenAI-compatible `/v1/audio/speech`, 15 s timeout | `MVG_OPENAI_TTS_URL`, `MVG_OPENAI_TTS_MODEL`, `MVG_OPENAI_TTS_VOICE`, `MVG_OPENAI_TTS_KEY` |
| `http` | generic hook: POST `{"text": ...}`, the body is the WAV, 15 s timeout | `MVG_HTTP_TTS_URL` |
| `gemini` | Gemini TTS API, 24 kHz mono PCM wrapped in WAV, 15 s timeout | `MVG_GEMINI_API_KEY` (or `GEMINI_API_KEY`), `MVG_GEMINI_VOICE`, `MVG_GEMINI_TTS_MODEL`, `MVG_GEMINI_TTS_URL` |
| `kokoro` | Kokoro TTS endpoint, 24 kHz mono PCM wrapped in WAV, 15 s timeout | `MVG_KOKORO_URL`, `MVG_KOKORO_VOICE`, `MVG_KOKORO_MODEL`, `MVG_KOKORO_API_KEY` (or `MVG_KOKORO_KEY`) |

Example: Kokoro-FastAPI serves an OpenAI-compatible endpoint at
`http://localhost:8880/v1/audio/speech`. To speak with it and fall back to
Piper:

```
MVG_TTS_CHAIN=openai,piper
MVG_OPENAI_TTS_URL=http://localhost:8880/v1/audio/speech
MVG_OPENAI_TTS_MODEL=kokoro
MVG_OPENAI_TTS_VOICE=af_sky
```

Example: Gemini TTS using the official Gemini audio generation API (`x-goog-api-key` header auth), falling back to Piper:

```
MVG_TTS_CHAIN=gemini,piper
MVG_GEMINI_API_KEY=your-gemini-key
MVG_GEMINI_VOICE=Schedar
```

Example: Dedicated Kokoro TTS endpoint (`/v1/audio/speech`), falling back to Piper:

```
MVG_TTS_CHAIN=kokoro,piper
MVG_KOKORO_URL=http://localhost:8880/v1/audio/speech
MVG_KOKORO_VOICE=af_heart
MVG_KOKORO_API_KEY=optional-bearer-key
```

**Speaker ID.** `MVG_SPEAKER_ID=0` turns speaker identification off on every
node. The default is `1`. The embedding server is `MVG_DESKTOP_EMBED_URL`.

**No classifier.** Without the classifier (`the plugin's classify module` is not
importable, so there is no paid key), listening still works on the agent's
name alone. An utterance that contains the name engages (score 1.0,
classifier `name-only`). One without the name does not (score 0.0). Wake
phrases work as before.

**Brain.** `MVG_BRAIN` is `plugin` (the plugin's `brain_call`) or `http` (POST to
`MVG_BRAIN_URL`, see "Brain: `MVG_BRAIN=http` contract"). See "Per-node brain routing" for per-node targets.

**Startup lines.** Before it starts serving, the gateway prints one line per
backend. Each probe has a 1 s timeout and never raises. Nothing calls a paid
API at startup.

```
stt: <WHISPER_URL> reachable|unreachable
tts: chain=<names> desktop:reachable|unreachable piper:present|missing <name>:configured|missing config ...
speaker-id: on|off embed=<MVG_DESKTOP_EMBED_URL> reachable|unreachable
classifier: present
classifier: absent (name-only)
brain: <MVG_BRAIN> <MVG_BRAIN_URL or the brain target>
```

Probes count any HTTP answer as reachable. A POST-only endpoint that answers
405 is up.

## Config (env overrides, all optional)

Every variable below is also listed, commented, in `.env.example`. Secrets have no default.

| Var | Default |
|---|---|
| `MVG_HOME` | `~/.mvg` — state root: watch tokens, piper config and binaries, voice capture, speakers, the desktop TTS voice clip |
| `MVG_ENABLE_VISION` / `MVG_VISION_ENABLED` | `0` — `1` enables multimodal camera frame ingest (`image`, `image_b64`, `burst_images_b64`, max 5 MiB) on `/voice/turn` |
| `MVG_ENABLE_ADPCM` / `MVG_ADPCM_ENABLED` | `0` — `1` enables BLE 4-bit IMA ADPCM audio decode (`APCM` header -> 16 kHz PCM WAV, max 512 KiB) on `/voice/turn` |
| `MVG_VISION_ADPCM` | `0` — `1` convenience alias enabling both vision frames and ADPCM decode |
| `MVG_ENABLE_MIC_REOPEN` | `0` — `1` enables opt-in auto mic reopening by including `expects_reply` (boolean detected via question cues and punctuation) in `/ask` and `/ask/stream` SSE `reply` and `done` payloads |
| `MVG_ARTIFACT_WEBHOOK_URL` | unset — HTTP/HTTPS webhook URL to post replies containing URLs or markdown code blocks asynchronously via a background daemon queue without delaying TTS voice replies |
| `MVG_TRANSCRIPT_SINK_URL` | unset — HTTP/HTTPS sink URL to receive completed turn transcript records asynchronously with validated URL schemes and no cross-scheme redirect downgrade |
| `MVG_TRANSCRIPT_AUTH_TOKEN` | unset — bearer token sent as `Authorization: Bearer <token>` with requests to `MVG_TRANSCRIPT_SINK_URL` |
| `MVG_TRANSCRIPT_LOG_PATH` | unset — file path to append turn transcript JSONL records with strict `0o600` permissions and automated rotation |
| `MVG_TRANSCRIPT_MAX_BYTES` | `10485760` (10 MiB) — size cap for `MVG_TRANSCRIPT_LOG_PATH` before rotating the log file to `.1` |
| `MVG_SIDE_CHANNEL_QUEUE_SIZE` | `1000` — maximum bounded task capacity for the background side-channel webhook and transcript worker queue |
| `MVG_PLUGIN` | unset — module name of the plugin (see "Plugin") |
| `MVG_BRAIN` | `plugin` when the plugin has `brain_call`, else `http` — `plugin` or `http` |
| `MVG_BRAIN_URL` | unset (required for `http`) — endpoint the `http` brain POSTs to |
| `MVG_BRAIN_TOKEN` | unset — if set, sent as `Authorization: Bearer <token>` |
| `MVG_HTTP_STREAM` | `0` — `1` enables streaming SSE mode for the `http` brain, requesting `Accept: text/event-stream` and incrementally consuming sentence/delta SSE lines |
| `MVG_BRAIN_DIRECT_AUDIO` | `0` — `1` forwards raw audio (`audio_b64`) in the brain payload JSON when captured on the turn |
| `MVG_MAX_SSE_CHARS` | `262144` (256 KiB) — maximum accumulated characters from an SSE brain stream |
| `MVG_DEFAULT_TARGET` | `agent` — brain target for a node with no profile (`MVG_HIVE_TARGET` is read as a fallback) |
| `MVG_HIVE_TIMEOUT` | `45` — seconds a brain call may take |
| `MVG_AGENT_NAME` | `the assistant` — the agent's name in prompts and the spoken apology |
| `MVG_AGENT_LABEL` | `agent` — speaker label of the agent's own replies in the addressee context, and its Piper `voice_mapping` key |
| `MVG_CLASSIFY_KEY` | `lomin.ambient` — the `classify_choice` task key for the addressee question |
| `MVG_LOG_NAME` | `lomin.jsonl` — request log file name under `MVG_LOG_DIR` |
| `MVG_OWNER_NAME` | `the owner` — who must confirm a spoken security request in text |
| `MVG_UNKNOWN_SPEAKER` | `unknown (answer for everyone present)` — the frame's speaker text when voice ID cannot place the speaker |
| `MVG_AGENT_NAMES` | unset — comma list of every spelling the agent's name can come out of Whisper as; empty means the name pre-gate never matches |
| `MVG_WAKE_NAMES` | same as `MVG_AGENT_NAMES` — names that count after a greeting ("hey <name>") to open an utterance |
| `MVG_PORT` | `9098` |
| `MVG_BIND` | `0.0.0.0` |
| `WHISPER_URL` | `http://127.0.0.1:9099/transcribe` |
| `MVG_PIPER_CONFIG` | `$MVG_HOME/piper-tts.json` |
| `MVG_LOG_DIR` | `<gateway dir>/logs` |
| `MVG_DESKTOP_TTS_URL` | `http://127.0.0.1:9097/tts` |
| `MVG_DESKTOP_TTS_TIMEOUT` | `8` (seconds) |
| `MVG_TTS_STREAM` | `1` — `/ask/stream` sends sub-sentence pcm `sentence` events (format=pcm, 24 kHz mono, text on the first chunk of each sentence only) read from the desktop `POST /tts/stream`; `0` = whole-sentence WAV events. Logs `first_audio_ms`/`tts_first_ms` |
| `MVG_DESKTOP_TTS_STREAM_URL` | derived: `MVG_DESKTOP_TTS_URL` + `/stream` |
| `MVG_TTS_STREAM_MIN_CHUNK_BYTES` | `4800` (~100 ms, the hub's own forwarding floor) |
| `MVG_TALKER_BACKENDS` | unset — comma list of `name=module:Class` plugin Talker backends for live voice (see `talker_backend.py`) |
| `MVG_TALKER_BACKEND` | unset — name of the Talker backend for `/voice/live` (`gemini`, `cascaded`, `mock`, or a plugin from `MVG_TALKER_BACKENDS`, or `talker_backend` in a node's `nodes.json` entry); unset means `/voice/live` answers 501 |
| `MVG_PERSONA_DIR` | unset — directory of per-target persona files (`<MVG_PERSONA_DIR>/<target>/voice_talker.md` and `voice.md`) used as fallback Talker instructions when the plugin's `talker_instructions(target, node_id)` hook returns nothing |
| `MVG_GEMINI_API_KEY` / `MVG_GEMINI_MODEL` / `MVG_GEMINI_VOICE` / `MVG_GEMINI_INSTRUCTIONS` / `MVG_GEMINI_URL` | unset / `gemini-3.8-live` / unset / unset / Google endpoint — settings for the built-in `gemini` Talker backend (`MVG_TALKER_BACKEND=gemini`, `gemini_live.py`); the key is required, `MVG_GEMINI_URL` overrides the endpoint |
| `MVG_TALKER_URL` / `MVG_TALKER_MODEL` / `MVG_TALKER_KEY` / `MVG_TALKER_TIMEOUT` / `MVG_TALKER_INSTRUCTIONS` | unset / `gpt-4o-mini` / unset / `15` / built-in prompt — settings for the built-in `cascaded` text-in/text-out Talker backend (`MVG_TALKER_BACKEND=cascaded`, `talker_backend.py`) |
| `MVG_LIVE_MAX_SEC` / `MVG_LIVE_IDLE_SEC` / `MVG_LIVE_MAX_FRAME` | `600` / `30` / `262144` — session length, idle timeout (seconds) and largest client frame (bytes) for `/voice/live` |
| `MVG_SPOKEN_CANCEL` | `0` — `1` makes a spoken "stop" / "cancel that" on `/voice/turn` cancel the running turn (see Cancel) |
| `MVG_CANCEL_PHRASES` | `stop,cancel,cancel that,never mind,nevermind,stop that` — comma list of exact cancel phrases |
| `MVG_CONTEXT_OPENER` | `0` — `1` speaks "One moment, <detail>" from the first stage event in place of the generic opener clip (see Context opener) |
| `MVG_SPOKEN_PROGRESS` | `0` — `1` speaks stage-event progress lines during long tasks (see Spoken progress) |
| `MVG_PROGRESS_MIN_GAP_SEC` / `MVG_PROGRESS_MAX_PER_TURN` / `MVG_PROGRESS_STAGES` | `12` / `3` / `tool_start,milestone` |
| `MVG_GAP_FILL` | `1` — filler clips while the brain is silent, pcm stream mode only (see "Gap-fill and latency metrics"); `0` disables |
| `MVG_GAP_FILL_DIR` | `<gateway dir>/gap-fill` |
| `MVG_SPLICE_RAW_AUDIO` | `0` — `1` passes the turn's WAV to the plugin's `brain_call_stream` as `audio_clips` (only if its signature accepts it); ADPCM and vision handling are unchanged |
| `MVG_AUDIO_CLIPS_MAX_BYTES` | `2097152` — total clip bytes over this are dropped and logged to `voice_fallback.log` (`audio_clips_over_cap`); the turn continues |
| `MVG_ORCHESTRATOR_PROMPT` | `0` — `1` appends a generic voice-orchestrator instruction (speak one sentence, then delegate to a subagent) to every framed brain prompt; turn records in `voice-latency.jsonl` carry `orchestrator` so runs with it on and off can be compared |
| `MVG_ORCHESTRATOR_TEXT` | built-in text — a site's own instruction block, replacing the default |
| `MVG_ORCHESTRATOR_TARGETS` | empty (all targets) — comma list of brain targets that get the block |
| `MVG_GAP_FILL_AFTER_MS` | `1200` — no real audio by this long after request start: emit one acknowledge/thinking clip |
| `MVG_GAP_FILL_LONG_MS` | `5000` — silence (audio already sent has finished playing) longer than this while the brain still works: emit one long_wait clip, at most one per turn |
| `MVG_TEXT_SENTENCES` | `0` — `1` makes `/ask/stream` send text-only `sentence` events (no `audio_b64`) and makes `/v1/audio/speech` stream (44-byte WAV header with unknown sizes, then 24 kHz mono PCM as synthesized; `"stream": false` in the request keeps the complete-WAV reply). For a hub's native streaming TTS client. Gap-fill is disabled in this mode: the hub latches a turn to brain-audio or remote-tts and rejects a mix, and a clip is audio |
| `MVG_FIRST_CLAUSE_EARLY` | `1` — cut the first utterance early (terminator with no trailing space yet, a comma/colon after >=20 chars and 4 words, or after 12 words) instead of waiting for the next partial; `0` disables. Both modes |
| `MVG_STT_CORRELATE_SEC` | `30` — how recent an `/stt` must be for its id/timing to be attached to the next `/ask/stream` turn record |
| `MVG_NODES_FILE` | `$MVG_HOME/nodes.json` (per-node profiles; see `nodes.example.json`) |
| `MVG_NODE_TARGETS_FILE` | `$MVG_HOME/node-targets.json` (fallback, read only if nodes.json is absent) |
| `MVG_NODE_TARGETS` | unset (JSON object, env; wins over the files) |
| `MVG_TRUSTED_KEY` | unset (shared secret for the `X-Voice-Claimed-Speaker` header; unset = claim ignored) |
| `MVG_VOICE_ID_GRACE_SEC` | `0.5` (voice ID grace after the transcript on /voice/turn) |
| `MVG_AMBIENT_THRESHOLD` | `0.6` |
| `MVG_AMBIENT_LAYA_TIMEOUT` | `0.5` (seconds) |
| `MVG_AMBIENT_LAYA_CONFIDENT` | `0.7` (Laya P(yes) that engages without Jev) |
| `MVG_AMBIENT_JEV_TIMEOUT` | `2.5` (seconds) |
| `MVG_AMBIENT_HAIKU_TIMEOUT` | `12` (seconds) |
| `MVG_AMBIENT_CONTEXT_MAX_ENTRIES` | `6` |
| `MVG_AMBIENT_CONTEXT_WINDOW_SEC` | `180` |
| `MVG_AMBIENT_FOLLOWUP_WINDOW_SEC` | `20` |
| `MVG_AMBIENT_SPEAKER_WAIT_SEC` | `0.5` (seconds, measured after STT finishes) |
| `MVG_STT_CACHE_TTL_SEC` | `60` |
| `MVG_DESKTOP_EMBED_URL` | `http://127.0.0.1:9096/embed` |
| `MVG_DESKTOP_EMBED_TIMEOUT` | `4` (seconds) |
| `MVG_DESKTOP_LAYA_URL` | `http://127.0.0.1:9121/v1/systemone` |
| `MVG_DESKTOP_LAYA_TIMEOUT` | `20` (seconds) |
| `MVG_SPEAKERS_PATH` | `$MVG_HOME/speakers.json` |
| `MVG_SPEAKER_THRESHOLD` | `0.70` |
| `MVG_SPEAKER_ID` | `1` — `0` turns speaker identification off on every node (`identify` false in `resolve_profile`) |
| `MVG_SILERO_MODEL` | unset (glob under `/opt/lomin/venv/.../openwakeword/resources/models/silero_vad.onnx`) |
| `MVG_SPEECH_THRESHOLD` | `0.5` (Silero per-window probability) |
| `MVG_MIN_SPEECH_MS` | `400` (voiced ms needed to pass the speech gate) |
| `MVG_SPEECH_GATE` | `1` (`0`/`false`/`off` disables the gate) |
| `MVG_TTS_CHAIN` | `desktop,piper` — comma list of TTS backends, tried in order; see "Backends". Unknown names are dropped with a stderr line; empty means `piper` |
| `MVG_ELEVENLABS_VOICE_ID` | unset — ElevenLabs voice id (required for `elevenlabs`) |
| `MVG_ELEVENLABS_MODEL` | `eleven_multilingual_v2` |
| `ELEVENLABS_API_KEY` | unset — key for the `elevenlabs` TTS backend (required for it) |
| `MVG_OPENAI_TTS_URL` | unset — OpenAI-compatible speech endpoint for the `openai` backend, e.g. `http://localhost:8880/v1/audio/speech` (required for it) |
| `MVG_OPENAI_TTS_MODEL` | `kokoro` |
| `MVG_OPENAI_TTS_VOICE` | `af_sky` |
| `MVG_OPENAI_TTS_KEY` | unset — bearer token sent to the `openai` endpoint, only when set |
| `MVG_HTTP_TTS_URL` | unset — generic `http` backend: POST `{"text": ...}`, the response body is the WAV (required for it) |
| `MVG_TTS` | unset — opt-in TTS backend selector (`gemini`, `kokoro`); returns `None`/off when unset |
| `MVG_GEMINI_API_KEY` | unset — API key for `gemini` TTS backend (passed via `x-goog-api-key` header, falls back to `GEMINI_API_KEY`) |
| `MVG_GEMINI_VOICE` | `Schedar` — default voice for Gemini TTS |
| `MVG_GEMINI_TTS_MODEL` | `gemini-2.5-flash-preview-tts` — model for Gemini TTS |
| `MVG_GEMINI_TTS_URL` | unset — custom endpoint URL for Gemini TTS |
| `MVG_KOKORO_URL` | `http://127.0.0.1:8880/v1/audio/speech` — Kokoro TTS endpoint URL |
| `MVG_KOKORO_VOICE` | `af_heart` — default voice for Kokoro TTS |
| `MVG_KOKORO_MODEL` | `kokoro` — default model for Kokoro TTS |
| `MVG_KOKORO_API_KEY` | unset — bearer token for Kokoro TTS (isolated from `MVG_OPENAI_TTS_KEY`) |
| `MVG_KOKORO_KEY` | unset — alias for `MVG_KOKORO_API_KEY` |
| `MVG_MAX_TTS_RESPONSE_BYTES` | `26214400` (25 MiB) — maximum response size cap for TTS backend HTTP responses |
| `MVG_HA_ENABLED` / `MVG_HA_FASTPATH` | `0` — `1` enables Home Assistant conversation intent fast-path |
| `MVG_HA_URL` | unset — Home Assistant base URL (e.g. `http://homeassistant.local:8123`) for `/api/conversation/process` |
| `MVG_HA_TOKEN` | unset — long-lived access token for Home Assistant REST API |
| `MVG_HA_INTENT_TOKEN` | unset — bearer token required for executing `/intent` actions via Home Assistant |
| `MVG_ENABLE_DEFER` | `0` — `1` enables asynchronous deferred voice answer store (`[[defer]]` marker detection and `/voice/pending` endpoints) |
| `MVG_DEVICE_TOKENS` | unset — per-device or shared bearer tokens (`node1:tok1,node2:tok2[,shared_token]`) enforced across all API endpoints |
| `MVG_AUTH_MAX_FAILURES` | `5` — maximum consecutive failed bearer auth attempts per client IP before lockout |
| `MVG_AUTH_LOCKOUT_SEC` | `60.0` — lockout duration in seconds after max failed auth attempts (returns 429 Too Many Requests) |

## Endpoints

| Method/Path | Wire format | Backend |
|---|---|---|
| `POST /stt` | multipart/form-data (part `audio`, `audio.wav`) or raw WAV body → `{"text": "..."}`. Serves a 60s cache hit (by sha256 of the WAV bytes) if `/ambient` already transcribed the same audio and engaged. | `WHISPER_URL` (default `http://<host>:9099/transcribe`) |
| `POST /ambient` | multipart/form-data (part `audio`, text field `node_id`) or raw WAV body with `?node_id=` → `{"engage": bool, "transcript": str, "score": float, "classifier": "wake-phrase"\|"pregate"\|"laya"\|"laya+jev"\|"laya+haiku"\|"jev"\|"haiku"\|"name-only"\|"none", "ms": {"stt": .., "classify": ..}}` | Whisper (via the existing `transcribe()`), then Laya → Jev → Haiku classification — see "Ambient gate" below |
| `POST /ask` | JSON `{prompt, session_id, effort, node_id, speaker, speaker_score}` → `text/event-stream` (`status`*, `reply`, `done`). Returns HTTP 400 on invalid JSON. | `the plugin's brain transport`'s `brain_call()`, target `agent` |
| `POST /ask/stream` | Same JSON request as `/ask` → `text/event-stream` (`status`*, `sentence`*, `reply`, `done`). Returns HTTP 400 on invalid JSON. | `brain_call_stream()` — speaks each sentence as soon as it's complete; see "MANDOS streaming" below |
| `POST /v1/audio/speech` | OpenAI-compatible JSON `{model, input, voice, response_format}` → **always 24 kHz mono 16-bit WAV bytes** (`response_format` is accepted but ignored for the actual audio — the edge client plays with `aplay -`, which cannot decode mp3). Long inputs are sentence-chunked and synthesized in parallel (`synthesize_chunked`); see "Every audio payload is 24 kHz mono 16-bit WAV" below. | desktop Chatterbox -> desktop Piper -> local Piper |
| `POST /embed` | raw WAV body → `{"embedding": [...], "model": "..."}`, straight proxy to the desktop's speaker-embedding server | `MVG_DESKTOP_EMBED_URL` — see "Speaker ID" below |
| `POST /v1/systemone` | JSON → JSON, straight proxy to the desktop Laya classifier | `MVG_DESKTOP_LAYA_URL` |
| `POST /intent` | JSON `{text, speaker, node, turn_id}` -> `204` (no match) or the same `text/event-stream` as `/ask/stream`. LAMMAS, see below. | `lammas.py` |
| `GET /health` | → `{"status": "ok", "upstreams": {...}}` | — |

Every request logs one JSON line to `logs/lomin.jsonl`
(endpoint, ms, outcome — no audio bytes, no keys); per-turn and per-`/stt` timing also goes to `logs/voice-latency.jsonl` (see "Gap-fill and latency metrics"). A TTS fallback to Piper
additionally logs one line to `logs/voice_fallback.log` with
`tts_engine: "piper"` and the reason the desktop TTS hop failed.

## Ambient gate: `/ambient` (talk to the agent without a wake word)

Instead of the edge device spotting a wake phrase locally, it sends every
speech segment here. This endpoint transcribes it (the same `transcribe()`
Whisper call `/stt` uses — no second STT pass, see the cache below) and
asks a classifier whether the utterance is addressed to the agent. It is
**separate from `/ask`**: the edge client decides what to do with
`engage: true` (typically: re-POST the same audio for a real `/stt` or
`/ask` turn), this endpoint never calls the brain itself.

Request: multipart/form-data (part `audio`, 16kHz mono WAV; text field
`node_id`) or a raw WAV body with `?node_id=` in the URL.

Response:
```json
{"engage": true, "transcript": "...", "score": 0.95,
 "classifier": "jev", "speaker": "mike", "speaker_score": 0.83,
 "ms": {"stt": 210.4, "classify": 340.1, "vad": 9.6}}
```

`speaker`/`speaker_score` come from the same best-effort speaker-ID path
described in "Speaker ID" below — `speaker` is `""` when nobody matched
(unconfigured, no match, or the identification failed for any reason).

**Stage 0, the speech gate (before Whisper).** Silero VAD runs over the
uploaded WAV first: openWakeWord's bundled `silero_vad.onnx` (default
`/opt/lomin/venv/lib/python3*/site-packages/openwakeword/resources/models/silero_vad.onnx`,
override `MVG_SILERO_MODEL`) through onnxruntime, 512-sample (32 ms)
windows, LSTM state reset per clip, stereo averaged and non-16 kHz audio
linearly resampled. Voiced ms = windows with p >= `MVG_SPEECH_THRESHOLD`
(0.5); below `MVG_MIN_SPEECH_MS` (400, the ear's
`AMBIENT_MIN_SPEECH_SECONDS`) the clip is answered `engage: false`,
`classifier: "no-speech"` with **no Whisper and no classify call**. It is
server-side because other edge clients may still use the energy
VAD. The log line carries `voiced_ms`, `speech_gate`
(`passed`/`rejected`/`unavailable`) and `vad_ms`. **Fails open:** a missing
onnxruntime/numpy/model, a non-16-bit or unparseable WAV, or any VAD error
logs `speech_gate: "unavailable"` and continues to Whisper. The model loads
once, lazily, under a lock (a failed load is retried after 5 minutes).
`MVG_SPEECH_GATE=0` disables it. Measured on the desktop: ~9-15 ms per
1-2 s clip (host 4 not measured).

**Classification, in order** (`classify_addressed_detail()`; up to two calls to
`the plugin's classify module`'s `classify()`, MAHANAXAR, each writing its own usage line to
`logs/classify-usage.jsonl`). Laya is a fast path for confident yeses only;
Jev gives a second opinion on everything else:
1. **Laya** (`layers=("laya",)`, call 1): the local classifier on the desktop
   (a local service on 127.0.0.1:9121 on the desktop, reached from the gateway host over
   an ssh reverse tunnel). A `choice` question
   `["yes","no"]`, "Is the speaker talking to the agent?", over a conversation turn
   list (`_build_ambient_turns`: one user turn per recent room line oldest
   first, an `agent: <reply>` turn if the agent replied within
   `MVG_AMBIENT_FOLLOWUP_WINDOW_SEC`, the utterance last, all
   `<speaker>: <text>`). Score is P(yes). Budget `MVG_AMBIENT_LAYA_TIMEOUT`
   (default 0.5s), no retry. P(yes) >= `MVG_AMBIENT_LAYA_CONFIDENT`
   (default 0.7) -> engage, `classifier: "laya"`, Jev is not called.
2. **Jev** (`layers=("jev","haiku")`, call 2), when Laya's P(yes) is below the
   confident bar, or Laya timed out or errored: via the Vercel AI Gateway
   (`https://ai-gateway.vercel.sh/v1/evaluate`,
   key at `$MVG_HOME/secrets/vercel-ai-gateway`, model `typesafe-ai/jev`),
   question key `addressed_to_agent`, `MVG_AMBIENT_JEV_TIMEOUT` (default
   2.5s), **single attempt** (`jev_retry=False`: classify()'s 1s-delay retry
   would blow the budget). Jev's score decides against the threshold, so a Jev
   yes overrides a Laya no. Voice-specific
   instructions cover Whisper's name mishearings (the agent/Lommin/Lowmin/Lemon/Loaming),
   what "addressed" means (a request or question, by name anywhere in the
   sentence, or an unnamed follow-up inside an open conversation — see
   below), and that mentioning the agent to someone else, or TV/music audio, is
   not addressed.
3. **Haiku fallback** (classify()'s `haiku` layer, `claude -p --model
   claude-haiku-4-5`) — only on a Jev *error or timeout*, never on a
   merely-low Jev score. Timeout `MVG_AMBIENT_HAIKU_TIMEOUT` (default 12s;
   measured ~8.8s on slow hardware). Haiku is slow, so the edge client's ambient
   timeout must exceed Laya + Jev + Haiku (0.5 + 2.5 + 12 s worst case).
4. If Laya answered but Jev and Haiku both fail: Laya's own P(yes) stands,
   `classifier: "laya"`, compared with the threshold (0.6). If nothing
   answered: `engage: false`, `classifier: "none"`. Never guesses.

`classifier` values: `"wake-phrase"` (a greeting plus the name opens the
utterance, e.g. "hey the agent", "okay Lommin"; score 1.0, no classifier called),
`"pregate"` (no name and no open conversation; score 0.0, no classifier called),
`"laya"` (confident, or Laya-only fallback),
`"laya+jev"` / `"laya+haiku"` (Laya ran and was not confident, the second
opinion decided), `"jev"` / `"haiku"` (Laya timed out or errored), `"none"`,
plus `"no-speech"` from the gate. When Laya answered, the response and log
line also carry `laya_score`.

**Laya dependency:** needs `laya-classifier.service` running on the desktop
and the tunnel carrying `-R 127.0.0.1:9121:127.0.0.1:9121`. With the desktop
off, the connection is refused (or times out at 0.5s), Laya is skipped and
Jev answers exactly as before Laya existed. Accepted trade-off: some false positives
on third-person mentions ("the agent told me ...").

Threshold: `MVG_AMBIENT_THRESHOLD` (default `0.6`) — `engage` is
`score >= threshold` and only when a classifier actually answered. With no classifier
the score is 1.0 when the name is present and 0.0 otherwise (classifier `name-only`).

**Filtering before classification:** an empty or very short (fewer than 2
alphanumeric characters) transcript, or one of Whisper's known silence
hallucinations ("Thank you.", "you", "Thanks for watching", ...), never
reaches a classifier at all — `engage: false`, `classifier: "none"`,
`score: 0.0` immediately.

**Rolling context, per `node_id`, in memory only:** the last 6 transcripts
from the last 3 minutes (`MVG_AMBIENT_CONTEXT_MAX_ENTRIES` /
`MVG_AMBIENT_CONTEXT_WINDOW_SEC`), plus the timestamp of the agent's last real
`/ask` reply on that node (the apology doesn't count — see `record_reply`
in `_handle_ask`). Within `MVG_AMBIENT_FOLLOWUP_WINDOW_SEC` (default 20s)
of that reply, the classifier is told the conversation is OPEN, so an
unnamed follow-up like "and tomorrow?" can still engage. Jev and Haiku
see the same state text (Laya gets the equivalent turn list) — each line is prefixed `<speaker>: ` (`unknown:`
when nobody matched), including the current utterance
(`_build_ambient_state`), so the classifier and any downstream reasoning
knows who said what, not an anonymous transcript.

**Privacy:** an ambient transcript that doesn't engage is never written to
disk anywhere — not the request log, not an error message, not the rolling
context's persistence (there isn't any; it's process memory, gone on
restart). The one request-log line per call
(`logs/lomin.jsonl`) carries metadata only: `node`,
`stage_ms` (`{stt, classify}`), `score`, `engaged`, `classifier`, `chars`
(character count) — never the transcript text, engaged or not. **The
speaker label is never logged either** — it stays in-memory only (the
rolling context and the speaker cache), the same posture as the transcript
text itself, since a name is still identifying information about who was
in the room.

## Speaker ID (SPEC-011)

The Go voice hub (`speaker.go`) already
does speaker identification for any turn that reaches it — but an ambient
utterance that doesn't engage never reaches the hub at all, so without this
it would stay permanently anonymous. This gateway runs the same
identification independently inside `/ambient` (`identify_speaker()`) so
every voice clip gets a speaker label, not just engaged ones.

**Architecture:** a desktop-hosted embedding server
(`desktop_embed.py`, this directory — speechbrain's ECAPA-TDNN,
`speechbrain/spkrec-ecapa-voxceleb`, on the desktop's RTX 4090), reached
over the same ssh reverse tunnel `desktop_tts.py` uses
(`its systemd unit`, now forwarding both `9097` for TTS and
`9096` for embedding over one ssh connection). `gateway.py`'s `POST /embed`
is a thin proxy to it (`MVG_DESKTOP_EMBED_URL`) — this is both what the Go
hub's own `DefaultEmbedClient` calls for an engaged turn, and what
`identify_speaker()` calls internally (via `proxy_embed()`) for an ambient
one.

**Matching (`identify_speaker`, `_load_fingerprints`, `_match_speaker`,
`_cosine_similarity`) is a direct Python port of `speaker.go`'s
`FingerprintStore`/`MatchSpeaker`/`CosineSimilarity`:**
- Fingerprints are read from `MVG_SPEAKERS_PATH` (default
  `<path>` — the same file
  the Go hub reads) and **reloaded whenever its mtime or size changes**, so
  enrolling a new speaker (`enroll-speaker.py` (not in this repo), below) never
  needs a gateway or hub restart.
- The embedding's `model` field must match `speakers.json`'s `model`
  exactly, or the match is refused (returns unmatched) rather than
  comparing embeddings from two different models.
- Cosine similarity against every enrolled speaker; the best match wins if
  it clears `MVG_SPEAKER_THRESHOLD` (default `0.70`, same default as the
  Go side's `GetThreshold()`).
- A fingerprint that can't be compared (wrong dimension, malformed) is
  skipped rather than aborting the whole match.

**Timing:** `identify_speaker()` runs in a background thread started
*before* STT, in parallel with the Whisper call — `_handle_ambient` only
waits up to `MVG_AMBIENT_SPEAKER_WAIT_SEC` (default 0.5s) for it, measured
from when STT finishes, not from when the request started. A slow or dead
embed hop therefore never meaningfully delays `/ambient`'s response; it
just means `speaker: ""` for that turn.

**Best-effort, always:** no fingerprints configured, an unreachable or slow
desktop embed server, a model mismatch, or a malformed response — every one
of these returns `("", 0.0)` from `identify_speaker()`, never raises, and
never fails or delays the ambient turn itself.

**Caching:** a match is cached by `sha256(wav_bytes)` alongside the
existing STT cache (`_speaker_cache`, same `MVG_STT_CACHE_TTL_SEC`,
populated only on engage) — a second `/ambient` call on the same bytes
(a retry, for instance) is served from the speaker cache instead of a
second desktop round trip. The Go hub's own engaged-turn path still does
its own `/embed` call regardless; this cache only helps `/ambient` calling
itself, not the hub.

### Desktop embedding server (`desktop_embed.py`)

Same pattern as `desktop_tts.py`: `POST /embed` (raw WAV body, any sample
rate/channel count — downmixed and resampled to 16kHz mono defensively) →
`{"embedding": [192 floats], "model": "speechbrain/spkrec-ecapa-voxceleb"}`,
`GET /health`. The model is loaded once at process start (`_load_model()`),
followed by one warmup encode, so the first real request doesn't pay for
CUDA kernel compilation. `MDE_IDLE_UNLOAD_MIN` (float minutes, default `0` = never unload) drops the model after that many idle minutes; the next `/embed` reloads it, and `/health` reports `"status": "idle"` meanwhile. Runs under the same shared venv as
`desktop_tts.py`, `$MVG_HOME/venvs/chatterbox` (needs `speechbrain` in
addition to what's already there — confirmed not to touch the installed
CUDA `torch`/`torchaudio` build). Rejects an empty, undecodable, or
too-short (<0.5s) body with 400. Installed as
`its systemd unit` (not installed by this PR — see
"Running").

### Enrollment: `enroll-speaker.py` (not in this repo)

Runs on the desktop.

```
enroll-speaker.py mike --from-pi 25
enroll-speaker.py mike --wav clip1.wav clip2.wav --dry-run
enroll-speaker.py --list
enroll-speaker.py --remove mike
```

`--from-pi SECONDS` (default 25) ssh's to the gateway host, stops
`its systemd unit`, records `SECONDS` of audio from the same
device the live client uses (`arecord -D mm_mic -f S16_LE -r 16000 -c 1`
— the host's `~/.asoundrc` `mm_mic` alias, not the raw hardware device
directly, so enrollment matches the exact capture path the hub embeds at
runtime), restarts the service in a `finally`/trap no matter what happens
in between, and scp's the WAV back. The recording is split into ~4s
chunks, near-silent chunks are dropped by a relative RMS floor, each
surviving chunk is embedded via the desktop embedding server directly
(`http://127.0.0.1:9096/embed` by default, `--embed-url` to override),
L2-normalized, averaged into a centroid, and the centroid is
L2-normalized again. Prints the pairwise chunk-to-centroid cosine
min/mean as a quality check — a low min means noisy or inconsistent audio,
worth re-recording. The `model` string enrolled is whatever the embedding
server itself reports, never a hardcoded constant, so the two can't drift.

The result is merged into `speakers.json` on the gateway host
(`<path>`) — read via ssh,
the target id replaced or added, `model` set/verified, written atomically
(tmp file + `mv`) after backing up the old file as `.bak-<timestamp>`.
Refuses to write if the existing file's `model` differs from the newly
enrolled embedding's model, unless `--force` (which also drops every other
enrolled speaker, since their embeddings are meaningless under a different
model). `--dry-run` prints what would be written instead of writing it.

### Config for the host's the hub hub

Add to the host's `voice_hub` config (`<path>`):

```yaml
voice_hub:
  speaker_id:
    embed_url: "http://<host>:9098/embed"
    fingerprints_path: "/config/speakers.json"
    threshold: 0.70
```

(`http://<host>:9098/embed` is this gateway's own `/embed` proxy —
the Go hub never talks to the desktop embedding server directly, same as
it never talks to the desktop TTS server directly.)

## STT cache

When `/ambient` engages, the edge client is expected to re-POST the same
WAV bytes to `/stt` (or the hub does, on its behalf) for the real
transcript rather than paying for a second Whisper pass. `/ambient` caches
the transcript by `sha256(wav_bytes)` for `MVG_STT_CACHE_TTL_SEC` (default
60s) **only when it engaged** — a non-engaging transcript is never cached,
consistent with the privacy rule above. `/stt` checks the cache first and
logs `ok:cache_hit` instead of calling Whisper again on a hit.

## /ask and the brain

Requests to `/ask` and `/ask/stream` (`/ask*`) require valid JSON in the request body; invalid JSON returns HTTP 400 (`{"error": "invalid JSON body"}` or `{"error": "request body must be a JSON object"}`).

The prompt is wrapped so the answering leg knows it's a spoken turn:

```
[VOICE via hub node <node_id>, speaker: <speaker or "unknown, answer for everyone present">] <prompt>

This reply will be spoken aloud: plain sentences, no markdown, no code, under 60 words.
```

The gateway dispatches the framed prompt through the plugin's `brain_call(target, prompt, timeout)` (or `http_brain_call` when `MVG_BRAIN=http`). Site-specific transport internals (such as queue tables, caller identity fields, or database wake polling) live inside the plugin; see [`examples/README.md`](examples/README.md) for a reference plugin and a queue-backed transport example.

If `brain_call` returns `"unanswered: <reason>"` (any reason, including one
with a parenthetical suffix like `"unanswered: timed_out (could not
deliver: ...)"`) or an empty reply, the gateway replies with a short spoken
apology ("the agent is busy right now, try again in a moment.") instead of
forwarding that text, and logs the reason. The blocking `brain_call()` runs
in a worker thread so `status` events keep pulsing (~every 5s) while it
waits; SSE headers are flushed immediately on connect, before that thread
even starts, since the hub's `ResponseHeaderTimeout` is watching.

## MANDOS streaming: `/ask/stream` (specs/2026-09-28-mandos-streaming.md)

`/ask` waits for the whole brain call answer before returning any text, so
nothing plays until the entire reply is known (measured: 271 chars took
9.7s end to end). `/ask/stream` fixes this: it consumes the brain call's
in-progress answer as the answering leg writes it (`brain_call_stream(target, prompt, timeout, on_partial, stages_out)`), cuts
complete sentences out of it with `sentence_segmenter.SentenceSegmenter`,
and synthesizes + emits each one as soon as it's ready — sentence N+1's
TTS runs while sentence N's audio is already on the wire. Sentences are
always emitted in source order regardless of which TTS call actually
finishes first.

Wire format (additive — `/ask` is unchanged, and this endpoint is not yet
consumed by hub.go/hub, see the spec's "What's out of scope"):

```
event: status    data: {"status": "thinking"}
event: sentence  data: {"text": "...", "engine": "chatterbox", "audio_b64": "..."}
event: sentence  data: {...}
event: reply     data: {"reply": "<full text, for transcript/logging>"}
event: done      data: {}
```

If the target never streams (an older leg, or a turn with no intermediate
text blocks — every turn before this PR), `sentence` events simply never
fire early and the reply still comes back as one sentence at the end,
same net behavior as `/ask` plus the trailing `reply`/`done` pair.

**Every audio payload is 24 kHz mono 16-bit WAV** .
The engine can change sentence to sentence within one streamed reply
(the desktop Chatterbox clone at 24 kHz, Piper at 22.05 kHz, or
another rate) — the the hub ear wants ONE format per turn, so every
`sentence` event's `audio_b64` and `/v1/audio/speech`'s response body are
passed through `normalize_wav()` before being sent. `engine` still reports
which engine actually spoke; only the audio's sample rate/channels/width is
normalized, not the label. Conversion is best-effort: if it fails for a
given sentence, the gateway logs the failure to `logs/voice_fallback.log`
(`tts_engine: "normalize-skip"`) and sends the original WAV rather than
dropping the sentence.

## Gap-fill and latency metrics

**Gap-fill (v3, ask-077dfb).** The brain takes 4-5 s to its first sentence. With
`MVG_GAP_FILL` on (default), `/ask/stream` (pcm mode) fills the silence with short
sounds, each in its own timing slot. Times are ms since request start unless noted.

- `MVG_GAP_FILL_AFTER_MS` (1200): no real audio yet -> one `vocal_open` clip
  ("Hm.", "Mm.", "Mhmm."), with no caption (`text: ""`).
- `MVG_GAP_FILL_MID_MS` (3500), and at least 800 ms of playback silence -> one
  effect: `handoff` when the transcript matches a handoff topic, else
  `sfx_home` / `sfx_calendar_lists` / `sfx_lookup` for the matching topic, else
  `sfx_generic`. Topical effects are off with `MVG_GAP_FILL_TOPICAL=0`.
- `MVG_GAP_FILL_SLOW_MS` (8000), and 800 ms of silence -> one `vocal_long` clip
  ("Hmmm."), once per turn.
- `MVG_GAP_FILL_TICK_MS` (3000): while still no real audio, a `sfx_tick` after
  each `TICK_MS` of silence, once the long vocal has played.
- `MVG_GAP_FILL_LONG_MS` (5000): once real audio has started, a `sfx_tick` after
  each `LONG_MS` of silence. No vocal clip ever plays after real audio.
- At most 6 ticks per turn (`GAP_FILL_MAX_TICKS`, not an env var). One clip per
  poll; each slot fires at most once. A category with no clips is silent (the
  slot still counts as used).
- A clip is emitted whole as pcm `sentence` events (24 kHz mono, `engine:
  "gap-fill"`, `filler: true`), never interleaved with real audio. Only `handoff`
  clips carry their caption (`text`) on the first chunk. The same clip never plays
  in consecutive turns.

Clips live in `data/voice/gap-fill/<category>/<slug>.wav` plus `manifest.json`.
Vocal clips (`vocal_open`, `vocal_long`, `handoff`) are rendered on the desktop by
`bin/render-gap-fill.py` from `data/voice/gap-fill-phrases.json`; machine effects
(`sfx_*`) are synthesized by `bin/render-gap-fill-sfx.py` (stdlib, deterministic). The phrase renderer skips existing files; `--force`
re-renders. Clips are committed. The gateway reads its own workspace checkout, so the gateway host gets
them through gitops. Missing/invalid clips just mean no filler. Not used in
`MVG_TTS_STREAM=0` (wav) mode. The request log line carries `gap_fill` (slugs).

**Latency log: `logs/voice-latency.jsonl`.** One `kind: "turn"` record per
`/ask/stream`: `ts, node_id, session_id, turn_id, request_start` (epoch ms),
`brain_first_text_ms, first_sentence_ms, gap_fill_first_ms, tts_first_ms,
first_audio_ms, last_audio_ms, total_ms, sentences, reply_chars, audio_s,
gap_fill, tts_engine, target, outcome`, plus `stt_id`/`stt_ms` when a recent
`/stt` (same node if both know it) is correlated. Stage times are ms since
request start. The same `turn_id` is sent as the first SSE event, `event: turn`
`{"turn_id": ...}`, so ears can log it. One `kind: "stt"` record per `/stt`:
`ts, stt_id, node_id` (from `?node_id=`, `X-Node-Id` or a multipart `node_id`
field; null if the hub sends none), `audio_s, stt_ms, chars, cache_hit`.

The Discord ear writes `logs/voice-latency-discord.jsonl` on the desktop
(the Discord ear's own README). `voice-latency-report.py --since 24h`
joins both (reads this file locally or via `ssh <gateway-host> cat`) and prints p50/p90 per
stage and node.

### Measuring latency

`tools/latency_summary.py` (stdlib only) reads a `voice-latency.jsonl` and prints, for the `turn` records, the count and the p50 and p90 (nearest rank) of `first_audio_ms` and `total_ms`. When records carry `opener_kind`, each kind gets its own block. Records missing a field are skipped for that field.

```bash
python3 tools/latency_summary.py logs/voice-latency.jsonl
```

Example output (fabricated placeholder numbers, to show the format only):

```
4 turns
  first_audio_ms: n=4 p50=1200 p90=1500
  total_ms: n=4 p50=4000 p90=5000
opener_kind=gap_fill
1 turns
  first_audio_ms: n=1 p50=1200 p90=1200
  total_ms: n=1 p50=4000 p90=4000
```

## Brain hand-off: stages and progress events

Every `/ask/stream`, `/voice/turn`, and `/voice/live` brain call passes a mutable `stages_out` dict to `brain_call_stream(target, prompt, timeout, on_partial, stages_out)`. A plugin can populate `stages_out` in two ways:

1. **Timing milestones (`stages` in `voice-latency.jsonl`):** Any top-level keys (such as `call_start_ms`, `first_partial_seen_ms`, `final_seen_ms`) are recorded on the turn record's `stages` object for latency analysis.
2. **Live progress events (`stages_out["events"]`):** Appending dicts of shape `{"stage": "tool_start" | "milestone", "detail": "...", "t": 1.2}` drives the context opener (`MVG_CONTEXT_OPENER=1`), spoken progress on `/voice/turn` (`MVG_SPOKEN_PROGRESS=1`), and live `ProgressRelay` updates (`[task progress] <detail>`) on `/voice/live`.

Site-specific queue tables, `from_agent` handling, broker long-polling, and `PRAGMA data_version` wake polling (`MVG_HIVE_POLL_SEC`) belong in the plugin rather than the gateway; see [`examples/README.md`](examples/README.md) for the full site-transport example and stage timestamp reference.

## TTS: always WAV, two-hop fallback chain

1. **Desktop TTS** (`MVG_DESKTOP_TTS_URL`, default
   `http://127.0.0.1:9097/tts`, `MVG_DESKTOP_TTS_TIMEOUT` default 8s) — a
   separate server (`desktop_tts.py`, this directory) running on the
   desktop , reached over an ssh reverse tunnel
   (the tunnel unit) because WSL2 is NAT-only and the gateway host
   cannot open a connection to it directly. This exists because the host's
   own local Piper takes ~19-20s for one sentence under load — well past the Go voice hub's TTS timeout (10s default,
   `config.GetTTSTimeoutSeconds`) — while the desktop's RTX 4090, idle,
   does the same synthesis in well under a second once its voice is
   loaded. This server itself tries two engines, in order:
   1. **Chatterbox Turbo**, zero-shot voice cloning cloned onto the agent's
      own voice from a reference clip (see below) — gives the agent his real voice instead of a generic one.
   2. **Piper** (in-process fallback on this same server), if Chatterbox
      failed to load or a specific call raised.
2. **Local Piper** (`$MVG_HOME/venvs/piper/bin/piper`, voice model
   `$MVG_HOME/piper-voices/en_US-lessac-medium.onnx`) — the last resort,
   run on the gateway host itself if both of the above fail. Already produces WAV
   directly, no ffmpeg step.

The `response_format` field in the request is accepted (so the hub's
request shape is honored) but the gateway always returns `audio/wav`; the
`X-TTS-Engine: chatterbox|piper-desktop|piper` response header
says which engine actually spoke — for the desktop hop, the gateway relays
whatever `desktop_tts.py` itself reports via its own `X-TTS-Engine`
header, rather than assuming Piper. Every fallback (hop failure) still logs one line to `logs/voice_fallback.log` with the reason
and which engine ultimately answered.

### Desktop TTS server and tunnel

`desktop_tts.py` runs on the desktop, binds `127.0.0.1:9097`, loads both
engines ONCE at startup (never per request — reloading a model or
re-embedding a voice-clone reference on every call is exactly the latency
this hop exists to avoid), and exposes `POST /tts` (`{"text": ...}` →
`audio/wav`, `X-TTS-Engine: chatterbox|piper` header) and `GET /health`.
`POST /tts/stream` (same request body) sends audio as Chatterbox produces
it: a streaming-WAV header (size fields 0xFFFFFFFF) then PCM16 mono 24 kHz,
EOF-terminated, headers flushed with the first audio chunk (`"format":"pcm"`
for bare s16le; `X-Sample-Rate` header). Speech tokens are decoded in
windows with context/lookahead tokens and a crossfaded overlap. If
Chatterbox is not loaded, or fails before the first chunk, it serves the
plain `/tts` reply instead. Flags: `MDT_STREAM_ENABLED`,
`MDT_STREAM_FIRST_TOKENS` (12), `MDT_STREAM_CHUNK_TOKENS` (30),
`MDT_STREAM_CONTEXT_TOKENS` (8), `MDT_STREAM_LOOKAHEAD_TOKENS` (6),
`MDT_STREAM_OVERLAP_SAMPLES` (720), `MDT_STREAM_WATERMARK`.

- **Chatterbox Turbo** (`ResembleAI/chatterbox-turbo`, MIT,
  `chatterbox-tts` pip package): loaded via
  `ChatterboxTurboTTS.from_pretrained(device="cuda")`, then
  `model.prepare_conditionals()` embeds the reference audio at
  `$MVG_HOME/voices/default/ref.wav` (env override `MDT_CHATTERBOX_REF_AUDIO`) — the agent's cloned voice. That reference is
  ~12s of clean, single-speaker reference speech, copied to a durable path on the desktop — never
  read from the slow 9p Windows-drive mount at runtime. Disable with
  `MDT_CHATTERBOX_ENABLED=0` to force Piper-only.
- **Piper** (`en_US-lessac-medium`, generic voice) — loaded via
  `piper.PiperVoice.load` as before, used whenever Chatterbox isn't
  loaded or a specific synthesis call raises.

Needs `torch` (CUDA build), `chatterbox-tts`, and `piper-tts` all in one
venv (`$MVG_HOME/venvs/chatterbox`) — never the system or workspace
Python. Piper voice files (`en_US-lessac-medium.onnx[.json]`) live in
`$MVG_HOME/piper-voices/` on both hosts; the desktop's copy was scp'd
from the host's.

`its systemd unit` opens
`ssh -N -R 127.0.0.1:9097:127.0.0.1:9097 -R 127.0.0.1:9096:127.0.0.1:9096 <gateway-host>`
from the desktop, so the host's own gateway process reaches the desktop
servers at its own `127.0.0.1:9097` (TTS) and `127.0.0.1:9096` (speaker
embedding) — the connection direction is desktop-initiated (the only
direction WSL2's NAT allows), the traffic direction is host calling the
desktop. `ControlMaster=no`/`ControlPath=none` are set explicitly because
`~/.ssh/config`'s `Host <gateway-host>` entry otherwise multiplexes onto a shared
`ControlPersist` master, which does not behave the way `Type=simple`
expects a systemd unit's own process to. The embedding forward was added
to this same tunnel rather than opening a second ssh connection — one
consequence is that with `ExitOnForwardFailure=yes`, a bad `9096` forward
now also takes the `9097` TTS forward down, and restarting the tunnel to
deploy an embedding change
briefly drops TTS too.

Three units are checked in here (`its systemd unit`,
`its systemd unit`, `its systemd unit`) but
their `ExecStart`s point at stable install paths
(`~/.local/lib/hub-desktop-tts/desktop_tts.py`,
`~/.local/lib/hub-desktop-embed/desktop_embed.py`), not this
worktree or the pre-merge workspace checkout — repoint them after this PR
merges if the install path changes. `HF_HUB_OFFLINE=1` in
`its systemd unit` only helps once the speechbrain model is
already cached locally (same caveat as the TTS unit's Chatterbox weights) —
the first run needs it unset, or the cache already warm, so `from_hparams`
can actually download the model.

## POST /voice/turn (the LAMMEN pipeline)

One request per utterance; SSE out. Send multipart `audio` (WAV) with text
fields, or a raw WAV body with the same flags in the query string.

| Field / query | Meaning |
|---|---|
| `node_id` | required; picks the nodes.json profile |
| `target` | optional; the agent that answers this request. Used only if the node's profile lists it in `targets`, else the node's `target` |
| `open_room` | `1` runs the addressed gate (default: the profile's) |
| `tts` | `stream` (PCM sentences, default) / `wav` / `none` |
| `session_id` | conversation continuity |
| `audio_chunk` | also emit hub-dialect `audio_chunk` events (default: profile) |
| `device` | optional; what the speaker is using, e.g. `iphone`, `watch`. Header `X-Voice-Device` is the fallback |
| `medium` | optional; how they reach the agent, e.g. `phone-app`, `watch-app`, `room-speaker`, `chat-voice`. Header `X-Voice-Medium` is the fallback |
| `image` / `image_b64` | optional; JPEG camera frame (binary in multipart `image`, or base64 string up to 5 MiB in `image_b64`), gated by `MVG_ENABLE_VISION` or `MVG_VISION_ENABLED` |
| `burst_images_b64` | optional; list of up to 5 base64 JPEG burst frames (each <= 5 MiB), gated by `MVG_ENABLE_VISION` or `MVG_VISION_ENABLED` |

Opt-in Ingest (`vision_adpcm.py`):
- **BLE IMA ADPCM Audio:** When `MVG_ENABLE_ADPCM=1` or `MVG_ADPCM_ENABLED=1`, requests with `APCM`-header audio are decoded into 16 kHz 16-bit mono PCM WAV (upload size capped at 512 KiB; larger payloads return HTTP 413).
- **Multimodal Camera Frames:** When `MVG_ENABLE_VISION=1` or `MVG_VISION_ENABLED=1`, camera frames uploaded via multipart or JSON are validated (base64 validated, max 5 MiB per frame) and forwarded to the brain. Set `MVG_VISION_ADPCM=1` to enable both.

`device` and `medium` are lowercased, stripped and kept only if they match
`^[a-z0-9][a-z0-9-]{0,31}$` (else dropped). Resolution for each: the request
field (cleaned), then the node's `nodes.json` value, then empty. The same two
fields (JSON body keys or the same headers) are read by `/ask`, `/ask/stream`
and `/intent`. Both are logged on a `/voice/turn` request when non-empty.

Watch route: a request carrying `X-Watch-Route` needs `Authorization: Bearer
<token>`, matched against the keys of the tokens file (`MVG_WATCH_TOKENS_FILE`,
default `$MVG_HOME/secrets/gotchi-watch-tokens.json`). The value is a node id
(forces that node, whatever `node_id` the request sends) or a JSON list of
allowed node ids (`{"<token>": ["watch", "phone"]}`): the
request's `node_id` is used if it is in the list, else the list's first entry.
An empty list or any other value type is a 401.

Headers: `X-Voice-Claimed-Speaker` + `X-Voice-Gateway-Key` (must equal env
`MVG_TRUSTED_KEY`; otherwise the claim is ignored, as it is whenever the env
var is unset). Never a body field.

Events: `turn{turn_id,node_id}`, `state{state: transcribing|classifying|thinking|synthesizing|speaking}` (`synthesizing`: the first sentence is going to TTS and nothing has played; a gap-fill clip, whose `sentence` carries `filler: true`, does not send `speaking`),
`transcript{text,transcript,speaker,voice_id}`, `ignored{reason: no-speech|hallucination|not-addressed, score?, classifier?}`
(terminal, no `done`), `status{status}` (keepalive), `sentence{text,engine,format?,sample_rate?,channels?,audio_b64?}`,
`audio_chunk{chunk_index,format,is_final,data,sample_rate?,channels?}` (opt-in), `reply{reply,text,tts_engine}`,
`done{duration_ms}`, `error{error,message}`.

Profiles (`nodes.json`): `target`, `targets`, `open_room`, `identify`, `speech_gate`,
`gap_fill`, `reply_cap_words`, `audio_chunk`, `capture_audio`, `device`, `medium`
(the last two are the node's default device/medium tags, overridden by the
request). Read per request.

Room ear: set `hub_url` to `http://<gateway>:9098/api/voice/interact?node_id=<gateway-host>-bench`
(or `/voice/turn?...`; `/api/voice/interact` is an alias so the ear's derived
heartbeat URL stays valid, and `/api/voice/heartbeat` answers 204).

Legacy endpoints (`/stt`, `/ambient`, `/ask`, `/ask/stream`) stay for one
release; each call logs `legacy_endpoint` and `node` in the request log.

## Client capture contract

`/voice/turn` receives a finished upload, so the gateway has no audio from before speech start. A clipped word start is fixed on the client, which should capture a lead-in. Every number below is a recommendation, not a measured guarantee.

- **Lead-in.** Keep a rolling buffer and send it from before the detected speech start. Recommended 600 ms; tune per device.
- **Resumed speech.** If the user starts again soon after an utterance (recommended window 3 s), the client may send one combined clip instead of two turns.
- **Format.** 16 kHz mono 16-bit PCM WAV, or the ADPCM framing (`APCM` header) when `MVG_ENABLE_ADPCM=1` (upload cap 512 KiB, larger payloads return HTTP 413).
- **Barge-in.** Flush queued playback as soon as the user starts speaking. A server-sent `clear_audio_buffer` event for this is described in `specs/2026-11-05-talker-thinker-live-architecture.md` and does not exist yet.

## Per-node brain routing

`node_id` picks the profile in `nodes.json`
(`MVG_NODE_TARGETS`, a JSON object in env, overrides a node's target); read
fresh on every call. If `nodes.json` is absent the old `node-targets.json`
(`{node: target}`) is read instead. A `node_id` with no profile — including a
blank one — uses the defaults and answers from `MVG_DEFAULT_TARGET` (`agent`). Only which leg answers changes; the reply still comes back out of the
device that asked.

A node may list `targets`, the agents a request may pick: a `/voice/turn` request's `target` field is used only if listed, else the node's `target`.

Every brain call (`/voice/turn`, `/ask`, `/ask/stream`) is framed by one helper,
`frame_voice_prompt`:

```
[VOICE via node <node_id> · device: <device> · medium: <medium> · speaker: <Name | Name (logged in) | unknown (answer for everyone present)> · voice ID: <verdict> · enrolled: <ids | none | unknown> · open room] <prompt>

This reply will be spoken aloud: plain sentences, no markdown, no code, under 60 words.
```

The `enrolled:` field is the voiceprint roster from speakers.json on every turn; the brain must answer "who is enrolled" from this field, not say it has not checked.

Verdict: `verified <id> <score>` / `not verified (best <score>)` / `unavailable`.
`open room` appears only when the addressed gate ran. `device: …` and
`medium: …` appear only when non-empty (each independently), right after the
node bit, e.g. `[VOICE via node phone · device: iphone · medium: phone-app · speaker: Mike · ...]`;
with neither set the line is unchanged.

The gateway passes the framed prompt to the plugin's `brain_call(target, prompt, timeout)` / `brain_call_stream(target, prompt, timeout, on_partial, stages_out)` (or `http_brain_call` when `MVG_BRAIN=http`). See [`examples/README.md`](examples/README.md) for site-specific transport examples.

If `brain_call` returns `"unanswered: <reason>"` (any reason, including one
with a parenthetical suffix like `"unanswered: timed_out (could not
deliver: ...)"`) or an empty reply, the gateway replies with a short spoken
apology ("the agent is busy right now, try again in a moment.") instead of
forwarding that text, and logs the reason. The blocking `brain_call()` runs
in a worker thread so `status` events keep pulsing (~every 5s) while it
waits; SSE headers are flushed immediately on connect, before that thread
even starts, since the hub's `ResponseHeaderTimeout` is watching.

`/ask`'s `node_id` picks which brain leg answers: `MVG_NODE_TARGETS` (a JSON object,
env) if set, else `node-targets.json` (legacy shape
`{"<gateway-host>-bench": "agent", "deck-node": "deck-node", "chat-voice": "agent"}`), read
fresh on every call so an edit doesn't need a restart. A `node_id` not in
the mapping — including a node this file has never heard of — falls back
to `MVG_DEFAULT_TARGET` (`agent`). Only which leg answers
changes; the reply still comes back out of the device that asked, because
the hub replies on that device's own stream, not through this routing.

## Running

```
python3 gateway.py
```

Under systemd, use `lomin.service.example` (`WorkingDirectory=/opt/lomin`, `EnvironmentFile=/etc/lomin.env`).

Two desktop-side helpers are optional and run on a GPU host, reached through an ssh reverse tunnel: `desktop_tts.py` (the Chatterbox/Piper TTS server, `MVG_DESKTOP_TTS_URL`) and `desktop_embed.py` (the speaker-embedding server, `MVG_DESKTOP_EMBED_URL`). See "Desktop TTS server and tunnel" and "Speaker ID" below.

## Running without a plugin

With no `MVG_PLUGIN`, the gateway runs on its own defaults: the brain is `http` (set `MVG_BRAIN_URL`), there is no fast path, no addressee classifier, and every node answers from `MVG_DEFAULT_TARGET`. Env values the plugin would have filled come from the table in "Config". Paths default under `MVG_HOME` (default `~/.mvg`).

## Tests

```
python3 -m pytest -q tests/
```

No network access: upstreams are local HTTP servers on port 0 or monkeypatched functions. `test_plugin.py` covers the loader (a fake plugin in `tmp_path`, a missing plugin name, the empty-name defaults). `test_standalone.py` covers the no-plugin defaults. Tests that need the plugin's `classify` module skip when it is not importable. `test_render_gap_fill_sfx.py` and `test_voice_latency_report.py` skip unless their scripts are present in the checkout.

## Fast path: `try_intent`

The plugin's `try_intent(text)` answers common structured commands with no brain call. The gateway owns no intents: the plugin decides what matches and what its handlers do.

**Placement.** In the turn worker, after STT, the hallucination filter and the addressed gate, before the brain. A hit (`hit: True`) speaks the plugin's reply through the normal sentence/TTS path, so `/voice/turn` and `/ask/stream` get it with no hub change. A miss, a `None` return, an exception, a handler error, or a handler slower than the plugin's own timeout goes to the brain unchanged.

**Return shape.** A hit: `{"hit": True, "intent", "reply", "match_ms", "handler_ms"}`. A miss: `{"hit": False, "intent": name-or-None, "match_ms", "handler_ms", "reason"}`, or `None`.

With no plugin, or a plugin without `try_intent`, every turn is a miss.

### `POST /intent` (for the hub hub)

Request, `Content-Type: application/json`:

```json
{"text": "turn on the living room lights", "speaker": "mike", "node": "kitchen-ear", "turn_id": null}
```

`speaker`, `node`, `turn_id` may be null/absent; `turn_id` is echoed if given.
`text` is required (400 otherwise, as for malformed JSON).

* **204 No Content**, empty body: no match, kill switch on, or the handler
  failed/timed out. Nothing was spoken; fall back to your brain.
* **200**, `Content-Type: text/event-stream`: matched and executed. The body is
  exactly what `/ask/stream` returns:

```
event: turn
data: {"turn_id": "abc-123"}

event: sentence
data: {"text": "Living Room lights on.", "engine": "...", "format": "pcm", "sample_rate": 24000, "channels": 1, "audio_b64": "..."}

event: reply
data: {"reply": "Living Room lights on."}

event: done
data: {}
```

**`tts` (optional): `stream` | `wav` | `none`.** Same meaning as on `/voice/turn`.
`"tts": "none"` keeps the SSE shape above but sends `sentence` events with text
only and runs no TTS here: use it when the hub speaks the reply itself. Absent =
the gateway's default (`MVG_TEXT_SENTENCES` / `MVG_TTS_STREAM`); anything else is 400.

SSE is the only response format for a match. A 204 means no match.
A Home Assistant proxy (Assist conversation API in front of this) maps HA's
`no_intent_match` error to 204 and its response speech to this SSE form.

The side effect has already happened by the time the 200 headers are sent, so
do not retry a 200 against the brain. Latency: the match itself is ~0.03 ms;
the call is the handler plus TTS.

Deploy: nothing is installed by this change. On the gateway host, `git pull` (the 5-minute
gitops cycle) then restart `lomin.service`; `intents/` must
ship with `gateway.py`. Check `logs/voice-latency.jsonl` for `intent` fields and
run `voice-latency-report.py --since 1d`.

## Side-Channel Webhooks, Transcripts & Mic Reopen

Lomin supports non-blocking side-channel notifications, turn transcript streaming, and opt-in question-driven microphone reopening (`side_channel.py`).

### Opt-In Auto Mic Reopen (`MVG_ENABLE_MIC_REOPEN`)
- **Default:** `0` (disabled). When disabled, `/ask` and `/ask/stream` SSE `reply` and `done` payloads remain strictly unmodified to preserve compatibility with existing nodes and clients.
- **Enabled (`MVG_ENABLE_MIC_REOPEN=1`):** Inspects the final spoken reply with `reply_expects_reply()`. If the reply ends with a question mark (accounting for whitespace, quotes, and emoji ornaments), the gateway attaches `"expects_reply": true` to the SSE `reply` and `done` events, indicating to the client device that its microphone should immediately reopen for a follow-up answer.

### Side-Channel Artifact Webhooks (`MVG_ARTIFACT_WEBHOOK_URL`)
- When configured, raw replies are checked for URLs (`https?://...`) and Markdown code blocks (```` ```...``` ````).
- If artifacts are detected, the spoken reply is augmented with a verbal cue ("*Sending the link to Discord.*") unless already cued.
- The raw reply is sent whole to the side-channel webhook payload without markdown stripping, so formatting, links, and code blocks are preserved intact in the destination channel.
- The webhook payload is dispatched via a dedicated background worker queue independent of transcript logging (`MVG_SIDE_CHANNEL_QUEUE_SIZE`, default 1000), ensuring slow or unreachable webhooks never block speech synthesis or drop transcript events.

### Turn Transcripts (`MVG_TRANSCRIPT_SINK_URL` and `MVG_TRANSCRIPT_LOG_PATH`)
- Turn records (including prompt, spoken reply, timing metrics, and node ID) are exported asynchronously via a dedicated background worker queue.
- **HTTP/HTTPS Sink (`MVG_TRANSCRIPT_SINK_URL`):** Dispatches JSON turn records. Validates URL schemes (only `http` and `https` permitted) and blocks insecure cross-scheme redirect downgrades. Supports optional bearer authorization via `MVG_TRANSCRIPT_AUTH_TOKEN`.
- **JSONL File Sink (`MVG_TRANSCRIPT_LOG_PATH`):** Appends turn records to a local JSONL file with strict `0o600` permissions (`-rw-------`). Automatically rotates the active log file to `.1` when size exceeds `MVG_TRANSCRIPT_MAX_BYTES` (default 10 MiB).

## Home Assistant Conversation Fast-Path & Authorization

When `MVG_HA_ENABLED=1` (or `MVG_HA_FASTPATH=1`), Lomin consults Home Assistant's `/api/conversation/process` before falling through to the brain.

- **Fast-Path Matching:** Natural-language control commands (e.g., turning on lights, running scripts, setting thermostats) match HA intents deterministically. Matched intents return immediately with `intent: "ha_conversation"` and the spoken confirmation text.
- **Security & Authorization:**
  - When `MVG_HA_INTENT_TOKEN` or `MVG_DEVICE_TOKENS` is configured, calls to `/intent` require authorization.
  - Authorization requires either a valid Bearer token (`Authorization: Bearer <token>`) matching `MVG_HA_INTENT_TOKEN` / device verifier, or a verified speaker identity via trusted gateway headers (`X-Voice-Claimed-Speaker` validated with `MVG_TRUSTED_KEY`) or biometric voice ID (`voice_id.status == "verified"`).
  - Unauthenticated requests passing an arbitrary `{"speaker": "..."}` JSON field are strictly rejected with HTTP 401 Unauthorized; the speaker field alone never bypasses authorization.
- **Circuit Breaker:** Consecutive timeouts or upstream network errors trigger an automated circuit breaker cooldown, failing fast to the brain to prevent turn stalls.

## Per-Device Bearer Auth & Deferred Async Voice Answers

### Endpoint Bearer Authentication (`MVG_DEVICE_TOKENS`)

When `MVG_DEVICE_TOKENS` is configured, Lomin enforces Bearer token authentication across all API endpoints:
- **Protected Endpoints:** `/voice/pending` (GET & POST), `/voice/turn`, `/ask`, `/ask/stream`, `/stt`, `/embed`, `/ambient`, `/intent`, and `/v1/audio/speech`.
- **Configuration Format:** Comma-separated per-device tokens and optional shared fallback:
  `MVG_DEVICE_TOKENS="node1:tok1,node2:tok2,shared_fallback"`
- **Lockout & Rate Limiting:**
  - Failed authentication attempts are tracked per client IP address.
  - After `MVG_AUTH_MAX_FAILURES` (default 5) consecutive failures within `MVG_AUTH_LOCKOUT_SEC` (default 60s), subsequent requests are immediately rejected with HTTP 429 Too Many Requests until the lockout window expires.
  - Successful authentication clears the failure history for the client.

### Deferred Voice Answers (`MVG_ENABLE_DEFER`)

When `MVG_ENABLE_DEFER=1`, the gateway supports deferred asynchronous voice replies:
- **Marker Detection:** Replies containing `[[defer]]` markers indicate long-running or asynchronous task results.
- **Marker Stripping:** The marker is stripped from spoken audio replies (`strip_defer_marker`) so it is never synthesized into TTS speech.
- **Pending Store:** The pending answer is stored in a thread-safe, memory-bounded store (`PendingVoiceAnswerStore`) indexed by node ID.
- **Client Retrieval (`/voice/pending`):**
  - `GET /voice/pending?node=<node>&pop=1`: Retrieves and pops the oldest pending answer for the node.
  - `POST /voice/pending`: Stores an asynchronous answer directly into the pending queue for subsequent voice delivery.


- **Request Validation:** Malformed JSON bodies on `/ask` and `/ask/stream` return `400 Bad Request` (`{"error": "invalid JSON body"}`).

## Live voice: `GET /voice/live` (WebSocket)

A bidirectional session driven by a Talker backend (`talker_backend.py`) and the coordination harness (`live_harness.py`). The WebSocket layer is a small stdlib RFC 6455 server (`live_ws.py`), so there is no optional dependency to install.

- Request: `GET /voice/live?node_id=<node>` with the usual WebSocket upgrade headers and, when `MVG_DEVICE_TOKENS` is set, the same bearer token as `/voice/turn` (bad or missing: 401). No backend configured: 501. No upgrade headers: 426.
- **`voice_mode` (`native` | `custom_tts`):** Set `"voice_mode"` in the node's `nodes.json` entry (default `"native"`; any other value returns HTTP `400`).
  - `"native"`: the Talker backend emits 24 kHz mono PCM16 directly (e.g. `GeminiLiveBackend` sets `responseModalities: ["AUDIO"]` with optional `MVG_GEMINI_VOICE`).
  - `"custom_tts"`: the Talker backend emits text (`GeminiLiveBackend` sets `responseModalities: ["TEXT"]` and omits `speechConfig`; `CascadedTextBackend` emits stripped text), and `LiveSession` synthesizes each reply through the gateway's `MVG_TTS_CHAIN` (`synthesize(text)`) into 24 kHz mono PCM16 binary frames. See [`specs/2026-11-07-custom-tts-talker.md`](specs/2026-11-07-custom-tts-talker.md).
- **Talker instructions & persona fallback (`MVG_PERSONA_DIR`):** System instructions for the Talker are resolved in order:
  1. The plugin's `talker_instructions(target, node_id)` hook (if defined and returning a non-empty string).
  2. `<MVG_PERSONA_DIR>/<target>/voice_talker.md` (if `MVG_PERSONA_DIR` is set and the file exists).
  3. `<MVG_PERSONA_DIR>/<target>/voice.md` (if `MVG_PERSONA_DIR` is set and the file exists).
  4. `MVG_GEMINI_INSTRUCTIONS` (for `gemini`) or `MVG_TALKER_INSTRUCTIONS` (for `cascaded`).
- **Gemini Live defaults, compression & auto-reconnect (`gemini_live.py`):**
  - Default model: `gemini-3.8-live` (override with `MVG_GEMINI_MODEL`).
  - Every setup handshake enables sliding-window context compression (`"contextWindowCompression": {"slidingWindow": {}}`) and session resumption (`"sessionResumption": {"handle": ...}`).
  - When the server sends a `sessionResumptionUpdate` (`resumable: true`), the backend stores `newHandle`. On a `goAway` frame (`{"type": "reconnect"}`), `LiveSession` automatically calls `backend.reconnect()` to reopen the WebSocket and resume with the saved handle.
- **Stage-event relay & task cancellation:**
  - When the Talker calls `call_thinker`, `/voice/live` registers the handoff's `task_id` in `_ACTIVE_TASKS` and attaches a live `stages_out` sink. Every stage event appended to `stages_out["events"]` is forwarded immediately to `LiveSession.on_stage_event` (`[task progress] <detail>`).
  - When the Talker calls `cancel_thinker`, `LiveSession` invokes `gateway.cancel_active_turn(node_id)`, which triggers the plugin's `brain_cancel(target, task_id)` hook.
- Client binary frames: 16 kHz mono PCM16. Server binary frames: 24 kHz mono PCM16 from the Talker (or synthesized in `custom_tts` / text-out mode).
- Text frames are JSON. Client to server: `{"event": "status"}` (speak the latest task progress), `{"event": "end"}`. Server to client: `{"event": "text"|"interrupted"|"turn_end"|"error", ...}`.
- Close codes: 1000 normal, 1009 frame larger than `MVG_LIVE_MAX_FRAME`, 1011 backend error, 4000 idle for `MVG_LIVE_IDLE_SEC`, 4001 longer than `MVG_LIVE_MAX_SEC`.
- Fallback: if the backend fails to open or errors mid-session, the server sends `{"event": "fallback", "endpoint": "/voice/turn", "reason": ...}` and then closes 1011. The client should continue the conversation on `/voice/turn`.
- One line per session in `lomin.jsonl`: endpoint `/voice/live`, outcome, node, backend.
- Text-out replies: a backend `text` event (without `"spoken": true`) is also synthesized through the normal TTS chain and sent as a binary 24 kHz PCM frame; a TTS failure sends a non-fatal `tts_error` event and the text still arrives.

### Choosing a voice mode

Two modes share one gateway:

1. **Live (`/voice/live`)**: a Talker (`gemini` in `"native"` or `"custom_tts"` `voice_mode`, `cascaded`, or a plugin from `MVG_TALKER_BACKENDS`) holds the conversation and hands slow work to the brain with `call_thinker`. Needs `MVG_TALKER_BACKEND`. Failure falls back to mode 2 (see Fallback above).
2. **Pipeline (`/voice/turn`)**: STT, brain, TTS per turn, with the delegate-first orchestrator prompt, spoken progress, spoken cancel and the context opener.

Live-mode keys: `MVG_TALKER_BACKEND`, `MVG_TALKER_BACKENDS`, `MVG_PERSONA_DIR`, `MVG_LIVE_MAX_SEC`, `MVG_LIVE_IDLE_SEC`, `MVG_LIVE_MAX_FRAME`, for the `gemini` backend `MVG_GEMINI_API_KEY`, `MVG_GEMINI_MODEL` (default `gemini-3.8-live`), `MVG_GEMINI_VOICE`, `MVG_GEMINI_INSTRUCTIONS`, `MVG_GEMINI_URL`, and for the `cascaded` backend `MVG_TALKER_URL`, `MVG_TALKER_MODEL`, `MVG_TALKER_KEY`, `MVG_TALKER_TIMEOUT`, `MVG_TALKER_INSTRUCTIONS`. All are in the env table and `.env.example`. Per-node choice: `talker_backend` and `voice_mode` (`"native"` | `"custom_tts"`) in the node's `nodes.json` entry.

Files: `talker_backend.py` (contract, `MockBackend`, `CascadedTextBackend`, loader), `live_harness.py` (`LiveSession`, `ProgressRelay`, and the four tools), `live_ws.py` (WebSocket bridge), `gemini_live.py` (Gemini Live backend), `tools/gemini_live_smoke.py` (live smoke test, including `--test-silent-progress`). Run `MVG_GEMINI_API_KEY=... python3 tools/gemini_live_smoke.py --test-silent-progress` to verify against the live Gemini endpoint.
