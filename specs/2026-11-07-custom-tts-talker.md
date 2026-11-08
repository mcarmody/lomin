# Custom TTS Talker Path (`voice_mode: "custom_tts"`)

Status: implemented (baseline `TEXT` modality + `LiveSession._speak_text` synthesis) and specified for incremental sentence streaming.
Context: issues #59, #60, #68; companion to the Talker + Thinker live voice architecture (`gemini_live.py`, `talker_backend.py`, `live_harness.py`, `gateway.py`).

## 1. Goal and non-goals

Goal: let a `/voice/live` Talker session use Lomin's own TTS chain (Chatterbox cloned voices, Kokoro, ElevenLabs, Piper) for spoken output while keeping low-latency live audio or text turn-taking and full `call_thinker` / `cancel_thinker` tool coordination.

Non-goals: replacing `voice_mode: "native"` when a provider's built-in voice is desired; duplicating TTS backend selection logic outside `tts_backends.py`; changing the client WebSocket wire contract on `GET /voice/live` (clients still send 16 kHz mono PCM16 binary frames and receive 24 kHz mono PCM16 binary frames plus JSON event frames).

## 2. Two `custom_tts` Talker topologies

A node selects custom TTS on `/voice/live` by setting `"voice_mode": "custom_tts"` in its `nodes.json` profile (validated by `_handle_voice_live` in `gateway.py`; values other than `"native"` or `"custom_tts"` return HTTP `400`).

### Path A: Native Audio-In to Text-Out (`GeminiLiveBackend` with `TEXT` modality)

1. **Setup handshake (`gemini_live.py`):**
   - When `config.get("voice_mode") == "custom_tts"`, `GeminiLiveBackend._handshake()` sets:
     ```json
     "generationConfig": {"responseModalities": ["TEXT"]}
     ```
     and omits `speechConfig` (which is only valid for `"AUDIO"` modality).
   - `tools` (`TOOL_DECLARATIONS`), `contextWindowCompression` (`{"slidingWindow": {}}`), `sessionResumption`, and `systemInstruction` are sent identically to `"native"` mode.
2. **Input path:**
   - Client 16 kHz PCM16 audio frames stream directly into `backend.send_audio(pcm16_16k)` (`realtimeInput.audio`), preserving Gemini Live's native audio understanding and server-side VAD/barge-in (`interrupted` events).
3. **Output path:**
   - The model emits `serverContent.modelTurn.parts[].text` deltas, which `server_message_events()` translates into `{"type": "text", "text": ..., "spoken": False}` events.

### Path B: Cascaded VAD + STT into a Fast Text Model (`CascadedTextBackend`)

1. **Backend (`talker_backend.py`):**
   - `CascadedTextBackend` (`MVG_TALKER_BACKEND=cascaded`) operates in text-in / text-out mode (`capabilities = frozenset({"text_out", "tool_calls", "silent_context"})`).
   - Upstream audio is segmented by VAD and transcribed by Whisper STT (or fed as text turns) into `backend.send_text(transcript, silent=False)`.
   - A fast text model (`MVG_TALKER_URL` / `MVG_TALKER_MODEL` / `MVG_TALKER_KEY`, or an injected responder) generates plain-spoken text and OpenAI-style `tool_calls`.
2. **Markdown stripping:**
   - All text emitted by `CascadedTextBackend` is cleaned via `text_utils.strip_markdown()` and emitted as `{"type": "text", "text": cleaned, "spoken": False}`.

## 3. Streaming through `SentenceSegmenter` and `tts_backends.synthesize()`

When `LiveSession` (`live_harness.py`) receives `{"type": "text", "text": chunk, "spoken": False}` and has a `speak` callback (`lambda text: synthesize(text)[0]` wired by `gateway.py`):

1. **Client text relay:**
   - The `{"type": "text", "text": chunk, "spoken": False}` event is forwarded immediately to the WebSocket client as a JSON text frame so captions/UI update without waiting for synthesis.
2. **Sentence segmentation (`SentenceSegmenter`):**
   - For multi-token or streaming `text` deltas emitted before `turn_end`, chunks are accumulated through `sentence_segmenter.SentenceSegmenter(early_first=FIRST_CLAUSE_EARLY)` so synthesis starts on the first complete clause or sentence rather than waiting for `turn_end`, while single-shot complete utterances flush immediately through `_speak_text(text)`.
   - On `turn_end` (or `interrupted`), any trailing buffered text in the segmenter is flushed (or discarded on `interrupted`).
3. **TTS synthesis & PCM normalization (`_speak_text`):**
   - Synthesis runs in a background executor thread via `self._speak(sentence)` (`gateway.synthesize` -> `tts_backends.synthesize`), walking `MVG_TTS_CHAIN` (`desktop`, `piper`, `elevenlabs`, `openai`, `http`, `gemini`, `kokoro`).
   - Returned WAV bytes are normalized to 24 kHz mono 16-bit PCM via `audio_utils.normalize_wav(wav, target_rate=24000)` and unwrapped to raw PCM16 bytes via `tts_backends.extract_pcm_from_wav_or_raw(wav)`.
   - `LiveSession` sends `{"type": "audio", "pcm": pcm, "rate": 24000, "synthesized": True}` to the client, which `live_ws.py` transmits as a binary WebSocket frame.
4. **Non-fatal TTS errors:**
   - If every engine in `MVG_TTS_CHAIN` fails, `LiveSession` emits `{"type": "tts_error", "message": ...}` to the client and keeps the live session open; the text frame has already been delivered.

## 4. Tool-call coordination in text-out mode

Tool coordination in `custom_tts` / text-out mode is identical to native audio mode and is owned entirely by `LiveSession` (`live_harness.py`):

1. **Tool declarations:**
   - Both `GeminiLiveBackend` (`TOOL_DECLARATIONS`) and `CascadedTextBackend` (`CASCADED_TOOLS`) expose the same four tools: `call_thinker(request)`, `cancel_thinker()`, `answer_prompt(prompt_id, decision)`, and `set_progress_updates(enabled)`.
2. **Acknowledge-then-delegate turn sequence:**
   - In `custom_tts` mode, the Talker emits a short spoken acknowledgment (`{"type": "text", "text": "Checking on that now.", "spoken": False}`) followed by `{"type": "tool_call", "call_id": ..., "name": "call_thinker", "args": {"request": ...}}`.
   - `LiveSession._dispatch()` synthesizes the acknowledgment through `_speak_text` and dispatches `_handle_tool("call_thinker", args)`, returning `{"status": "started", "handoff_id": n}` via `backend.send_tool_result(call_id, result)`.
3. **Silent progress updates (`ProgressRelay`):**
   - While the Thinker (`brain_call_stream`) runs in `LiveSession._work()`, stage events pushed to `session.on_stage_event(ev)` are formatted as `[task progress] <detail>` and sent via `backend.send_text(line, silent=True)` (`turnComplete: false` on Gemini Live; appended without triggering generation on `CascadedTextBackend`).
4. **Thinker result delivery:**
   - When the Thinker finishes, `LiveSession._work()` flushes `relay.finish()` and calls `backend.send_text(f"[thinker result, handoff {job['id']}] {text}", silent=False)`.
   - The Talker summarizes the result in plain text (`{"type": "text", "spoken": False}`), which `LiveSession` synthesizes into 24 kHz PCM audio via the gateway TTS chain.
5. **Cancellation (`cancel_thinker`):**
   - If the user barges in or asks to stop, the Talker invokes `cancel_thinker()`. `LiveSession._cancel()` marks the active job cancelled, drains queued handoffs, and calls `gateway.cancel_active_turn(node_id)` -> plugin `brain_cancel(target, task_id)`.

## 5. Verification

- `tests/test_gemini_live.py::test_custom_tts_voice_mode_requests_text_modality` verifies `responseModalities: ["TEXT"]` and absence of `speechConfig`.
- `tests/test_live_ws.py::test_voice_mode_validation` verifies HTTP `400` on invalid `voice_mode` and propagation of `"native"` / `"custom_tts"` into backend `open(config)`.
- `tests/test_live_harness.py` verifies text-out synthesis via `speak`, `tts_error` resilience, and tool-call round-trips.
