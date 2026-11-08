#!/usr/bin/env python3
"""
Lomin Agent Voice Gateway. Gives agents a voice: STT, brain and TTS backends over
plain HTTP. Site-specific parts (the brain transport, the addressee classifier,
the fast-path intents, default env values) come from a plugin named by
MVG_PLUGIN; see README.md, "Plugin".

Binds 0.0.0.0:9098 by default. Python 3.13 stdlib only, with optional,
guarded dependencies: /ambient's classifier is the plugin's `classify` module
(without it /ambient engages on the agent's name alone, classifier
"name-only"), and /ambient's pre-STT speech gate needs onnxruntime + numpy
plus Silero's ONNX model (missing -> the gate fails open and the clip goes to
Whisper as before).

LAMMEN pipeline (specs/2026-10-02-lammen-voice-pipeline.md): POST /voice/turn
is the one endpoint every voice input should call. One request per utterance,
SSE out. Request: multipart `audio` (WAV) plus text fields, or a raw WAV body
with the same flags in the query string. Fields/query: node_id (required),
open_room, tts (stream|wav|none), session_id, audio_chunk. Stages, fixed
order: speech gate -> Whisper || voice ID (ID gets MVG_VOICE_ID_GRACE_SEC,
0.5 s, after the transcript) -> hallucination filter -> addressed gate (only
when open_room) -> brain (one framing line, frame_voice_prompt) -> sentence
TTS with gap-fill (vocal opener at ~1.2 s, topical or generic machine
effect at ~3.5 s, long vocal at ~8 s, machine ticks; see _maybe_gap_fill).
Events: turn, state, transcript, ignored, status,
sentence, audio_chunk (ear compat, opt-in), reply, done, error. Per-node
profiles and flags live in nodes.json (see resolve_profile); /stt, /ambient,
/ask and /ask/stream are legacy, kept one release, and each call logs
`legacy_endpoint` (+ `node`) in the request log.

claimed_speaker (a login identity, shown as `speaker: alice (logged in)`; it
never upgrades the voice ID verdict) is trusted ONLY from the request header
X-Voice-Claimed-Speaker together with the shared-secret header
X-Voice-Gateway-Key equal to env MVG_TRUSTED_KEY. If MVG_TRUSTED_KEY is unset
or empty the claim is ignored entirely, and it is never read from a body
field or the query string. The same rule applies on /ask and /ask/stream.

Voice ID verdict (always in the framing and in `transcript.voice_id`):
`verified <id> <score>` (>= MVG_SPEAKER_THRESHOLD) | `not verified (best
<score>)` | `unavailable` (no prints, embed down, or identify off). Legacy
callers (the hub) send `speaker` (set only on a match) + `speaker_score`;
that is read as the voice ID match.

Endpoints (wire formats fixed by the calling hub; read its client before
changing this):

    POST /stt                  multipart/form-data (part "audio", audio.wav)
                                or a raw WAV body -> {"text": "..."}
                                Serves a cached transcript (60s, by sha256
                                of the WAV bytes) if /ambient already
                                transcribed the same audio and engaged.
    POST /ambient               multipart/form-data (part "audio", text
                                field "node_id") or a raw WAV body with
                                ?node_id= -> {"engage", "transcript",
                                "score", "classifier", "speaker",
                                "speaker_score", "ms"}. Ambient-wake gate:
                                transcribes every speech segment and asks a
                                classifier (Jev, then Haiku on
                                error/timeout) whether it's addressed to
                                the agent. See "Ambient gate" below. Also runs
                                best-effort speaker ID in parallel with STT
                                (SPEC-011, identify_speaker()) — this is the
                                one path an engaged voice turn does NOT take
                                through the hub's own speaker-ID client,
                                since a non-engaging utterance never reaches
                                the hub at all.
    POST /ask                  JSON {prompt, session_id, effort, node_id,
                                speaker, speaker_score}
                                -> text/event-stream: status*, reply, done
    POST /v1/audio/speech      OpenAI-compatible JSON {model, input, voice,
                                response_format} -> ALWAYS audio/wav bytes
                                (the edge client plays with `aplay -`, which
                                cannot decode mp3 — response_format is
                                accepted but ignored for the actual bytes).
                                Header X-TTS-Engine:
                                    chatterbox|piper-desktop|piper
    POST /voice/turn           Requests carrying header X-Watch-Route (set by the
                                nginx /gotchi-api/ location, off-LAN) must send
                                "Authorization: Bearer <token>" matching a key in
                                WATCH_TOKENS_PATH, else 401; the mapped node id
                                overrides any node_id in the request. A token's
                                value is a node id, or a list of allowed node
                                ids (the request's node_id is used if listed,
                                else the list's first).
    POST /embed                 raw WAV body -> {"embedding": [...], "model": "..."}
                                Proxies straight through to the desktop's
                                speaker-embedding server (MVG_DESKTOP_EMBED_URL)
                                over the same ssh reverse tunnel desktop_tts.py
                                uses.
    POST /v1/systemone         JSON (Jev /v1/systemone wire protocol) -> JSON; straight proxy to the desktop Laya classifier (MVG_DESKTOP_LAYA_URL) over the ssh reverse tunnel.
    GET  /health                -> {"status": "ok", ...upstream booleans}

TTS fallback chain: a desktop-hosted TTS server reached over an ssh reverse
tunnel (MVG_DESKTOP_TTS_URL, default http://127.0.0.1:9097/tts) — Chatterbox
Turbo cloned onto the agent's own voice if it loaded, else that same desktop
server's in-process Piper fallback — then this host's own local Piper binary
(19-20s for one sentence under load — the last resort, not the normal path).

Per-node brain routing: node_id picks the profile in nodes.json (MVG_NODES_FILE,
default $MVG_HOME/nodes.json; target, open_room, identify, speech_gate,
gap_fill, reply_cap_words, audio_chunk); MVG_NODE_TARGETS (a JSON object, env)
overrides a node's target. If nodes.json is absent the old node-targets.json
({node: target}) is read instead. Any node_id not in the mapping falls back to
DEFAULT_TARGET (MVG_DEFAULT_TARGET, default "agent"). A node may list `targets`,
the agents a request may pick: a `/voice/turn` request's `target` field is used
only if listed, else the node's `target`. Only the leg that answers
changes — the reply still comes back out of the device that asked, because the
hub replies on that device's own stream.

Every request logs one JSON line to logs/lomin.jsonl
(endpoint, ms, outcome — no audio bytes, no keys). TTS fallbacks to Piper
additionally log one JSON line to logs/voice_fallback.log.

Ambient gate (/ambient): every speech segment picked up by the edge client
is sent here instead of only a wake-word-triggered one. A transcript that
doesn't engage is NEVER written to disk anywhere — not this endpoint's log
line (metadata only: node, ms, score, engaged, classifier, character
count), not an error message, not the rolling context (in-memory only, per
node_id: last 6 transcripts within 3 minutes, plus the timestamp of the
agent's last /ask reply on that node so an unnamed follow-up within
MVG_AMBIENT_FOLLOWUP_WINDOW_SEC of a reply can still engage). Classifier
chain (classify_addressed_detail; the plugin's classify() = MAHANAXAR):
A wake phrase that opens the utterance ("hey <name>", WAKE_PHRASE_RE)
bypasses everything: engage with score 1.0, classifier "wake-phrase", no
classifier called. Otherwise a deterministic pre-gate runs before any
classifier: no agent-name variant
(AGENT_NAME_VARIANTS, word-boundary) and no open conversation on the node ->
not addressed, classifier "pregate", score 0.0, no classifier called. Laya
may engage alone only when the name was found; an unnamed follow-up needs Jev.
Laya (local classifier on the desktop via the TTS tunnel,
MVG_AMBIENT_LAYA_TIMEOUT default 0.5 s) is a fast path for confident yeses
only: P(yes) >= MVG_AMBIENT_LAYA_CONFIDENT (0.7) engages with classifier
"laya" and Jev is not called; below that, or on a Laya error/timeout, Jev via
the Vercel AI Gateway decides (single attempt, no retry; classifier
"laya+jev" if Laya answered, else "jev"); Haiku (`claude -p`, ~9 s on slow
hardware, hence MVG_AMBIENT_HAIKU_TIMEOUT, default 12 s) only on a Jev error or
timeout, never on a merely-low Jev score. The edge client's ambient timeout
must exceed Jev + Haiku. classify() writes the usage line
(logs/classify-usage.jsonl). If every layer fails, engage is always False.

Ambient gate, stage 0 (speech gate): before Whisper, Silero VAD
(openWakeWord's bundled silero_vad.onnx, via onnxruntime; 512-sample
windows, LSTM state reset per clip) measures voiced ms (windows with
p >= MVG_SPEECH_THRESHOLD). Under MVG_MIN_SPEECH_MS the clip is answered
engage=false, classifier "no-speech", with no STT and no classify call. Log
line carries voiced_ms, speech_gate (passed/rejected/unavailable) and
vad_ms. Fails open ("unavailable") on any VAD problem. Server-side because
other edge clients may still use the energy VAD.

Config is read at runtime from its home paths, never copied into this
worktree. Default state root: $MVG_HOME (default ~/.mvg).
    $MVG_HOME/piper-tts.json        -> piper.binary / voices_dir (optional;
                                        falls back to the same defaults
                                        skills/piper-tts/scripts/tts.py uses)

The brain for /ask and /voice/turn is the plugin's brain_call, or plain HTTP
(MVG_BRAIN=http, see http_brain_call). See README.md.
"""

from __future__ import annotations

import base64
import collections
import hashlib
import hmac
import importlib
import inspect
import json
import math
import os
import random
import re
import struct
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import wave
from concurrent.futures import Future, ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlsplit
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sentence_segmenter import SentenceSegmenter, segment_full_text  # noqa: E402
# Moved out of this file (ERED LOMIN P1b). Re-exported here so gateway.<name>
# keeps resolving for callers and tests.
from text_utils import strip_markdown, stable_markdown_prefix, unspoken_delta  # noqa: E402,F401
import tts_backends  # noqa: E402  (ERED LOMIN P1b: swappable TTS chain)
from multipart import parse_multipart_parts, parse_multipart_audio  # noqa: E402,F401
from audio_utils import (  # noqa: E402,F401
    AUDIO_TARGET_CHANNELS, AUDIO_TARGET_RATE, AUDIO_TARGET_SAMPWIDTH,
    normalize_wav, pcm_to_wav,
)
from vision_adpcm import (  # noqa: E402,F401
    ADPCM_MAGIC,
    MAX_IMAGE_B64_BYTES,
    MAX_IMAGE_BYTES,
    decode_adpcm_to_wav,
    extract_vision_payloads,
    is_valid_image_bytes,
)
from ha_fastpath import try_ha_conversation  # noqa: E402,F401
from side_channel import (  # noqa: E402,F401
    emit_turn_transcript,
    forward_artifacts_to_webhook,
    reply_expects_reply,
)
from pending_and_auth import (  # noqa: E402,F401
    BearerTokenVerifier,
    PendingVoiceAnswerStore,
    get_device_token_verifier,
    get_pending_store,
    has_defer_marker,
    is_defer_enabled,
    strip_defer_marker,
)

# --- Plugin ------------------------------------------------------------
# MVG_PLUGIN names an importable module that supplies household parts.
# Every hook is optional:
#   apply_defaults() -> None                     fill env defaults (setdefault only)
#   brain_call(target, prompt, timeout) -> str   the agent answers one framed prompt
#   brain_call_stream(target, prompt, timeout, on_partial, stages_out[, audio_clips=]) -> str
#                                                 audio_clips: list of 16 kHz WAV bytes, passed only
#                                                 if the signature accepts it and MVG_SPLICE_RAW_AUDIO=1
#   classify                                      module with classify_choice(...) (addressee classifier)
#   try_intent(text) -> dict | None               deterministic fast path
PLUGIN_NAME = os.environ.get("MVG_PLUGIN", "").strip()
_plugin = None
if PLUGIN_NAME:
    try:
        _plugin = importlib.import_module(PLUGIN_NAME)
    except Exception as exc:  # a named plugin that fails is loud, never silent
        print(f"lomin: plugin {PLUGIN_NAME!r} failed to load ({exc}); running without it",
              file=sys.stderr)
if _plugin is not None and hasattr(_plugin, "apply_defaults"):
    _plugin.apply_defaults()
_PLUGIN_MOD = _plugin
_plugin_brain_call = getattr(_plugin, "brain_call", None)
_plugin_brain_call_stream = getattr(_plugin, "brain_call_stream", None)
_plugin_brain_cancel = getattr(_plugin, "brain_cancel", None)
_plugin_talker_instructions = getattr(_plugin, "talker_instructions", None)
# Kept as `_jev_classifier`: test_ambient.py monkeypatches its attributes.
_jev_classifier = getattr(_plugin, "classify", None)
_try_intent = getattr(_plugin, "try_intent", None) or (lambda text: None)
_NO_INTENT = {"hit": False, "intent": None, "match_ms": 0.0, "handler_ms": None, "reason": "miss"}


def _is_ha_authorized(
    auth_header: str | None = None,
    node_id: str = "",
    speaker: str = "",
    speaker_score: Any = None,
    voice_id: dict | None = None,
    claimed_speaker: str = "",
) -> bool:
    """Verify whether caller is authorized to execute Home Assistant control."""
    ha_intent_token = os.environ.get("MVG_HA_INTENT_TOKEN", "").strip()
    device_verifier = get_device_token_verifier()
    tokens_configured = bool(ha_intent_token) or device_verifier.is_enabled
    if not tokens_configured:
        return True

    # 1. Check verified speaker identity
    if claimed_speaker:
        return True
    if isinstance(voice_id, dict) and voice_id.get("status") == "verified":
        return True

    # 2. Check bearer token in Authorization header
    if auth_header:
        token = BearerTokenVerifier._extract_token(auth_header)
        if token:
            if ha_intent_token and hmac.compare_digest(token.encode("utf-8"), ha_intent_token.encode("utf-8")):
                return True
            if device_verifier.is_enabled and device_verifier.verify(auth_header, node_id):
                return True

    return False


def _intent(
    text: str,
    has_images: bool = False,
    *,
    ha_authorized: bool = True,
) -> dict:
    """The fast path's result in the shape the call sites read. A plugin that
    answers None, or raises, or returns {'hit': False}, is a miss: Home Assistant
    fast-path is consulted before falling through to the brain."""
    try:
        res = _try_intent(text)
    except Exception as exc:  # a broken plugin must never lose a turn
        return {**_NO_INTENT, "reason": f"intent_error:{type(exc).__name__}"}

    plugin_miss = isinstance(res, dict) and res.get("hit") is False
    if res is not None and not plugin_miss:
        return res

    ha_enabled = bool(HA_ENABLED) or (
        os.environ.get("MVG_HA_ENABLED", "0") == "1"
        or os.environ.get("MVG_HA_FASTPATH", "0") == "1"
    )
    if ha_enabled:
        if not ha_authorized:
            return {**_NO_INTENT, "reason": "ha_unauthorized"}
        t0 = time.time()
        try:
            matched, speech, meta = try_ha_conversation(text, has_images=has_images)
            elapsed_ms = round((time.time() - t0) * 1000, 2)
            if matched:
                return {
                    "hit": True,
                    "intent": "ha_conversation",
                    "reply": speech,
                    "match_ms": elapsed_ms,
                    "handler_ms": elapsed_ms,
                    "reason": meta.get("reason", "action_done"),
                    "meta": meta,
                }
            return {
                **_NO_INTENT,
                "match_ms": elapsed_ms,
                "reason": meta.get("reason", "miss"),
            }
        except Exception as exc:
            elapsed_ms = round((time.time() - t0) * 1000, 2)
            return {
                **_NO_INTENT,
                "match_ms": elapsed_ms,
                "reason": f"ha_error:{type(exc).__name__}",
            }

    if res is not None:
        return res
    return dict(_NO_INTENT)

# --- Paths -------------------------------------------------------------

# State root (secrets, piper, voice capture). Default ~/.mvg.
MVG_HOME = Path(os.environ.get("MVG_HOME") or (Path.home() / ".mvg"))
LOG_DIR = Path(os.environ.get("MVG_LOG_DIR") or (Path(__file__).resolve().parent / "logs"))

# --- Config --------------------------------------------------------------

PORT = int(os.environ.get("MVG_PORT", "9098"))
# Largest request body the gateway will read (bytes). 20 MiB is about 10
# minutes of 16 kHz mono 16-bit WAV, far above any real turn.
MAX_BODY_BYTES = int(os.environ.get("MVG_MAX_BODY_BYTES", str(20 * 1024 * 1024)))
BIND = os.environ.get("MVG_BIND", "0.0.0.0")
# The agent's name, as the prompts and the spoken apology say it.
AGENT_NAME = os.environ.get("MVG_AGENT_NAME", "the assistant")
# The speaker label the agent's own replies carry in the addressee context, and its Piper voice_mapping key.
AGENT_LABEL = os.environ.get("MVG_AGENT_LABEL", "agent")
# The classify_choice task key for the addressee question (a site's classifier may key prompts and logs on it).
CLASSIFY_KEY = os.environ.get("MVG_CLASSIFY_KEY", "lomin.ambient")
# The request log file name under LOG_DIR.
LOG_NAME = os.environ.get("MVG_LOG_NAME", "lomin.jsonl")

# Multimodal camera & BLE IMA ADPCM ingest flags (Chunk 1 parity)
ENABLE_VISION = (
    os.environ.get("MVG_ENABLE_VISION", "0") == "1"
    or os.environ.get("MVG_VISION_ENABLED", "0") == "1"
    or os.environ.get("MVG_VISION_ADPCM", "0") == "1"
)
ENABLE_ADPCM = (
    os.environ.get("MVG_ENABLE_ADPCM", "0") == "1"
    or os.environ.get("MVG_ADPCM_ENABLED", "0") == "1"
    or os.environ.get("MVG_VISION_ADPCM", "0") == "1"
)


def is_vision_enabled() -> bool:
    return bool(ENABLE_VISION) or (
        os.environ.get("MVG_ENABLE_VISION", "0") == "1"
        or os.environ.get("MVG_VISION_ENABLED", "0") == "1"
        or os.environ.get("MVG_VISION_ADPCM", "0") == "1"
    )


def is_adpcm_enabled() -> bool:
    return bool(ENABLE_ADPCM) or (
        os.environ.get("MVG_ENABLE_ADPCM", "0") == "1"
        or os.environ.get("MVG_ADPCM_ENABLED", "0") == "1"
        or os.environ.get("MVG_VISION_ADPCM", "0") == "1"
    )

# Streaming SSE HTTP brain & direct-audio flags (Chunk 2 parity)
HTTP_BRAIN_STREAM = os.environ.get("MVG_HTTP_STREAM", "0") == "1"
BRAIN_DIRECT_AUDIO = os.environ.get("MVG_BRAIN_DIRECT_AUDIO", "0") == "1"
MAX_SSE_CHARS = int(os.environ.get("MVG_MAX_SSE_CHARS", str(256 * 1024)))
MAX_AUDIO_B64_BYTES = int(os.environ.get("MVG_MAX_AUDIO_B64_BYTES", str(10 * 1024 * 1024)))
MAX_AUDIO_BYTES = int(os.environ.get("MVG_MAX_AUDIO_BYTES", str(10 * 1024 * 1024)))


def is_http_stream_enabled() -> bool:
    return bool(HTTP_BRAIN_STREAM) or (os.environ.get("MVG_HTTP_STREAM", "0") == "1")


def validate_multimodal_payload(req: dict) -> tuple[int | None, str | None]:
    """Validate type, base64 validity, and max size of multimodal fields.

    Checks image_b64, burst_images_b64, audio_b64, as well as raw image
    and burst_images bytes if present in req.
    Returns (status_code, error_message) on failure, or (None, None) if valid.
    """
    if not isinstance(req, dict):
        return 400, "request body must be a JSON object"

    # Opt-in only: with the vision/ADPCM flags and MVG_BRAIN_DIRECT_AUDIO unset these
    # fields are ignored exactly as on main.
    vision_on = is_vision_enabled()
    audio_on = bool(BRAIN_DIRECT_AUDIO) or (os.environ.get("MVG_BRAIN_DIRECT_AUDIO", "0") == "1")
    if not vision_on and not audio_on:
        return None, None

    # Raw image inputs: image or image_bytes
    raw_img = (req.get("image") if "image" in req else req.get("image_bytes")) if vision_on else None
    if raw_img is not None:
        if isinstance(raw_img, (bytes, bytearray)):
            if len(raw_img) > MAX_IMAGE_BYTES:
                return 413, f"image payload exceeds maximum allowed size ({MAX_IMAGE_BYTES} bytes)"
            if not is_valid_image_bytes(raw_img):
                return 400, "image is not a valid image format (must be JPEG, PNG, or WEBP)"
        elif isinstance(raw_img, str):
            if len(raw_img) > MAX_IMAGE_B64_BYTES:
                return 413, f"image payload exceeds maximum allowed size ({MAX_IMAGE_B64_BYTES} bytes)"
            try:
                decoded = base64.b64decode(raw_img, validate=True)
            except Exception:
                return 400, "image is not valid base64"
            if len(decoded) > MAX_IMAGE_BYTES:
                return 413, f"image decoded data exceeds maximum allowed size ({MAX_IMAGE_BYTES} bytes)"
            if not is_valid_image_bytes(decoded):
                return 400, "image is not a valid image format (must be JPEG, PNG, or WEBP)"
        else:
            return 400, "image must be bytes or string"

    # Raw burst_images list
    if vision_on and "burst_images" in req and req["burst_images"] is not None:
        val = req["burst_images"]
        if not isinstance(val, list):
            return 400, "burst_images must be a list"
        for idx, item in enumerate(val):
            if isinstance(item, (bytes, bytearray)):
                if len(item) > MAX_IMAGE_BYTES:
                    return 413, f"burst_images[{idx}] payload exceeds maximum allowed size ({MAX_IMAGE_BYTES} bytes)"
                if not is_valid_image_bytes(item):
                    return 400, f"burst_images[{idx}] is not a valid image format (must be JPEG, PNG, or WEBP)"
            elif isinstance(item, str):
                if len(item) > MAX_IMAGE_B64_BYTES:
                    return 413, f"burst_images[{idx}] payload exceeds maximum allowed size ({MAX_IMAGE_B64_BYTES} bytes)"
                try:
                    decoded = base64.b64decode(item, validate=True)
                except Exception:
                    return 400, f"burst_images[{idx}] is not valid base64"
                if len(decoded) > MAX_IMAGE_BYTES:
                    return 413, f"burst_images[{idx}] decoded data exceeds maximum allowed size ({MAX_IMAGE_BYTES} bytes)"
                if not is_valid_image_bytes(decoded):
                    return 400, f"burst_images[{idx}] is not a valid image format (must be JPEG, PNG, or WEBP)"
            else:
                return 400, f"burst_images[{idx}] must be bytes or string"

    # 1. image_b64
    if vision_on and "image_b64" in req and req["image_b64"] is not None:
        val = req["image_b64"]
        if not isinstance(val, str):
            return 400, "image_b64 must be a string"
        if len(val) > MAX_IMAGE_B64_BYTES:
            return 413, f"image_b64 payload exceeds maximum allowed size ({MAX_IMAGE_B64_BYTES} bytes)"
        try:
            decoded = base64.b64decode(val, validate=True)
        except Exception:
            return 400, "image_b64 is not valid base64"
        if len(decoded) > MAX_IMAGE_BYTES:
            return 413, f"image_b64 decoded data exceeds maximum allowed size ({MAX_IMAGE_BYTES} bytes)"
        if not is_valid_image_bytes(decoded):
            return 400, "image_b64 is not a valid image format (must be JPEG, PNG, or WEBP)"

    # 2. burst_images_b64
    if vision_on and "burst_images_b64" in req and req["burst_images_b64"] is not None:
        val = req["burst_images_b64"]
        if not isinstance(val, list):
            return 400, "burst_images_b64 must be a list"
        for idx, item in enumerate(val):
            if not isinstance(item, str):
                return 400, f"burst_images_b64[{idx}] must be a string"
            if len(item) > MAX_IMAGE_B64_BYTES:
                return 413, f"burst_images_b64[{idx}] payload exceeds maximum allowed size ({MAX_IMAGE_B64_BYTES} bytes)"
            try:
                decoded = base64.b64decode(item, validate=True)
            except Exception:
                return 400, f"burst_images_b64[{idx}] is not valid base64"
            if len(decoded) > MAX_IMAGE_BYTES:
                return 413, f"burst_images_b64[{idx}] decoded data exceeds maximum allowed size ({MAX_IMAGE_BYTES} bytes)"
            if not is_valid_image_bytes(decoded):
                return 400, f"burst_images_b64[{idx}] is not a valid image format (must be JPEG, PNG, or WEBP)"

    # 3. audio_b64
    if audio_on and "audio_b64" in req and req["audio_b64"] is not None:
        val = req["audio_b64"]
        if not isinstance(val, str):
            return 400, "audio_b64 must be a string"
        if len(val) > MAX_AUDIO_B64_BYTES:
            return 413, f"audio_b64 payload exceeds maximum allowed size ({MAX_AUDIO_B64_BYTES} bytes)"
        try:
            decoded = base64.b64decode(val, validate=True)
        except Exception:
            return 400, "audio_b64 is not valid base64"
        if len(decoded) > MAX_AUDIO_BYTES:
            return 413, f"audio_b64 decoded data exceeds maximum allowed size ({MAX_AUDIO_BYTES} bytes)"

    return None, None

# Home Assistant conversation intent fast-path flags (Chunk 3 parity)
HA_ENABLED = (
    os.environ.get("MVG_HA_ENABLED", "0") == "1"
    or os.environ.get("MVG_HA_FASTPATH", "0") == "1"
)
HA_URL = os.environ.get("MVG_HA_URL", "")
HA_TOKEN = os.environ.get("MVG_HA_TOKEN", "")
HA_INTENT_TOKEN = os.environ.get("MVG_HA_INTENT_TOKEN", "")

WHISPER_URL = os.environ.get("WHISPER_URL", "http://127.0.0.1:9099/transcribe")
WHISPER_HEALTH_URL = os.environ.get(
    "WHISPER_HEALTH_URL",
    WHISPER_URL.rsplit("/", 1)[0] + "/health" if "/" in WHISPER_URL else WHISPER_URL,
)

DEFAULT_TARGET = (os.environ.get("MVG_DEFAULT_TARGET") or os.environ.get("MVG_HIVE_TARGET")
                  or "agent")
HIVE_TIMEOUT = int(os.environ.get("MVG_HIVE_TIMEOUT", "45"))

NODE_TARGETS_PATH = Path(
    os.environ.get("MVG_NODE_TARGETS_FILE")
    or (MVG_HOME / "node-targets.json")
)

NODES_PATH = Path(
    os.environ.get("MVG_NODES_FILE")
    or (MVG_HOME / "nodes.json")
)
# Gotchi Apple Watch bearer tokens: JSON {"<token>": "<node id>"}, read per call.
WATCH_TOKENS_PATH = Path(
    os.environ.get("MVG_WATCH_TOKENS_FILE", str(MVG_HOME / "secrets" / "gotchi-watch-tokens.json"))
)
# /voice/turn: how long voice ID may keep running after the transcript is in.
VOICE_ID_GRACE_SEC = float(os.environ.get("MVG_VOICE_ID_GRACE_SEC", "0.5"))

PIPER_CONFIG_PATH = Path(
    os.environ.get("MVG_PIPER_CONFIG") or (MVG_HOME / "piper-tts.json")
)
PIPER_BINARY_DEFAULT = str(MVG_HOME / "venvs" / "piper" / "bin" / "piper")
PIPER_VOICES_DIR_DEFAULT = str(MVG_HOME / "piper-voices")
PIPER_VOICE_MODEL_DEFAULT = "en_US-lessac-medium"
PIPER_TIMEOUT = float(os.environ.get("MVG_PIPER_TIMEOUT", "45"))

DESKTOP_TTS_URL = os.environ.get("MVG_DESKTOP_TTS_URL", "http://127.0.0.1:9097/tts")
DESKTOP_TTS_TIMEOUT = float(os.environ.get("MVG_DESKTOP_TTS_TIMEOUT", "25"))
# Sub-sentence PCM streaming on /ask/stream (desktop_tts.py POST /tts/stream).
# MVG_TTS_STREAM=0 restores whole-sentence WAV events. The stream URL is
# derived from MVG_DESKTOP_TTS_URL (".../tts" -> ".../tts/stream") unless
# MVG_DESKTOP_TTS_STREAM_URL is set.
TTS_STREAM = os.environ.get("MVG_TTS_STREAM", "1").strip().lower() not in ("0", "false", "off", "no", "")
DESKTOP_TTS_STREAM_URL = os.environ.get("MVG_DESKTOP_TTS_STREAM_URL", "")
# Coalesce upstream reads into events of at least this many bytes (~100 ms of
# 24 kHz s16le mono; the hub buffers to the same floor before forwarding, so
# smaller events would add nothing but overhead).
TTS_STREAM_MIN_CHUNK_BYTES = int(os.environ.get("MVG_TTS_STREAM_MIN_CHUNK_BYTES", "4800"))

# Text-only sentences (MVG_TEXT_SENTENCES=1, default off): /ask/stream emits
# `sentence` events with text and no audio, and the hub speaks each one itself
# through its configured tts_url (this gateway's POST /v1/audio/speech, which
# then streams WAV-header-then-PCM). The hub latches a turn to brain-audio or
# remote-tts on its first sentence event and rejects a mix, so in this mode
# there is no pre-rendered gap-fill (a clip would be audio) and gap-fill is
# disabled; /v1/audio/speech streams unless the request says "stream": false.
TEXT_SENTENCES = os.environ.get("MVG_TEXT_SENTENCES", "0").strip().lower() in ("1", "true", "on", "yes")
# Flush the first clause of a reply early (segmenter early_first) instead of
# waiting for a whitespace after the first terminator. Both modes.
FIRST_CLAUSE_EARLY = os.environ.get("MVG_FIRST_CLAUSE_EARLY", "1").strip().lower() not in ("0", "false", "off", "no", "")

DESKTOP_EMBED_URL = os.environ.get("MVG_DESKTOP_EMBED_URL", "http://127.0.0.1:9096/embed")
DESKTOP_EMBED_TIMEOUT = float(os.environ.get("MVG_DESKTOP_EMBED_TIMEOUT", "4"))
DESKTOP_LAYA_URL = os.environ.get("MVG_DESKTOP_LAYA_URL", "http://127.0.0.1:9121/v1/systemone")
DESKTOP_LAYA_TIMEOUT = float(os.environ.get("MVG_DESKTOP_LAYA_TIMEOUT", "20"))

# --- Speaker identification for /ambient (SPEC-011) ------------------------
# Mirrors the hub's speaker.go FingerprintStore /
# MatchSpeaker so /ambient (which never reaches the Go hub for a
# non-engaging utterance) gets the same speaker label an engaged /ask turn
# would, instead of leaving every ambient-only turn anonymous.
SPEAKERS_PATH = Path(
    os.environ.get("MVG_SPEAKERS_PATH") or str(MVG_HOME / "speakers.json")
)
SPEAKER_THRESHOLD = float(os.environ.get("MVG_SPEAKER_THRESHOLD", "0.70"))
# Speaker ID runs in a background thread started before STT; this is the
# most extra time /ambient will wait for it AFTER STT finishes, so a slow
# or dead embed hop never meaningfully delays the ambient response.
AMBIENT_SPEAKER_WAIT_SEC = float(os.environ.get("MVG_AMBIENT_SPEAKER_WAIT_SEC", "0.5"))

STATUS_INTERVAL_SEC = float(os.environ.get("MVG_STATUS_INTERVAL_SEC", "5"))

# --- Ambient gate config ---------------------------------------------------

AMBIENT_THRESHOLD = float(os.environ.get("MVG_AMBIENT_THRESHOLD", "0.6"))
# Laya (local classifier on the desktop, primary): if it takes longer than
# this, or errors, Jev answers instead.
AMBIENT_LAYA_TIMEOUT = float(os.environ.get("MVG_AMBIENT_LAYA_TIMEOUT", "0.5"))
# Laya P(yes) at or above this engages on its own (Jev not called); below it,
# Jev gives the second opinion.
AMBIENT_LAYA_CONFIDENT = float(os.environ.get("MVG_AMBIENT_LAYA_CONFIDENT", "0.7"))
# The option key the addressee classifier answers with. A plugin may rename it
# (a site whose classifier models were built on its own key). A value that is
# not a lowercase identifier is refused on stderr and the default is used.
AMBIENT_ADDRESSED = getattr(_plugin, "AMBIENT_ADDRESSED", None) or "addressed_to_agent"
if not (isinstance(AMBIENT_ADDRESSED, str)
        and re.fullmatch(r"[a-z][a-z0-9_]{0,63}", AMBIENT_ADDRESSED)):
    print(f"lomin: plugin AMBIENT_ADDRESSED '{AMBIENT_ADDRESSED}' is invalid; "
          "using addressed_to_agent", file=sys.stderr)
    AMBIENT_ADDRESSED = "addressed_to_agent"
AMBIENT_LAYA_QUESTION = (
    f"Who is the LAST line addressed to? Choose {AMBIENT_ADDRESSED} only if "
    f"the speaker names {AGENT_NAME}, the AI assistant, or is directly "
    f"following up on {AGENT_NAME}'s own reply just above it. People talking to "
    "each other (even questions, even \"you\") are "
    "conversation_between_people."
)
# A plugin may supply its own wording (a site whose classifier was tuned on it).
AMBIENT_LAYA_QUESTION = getattr(_plugin, "AMBIENT_LAYA_QUESTION", None) or AMBIENT_LAYA_QUESTION
# MÁHANAXAR stage 4: the addressed-to-agent yes/no is a choice over these.
# Engagement is decided on P(addressed) against AMBIENT_THRESHOLD /
# AMBIENT_LAYA_CONFIDENT, exactly the old cuts.
AMBIENT_OPTIONS = {
    AMBIENT_ADDRESSED: f"the LAST utterance is a request or question for {AGENT_NAME} to act on or answer",
    "conversation_between_people": f"people in the room are talking to each other, not to {AGENT_NAME}",
    "background_media": "TV, radio, music, or a video call playing in the room",
    "unclear": f"it cannot be told whether the LAST utterance is meant for {AGENT_NAME}",
}
AMBIENT_JEV_TIMEOUT = float(os.environ.get("MVG_AMBIENT_JEV_TIMEOUT", "2.5"))
# Haiku is the fallback layer (only runs when Jev errors/times out); `claude -p
# --model claude-haiku-4-5` measured ~8.8 s on slow hardware.
AMBIENT_HAIKU_TIMEOUT = float(os.environ.get("MVG_AMBIENT_HAIKU_TIMEOUT", "12"))

# Rolling per-node context: how many transcripts to remember and for how long.
AMBIENT_CONTEXT_MAX_ENTRIES = int(os.environ.get("MVG_AMBIENT_CONTEXT_MAX_ENTRIES", "6"))
AMBIENT_CONTEXT_WINDOW_SEC = float(os.environ.get("MVG_AMBIENT_CONTEXT_WINDOW_SEC", "180"))
# How long after the agent's last reply on a node a conversation is considered
# "open" (an unnamed follow-up can still engage). Counted from when the node
# finishes PLAYING the reply (see extend_reply_playback), not from text-ready.
AMBIENT_FOLLOWUP_WINDOW_SEC = float(os.environ.get("MVG_AMBIENT_FOLLOWUP_WINDOW_SEC", "30"))

# STT cache (used by /stt when /ambient already transcribed the same bytes).
STT_CACHE_TTL_SEC = float(os.environ.get("MVG_STT_CACHE_TTL_SEC", "60"))

# Whisper's own well-known silence hallucinations — never worth classifying.
AMBIENT_HALLUCINATIONS = {
    "thank you.", "thank you", "thanks for watching", "thanks for watching!",
    "you", "bye.", "bye", "okay.", "okay",
}
# Fewer than this many alphanumeric characters isn't a real utterance.
AMBIENT_MIN_ALNUM_CHARS = 2

# Every spelling the agent's name can come out of Whisper as. One list (env
# MVG_AGENT_NAMES, comma-separated), shared by the deterministic pre-gate
# (_name_mentioned) and the classifier instruction text below. Empty by
# default: with no names, the name pre-gate never matches.
AGENT_NAME_VARIANTS = tuple(
    n.strip() for n in os.environ.get("MVG_AGENT_NAMES", "").split(",") if n.strip()
)
# Names that open an utterance with a greeting ("hey <name>") are an explicit
# address. Env MVG_WAKE_NAMES; defaults to MVG_AGENT_NAMES. Names that are
# only mis-transcriptions (not wake words) belong in MVG_AGENT_NAMES alone.
_WAKE_NAMES = tuple(
    n.strip() for n in os.environ.get("MVG_WAKE_NAMES", ",".join(AGENT_NAME_VARIANTS)).split(",")
    if n.strip()
)
_AGENT_NAME_RE = (
    re.compile(
        r"\b(?:" + "|".join(re.escape(v) for v in AGENT_NAME_VARIANTS) + r")\b",
        re.IGNORECASE,
    )
    if AGENT_NAME_VARIANTS else None
)
# A greeting plus the name at the very start of the utterance ("hey <name> what
# time is it", "Hey, <name>.") is an explicit address: it engages without any
# classifier. Anchored at the start, so "I said hey <name> earlier" does not
# qualify. With no wake names the pattern never matches.
WAKE_PHRASE_RE = (
    re.compile(
        r"^[\W_]*(?:hey|hi|hello|okay|ok|yo)[,.!]?\s+(?:"
        + "|".join(re.escape(n.lower()) for n in _WAKE_NAMES) + r")\b",
        re.IGNORECASE,
    )
    if _WAKE_NAMES else re.compile(r"(?!)")
)


def _name_mentioned(transcript: str) -> bool:
    """True if the transcript contains any agent-name variant as a whole word
    (case-insensitive; "<name>'s" counts, "<name>cow" does not). Always False
    when no names are configured."""
    if _AGENT_NAME_RE is None:
        return False
    return bool(_AGENT_NAME_RE.search(transcript or ""))


def _variants_phrase() -> str:
    names = AGENT_NAME_VARIANTS
    if not names:
        return ""
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + ", or " + names[-1]


VOICE_ADDRESSED_INSTRUCTIONS = (
    "This is a transcript of spoken audio picked up by a microphone in a "
    "room, produced by automatic speech recognition (Whisper), which is "
    "imperfect: the AI assistant, called " + AGENT_NAME + ", "
    + ("may be transcribed as " + _variants_phrase() + ". " if _variants_phrase() else "")
    + "Decide "
    f"whether the LAST utterance is addressed TO {AGENT_NAME} — a request or "
    "question for the assistant to act on or answer — either by naming "
    f"{AGENT_NAME} anywhere in the sentence (e.g. \"what's the weather, {AGENT_NAME}\") or "
    f"as a direct follow-up continuing an open conversation with {AGENT_NAME} (see "
    "the conversation-state note below). Mentioning "
    f"{AGENT_NAME} to another person "
    f"(\"ask {AGENT_NAME} later\", \"{AGENT_NAME} said the meeting's at 5\") is NOT "
    f"addressed to {AGENT_NAME}. Audio from a TV, radio, music, or a video call "
    f"playing in the room is NOT addressed to {AGENT_NAME}, even if it happens to "
    "say the name. Each line below is prefixed with a speaker label (a "
    "name, or \"unknown\" when speaker identification didn't match "
    "anyone) followed by a colon, e.g. \"alice: what's the weather\" — that "
    "label identifies who SAID the line, not who it is addressed to."
)
VOICE_ADDRESSED_INSTRUCTIONS = getattr(_plugin, "VOICE_ADDRESSED_INSTRUCTIONS", None) or VOICE_ADDRESSED_INSTRUCTIONS


# --- Ambient rolling context (in-memory only; never written to disk) ------

_ambient_lock = threading.Lock()
# node_id -> deque[(ts, speaker_label, transcript)]
_ambient_context: dict[str, "collections.deque"] = {}
# node_id -> (ts, reply text) of the agent's last /ask reply on that node
_ambient_last_reply: dict[str, tuple] = {}
# sha256(wav bytes) -> (ts, transcript)
_stt_cache: dict[str, tuple] = {}
# sha256(wav bytes) -> (ts, speaker, score) — see identify_speaker()
_speaker_cache: dict[str, tuple] = {}


def _ambient_reset_state() -> None:
    """Test-only helper — module globals above are shared mutable state
    across requests/tests."""
    with _ambient_lock:
        _ambient_context.clear()
        _ambient_last_reply.clear()
        _stt_cache.clear()
        _speaker_cache.clear()
    with _fp_lock:
        _fp_cache["path"] = None
        _fp_cache["mtime"] = None
        _fp_cache["size"] = None
        _fp_cache["data"] = None


def _clip_reply(reply: str, limit: int = 400) -> str:
    reply = " ".join((reply or "").split())
    return reply if len(reply) <= limit else reply[:limit].rstrip() + "..."


def record_reply(node_id: str, reply: str) -> None:
    """Called by /ask once it has a real (non-apology) reply for a node, so
    /ambient knows a conversation is open on that node — and can show the
    classifier what was actually said, so it can tell an unnamed follow-up
    ("and tomorrow?") is continuing that specific reply."""
    if not node_id:
        return
    reply = strip_defer_marker(reply)
    with _ambient_lock:
        now = _now()
        _ambient_last_reply[node_id] = (now, reply)
        # The agent's own reply is part of the rolling room context for the whole
        # AMBIENT_CONTEXT_WINDOW_SEC (not only the follow-up window), so a
        # side conversation after the agent answered reads correctly.
        dq = _ambient_context.setdefault(
            node_id, collections.deque(maxlen=AMBIENT_CONTEXT_MAX_ENTRIES)
        )
        dq.append((now, AGENT_LABEL, _clip_reply(reply)))


def extend_reply_playback(node_id: str, play_end: "float | None") -> None:
    """Push the follow-up window's start out to when the node finishes
    PLAYING the reply (wall-clock `play_end`), since record_reply stamps at
    text-ready, before the audio is synthesised and played. Only ever moves
    the stamp later; unknown audio length (None) keeps record_reply's stamp.
    The rolling-context entry keeps its original timestamp."""
    if not node_id or play_end is None:
        return
    with _ambient_lock:
        last = _ambient_last_reply.get(node_id)
        if last is not None and play_end > last[0]:
            _ambient_last_reply[node_id] = (play_end, last[1])


def _speaker_label(speaker: str) -> str:
    """"mike" if matched, "unknown" otherwise — what the classifier and the
    rolling context both show instead of an anonymous line."""
    return speaker if speaker else "unknown"


def _record_ambient_transcript(node_id: str, transcript: str, speaker: str = "") -> None:
    if not node_id:
        return
    with _ambient_lock:
        dq = _ambient_context.setdefault(
            node_id, collections.deque(maxlen=AMBIENT_CONTEXT_MAX_ENTRIES)
        )
        dq.append((_now(), _speaker_label(speaker), transcript))


def _ambient_context_lines(node_id: str) -> list[str]:
    """Last AMBIENT_CONTEXT_MAX_ENTRIES transcripts within
    AMBIENT_CONTEXT_WINDOW_SEC, oldest first, excluding the current
    utterance (that's appended separately by the caller). Each line is
    "<speaker>: <text>" ("unknown" when speaker ID didn't match anyone) so
    the classifier sees who said what, not an anonymous transcript."""
    if not node_id:
        return []
    cutoff = _now() - AMBIENT_CONTEXT_WINDOW_SEC
    with _ambient_lock:
        dq = _ambient_context.get(node_id)
        if not dq:
            return []
        return [f"{label}: {text}" for ts, label, text in dq if ts >= cutoff]


def _conversation_open(node_id: str) -> bool:
    """A conversation with the agent is open on the node: the agent replied within
    AMBIENT_FOLLOWUP_WINDOW_SEC. Same rule as _conversation_open_note."""
    if not node_id:
        return False
    with _ambient_lock:
        last = _ambient_last_reply.get(node_id)
    return last is not None and _now() - last[0] <= AMBIENT_FOLLOWUP_WINDOW_SEC


def _conversation_open_note(node_id: str) -> str:
    if not node_id:
        return f"No open conversation with {AGENT_NAME} on this device."
    with _ambient_lock:
        last = _ambient_last_reply.get(node_id)
    if last is None:
        return f"No open conversation with {AGENT_NAME} on this device."
    ts, reply = last
    age = _now() - ts
    if age <= AMBIENT_FOLLOWUP_WINDOW_SEC:
        return (
            f"{_display_name(AGENT_NAME)} replied on this device {age:.0f}s ago — a conversation is "
            f"OPEN, so an unnamed follow-up (e.g. \"and tomorrow?\") can be "
            f"addressed to {AGENT_NAME}. {_display_name(AGENT_NAME)}'s last reply was: {reply!r}"
        )
    return f"No open conversation with {AGENT_NAME} on this device right now."


def _build_ambient_state(node_id: str, transcript: str, speaker: str = "") -> str:
    lines = _ambient_context_lines(node_id)
    parts = []
    if lines:
        parts.append("Earlier in this room (oldest first):")
        parts.extend(f"- {line}" for line in lines)
    parts.append(_conversation_open_note(node_id))
    parts.append(f"LAST utterance (classify this one): {_speaker_label(speaker)}: {transcript}")
    return "\n".join(parts)


def _build_ambient_turns(node_id: str, transcript: str, speaker: str = "") -> list[dict]:
    """Laya's state: a conversation turn list built from the same in-memory
    context as _build_ambient_state. One user turn per recent room line
    (oldest first, "<speaker>: <text>"; the agent's own replies appear in the same
    context as "agent: <reply>" for the whole context window), and last the
    utterance itself "<speaker>: <text>"."""
    turns = [{"role": "user", "content": line} for line in _ambient_context_lines(node_id)]
    turns.append({"role": "user", "content": f"{_speaker_label(speaker)}: {transcript}"})
    return turns


def is_hallucination_or_too_short(transcript: str) -> bool:
    t = (transcript or "").strip()
    if t.lower().rstrip(".!") in AMBIENT_HALLUCINATIONS:
        return True
    alnum = sum(1 for c in t if c.isalnum())
    return alnum < AMBIENT_MIN_ALNUM_CHARS


# Echo filter: a transcript that is mostly a run of the agent's own last reply is
# the room speaker heard by the mic, not a person. Dropped before the brain.
ECHO_WINDOW_SEC = float(os.environ.get("MVG_ECHO_WINDOW_SEC", "120"))
ECHO_MIN_TOKENS = int(os.environ.get("MVG_ECHO_MIN_TOKENS", "3"))
ECHO_MIN_RATIO = float(os.environ.get("MVG_ECHO_MIN_RATIO", "0.8"))


def _echo_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", (text or "").lower())


def is_silent_sentinel(raw: str) -> bool:
    """True when the whole brain reply is the word PASS (any case, optional
    trailing punctuation): the brain chose silence (NIENNA N-2026-10-13-069)."""
    return raw.strip().rstrip(".!?").strip().upper() == "PASS"


def is_reply_echo(node_id: str, transcript: str) -> bool:
    """True when at least ECHO_MIN_RATIO of the transcript's words sit in
    contiguous runs of the node's last reply, heard within ECHO_WINDOW_SEC."""
    if not node_id:
        return False
    with _ambient_lock:
        last = _ambient_last_reply.get(node_id)
    if not last or _now() - last[0] > ECHO_WINDOW_SEC:
        return False
    t = _echo_tokens(transcript)
    r = _echo_tokens(last[1])
    if len(t) < ECHO_MIN_TOKENS or not r:
        return False
    import difflib
    sm = difflib.SequenceMatcher(None, t, r, autojunk=False)
    matched = sum(b.size for b in sm.get_matching_blocks() if b.size >= 2)
    return matched / len(t) >= ECHO_MIN_RATIO


# --- Ambient classifier: MAHANAXAR (plugin classify module) ------------------------

def decide_addressed(node_id: str, transcript: str, speaker: str = "") -> dict:
    """Pre-gate + classifier chain for one utterance (both /ambient and
    /voice/turn call this). Deterministic pre-gate first: no agent-name
    variant in the utterance AND no open conversation on the node means it
    cannot be addressed to the agent (VOICE_ADDRESSED_INSTRUCTIONS' own
    definition), so return score 0.0 / classifier "pregate" without calling
    Laya, Jev or Haiku. Otherwise run classify_addressed_detail, telling it
    whether the name was found (Laya may only engage alone when it was).

    Before the pre-gate: a wake phrase at the utterance start (WAKE_PHRASE_RE)
    is an explicit address, so score 1.0 / classifier "wake-phrase" with no
    classifier called."""
    if WAKE_PHRASE_RE.match(transcript or ""):
        return {"score": 1.0, "classifier": "wake-phrase",
                "reason": "wake phrase at utterance start", "laya_score": None}
    named = _name_mentioned(transcript)
    if not named and not _conversation_open(node_id):
        return {"score": 0.0, "classifier": "pregate",
                "reason": "no agent name and no open conversation",
                "laya_score": None}
    state = _build_ambient_state(node_id, transcript, speaker)
    turns = _build_ambient_turns(node_id, transcript, speaker)
    # With no MVG_AGENT_NAMES configured there is no name to look for, so
    # "name absent" carries no signal: let Laya engage as it would after a
    # named utterance instead of silently disabling the fast path.
    name_present = named or _AGENT_NAME_RE is None
    return classify_addressed_detail(state, turns, name_present=name_present)


def _addressed_score(result) -> float:
    """P(addressed_to_agent) for a choice Result: the layer's own probability
    when it reported one, else (Haiku, which returns no probabilities) 1.0
    for an addressed_to_agent choice and 0.0 for any other."""
    probs = result.probabilities
    if probs and AMBIENT_ADDRESSED in probs:
        return float(probs[AMBIENT_ADDRESSED])
    return 1.0 if result.choice == AMBIENT_ADDRESSED else 0.0


def classify_addressed_detail(state: str, laya_state=None, name_present: bool = True) -> dict:
    """The /ambient decision rule. Returns {"score", "classifier", "reason",
    "laya_score"} (laya_score None unless Laya answered). Laya is a fast path
    for confident yeses only; Jev gives a second opinion on everything else:

    1. Laya (own classify_choice() call, layers=("laya",), AMBIENT_LAYA_TIMEOUT,
       choice over AMBIENT_OPTIONS on the turn list `laya_state`) answers with
       P(addressed_to_agent) >= AMBIENT_LAYA_CONFIDENT -> classifier "laya", Jev not called.
    2. Laya P(yes) below that -> second classify() call, layers
       ("jev", "haiku") (Haiku only on a Jev error or timeout). That score
       decides against AMBIENT_THRESHOLD, so a Jev yes overrides Laya's no.
       classifier "laya+jev" / "laya+haiku".
    3. Laya timeout or error -> the same Jev/Haiku call, classifier
       "jev" / "haiku" exactly as before Laya existed.
    4. Laya answered but Jev AND Haiku both failed -> Laya's own P(yes),
       classifier "laya" (engage still needs AMBIENT_THRESHOLD).
    5. Nothing answered -> (0.0, "none").

    name_present=False (open-conversation follow-up with no agent name): Laya
    is only recorded, never decisive. It cannot engage alone (step 1) and
    cannot be the fallback verdict (step 4: Laya answered, Jev/Haiku failed
    -> "none", score 0.0, laya_score kept); Jev/Haiku must confirm.

    Haiku is slow (~9 s on slow hardware), so the edge client's ambient timeout
    must exceed Laya + Jev + Haiku. jev_retry=False: the 1s retry delay plus
    a second Jev attempt only delays the fallback. Each classify() call
    writes its own usage line (logs/classify-usage.jsonl)."""
    def out(score, classifier, reason="", laya_score=None):
        return {"score": score, "classifier": classifier, "reason": reason,
                "laya_score": laya_score}

    if _jev_classifier is None:
        # No classifier (no paid key, or the bridge is absent): listen on the
        # agent's name alone. Without a name the utterance is not addressed.
        if name_present:
            return out(1.0, "name-only", "no classifier: engaged on the agent's name")
        return out(0.0, "name-only", "no classifier and no name")

    laya_p = None
    try:
        _k, lres = _jev_classifier.classify_choice(
            CLASSIFY_KEY,
            state,
            VOICE_ADDRESSED_INSTRUCTIONS,
            AMBIENT_OPTIONS,
            question_key=AMBIENT_ADDRESSED,
            layers=("laya",),
            laya_state=laya_state,
            laya_question=AMBIENT_LAYA_QUESTION,
            laya_timeout=AMBIENT_LAYA_TIMEOUT,
        )
        # Laya must report P(addressed_to_agent) itself; a bare choice with no
        # probabilities is treated as "Laya gave no usable score" (falls to Jev).
        if lres.ok and lres.probabilities and AMBIENT_ADDRESSED in lres.probabilities:
            laya_p = float(lres.probabilities[AMBIENT_ADDRESSED])
    except Exception:
        laya_p = None
    if name_present and laya_p is not None and laya_p >= AMBIENT_LAYA_CONFIDENT:
        return out(laya_p, "laya", "", laya_p)

    try:
        _k, result = _jev_classifier.classify_choice(
            CLASSIFY_KEY,
            state,
            VOICE_ADDRESSED_INSTRUCTIONS,
            AMBIENT_OPTIONS,
            question_key=AMBIENT_ADDRESSED,
            layers=("jev", "haiku"),
            timeout=AMBIENT_JEV_TIMEOUT,
            haiku_timeout=AMBIENT_HAIKU_TIMEOUT,
            jev_retry=False,
        )
    except Exception as exc:
        result = None
        reason = f"classify raised: {type(exc).__name__}"
    if result is not None and result.ok and result.choice is not None:
        name = result.layer if laya_p is None else f"laya+{result.layer}"
        return out(_addressed_score(result), name, "", laya_p)
    if result is not None:
        reason = "; ".join(f"{k}: {v}" for k, v in result.errors.items())
    if laya_p is not None:
        if not name_present:
            return out(0.0, "none", reason, laya_p)
        return out(laya_p, "laya", reason, laya_p)
    return out(0.0, "none", reason)


def classify_addressed(state: str, laya_state=None) -> tuple[float, str, str]:
    """(score, classifier, reason) view of classify_addressed_detail();
    classifier is "laya"/"laya+jev"/"laya+haiku"/"jev"/"haiku"/"name-only"
    (no classifier installed; see classify_addressed_detail)/"none"."""
    d = classify_addressed_detail(state, laya_state)
    return d["score"], d["classifier"], d["reason"]


# --- Ambient speech gate: Silero VAD before Whisper -----------------------
#
# A clip that is only noise must never reach Whisper or the classifier.
# The ear screens with Silero itself, but other edge clients may
# still use the energy VAD, so the gateway re-checks server-side. Fail open:
# anything that stops the gate from deciding (no onnxruntime/numpy, no
# model, odd WAV, VAD error) lets the clip through to Whisper as before.

SILERO_WINDOW_SAMPLES = 512          # 32 ms @ 16 kHz
SILERO_SAMPLE_RATE = 16000
SILERO_WINDOW_MS = SILERO_WINDOW_SAMPLES * 1000 // SILERO_SAMPLE_RATE
SILERO_MODEL_DEFAULT_GLOB = (
    "/opt/lomin/venv/lib/python3*/site-packages/"
    "openwakeword/resources/models/silero_vad.onnx"
)
SILERO_MODEL_PATH = os.environ.get("MVG_SILERO_MODEL", "")
SPEECH_THRESHOLD = float(os.environ.get("MVG_SPEECH_THRESHOLD", "0.5"))
MIN_SPEECH_MS = float(os.environ.get("MVG_MIN_SPEECH_MS", "400"))
SPEECH_GATE_ENABLED = os.environ.get("MVG_SPEECH_GATE", "1") not in ("0", "false", "off")
_SILERO_RETRY_SEC = 300.0  # after a failed model load, don't retry every clip

_silero_lock = threading.Lock()
_silero_session = None
_silero_failed_at = 0.0


def _silero_model_file() -> str:
    if SILERO_MODEL_PATH:
        return SILERO_MODEL_PATH
    import glob
    hits = sorted(glob.glob(SILERO_MODEL_DEFAULT_GLOB))
    return hits[-1] if hits else SILERO_MODEL_DEFAULT_GLOB


def _silero_get():
    """Load the ONNX session once, lazily, thread-safe. Returns None when
    onnxruntime/numpy/the model is unavailable (gate then fails open).
    Sessions are safe to share across threads; the LSTM state lives in each
    clip's own run, never on the session."""
    global _silero_session, _silero_failed_at
    if _silero_session is not None:
        return _silero_session
    with _silero_lock:
        if _silero_session is not None:
            return _silero_session
        if _silero_failed_at and time.monotonic() - _silero_failed_at < _SILERO_RETRY_SEC:
            return None
        try:
            import numpy  # noqa: F401
            import onnxruntime as ort
            opts = ort.SessionOptions()
            opts.inter_op_num_threads = 1
            opts.intra_op_num_threads = 1
            sess = ort.InferenceSession(
                _silero_model_file(), sess_options=opts,
                providers=["CPUExecutionProvider"])
            names = {i.name for i in sess.get_inputs()}
            if not {"input", "h", "c", "sr"} <= names:
                raise RuntimeError(f"unsupported silero model inputs {sorted(names)}")
            _silero_session = sess
            return sess
        except Exception as exc:
            _silero_failed_at = time.monotonic()
            print(f"speech gate: silero unavailable ({type(exc).__name__}: {exc})"[:300],
                  file=sys.stderr, flush=True)
            return None


def _wav_to_pcm16k(wav_bytes: bytes):
    """WAV bytes -> mono int16 numpy array at 16 kHz. Stereo is averaged
    and other sample rates are linearly resampled; anything that is not
    16-bit PCM raises ValueError (gate fails open)."""
    import numpy as np
    with wave.open(BytesIO(wav_bytes), "rb") as w:
        if w.getsampwidth() != 2:
            raise ValueError(f"sample width {w.getsampwidth()} unsupported")
        rate, channels = w.getframerate(), w.getnchannels()
        raw = w.readframes(w.getnframes())
    pcm = np.frombuffer(raw, dtype="<i2")
    if channels > 1:
        pcm = pcm[: len(pcm) // channels * channels].reshape(-1, channels).mean(axis=1)
    if rate != SILERO_SAMPLE_RATE and len(pcm):
        n_out = int(len(pcm) * SILERO_SAMPLE_RATE / rate)
        pcm = np.interp(np.linspace(0, len(pcm) - 1, n_out), np.arange(len(pcm)), pcm)
    return np.asarray(pcm).astype(np.int16)


def silero_window_probs(session, pcm) -> list:
    """Per-512-sample-window speech probabilities, LSTM state reset at clip
    start (fresh zeros). A trailing partial window is dropped."""
    import numpy as np
    h = np.zeros((2, 1, 64), dtype=np.float32)
    c = np.zeros((2, 1, 64), dtype=np.float32)
    sr = np.array(SILERO_SAMPLE_RATE, dtype=np.int64)
    probs = []
    for i in range(len(pcm) // SILERO_WINDOW_SAMPLES):
        chunk = (pcm[i * SILERO_WINDOW_SAMPLES:(i + 1) * SILERO_WINDOW_SAMPLES]
                 / 32767).astype(np.float32)
        out, h, c = session.run(None, {"input": chunk[None, :], "h": h, "c": c, "sr": sr})
        probs.append(float(out[0][0]))
    return probs


def voiced_ms(probs, threshold: float = None) -> float:
    """Milliseconds of audio in windows scoring >= threshold."""
    thr = SPEECH_THRESHOLD if threshold is None else threshold
    return sum(1 for p in probs if p >= thr) * SILERO_WINDOW_MS


def speech_gate_decision(voiced: float, min_ms: float = None) -> bool:
    """True when the clip has enough voiced audio to be worth transcribing."""
    return voiced >= (MIN_SPEECH_MS if min_ms is None else min_ms)


def speech_gate(wav_bytes: bytes) -> tuple:
    """Returns (status, voiced_ms, vad_ms); status is "passed", "rejected"
    or "unavailable" (fail open: the caller continues to Whisper)."""
    if not SPEECH_GATE_ENABLED:
        return "unavailable", None, 0.0
    t0 = time.perf_counter()
    try:
        session = _silero_get()
        if session is None:
            return "unavailable", None, round((time.perf_counter() - t0) * 1000, 1)
        pcm = _wav_to_pcm16k(wav_bytes)
        t_vad = time.perf_counter()
        voiced = voiced_ms(silero_window_probs(session, pcm))
        vad_ms = round((time.perf_counter() - t_vad) * 1000, 1)
    except Exception:
        return "unavailable", None, round((time.perf_counter() - t0) * 1000, 1)
    return ("passed" if speech_gate_decision(voiced) else "rejected"), voiced, vad_ms


# --- Logging --------------------------------------------------------------

_log_lock = threading.Lock()


def _log_line(path: Path, record: dict) -> None:
    """Append one JSON line. Never raises — logging must not break a request."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with _log_lock:
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass


# Old stage endpoints kept for one release; every call is flagged in the
# request log (`legacy_endpoint`, plus `node`) so remaining callers can be found.
LEGACY_ENDPOINTS = ("/stt", "/ambient", "/ask", "/ask/stream")
_REQ = threading.local()  # per-request (thread-per-connection) node id


def _set_req_node(node_id: str) -> None:
    _REQ.node = node_id or ""


def watch_node_for(auth_header: str, requested_node: str = "") -> "str | None":
    """Node id for a valid `Bearer <token>` header, else None. Never raises.
    A token's value is a node id (string) or a list of allowed node ids: a
    list yields `requested_node` if listed, else its first element."""
    try:
        scheme, _, token = (auth_header or "").strip().partition(" ")
        token = token.strip()
        if scheme.lower() != "bearer" or not token:
            return None
        tokens = json.loads(WATCH_TOKENS_PATH.read_text())
        if not isinstance(tokens, dict):
            return None
        found = None
        for key, node in tokens.items():
            if hmac.compare_digest(str(key).encode(), token.encode()):
                found = node
        if isinstance(found, str):
            return found or None
        if isinstance(found, list):
            nodes = [n for n in found if isinstance(n, str) and n]
            if not nodes:
                return None
            return requested_node if requested_node in nodes else nodes[0]
        return None
    except Exception:
        return None


_TAG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,31}$")


def clean_tag(v) -> str:
    """A device/medium tag: lowercase, stripped, `^[a-z0-9][a-z0-9-]{0,31}$`, else ''."""
    if not isinstance(v, str):
        return ""
    v = v.strip().lower()
    return v if _TAG_RE.match(v) else ""


def resolve_device_medium(node_id: str, device=None, medium=None) -> tuple[str, str]:
    """(device, medium): the cleaned request value, else the node's nodes.json
    profile value, else ''."""
    prof = resolve_profile(node_id)
    return (clean_tag(device) or clean_tag(prof.get("device")),
            clean_tag(medium) or clean_tag(prof.get("medium")))


def log_request(endpoint: str, ms: float, outcome: str, **extra) -> None:
    if endpoint in LEGACY_ENDPOINTS:
        extra.setdefault("legacy_endpoint", endpoint)
        extra.setdefault(
            "node", extra.get("node_id") or getattr(_REQ, "node", "") or "unknown")
    rec = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "endpoint": endpoint,
        "ms": round(ms, 1),
        "outcome": outcome,
    }
    rec.update(extra)
    _log_line(LOG_DIR / LOG_NAME, rec)


def log_fallback(reason: str, tts_engine: str = "piper", **extra) -> None:
    rec = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tts_engine": tts_engine,
        "reason": reason,
    }
    rec.update(extra)
    _log_line(LOG_DIR / "voice_fallback.log", rec)


# --- Latency metrics (logs/voice-latency.jsonl) --------------------------
# One record per /ask/stream turn (kind "turn") and one per /stt call (kind
# "stt"). bin/voice-latency-report.py rolls them up. The hub calls /stt just
# before /ask/stream, so the most recent /stt is remembered in memory and its
# id/timing is attached to the turn record when it is recent enough (and the
# node matches, when both sides know it).

LATENCY_LOG_NAME = "voice-latency.jsonl"
STT_CORRELATE_SEC = float(os.environ.get("MVG_STT_CORRELATE_SEC", "30"))
# Most recent /stt per node ("" when the node is unknown), plus the newest
# overall for callers that do not know their node. Guarded by _last_stt_lock:
# two rooms talking at once must not clobber each other's correlation.
_last_stt: dict[str, dict] = {}
_last_stt_any: dict = {}
_last_stt_lock = threading.Lock()


def _iso_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def log_latency(record: dict) -> None:
    rec = {"ts": _iso_now()}
    rec.update(record)
    _log_line(LOG_DIR / LATENCY_LOG_NAME, rec)


def _wav_seconds(wav_bytes: bytes) -> float | None:
    try:
        with wave.open(BytesIO(wav_bytes), "rb") as w:
            return round(w.getnframes() / float(w.getframerate()), 3)
    except Exception:
        return None


def _note_stt(node_id: str, audio_s, stt_ms: float, chars: int, cache_hit: bool) -> None:
    stt_id = uuid.uuid4().hex[:12]
    now = time.time()
    rec = {"stt_id": stt_id, "node_id": node_id, "t_end": now,
           "stt_ms": round(stt_ms, 1)}
    with _last_stt_lock:
        _last_stt[node_id or ""] = rec
        _last_stt_any.clear()
        _last_stt_any.update(rec)
    log_latency({"kind": "stt", "stt_id": stt_id, "node_id": node_id or None,
                 "audio_s": audio_s, "stt_ms": round(stt_ms, 1), "chars": chars,
                 "cache_hit": cache_hit})


def _correlated_stt(node_id: str, request_start: float) -> dict:
    with _last_stt_lock:
        if node_id:
            # This node's own /stt, else one from a caller that sent no node.
            s = dict(_last_stt.get(node_id) or _last_stt.get("") or {})
        else:
            s = dict(_last_stt_any)
    if not s or request_start - s["t_end"] > STT_CORRELATE_SEC or request_start < s["t_end"] - 1:
        return {}
    if s.get("node_id") and node_id and s["node_id"] != node_id:
        return {}
    return {"stt_id": s["stt_id"], "stt_ms": s["stt_ms"]}


# --- Gap-fill clips (data/voice/gap-fill: vocal sounds from bin/render-gap-fill.py,
# machine effects from bin/render-gap-fill-sfx.py)
# Short sounds played while the brain is silent, each in its own timing slot
# (vocal_open, sfx_generic / sfx_<topic> / handoff, vocal_long, sfx_tick). Loaded from the
# gateway's own workspace checkout (the WAVs arrive via git). Every clip is
# 24 kHz mono PCM16 so it can ride the same pcm `sentence` events as real
# audio.

GAP_FILL = os.environ.get("MVG_GAP_FILL", "1").strip().lower() not in ("0", "false", "off", "no", "")
# Slot timings, ms since the request started (or, for the ticks, ms of playback
# silence). See _maybe_gap_fill for the rules.
GAP_FILL_AFTER_MS = float(os.environ.get("MVG_GAP_FILL_AFTER_MS", "1200"))   # opener vocal
GAP_FILL_MID_MS = float(os.environ.get("MVG_GAP_FILL_MID_MS", "3500"))       # topical effect
GAP_FILL_SLOW_MS = float(os.environ.get("MVG_GAP_FILL_SLOW_MS", "8000"))     # long vocal
GAP_FILL_TICK_MS = float(os.environ.get("MVG_GAP_FILL_TICK_MS", "3000"))     # tick spacing before the reply
GAP_FILL_LONG_MS = float(os.environ.get("MVG_GAP_FILL_LONG_MS", "5000"))     # silence after reply audio before a tick
GAP_FILL_QUIET_MS = 800.0  # silence required before the mid, long and tick-before-reply slots
GAP_FILL_MAX_TICKS = 6
GAP_FILL_DIR = Path(os.environ.get("MVG_GAP_FILL_DIR") or (Path(__file__).resolve().parent / "gap-fill"))
GAP_FILL_VOCAL_OPEN = ("vocal_open",)
GAP_FILL_VOCAL_LONG = ("vocal_long",)
GAP_FILL_SFX_GENERIC = ("sfx_generic",)
GAP_FILL_SFX_TICK = ("sfx_tick",)
GAP_FILL_TOPICAL = os.environ.get("MVG_GAP_FILL_TOPICAL", "1").strip().lower() not in ("0", "false", "off", "no", "")
# Topical effects: the first topic whose pattern matches the turn's transcript
# picks the mid-slot pool. Order matters: handoff beats home beats
# calendar_lists beats lookup.
GAP_FILL_TOPICS = tuple(
    (category, re.compile(pattern, re.IGNORECASE))
    for category, pattern in (
        ("handoff", r"\b(queue|build queue|hand (it )?off|pass (that|it) (on|to)|have (the other agent|the primary)|tell (the other agent|the primary)|ask (the other agent|the primary))\b"),
        ("home", r"\b(lights?|lamp|thermostat|temperature|heat(er|ing)?|fan|lock|unlock|door|garage|blinds?|switch|turn (on|off))\b"),
        ("calendar_lists", r"\b(calendar|schedule|appointments?|meetings?|remind(er)?s?|lists?|shopping|grocer(y|ies)|to-?do|tasks?)\b"),
        ("lookup", r"\b(look (it |that )?up|search|find out|what is|what'?s|who is|who was|when did|how many|news|weather|latest)\b"),
    )
)
GAP_FILL_CHUNK_BYTES = 9600  # 200 ms of 24 kHz s16le mono

_gap_cache: dict = {}
_gap_lock = threading.Lock()
_gap_prev_turn: set = set()   # slugs played in the previous turn
_gap_rng = random.Random()


def load_gap_clips(directory: Path | None = None) -> dict:
    """{category: [{"slug", "text", "pcm"}]} read from manifest.json. Cached
    per directory; clips that are not 24 kHz mono PCM16 are skipped."""
    directory = Path(directory or GAP_FILL_DIR)
    key = str(directory)
    with _gap_lock:
        if key in _gap_cache:
            return _gap_cache[key]
    clips: dict = {}
    try:
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        for rel, meta in (manifest.get("clips") or {}).items():
            try:
                with wave.open(str(directory / rel), "rb") as w:
                    if (w.getframerate(), w.getnchannels(), w.getsampwidth()) != (
                            AUDIO_TARGET_RATE, AUDIO_TARGET_CHANNELS, AUDIO_TARGET_SAMPWIDTH):
                        continue
                    pcm = w.readframes(w.getnframes())
            except Exception:
                continue
            if not pcm:
                continue
            category = meta.get("category") or rel.split("/", 1)[0]
            clips.setdefault(category, []).append(
                {"slug": rel[:-4] if rel.endswith(".wav") else rel,
                 "category": category, "text": meta.get("text") or "", "pcm": pcm})
    except Exception:
        clips = {}
    with _gap_lock:
        _gap_cache[key] = clips
    return clips


def pick_gap_clip(categories, used=()) -> dict | None:
    """One random clip from the given categories. Never a clip already
    played this turn (`used`) or anywhere in the previous turn, when there
    is an alternative."""
    clips = load_gap_clips()
    pool = [c for cat in categories for c in clips.get(cat, [])]
    if not pool:
        return None
    with _gap_lock:
        avoid = set(used) | _gap_prev_turn
        options = ([c for c in pool if c["slug"] not in avoid]
                   or [c for c in pool if c["slug"] not in used] or pool)
        return _gap_rng.choice(options)


def effect_categories(prompt) -> tuple:
    """Mid-slot pool for one turn: `("handoff",)` for a handoff match,
    `("sfx_<topic>",)` for the other topics when MVG_GAP_FILL_TOPICAL is on.
    An empty or non-string prompt, or no topic match, gets the generic effects."""
    if not GAP_FILL_TOPICAL or not isinstance(prompt, str) or not prompt.strip():
        return GAP_FILL_SFX_GENERIC
    for category, pattern in GAP_FILL_TOPICS:
        if pattern.search(prompt):
            return ("handoff",) if category == "handoff" else (f"sfx_{category}",)
    return GAP_FILL_SFX_GENERIC


def _gap_turn_done(used) -> None:
    """Remember this turn's clips so the next turn avoids them. A turn with
    no filler leaves the previous memory alone."""
    global _gap_prev_turn
    if used:
        with _gap_lock:
            _gap_prev_turn = set(used)


def _safe_normalize_wav(wav_bytes: bytes) -> bytes:
    """normalize_wav(), but never raises: a failed conversion logs and sends
    the ORIGINAL wav bytes rather than dropping the sentence — a mismatched
    format on one sentence is better than silence."""
    try:
        return normalize_wav(wav_bytes)
    except Exception as exc:
        log_fallback(f"normalize_wav_failed:{exc}", tts_engine="normalize-skip")
        return wav_bytes


# Overridable by tests.
NORMALIZE_WAV = _safe_normalize_wav


# --- STT (Whisper) ----------------------------------------------------------

def transcribe(wav_bytes: bytes) -> str:
    req = Request(
        WHISPER_URL,
        data=wav_bytes,
        headers={"Content-Type": "audio/wav"},
        method="POST",
    )
    with urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode())
    return (data.get("text") or "").strip()


def whisper_reachable() -> bool:
    try:
        with urlopen(WHISPER_HEALTH_URL, timeout=3):
            return True
    except Exception:
        return False


def _stt_cache_get(sha: str) -> str | None:
    with _ambient_lock:
        entry = _stt_cache.get(sha)
    if entry is None:
        return None
    ts, transcript = entry
    if _now() - ts > STT_CACHE_TTL_SEC:
        return None
    return transcript


def _stt_cache_put(sha: str, transcript: str) -> None:
    now = _now()
    with _ambient_lock:
        _stt_cache[sha] = (now, transcript)
        # Prune expired entries here rather than running a separate sweep
        # thread — the cache is small (one engaged utterance at a time in
        # practice) and this keeps "cached for 60s" honest instead of
        # holding transcripts in memory indefinitely.
        expired = [k for k, (ts, _) in _stt_cache.items() if now - ts > STT_CACHE_TTL_SEC]
        for k in expired:
            del _stt_cache[k]


def _speaker_cache_get(sha: str) -> tuple[str, float] | None:
    with _ambient_lock:
        entry = _speaker_cache.get(sha)
    if entry is None:
        return None
    ts, speaker, score = entry
    if _now() - ts > STT_CACHE_TTL_SEC:
        return None
    return speaker, score


def _speaker_cache_put(sha: str, speaker: str, score: float) -> None:
    now = _now()
    with _ambient_lock:
        _speaker_cache[sha] = (now, speaker, score)
        expired = [k for k, (ts, _, _) in _speaker_cache.items() if now - ts > STT_CACHE_TTL_SEC]
        for k in expired:
            del _speaker_cache[k]


# --- Speaker identification for /ambient (SPEC-011) -------------------------
# Ports the hub's speaker.go FingerprintStore /
# CosineSimilarity / MatchSpeaker to Python so /ambient — which sees every
# speech segment and never reaches the Go hub for a non-engaging one — can
# label a speaker exactly the way an engaged /ask turn (routed through the
# hub's own Go client, this gateway's /embed proxy) would.

_fp_lock = threading.Lock()
_fp_cache: dict = {"path": None, "mtime": None, "size": None, "data": None}


def _load_fingerprints() -> dict | None:
    """Reads SPEAKERS_PATH, re-reading only when its path, mtime or size
    changed — same "reload on change, no restart needed" contract as the
    Go hub's FingerprintStore.Load(). Returns None (never raises) on any
    problem: missing file, bad JSON, or a shape that doesn't look like
    {"model": ..., "speakers": [...]}. This path is best-effort; nothing
    here may block or fail an /ambient turn."""
    try:
        st = SPEAKERS_PATH.stat()
    except OSError:
        return None
    key = str(SPEAKERS_PATH)
    with _fp_lock:
        if (
            _fp_cache["path"] == key
            and _fp_cache["mtime"] == st.st_mtime
            and _fp_cache["size"] == st.st_size
            and _fp_cache["data"] is not None
        ):
            return _fp_cache["data"]
    try:
        data = json.loads(SPEAKERS_PATH.read_text())
    except Exception:
        return None
    if not isinstance(data, dict) or not isinstance(data.get("speakers"), list):
        return None
    with _fp_lock:
        _fp_cache.update(path=key, mtime=st.st_mtime, size=st.st_size, data=data)
    return data


def enrolled_roster() -> str:
    """`enrolled: alice, bob` / `enrolled: none` / `enrolled: unknown` for the
    voice frame. "none" only when the file loaded and speakers is empty; any
    read failure is "unknown", never "none". Reuses _load_fingerprints()."""
    data = _load_fingerprints()
    if data is None:
        return "enrolled: unknown"
    ids = []
    for sp in data.get("speakers") or []:
        sid = (sp or {}).get("id") if isinstance(sp, dict) else None
        if sid:
            ids.append(str(sid))
    return "enrolled: " + (", ".join(ids) if ids else "none")


def _cosine_similarity(a: list, b: list) -> float:
    if not a or not b or len(a) != len(b):
        raise ValueError("dimension mismatch or empty vector")
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        raise ValueError("zero-length vector")
    return dot / (na * nb)


def _match_speaker(embedding: list, speakers: list, threshold: float) -> tuple[str, float]:
    """Best enrolled speaker for `embedding`, if its cosine similarity clears
    `threshold`. A fingerprint that can't be compared (bad shape, wrong
    dimension) is skipped rather than aborting the whole match, same as the
    Go MatchSpeaker. The best score is returned even when it is below the
    threshold (so a caller can say "not verified (best 0.41)"), but the name
    is "" below it: a name is never returned under the threshold."""
    best_id, best = "", float("-inf")
    for sp in speakers or []:
        sid = (sp or {}).get("id")
        emb = (sp or {}).get("embedding")
        if not sid or not emb:
            continue
        try:
            sim = _cosine_similarity(embedding, emb)
        except Exception:
            continue
        if sim > best:
            best, best_id = sim, sid
    if not best_id:
        return "", 0.0
    _note_best_id(best_id)
    if best < threshold:
        return "", round(best, 4)
    return best_id, round(best, 4)


_BEST = threading.local()


def _note_best_id(best_id: str) -> None:
    """Side channel: _match_speaker's 2-tuple contract has no room for the
    best-matching id when it is below threshold, so park it per thread for
    identify_speaker_detail to collect (voice-ID logging / capture only)."""
    _BEST.id = best_id


def identify_speaker_detail(wav_bytes: bytes, sha: str) -> dict:
    """Voice ID with its verdict: {"status": "verified"|"not_verified"|
    "unavailable", "id", "score"}. verified = a name at or above
    SPEAKER_THRESHOLD; not_verified = prints compared but the best score fell
    below it (score is that best score, id is ""); unavailable = no prints,
    embed down, model mismatch or nothing comparable. Also carries "best": the
    best-matching enrolled id even below threshold ("" when none). Never
    raises."""
    _BEST.id = ""
    try:
        name, score = identify_speaker(wav_bytes, sha)
    except Exception:
        return {"status": "unavailable", "id": "", "score": 0.0, "best": ""}
    best = name or getattr(_BEST, "id", "") or ""
    if name:
        return {"status": "verified", "id": name, "score": score, "best": best}
    if score:  # 0.0 is what every failure path returns
        return {"status": "not_verified", "id": "", "score": score, "best": best}
    return {"status": "unavailable", "id": "", "score": 0.0, "best": ""}


def identify_speaker(wav_bytes: bytes, sha: str) -> tuple[str, float]:
    """Best-effort speaker match for one utterance's WAV bytes: ("", 0.0) on
    ANY failure — no fingerprints configured, desktop embed hop down or
    slow, a model mismatch, or a malformed response. Never raises. Checks
    the speaker cache first (see _speaker_cache_put, written alongside the
    STT cache on engage) so a repeat of the same bytes doesn't re-embed."""
    cached = _speaker_cache_get(sha)
    if cached is not None:
        return cached
    fp_file = _load_fingerprints()
    if not fp_file or not fp_file.get("speakers"):
        return "", 0.0
    try:
        status, body = proxy_embed(wav_bytes)
        if status != 200:
            return "", 0.0
        data = json.loads(body)
        embedding = data.get("embedding")
        model = data.get("model")
    except Exception:
        return "", 0.0
    if not embedding:
        return "", 0.0
    fp_model = fp_file.get("model")
    if fp_model and model and fp_model != model:
        # Refuse rather than compare embeddings from two different models —
        # same posture as the Go FingerprintIdentifier.Identify.
        return "", 0.0
    return _match_speaker(embedding, fp_file["speakers"], SPEAKER_THRESHOLD)


# --- Brain (the plugin brain or plain HTTP) ---------------------------------------------

PROFILE_DEFAULTS = {
    "target": None,          # None -> DEFAULT_TARGET
    "targets": [],           # agents a request may pick with `target` (empty: none)
    "open_room": False,      # per request, profile default
    "identify": True,
    "speech_gate": True,
    "gap_fill": True,
    "reply_cap_words": 60,
    "audio_chunk": False,    # also emit the hub-dialect `audio_chunk` events (ear)
    "capture_audio": False,  # opt-in: save gated turn audio + sidecar for enrollment
    "device": "",            # what the speaker holds/stands at, e.g. iphone (request overrides)
    "medium": "",            # how they reach us, e.g. phone-app, watch-app, room-speaker
}


CAPTURE_DIR = Path(os.environ.get(
    "MVG_CAPTURE_DIR", str(MVG_HOME / "data" / "voice-capture")))
CAPTURE_RETENTION_HOURS = float(os.environ.get("MVG_CAPTURE_RETENTION_HOURS", "48"))


def _prune_capture_dir(d: Path, retention_hours: float, now: float = None) -> None:
    cutoff = (time.time() if now is None else now) - retention_hours * 3600
    for f in d.iterdir():
        if f.suffix in (".wav", ".json") and f.is_file() and f.stat().st_mtime < cutoff:
            try:
                f.unlink()
            except OSError:
                pass


def _capture_write(node_id: str, turn_id: str, wav_bytes: bytes, transcript: str, vid: dict) -> None:
    try:
        d = Path(CAPTURE_DIR) / re.sub(r"[^A-Za-z0-9._-]", "_", node_id or "unknown")
        d.mkdir(parents=True, exist_ok=True)
        best = vid.get("best") or "none"
        best_fs = re.sub(r"[^A-Za-z0-9.-]", "_", best)
        score = float(vid.get("score") or 0.0)
        stem = f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}_{turn_id}_{best_fs}_{score:.3f}"
        (d / f"{stem}.wav").write_bytes(wav_bytes)
        (d / f"{stem}.json").write_text(json.dumps({
            "turn_id": turn_id, "node_id": node_id, "transcript": transcript,
            "voice_id": voice_id_verdict(vid), "voice_id_status": vid.get("status"),
            "best_id": vid.get("best") or "", "score": round(score, 3),
        }))
        _prune_capture_dir(d, CAPTURE_RETENTION_HOURS)
    except Exception as exc:  # best-effort, but never silent
        log_request("capture", 0.0, "error:capture_write", node_id=node_id, error=repr(exc)[:200])


def capture_turn_audio(node_id: str, turn_id: str, wav_bytes: bytes, transcript: str, vid: dict) -> None:
    """Best-effort, off the request path: never raises, never blocks a turn."""
    threading.Thread(target=_capture_write, args=(node_id, turn_id, wav_bytes, transcript, dict(vid)),
                     daemon=True).start()


def _env_targets() -> dict:
    raw = os.environ.get("MVG_NODE_TARGETS")
    if raw:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
    return {}


def _node_profiles() -> dict:
    """node_id -> profile dict. nodes.json is the source; if it is absent the
    old node-targets.json ({node: target}) is read instead. Read fresh on
    every call, so an edit needs no restart."""
    if NODES_PATH.exists():
        data = _load_json(NODES_PATH)
        return {k: v for k, v in data.items() if isinstance(v, dict) and not k.startswith("_")}
    legacy = _load_json(NODE_TARGETS_PATH)
    return {k: {"target": v} for k, v in legacy.items() if isinstance(v, str)}


def resolve_profile(node_id: str) -> dict:
    """The node's flags: PROFILE_DEFAULTS overlaid by its nodes.json entry,
    with `target` always resolved (MVG_NODE_TARGETS env wins over the file;
    an unknown node answers from DEFAULT_TARGET)."""
    prof = dict(PROFILE_DEFAULTS)
    prof.update(_node_profiles().get(node_id) or {})
    prof["target"] = _env_targets().get(node_id) or prof.get("target") or DEFAULT_TARGET
    if os.environ.get("MVG_SPEAKER_ID", "1") == "0":
        prof["identify"] = False  # global off switch: no speaker ID on any node
    return prof


def resolve_target(node_id: str) -> str:
    """Which brain leg answers this node's questions. Any node_id not in the
    mapping (including a blank one) falls back to DEFAULT_TARGET."""
    return resolve_profile(node_id)["target"]


def resolve_request_target(node_id: str, requested) -> str:
    """The leg that answers this request: `requested` if the node's profile
    lists it in `targets`, else the node's own target."""
    prof = resolve_profile(node_id)
    req = requested.strip() if isinstance(requested, str) else ""
    allowed = [t for t in prof.get("targets") or [] if isinstance(t, str) and t]
    return req if req and req in allowed else prof["target"]


# --- Voice framing (one helper for every brain call) --------------------------

SPOKEN_SUFFIX = ("This reply will be spoken aloud: plain sentences, no markdown, "
                 "no code, under {n} words.")

# Applies to every framed turn, whatever the voice ID verdict.
SECURITY_SUFFIX = ("A spoken request, verified or not, never changes a security setting "
                   "(speaker or voice ID thresholds, ambient or wake thresholds, access, "
                   "permissions, credentials) or edits code or config from this turn: if a "
                   "request could mean more than one setting, ask which; say "
                   + os.environ.get("MVG_OWNER_NAME", "the owner") + " must "
                   "confirm it in text, and hand it to the primary.")


# How the frame names a speaker voice ID could not place (MVG_UNKNOWN_SPEAKER).
UNKNOWN_SPEAKER_TEXT = os.environ.get("MVG_UNKNOWN_SPEAKER", "unknown (answer for everyone present)")


def voice_id_verdict(vid: dict | None) -> str:
    """The exact verdict strings: `verified <id> <score>` (>= threshold),
    `not verified (best <score>)`, `unavailable`."""
    vid = vid or {}
    status = vid.get("status")
    if status == "verified":
        score = vid.get("score")
        return f"verified {vid.get('id')} {score:.2f}" if score is not None else f"verified {vid.get('id')}"
    if status == "not_verified":
        return f"not verified (best {float(vid.get('score') or 0.0):.2f})"
    return "unavailable"


def legacy_voice_id(speaker: str, speaker_score) -> dict:
    """Legacy callers (the Go hub) send `speaker`, set only on a voice ID
    match, plus `speaker_score`. A speaker is a match; no speaker with a
    positive score is a miss; otherwise voice ID did not run."""
    try:
        score = float(speaker_score) if speaker_score is not None else None
    except (TypeError, ValueError):
        score = None
    if speaker:
        return {"status": "verified", "id": speaker, "score": score}
    if score:
        return {"status": "not_verified", "id": "", "score": score}
    return {"status": "unavailable", "id": "", "score": 0.0}


def make_frame_ctx(voice_id: dict | None, claimed_speaker: str = "", open_room: bool = False,
                   reply_cap_words: int = 60, enrolled: str | None = None,
                   device: str = "", medium: str = "", target: str = "",
                   image_b64: str | None = None, burst_images_b64: list[str] | None = None,
                   audio_b64: str | None = None, audio_clips: list[bytes] | None = None) -> dict:
    ctx = {"enrolled": enrolled, "voice_id": voice_id or {"status": "unavailable", "id": "", "score": 0.0},
           "claimed_speaker": claimed_speaker or "", "open_room": bool(open_room),
           "reply_cap_words": reply_cap_words, "device": device or "", "medium": medium or "",
           "target": target or ""}
    if image_b64:
        ctx["image_b64"] = image_b64
    if burst_images_b64:
        ctx["burst_images_b64"] = burst_images_b64
    if audio_b64:
        ctx["audio_b64"] = audio_b64
    if audio_clips:
        ctx["audio_clips"] = audio_clips
    return ctx


def audio_clips_max_bytes() -> int:
    try:
        return max(0, int(os.environ.get("MVG_AUDIO_CLIPS_MAX_BYTES", str(2 * 1024 * 1024))))
    except ValueError:
        return 2 * 1024 * 1024


def collect_audio_clips(wav_bytes: bytes | None) -> list[bytes] | None:
    """The raw-audio clips handed to a plugin brain (`audio_clips`). None unless
    MVG_SPLICE_RAW_AUDIO=1 and there is audio; over MVG_AUDIO_CLIPS_MAX_BYTES the
    clips are dropped and the drop is logged, and the turn goes on without them."""
    if os.environ.get("MVG_SPLICE_RAW_AUDIO", "0").strip().lower() not in ("1", "true", "yes", "on"):
        return None
    if not wav_bytes:
        return None
    clips = [bytes(wav_bytes)]
    if sum(len(c) for c in clips) > audio_clips_max_bytes():
        log_fallback("audio_clips_over_cap", tts_engine="audio-clips-skip",
                     bytes=sum(len(c) for c in clips), cap=audio_clips_max_bytes())
        return None
    return clips


# Orchestrator instruction (Mode 2, specs/2026-11-05-talker-thinker-live-architecture.md 6.7).
# Generic on purpose: no tool, agent or household names. A site adds its own half
# with MVG_ORCHESTRATOR_TEXT. Off unless MVG_ORCHESTRATOR_PROMPT=1; MVG_ORCHESTRATOR_TARGETS
# (comma list) limits it to those brain targets.
ORCHESTRATOR_TEXT = ("You are the voice orchestrator. Speak one specific sentence first, then delegate. "
                     "Do not do the work yourself: hand it to a subagent and choose that subagent's "
                     "model and effort for the task. If a follow-up arrives while a task runs, say so "
                     "and do not start a second task silently. Progress lines come from the harness: "
                     "do not invent progress.")


def orchestrator_enabled(target: str = "") -> bool:
    if os.environ.get("MVG_ORCHESTRATOR_PROMPT", "0").strip().lower() not in ("1", "true", "yes", "on"):
        return False
    only = [t.strip() for t in os.environ.get("MVG_ORCHESTRATOR_TARGETS", "").split(",") if t.strip()]
    return not only or target in only


def _display_name(name: str) -> str:
    return name[:1].upper() + name[1:]


def frame_voice_prompt(prompt: str, node_id: str, ctx: dict) -> str:
    """The single framing line every brain call carries:
    `[VOICE via node lammen · speaker: Alice · voice ID: verified alice 0.82 · enrolled: alice · open room] <text>`
    followed by the spoken-reply suffix. When the ctx carries a device and/or
    medium they follow the node bit: `VOICE via node apple-phone · device: iphone · medium: phone-app · speaker: ...`. A login identity reads
    `speaker: Alice (logged in)` and never changes the voice ID verdict."""
    vid = ctx.get("voice_id") or {}
    if ctx.get("claimed_speaker"):
        who = f"{_display_name(ctx['claimed_speaker'])} (logged in)"
    elif vid.get("status") == "verified":
        who = _display_name(vid.get("id") or "")
    else:
        who = UNKNOWN_SPEAKER_TEXT
    bits = [f"VOICE via node {node_id}"]
    if ctx.get("device"):
        bits.append(f"device: {ctx['device']}")
    if ctx.get("medium"):
        bits.append(f"medium: {ctx['medium']}")
    bits += [f"speaker: {who}", f"voice ID: {voice_id_verdict(vid)}"]
    bits.append(ctx.get("enrolled") or enrolled_roster())
    if ctx.get("open_room"):
        bits.append("open room")
    cap = ctx.get("reply_cap_words") or 60
    tail = SPOKEN_SUFFIX.format(n=cap) + " " + SECURITY_SUFFIX
    if orchestrator_enabled(ctx.get("target") or ""):
        tail += " " + (os.environ.get("MVG_ORCHESTRATOR_TEXT", "").strip() or ORCHESTRATOR_TEXT)
    return f"[{' · '.join(bits)}] {prompt}\n\n" + tail


# Per-thread framing context for the brain worker thread (same pattern as
# _STAGE_SINK below). Unset -> frame from the legacy speaker/speaker_score args.
_FRAME_CTX = threading.local()


def _frame_for_call(prompt: str, node_id: str, speaker: str, speaker_score) -> str:
    ctx = getattr(_FRAME_CTX, "ctx", None) or make_frame_ctx(legacy_voice_id(speaker, speaker_score))
    return frame_voice_prompt(prompt, node_id, ctx)


def _call_target(node_id: str) -> str:
    """The per-request target carried in this thread's frame context, else the node's target."""
    ctx = getattr(_FRAME_CTX, "ctx", None) or {}
    return ctx.get("target") or resolve_target(node_id)


def plugin_brain_call(prompt: str, session_id: str, node_id: str, speaker: str,
                      speaker_score, timeout: int) -> str:
    if _plugin_brain_call is None:
        raise RuntimeError("no plugin brain")
    wrapped = _frame_for_call(prompt, node_id, speaker, speaker_score)
    return _plugin_brain_call(_call_target(node_id), wrapped, timeout)


# MVG_BRAIN=plugin|http. Default: plugin when the loaded plugin has brain_call, else http.
MVG_BRAIN = os.environ.get("MVG_BRAIN") or ("plugin" if _plugin_brain_call is not None else "http")
# Read at call time from these module attributes, so tests can point them at a local server.
MVG_BRAIN_URL = os.environ.get("MVG_BRAIN_URL", "")
MVG_BRAIN_TOKEN = os.environ.get("MVG_BRAIN_TOKEN", "")


from streaming_brain import consume_brain_sse_lines as _sb_consume_brain_sse_lines


def consume_brain_sse_lines(
    lines,
    on_partial=None,
    max_stream_seconds: float | None = None,
    max_chars: int | None = None,
    read_timeout: float | None = None,
    incremental: bool = False,
) -> str:
    """Consume Server-Sent Events (SSE) lines from an HTTP brain stream via streaming_brain."""
    eff_max_chars = MAX_SSE_CHARS if max_chars is None else max_chars
    eff_read_timeout = max_stream_seconds if read_timeout is None else read_timeout
    return _sb_consume_brain_sse_lines(
        lines,
        on_partial=on_partial,
        max_chars=eff_max_chars,
        read_timeout=eff_read_timeout,
        max_stream_seconds=max_stream_seconds,
        raise_on_timeout=False,
        incremental=incremental,
    )


def build_http_brain_body(prompt: str, session_id: str, node_id: str, speaker: str,
                          speaker_score,
                          image_b64: str | None = None, burst_images_b64: list[str] | None = None,
                          audio_b64: str | None = None) -> tuple[dict, bytes]:
    """Shared request body builder for http_brain_call and http_brain_call_stream."""
    body_dict = {
        "prompt": _frame_for_call(prompt, node_id, speaker, speaker_score),
        "session_id": session_id,
        "node_id": node_id,
        "target": _call_target(node_id),
    }
    enable_vision = is_vision_enabled()
    if enable_vision:
        ctx = getattr(_FRAME_CTX, "ctx", None) or {}
        img = image_b64 or ctx.get("image_b64")
        burst = burst_images_b64 or ctx.get("burst_images_b64")
        if img:
            body_dict["image_b64"] = img
        if burst:
            body_dict["burst_images_b64"] = burst
    brain_direct_audio = bool(BRAIN_DIRECT_AUDIO) or (os.environ.get("MVG_BRAIN_DIRECT_AUDIO", "0") == "1")
    if brain_direct_audio:
        ctx = getattr(_FRAME_CTX, "ctx", None) or {}
        aud = audio_b64 or ctx.get("audio_b64")
        if aud:
            body_dict["audio_b64"] = aud
    return body_dict, json.dumps(body_dict).encode("utf-8")


def http_brain_call(prompt: str, session_id: str, node_id: str, speaker: str,
                    speaker_score, timeout: int,
                    image_b64: str | None = None, burst_images_b64: list[str] | None = None,
                    audio_b64: str | None = None) -> str:
    """Brain over plain HTTP (standalone). POSTs the framed prompt as JSON to
    MVG_BRAIN_URL and returns the response's "reply" string.

    When MVG_HTTP_STREAM is enabled, also accepts and consumes text/event-stream.
    """
    if not MVG_BRAIN_URL:
        raise RuntimeError("MVG_BRAIN_URL not set")
    is_stream = is_http_stream_enabled()
    _, body = build_http_brain_body(prompt, session_id, node_id, speaker, speaker_score,
                                    image_b64=image_b64, burst_images_b64=burst_images_b64,
                                    audio_b64=audio_b64)
    headers = {"Content-Type": "application/json"}
    if is_stream:
        headers["Accept"] = "text/event-stream, application/json"
    if MVG_BRAIN_TOKEN:
        headers["Authorization"] = f"Bearer {MVG_BRAIN_TOKEN}"
    req = Request(MVG_BRAIN_URL, data=body, headers=headers, method="POST")
    with urlopen(req, timeout=timeout) as resp:
        headers_obj = getattr(resp, "headers", None)
        content_type = headers_obj.get("Content-Type", "") if hasattr(headers_obj, "get") else ""
        if is_stream and "text/event-stream" in content_type:
            return consume_brain_sse_lines(resp, on_partial=None, max_stream_seconds=timeout)
        data = json.loads(resp.read().decode("utf-8"))
    reply = data.get("reply")
    if not isinstance(reply, str):
        raise RuntimeError("MVG_BRAIN_URL response has no reply string")
    return reply


def http_brain_call_stream(prompt: str, session_id: str, node_id: str, speaker: str,
                           speaker_score, timeout: int, on_partial,
                           image_b64: str | None = None, burst_images_b64: list[str] | None = None,
                           audio_b64: str | None = None) -> str:
    """Streaming shape for the http brain: incrementally consumes SSE lines via
    consume_brain_sse_lines when HTTP_BRAIN_STREAM is enabled (or Content-Type
    is text/event-stream); otherwise preserves the standard JSON path."""
    if not MVG_BRAIN_URL:
        raise RuntimeError("MVG_BRAIN_URL not set")

    is_stream = is_http_stream_enabled()
    _, body = build_http_brain_body(prompt, session_id, node_id, speaker, speaker_score,
                                    image_b64=image_b64, burst_images_b64=burst_images_b64,
                                    audio_b64=audio_b64)
    headers = {"Content-Type": "application/json"}
    if is_stream:
        headers["Accept"] = "text/event-stream, application/json"
    if MVG_BRAIN_TOKEN:
        headers["Authorization"] = f"Bearer {MVG_BRAIN_TOKEN}"

    req = Request(MVG_BRAIN_URL, data=body, headers=headers, method="POST")
    with urlopen(req, timeout=timeout) as resp:
        headers_obj = getattr(resp, "headers", None)
        content_type = headers_obj.get("Content-Type", "") if hasattr(headers_obj, "get") else ""
        if "text/event-stream" in content_type or (is_stream and "application/json" not in content_type):
            reply = consume_brain_sse_lines(resp, on_partial=on_partial, max_stream_seconds=timeout)
        else:
            data = json.loads(resp.read().decode("utf-8"))
            reply = data.get("reply")
            if not isinstance(reply, str):
                raise RuntimeError("MVG_BRAIN_URL response has no reply string")
            on_partial(reply)
    return reply


# Overridable by tests so no test run touches the real brain transport.
BRAIN_CALL = http_brain_call if MVG_BRAIN == "http" else plugin_brain_call


def _accepts_kwarg(fn, name: str) -> bool:
    """True if fn takes a keyword `name` (or **kwargs). Checked per call so a
    patched hook is honoured; a signature that cannot be read counts as no."""
    try:
        params = inspect.signature(fn).parameters
    except (ValueError, TypeError):
        return False
    return name in params or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())


def plugin_brain_call_stream(prompt: str, session_id: str, node_id: str, speaker: str,
                             speaker_score, timeout: int, on_partial,
                             task_id: str | None = None,
                             audio_clips: list[bytes] | None = None) -> str:
    """Streaming brain through the plugin. With a plugin brain_call_stream, that
    hook gets on_partial and the per-turn stage sink. Without one, the plugin's
    plain brain_call answers and on_partial fires once with the full reply."""
    if _plugin_brain_call_stream is None:
        reply = plugin_brain_call(prompt, session_id, node_id, speaker, speaker_score, timeout)
        on_partial(reply)
        return reply
    wrapped = _frame_for_call(prompt, node_id, speaker, speaker_score)
    extra = {}
    if task_id is not None and _accepts_kwarg(_plugin_brain_call_stream, "task_id"):
        extra["task_id"] = task_id
    if audio_clips is not None and _accepts_kwarg(_plugin_brain_call_stream, "audio_clips"):
        extra["audio_clips"] = audio_clips
    return _plugin_brain_call_stream(
        _call_target(node_id), wrapped, timeout, on_partial,
        getattr(_STAGE_SINK, "stages", None), **extra,
    )


# Overridable by tests so no test run touches the real brain transport.
STREAM_BRAIN_CALL = http_brain_call_stream if MVG_BRAIN == "http" else plugin_brain_call_stream

# Per-thread sink for brain-call stage timestamps (the turn record's `stages`).
# A thread-local rather than a new STREAM_BRAIN_CALL argument so every
# existing override of that hook keeps its signature.
_STAGE_SINK = threading.local()

# Active brain call per node: {node_id: {"task_id", "target", "started", "cancel_event", "wake"}}.
# A turn registers on entry and always removes its entry on exit (success, error, cancel).
_ACTIVE_TASKS: dict = {}
_ACTIVE_TASKS_LOCK = threading.Lock()


def _register_task(node_id: str, target: str, task_id: str | None = None) -> dict:
    task = {"task_id": task_id or uuid.uuid4().hex[:12], "target": target, "started": time.time(),
            "cancel_event": threading.Event(), "wake": None}
    with _ACTIVE_TASKS_LOCK:
        _ACTIVE_TASKS[node_id or ""] = task
    return task


def _unregister_task(node_id: str, task: dict) -> None:
    with _ACTIVE_TASKS_LOCK:
        if _ACTIVE_TASKS.get(node_id or "") is task:
            del _ACTIVE_TASKS[node_id or ""]


def _resolve_talker_instructions(target: str, node_id: str) -> str | None:
    """Resolve system instructions for a live Talker session.

    Checks the plugin's `talker_instructions(target, node_id)` hook first, then
    falls back to `<MVG_PERSONA_DIR>/<target>/voice_talker.md` followed by
    `<MVG_PERSONA_DIR>/<target>/voice.md`."""
    hook = _plugin_talker_instructions
    if hook is None:
        for mod in (_PLUGIN_MOD, _plugin):
            if mod is not None:
                hook = getattr(mod, "talker_instructions", None)
                if callable(hook):
                    break
    if callable(hook):
        try:
            res = hook(target, node_id)
            if isinstance(res, str) and res.strip():
                return res.strip()
        except Exception:
            pass
    persona_dir = os.environ.get("MVG_PERSONA_DIR", "").strip()
    target_clean = (target or "").strip()
    if persona_dir and target_clean and "/" not in target_clean and "\\" not in target_clean and ".." not in target_clean and target_clean != ".":
        base = Path(persona_dir) / target_clean
        for filename in ("voice_talker.md", "voice.md"):
            p = base / filename
            try:
                if p.is_file():
                    text = p.read_text(encoding="utf-8").strip()
                    if text:
                        return text
            except OSError:
                continue
    return None


def cancel_active_turn(node_id: str) -> str:
    """Cancel the node's running brain turn. Returns "cancelled" (the plugin's
    brain_cancel hook ran and returned truthy), "stopped_speech_only" (no hook,
    or it returned falsy or raised: the turn stops speaking and drops the
    result, the brain may still be running) or "no_active_turn" (nothing
    running, or already cancelled)."""
    with _ACTIVE_TASKS_LOCK:
        task = _ACTIVE_TASKS.get(node_id or "")
        if task is None or task["cancel_event"].is_set():
            return "no_active_turn"
        task["cancel_event"].set()
        target, task_id, wake = task["target"], task["task_id"], task["wake"]
    if wake is not None:
        wake.set()
    hook = _plugin_brain_cancel
    if hook is None:
        return "stopped_speech_only"
    try:
        return "cancelled" if hook(target, task_id) else "stopped_speech_only"
    except Exception as exc:
        log_fallback(f"brain_cancel_failed:{str(exc)[:200]}", tts_engine="cancel-skip")
        return "stopped_speech_only"

STAGE_DETAIL_MAX = 120


def normalize_stage_event(raw) -> dict | None:
    """Normalise a brain stage event to `{"stage": str, "detail": str, "t": float}`.

    For consumers of stage events (progress narration); it does not touch what
    is written to the turn record's `stages`. Accepts a dict with `stage` (or
    `name` / `event`), optional `detail` (or `message` / `text`), optional `t`
    in seconds (or `t_s`, or `t_ms` / `ms` in milliseconds). Missing detail
    becomes "", missing or bad `t` becomes 0.0, `detail` is cut to
    STAGE_DETAIL_MAX characters. Anything that is not a dict, or has no usable
    stage name, gives None."""
    if not isinstance(raw, dict):
        return None
    stage = ""
    for key in ("stage", "name", "event"):
        v = raw.get(key)
        if isinstance(v, str) and v.strip():
            stage = v.strip()
            break
    if not stage:
        return None
    detail = ""
    for key in ("detail", "message", "text"):
        v = raw.get(key)
        if isinstance(v, str) and v.strip():
            detail = " ".join(v.split())[:STAGE_DETAIL_MAX]
            break
    t = 0.0
    for key, scale in (("t", 1.0), ("t_s", 1.0), ("t_ms", 0.001), ("ms", 0.001)):
        v = raw.get(key)
        if isinstance(v, (int, float)) and not isinstance(v, bool) and v >= 0:
            t = float(v) * scale
            break
    return {"stage": stage, "detail": detail, "t": t}


# --- Spoken progress (MVG_SPOKEN_PROGRESS=1) ----------------------------------

def context_opener_enabled() -> bool:
    return os.environ.get("MVG_CONTEXT_OPENER", "0").strip().lower() in ("1", "true", "on", "yes")


def spoken_progress_enabled() -> bool:
    return os.environ.get("MVG_SPOKEN_PROGRESS", "0").strip().lower() in ("1", "true", "on", "yes")


def _env_num(name: str, default, cast):
    try:
        return cast(os.environ.get(name, "").strip() or default)
    except ValueError:
        return cast(default)


class ProgressNarrator:
    """Decides when a stage event becomes one spoken progress line. Pure: the
    caller supplies the clock and the playback / reply state. A line is allowed
    only when nothing is playing, the real reply has not started, the event's
    stage is allowlisted, at least min_gap seconds have passed since the last
    line, and fewer than max_lines have been spoken. The newest unseen eligible
    event wins; older unseen ones are dropped, never queued."""

    def __init__(self, events_source, clock=time.time, min_gap=None, max_lines=None, allow=None):
        self.source = events_source  # callable -> list of raw events
        self.clock = clock
        self.min_gap = min_gap if min_gap is not None else _env_num("MVG_PROGRESS_MIN_GAP_SEC", "12", float)
        self.max_lines = max_lines if max_lines is not None else _env_num("MVG_PROGRESS_MAX_PER_TURN", "3", int)
        raw = os.environ.get("MVG_PROGRESS_STAGES", "tool_start,milestone")
        self.allow = frozenset(allow if allow is not None else (x.strip() for x in raw.split(",") if x.strip()))
        self.seen = 0
        self.lines = 0
        self.last_spoken = None

    def next_line(self, playing: bool, reply_started: bool):
        try:
            events = list(self.source() or [])
        except Exception:
            return None
        fresh, self.seen = events[self.seen:], len(events)
        if reply_started or playing or self.lines >= self.max_lines:
            return None
        now = self.clock()
        if self.last_spoken is not None and now - self.last_spoken < self.min_gap:
            return None
        for raw in reversed(fresh):
            ev = normalize_stage_event(raw)
            if ev and ev["stage"] in self.allow and ev["detail"]:
                self.lines += 1
                self.last_spoken = now
                return ev["detail"]
        return None

APOLOGY = f"{_display_name(AGENT_NAME)} is busy right now, try again in a moment."


# --- Spoken cancel (MVG_SPOKEN_CANCEL=1) -------------------------------------

_DEFAULT_CANCEL_PHRASES = ("stop", "cancel", "cancel that", "never mind", "nevermind", "stop that")
_CANCEL_ACKS = {
    "cancelled": "Cancelled.",
    "stopped_speech_only": "Stopped talking, but the task may still be running.",
}


def _cancel_phrases() -> frozenset:
    raw = os.environ.get("MVG_CANCEL_PHRASES", "")
    items = [_norm_cancel_text(p) for p in raw.split(",")] if raw.strip() else list(_DEFAULT_CANCEL_PHRASES)
    return frozenset(p for p in items if p)


def _norm_cancel_text(text: str) -> str:
    cleaned = "".join(c if (c.isalnum() or c.isspace() or c == "'") else " " for c in (text or "").lower())
    return " ".join(cleaned.replace("'", "").split())


def is_cancel_phrase(text: str) -> bool:
    """True when the transcript is a spoken cancel: an exact match on the phrase
    set, or at most 3 words starting with "stop" or "cancel". Never a substring
    match, so "stop the deploy script from restarting" is not a cancel."""
    norm = _norm_cancel_text(text)
    if not norm:
        return False
    if norm in _cancel_phrases():
        return True
    words = norm.split()
    return len(words) <= 3 and words[0] in ("stop", "cancel")


def spoken_cancel_enabled() -> bool:
    return os.environ.get("MVG_SPOKEN_CANCEL", "0").strip().lower() in ("1", "true", "on", "yes")


# --- TTS ---------------------------------------------------------------------

def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _now() -> float:
    """Indirection so tests can move the clock without patching time.time
    everywhere synthesize() reads it."""
    return time.time()


def _piper_binary_path() -> str:
    """The piper executable: piper.binary in PIPER_CONFIG_PATH, else the default."""
    piper_cfg = _load_json(PIPER_CONFIG_PATH).get("piper", {})
    return os.path.expanduser(piper_cfg.get("binary", PIPER_BINARY_DEFAULT))


def piper_speak(text: str) -> bytes:
    """Returns WAV bytes from local Piper. Raises on any failure."""
    cfg = _load_json(PIPER_CONFIG_PATH)
    piper_cfg = cfg.get("piper", {})
    binary = _piper_binary_path()
    voices_dir = Path(os.path.expanduser(piper_cfg.get("voices_dir", PIPER_VOICES_DIR_DEFAULT)))
    model_name = (cfg.get("voice_mapping") or {}).get(AGENT_LABEL) or cfg.get(
        "default_voice", PIPER_VOICE_MODEL_DEFAULT
    )
    model_path = voices_dir / f"{model_name}.onnx"
    config_json_path = voices_dir / f"{model_name}.onnx.json"

    if not os.path.exists(binary):
        raise RuntimeError(f"piper binary not found: {binary}")
    if not model_path.exists():
        raise RuntimeError(f"piper voice model not found: {model_path}")

    tmp_fd, tmp_wav = tempfile.mkstemp(suffix=".wav", prefix="mvg-piper-")
    os.close(tmp_fd)
    try:
        cmd = [binary, "-m", str(model_path), "-f", tmp_wav]
        if config_json_path.exists():
            cmd += ["-c", str(config_json_path)]
        proc = subprocess.run(
            cmd, input=text, capture_output=True, text=True, timeout=PIPER_TIMEOUT
        )
        if proc.returncode != 0:
            raise RuntimeError(f"piper exited {proc.returncode}: {proc.stderr[:300]}")
        data = Path(tmp_wav).read_bytes()
        if not data:
            raise RuntimeError("piper produced an empty file")
        return data
    finally:
        try:
            os.unlink(tmp_wav)
        except OSError:
            pass


def desktop_tts_speak(text: str) -> tuple[bytes, str]:
    """Returns (wav_bytes, engine) from the desktop's TTS server, reached
    over the ssh reverse tunnel at MVG_DESKTOP_TTS_URL (default
    http://127.0.0.1:9097/tts on this box's own loopback). Raises on any
    failure, including a non-WAV body (a tunnel that's up but proxying to
    the wrong thing, or an error page, would otherwise look like success).

    `engine` comes from the desktop server's own X-TTS-Engine response
    header — that server now tries Chatterbox
    (the agent's cloned voice) before falling back in-process to Piper, so a
    hardcoded "piper-desktop" label here would misreport which one
    actually spoke. Falls back to "piper-desktop" if the header is
    missing, for an older desktop server that predates this change."""
    payload = json.dumps({"text": text}).encode("utf-8")
    req = Request(
        DESKTOP_TTS_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(req, timeout=DESKTOP_TTS_TIMEOUT) as resp:
        data = resp.read()
        engine = resp.headers.get("X-TTS-Engine") or "piper-desktop"
    if not data or not data.startswith(b"RIFF"):
        raise RuntimeError("desktop tts returned a non-WAV or empty body")
    return data, f"{engine}-desktop" if engine == "piper" else engine


# --- Speaker embedding proxy (SPEC-011) --------------------------------------

def proxy_embed(wav_bytes: bytes) -> tuple[int, bytes]:
    """POSTs wav_bytes to DESKTOP_EMBED_URL (desktop_embed.py, reached over
    the same ssh reverse tunnel desktop_tts_speak uses) and returns
    (status_code, response_body_bytes) straight through.

    An HTTPError is the upstream's OWN response (e.g. 400 for empty/too-short
    audio, 503 while its model is still loading) and is passed through
    exactly — a blanket except->502 here would swallow that distinction and
    hide it from the hub as a generic "gateway can't reach you" failure.
    Only an unreachable upstream (dead tunnel, connection refused, timeout)
    becomes a 502 from THIS gateway."""
    req = Request(
        DESKTOP_EMBED_URL, data=wav_bytes,
        headers={"Content-Type": "audio/wav"}, method="POST",
    )
    try:
        with urlopen(req, timeout=DESKTOP_EMBED_TIMEOUT) as resp:
            return resp.status, resp.read()
    except HTTPError as exc:
        return exc.code, exc.read()
    except Exception as exc:
        return 502, json.dumps({"error": f"embed upstream unreachable: {exc}"}).encode("utf-8")


def proxy_systemone(body: bytes) -> tuple[int, bytes]:
    """POSTs a Jev /v1/systemone JSON body to DESKTOP_LAYA_URL (the desktop
    Laya classifier, laya.serve, reached over the same ssh reverse tunnel
    desktop_tts_speak uses) and returns (status_code, response_body_bytes).

    Callers get the upstream's own status: a 2xx is the classifier's answer,
    an HTTPError is the upstream's OWN response and is passed through exactly.
    Only an unreachable upstream (dead tunnel, connection refused, timeout)
    becomes a 502 from THIS gateway. This is a straight proxy with no retry
    and no fallback layer; the callers' own layering decides what to do on
    failure."""
    req = Request(
        DESKTOP_LAYA_URL, data=body,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urlopen(req, timeout=DESKTOP_LAYA_TIMEOUT) as resp:
            return resp.status, resp.read()
    except HTTPError as exc:
        return exc.code, exc.read()
    except Exception as exc:
        return 502, json.dumps({"error": f"laya upstream unreachable: {exc}"}).encode("utf-8")


def _desktop_backend(text: str) -> bytes:
    """desktop_tts_speak() as a chain backend: WAV only. synthesize() calls
    desktop_tts_speak() itself for this entry, to keep the engine label."""
    return desktop_tts_speak(text)[0]


# Swappable TTS backends (ERED LOMIN P1b). Every entry is (text) -> wav bytes.
# The lambdas look up piper_speak / desktop_tts_speak at call time, so a test
# or a monkeypatch of those module names still takes effect.
TTS_BACKENDS = {
    "desktop": _desktop_backend,
    "piper": lambda text: piper_speak(text),
    "elevenlabs": lambda text: tts_backends.elevenlabs_speak(text),
    "openai": lambda text: tts_backends.openai_speech_speak(text),
    "http": lambda text: tts_backends.http_tts_speak(text),
    "gemini": lambda text: tts_backends.gemini_speak(text),
    "kokoro": lambda text: tts_backends.kokoro_speak(text),
}
# Reason labels for the fallback log. "desktop-tts" is the label this chain
# has always logged; every other entry is its own name.
_TTS_REASON_LABEL = {"desktop": "desktop-tts"}
TTS_CHAIN_DEFAULT = "desktop,piper"


def _tts_chain_from_env() -> list[str]:
    """MVG_TTS_CHAIN (comma list, default "desktop,piper"), read now. Unknown
    names are dropped with one stderr line. An empty result is ["piper"]."""
    raw = os.environ.get("MVG_TTS_CHAIN", TTS_CHAIN_DEFAULT)
    names = [n.strip().lower() for n in raw.split(",") if n.strip()]
    unknown = [n for n in names if n not in TTS_BACKENDS]
    if unknown:
        print(f"gateway: MVG_TTS_CHAIN: unknown backend(s) dropped: {', '.join(unknown)}",
              file=sys.stderr)
    chain = [n for n in names if n in TTS_BACKENDS]
    return chain or ["piper"]


TTS_CHAIN = _tts_chain_from_env()


def synthesize(text: str) -> tuple[bytes, str]:
    """Returns (wav_bytes, engine) — always WAV regardless of what the
    caller asked for (the edge client's `aplay -` cannot decode mp3).

    Walks TTS_CHAIN in order (default: desktop TTS, the Chatterbox clone of
    the agent's voice that itself falls back to desktop Piper, then this box's own local
    Piper as the last resort). The first success wins and returns its engine
    name; the desktop entry keeps the label desktop_tts_speak() returns. Each
    failure is recorded as "<name>:<error>". If an entry after the first one
    speaks, the failures are logged with log_fallback(). If every entry fails,
    raises RuntimeError with all the reasons."""
    text = strip_defer_marker(text)
    reasons = []
    for idx, name in enumerate(TTS_CHAIN):
        try:
            if name == "desktop" and TTS_BACKENDS.get("desktop") is _desktop_backend:
                wav, engine = desktop_tts_speak(text)
            else:
                wav, engine = TTS_BACKENDS[name](text), name
        except Exception as exc:
            reasons.append(f"{_TTS_REASON_LABEL.get(name, name)}:{str(exc)[:200]}")
            continue
        if idx > 0:
            log_fallback("; ".join(reasons), tts_engine=name)
        return wav, engine
    raise RuntimeError("; ".join(reasons))


# --- sub-sentence PCM streaming (/ask/stream) --------------------------------

def _desktop_tts_stream_url() -> str:
    if DESKTOP_TTS_STREAM_URL:
        return DESKTOP_TTS_STREAM_URL
    return DESKTOP_TTS_URL.rstrip("/") + "/stream"


def streaming_wav_header(sample_rate: int, channels: int = 1) -> bytes:
    """44-byte canonical PCM WAV header for a stream of unknown length (RIFF
    and data sizes 0xFFFFFFFF). Exactly 44 bytes with "data" at offset 36, no
    extra chunks: the hub's TTS client only treats a body as WAV if so."""
    return (b"RIFF" + struct.pack("<I", 0xFFFFFFFF) + b"WAVEfmt "
            + struct.pack("<IHHIIHH", 16, 1, channels, sample_rate,
                          sample_rate * channels * 2, channels * 2, 16)
            + b"data" + struct.pack("<I", 0xFFFFFFFF))


def _pcm_slices_from_wav(wav_bytes: bytes, slice_bytes: int = 9600):
    """A full WAV (any engine/rate) as 24 kHz mono s16le slices, so a turn
    that latched to pcm never switches format. Raises on an unparseable WAV."""
    wav_bytes = normalize_wav(wav_bytes)
    with wave.open(BytesIO(wav_bytes), "rb") as w:
        raw = w.readframes(w.getnframes())
    if not raw:
        raise RuntimeError("empty wav")
    return [raw[i:i + slice_bytes] for i in range(0, len(raw), slice_bytes)]


def _pcm_body_iter(resp, first: bytes):
    """Yield even-length coalesced PCM chunks from an EOF-terminated body.
    Closes `resp` when exhausted or abandoned (closing the socket is what
    tells the desktop to stop synthesizing)."""
    carry = first
    try:
        while True:
            if len(carry) >= max(2, TTS_STREAM_MIN_CHUNK_BYTES):
                cut = len(carry) - (len(carry) % 2)
                yield carry[:cut]
                carry = carry[cut:]
                continue
            piece = resp.read1(8192)
            if not piece:
                break
            carry += piece
        cut = len(carry) - (len(carry) % 2)
        if cut:
            yield carry[:cut]
    finally:
        try:
            resp.close()
        except Exception:
            pass


def _wav_as_pcm_stream(text: str, wav: bytes, engine: str):
    """(engine, iterator of PCM chunks) for a WAV, re-running the full
    synthesize() chain if this WAV cannot be converted."""
    try:
        chunks = _pcm_slices_from_wav(wav)
    except Exception as exc:
        log_fallback(f"stream_wav_convert_failed:{str(exc)[:200]}", tts_engine="stream-wav")
        wav, engine = synthesize(text)
        chunks = _pcm_slices_from_wav(wav)
    return engine, iter(chunks)


def synthesize_stream(text: str):
    """Returns (engine, iterator of PCM16 mono 24 kHz byte chunks).

    Tries the desktop's POST /tts/stream {"format": "pcm"}; the first chunk
    is read before returning, so any failure up to first audio falls back
    to synthesize() (desktop /tts, Piper) with the result
    converted to PCM. A desktop that answers the stream request with a full
    WAV (Chatterbox unavailable) is handled the same way. Raises only if
    every engine fails."""
    text = strip_defer_marker(text)
    if not TTS_CHAIN or TTS_CHAIN[0] != "desktop":
        # Desktop streaming is not first in the chain: take the same path as
        # a failed desktop stream, so the chain (synthesize) decides.
        wav, engine = synthesize(text)
        return _wav_as_pcm_stream(text, wav, engine)
    resp = None
    try:
        req = Request(
            _desktop_tts_stream_url(),
            data=json.dumps({"text": text, "format": "pcm"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        resp = urlopen(req, timeout=DESKTOP_TTS_TIMEOUT)
        ctype = (resp.headers.get("Content-Type") or "").lower()
        engine = resp.headers.get("X-TTS-Engine") or "chatterbox-stream"
        if ctype.startswith("audio/l16"):
            rate = int(resp.headers.get("X-Sample-Rate") or AUDIO_TARGET_RATE)
            if rate != AUDIO_TARGET_RATE:
                raise RuntimeError(f"desktop stream rate {rate} != {AUDIO_TARGET_RATE}")
            first = resp.read1(8192)
            if not first:
                raise RuntimeError("desktop stream closed before first audio")
            it = _pcm_body_iter(resp, first)
            resp = None  # now owned by the iterator
            return engine, it
        data = resp.read()
        if not data.startswith(b"RIFF"):
            raise RuntimeError("desktop stream returned a non-WAV, non-PCM body")
        return _wav_as_pcm_stream(
            text, data, f"{engine}-desktop" if engine == "piper" else engine)
    except Exception as exc:
        log_fallback(f"desktop_stream_failed:{str(exc)[:200]}", tts_engine="stream-fallback")
    finally:
        if resp is not None:
            try:
                resp.close()
            except Exception:
                pass
    wav, engine = synthesize(text)
    return _wav_as_pcm_stream(text, wav, engine)


# --- sentence-parallel synthesis (non-streaming callers) --------------------
# MANDOS build (specs/2026-09-28-mandos-streaming.md): "sentence-chunk the
# TTS server itself for any long single input, so even non-streaming callers
# get sentence-parallel synthesis where that's faster." Both TTS engines in
# the chain (Chatterbox, desktop_tts.py's own comment says so explicitly;
# Piper measured similarly) have latency that scales with OUTPUT length, so
# N sentences synthesized concurrently and concatenated generally beats one
# serial call over their sum.

SENTENCE_CHUNK_MIN_CHARS = int(os.environ.get("MVG_SENTENCE_CHUNK_MIN_CHARS", "120"))
SENTENCE_CHUNK_MAX_WORKERS = int(os.environ.get("MVG_SENTENCE_CHUNK_MAX_WORKERS", "4"))


def _concat_wavs(wav_list: list[bytes]) -> bytes:
    """Concatenate same-format WAV byte-strings, in order, into one WAV.
    Raises ValueError on any format mismatch (different sample rate/width/
    channel count between chunks) — the caller treats that as "don't risk
    a garbled/mis-pitched result" and falls back to a single serial call."""
    params = None
    frames = []
    for wav_bytes in wav_list:
        with wave.open(BytesIO(wav_bytes), "rb") as w:
            p = (w.getnchannels(), w.getsampwidth(), w.getframerate())
            if params is None:
                params = p
            elif p != params:
                raise ValueError(f"wav format mismatch: {p} != {params}")
            frames.append(w.readframes(w.getnframes()))
    if params is None:
        raise ValueError("no wav chunks to concatenate")
    buf = BytesIO()
    with wave.open(buf, "wb") as out:
        out.setnchannels(params[0])
        out.setsampwidth(params[1])
        out.setframerate(params[2])
        out.writeframes(b"".join(frames))
    return buf.getvalue()


def synthesize_chunked(text: str) -> tuple[bytes, str]:
    """Sentence-parallel variant of synthesize(): for text long enough to
    be worth splitting, synthesize each sentence concurrently and
    concatenate. Falls back to the plain single-shot synthesize() for
    short text (not worth thread overhead), a single-sentence input, or
    when the concurrent calls landed on inconsistent engines/formats
    across sentences (rare — only if a fallback boundary, like the desktop
    TTS failing mid-request and Piper taking over, is crossed between
    sentences) — that fallback re-synthesizes the whole text serially, so
    it's always correct, just not always faster."""
    if len(text) < SENTENCE_CHUNK_MIN_CHARS:
        return synthesize(text)
    sentences = segment_full_text(text)
    if len(sentences) <= 1:
        return synthesize(text)
    with ThreadPoolExecutor(max_workers=min(SENTENCE_CHUNK_MAX_WORKERS, len(sentences))) as ex:
        results = list(ex.map(synthesize, sentences))
    engines = {engine for _, engine in results}
    try:
        combined = _concat_wavs([wav for wav, _ in results])
    except ValueError as exc:
        log_fallback(f"chunked_concat_mismatch:{exc}", tts_engine="chunked-fallback")
        return synthesize(text)
    engine_label = results[0][1] if len(engines) == 1 else "mixed:" + "+".join(sorted(engines))
    return combined, engine_label


# --- HTTP handler ------------------------------------------------------------

class _BodyRejected(Exception):
    """Raised by GatewayHandler._read_body after it has sent a 400/413."""


class GatewayHandler(BaseHTTPRequestHandler):
    server_version = "LominVoiceGateway/1.0"
    protocol_version = "HTTP/1.0"  # no Content-Length on /ask -> EOF ends the stream

    def log_message(self, *_args):
        pass

    # -- helpers --

    def _read_body(self) -> bytes:
        """Read the request body, refusing a missing-or-bad or oversized
        Content-Length before reading it (a wedged mic stream must not be able
        to OOM the gateway). Raises _BodyRejected after sending the error."""
        raw = self.headers.get("Content-Length", 0) or 0
        try:
            length = int(raw)
        except (TypeError, ValueError):
            length = -1
        if length < 0:
            self._reject_body(400, "bad Content-Length")
        if length > MAX_BODY_BYTES:
            self._reject_body(413, f"body too large ({length} > {MAX_BODY_BYTES} bytes)")
        return self.rfile.read(length) if length else b""

    def _reject_body(self, code: int, msg: str) -> None:
        self.close_connection = True  # the unread body stays on the socket
        self._send_json(code, {"error": msg})
        raise _BodyRejected(msg)

    def _send_json(self, code: int, obj: dict) -> None:
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _send_bytes(self, code: int, body: bytes, content_type: str, extra_headers=None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (extra_headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _get_client_id(self, node_id: str = "") -> str:
        if hasattr(self, "client_address") and self.client_address and isinstance(self.client_address, (tuple, list)):
            ip = str(self.client_address[0]).strip()
            if ip:
                return ip
        if hasattr(self, "headers") and self.headers:
            xff = self.headers.get("X-Forwarded-For")
            if xff:
                ip = xff.split(",")[0].strip()
                if ip:
                    return ip
        return node_id or "127.0.0.1"

    def _check_bearer_auth(self, node_id: str = "", endpoint: str = "", t0: float = 0.0) -> bool:
        verifier = get_device_token_verifier()
        if not verifier.is_enabled:
            return True
        client_id = self._get_client_id(node_id)
        auth_header = self.headers.get("Authorization", "")
        if verifier.verify(auth_header, node_id):
            return True
        if verifier.is_locked_out(client_id):
            self._send_json(429, {"error": "too many failed attempts, locked out"})
            if t0:
                log_request(endpoint, (time.time() - t0) * 1000, "error:locked_out", node=node_id)
            return False
        verifier.record_failure(client_id)
        self._send_json(401, {"error": "unauthorized"})
        if t0:
            log_request(endpoint, (time.time() - t0) * 1000, "error:unauthorized", node=node_id)
        return False

    # -- routes --

    def do_GET(self):
        path = urlsplit(self.path).path
        if path == "/health":
            return self._handle_health()
        if path == "/voice/pending" and is_defer_enabled():
            return self._handle_pending_get()
        if path == "/voice/live":
            return self._handle_voice_live()
        self._send_json(404, {"error": "not found"})

    def do_POST(self):
        try:
            return self._dispatch_post()
        except _BodyRejected:
            return

    def _dispatch_post(self):
        path = urlsplit(self.path).path
        if path == "/stt":
            return self._handle_stt()
        if path == "/ambient":
            return self._handle_ambient()
        if path == "/ask":
            return self._handle_ask()
        if path == "/ask/stream":
            return self._handle_ask_stream()
        if path == "/intent":
            return self._handle_intent()
        if path in ("/voice/turn", "/api/voice/interact"):
            # /api/voice/interact is the hub's path: an ear whose hub_url
            # points here keeps deriving a working heartbeat URL from it.
            return self._handle_voice_turn()
        if path == "/api/voice/heartbeat":
            self._read_body()
            self.send_response(204)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if path == "/v1/audio/speech":
            return self._handle_speech()
        if path == "/embed":
            return self._handle_embed()
        if path == "/v1/systemone":
            return self._handle_systemone()
        if path == "/voice/pending" and is_defer_enabled():
            return self._handle_pending_post()
        self._send_json(404, {"error": "not found"})

    # -- /health --

    def _handle_health(self):
        t0 = time.time()
        upstreams = {"whisper": whisper_reachable()}
        piper_cfg = _load_json(PIPER_CONFIG_PATH).get("piper", {})
        piper_binary = os.path.expanduser(piper_cfg.get("binary", PIPER_BINARY_DEFAULT))
        upstreams["piper_binary_present"] = os.path.exists(piper_binary)
        self._send_json(200, {"status": "ok", "upstreams": upstreams})
        log_request("/health", (time.time() - t0) * 1000, "ok")

    # -- /voice/pending (deferred async voice answers) --

    def _handle_pending_get(self):
        t0 = time.time()
        parsed = urlsplit(self.path)
        qs = parse_qs(parsed.query)
        node = (qs.get("node") or qs.get("node_id") or [""])[0].strip()
        if not self._check_bearer_auth(node, "/voice/pending", t0):
            return
        if not node:
            self._send_json(400, {"error": "node parameter required"})
            log_request("/voice/pending", (time.time() - t0) * 1000, "error:no_node")
            return

        pop_param = qs.get("pop", ["1"])[0].lower()
        peek_param = qs.get("peek", ["0"])[0].lower()
        should_pop = pop_param in ("1", "true", "yes") and peek_param not in ("1", "true", "yes")

        store = get_pending_store()
        item = store.pop_for_node(node) if should_pop else store.peek_for_node(node)
        has_pending = item is not None
        self._send_json(200, {
            "status": "ok",
            "node": node,
            "has_pending": has_pending,
            "pending": item,
            "item": item,
            "popped": should_pop and has_pending,
        })
        log_request("/voice/pending", (time.time() - t0) * 1000, "ok", node=node, has_pending=has_pending)

    def _handle_pending_post(self):
        t0 = time.time()
        body = self._read_body()
        try:
            req = json.loads(body.decode("utf-8")) if body else {}
        except Exception:
            req = {}
        node = (req.get("node_id") or req.get("node") or "").strip()
        if not self._check_bearer_auth(node, "/voice/pending", t0):
            return
        answer = (req.get("answer") or "").strip()
        prompt = (req.get("prompt") or "").strip()
        topic = (req.get("topic") or "").strip()

        if not node:
            self._send_json(400, {"error": "node_id required"})
            log_request("/voice/pending", (time.time() - t0) * 1000, "error:no_node")
            return
        if not answer:
            self._send_json(400, {"error": "answer required"})
            log_request("/voice/pending", (time.time() - t0) * 1000, "error:no_answer", node=node)
            return

        store = get_pending_store()
        item = store.store(node_id=node, prompt=prompt, answer=answer, topic=topic)
        self._send_json(200, {
            "status": "ok",
            "stored": True,
            "pending": item,
            "item": item,
            "node": node,
        })
        log_request("/voice/pending", (time.time() - t0) * 1000, "ok", node=node)

    # -- /stt --

    def _handle_stt(self):
        t0 = time.time()
        content_type = self.headers.get("Content-Type", "")
        body = self._read_body()
        wav_bytes = None
        node_id = (self.headers.get("X-Node-Id") or
                   (parse_qs(urlsplit(self.path).query).get("node_id") or [""])[0])
        if content_type.startswith("multipart/form-data"):
            parts = parse_multipart_parts(body, content_type)
            wav_bytes = parts.get("audio")
            node_id = node_id or parts.get("node_id", b"").decode("utf-8", "replace").strip()
        elif content_type in ("audio/wav", "application/octet-stream", "") :
            wav_bytes = body
        _set_req_node(node_id)
        if not self._check_bearer_auth(node_id, "/stt", t0):
            return
        if not wav_bytes:
            self._send_json(400, {"error": "no audio payload found"})
            log_request("/stt", (time.time() - t0) * 1000, "error:no_audio")
            return

        sha = hashlib.sha256(wav_bytes).hexdigest()
        cached = _stt_cache_get(sha)
        if cached is not None:
            ms = (time.time() - t0) * 1000
            log_request("/stt", ms, "ok:cache_hit", bytes=len(wav_bytes))
            _note_stt(node_id, _wav_seconds(wav_bytes), ms, len(cached), True)
            self._send_json(200, {"text": cached})
            return

        try:
            text = transcribe(wav_bytes)
        except Exception as exc:
            self._send_json(502, {"error": f"stt upstream failed: {exc}"})
            log_request("/stt", (time.time() - t0) * 1000, "error:upstream")
            return
        ms = (time.time() - t0) * 1000
        log_request("/stt", ms, "ok", bytes=len(wav_bytes))
        _note_stt(node_id, _wav_seconds(wav_bytes), ms, len(text or ""), False)
        self._send_json(200, {"text": text})

    # -- /ambient --

    def _handle_ambient(self):
        t0 = time.time()
        parsed = urlsplit(self.path)
        content_type = self.headers.get("Content-Type", "")
        body = self._read_body()
        wav_bytes = None
        node_id = ""

        if content_type.startswith("multipart/form-data"):
            parts = parse_multipart_parts(body, content_type)
            wav_bytes = parts.get("audio")
            if "node_id" in parts:
                node_id = parts["node_id"].decode("utf-8", "replace").strip()
        elif content_type in ("audio/wav", "application/octet-stream", ""):
            wav_bytes = body

        if not node_id:
            qs = parse_qs(parsed.query)
            node_id = (qs.get("node_id") or [""])[0].strip()
        _set_req_node(node_id)
        if not self._check_bearer_auth(node_id, "/ambient", t0):
            return

        if not wav_bytes:
            self._send_json(400, {"error": "no audio payload found"})
            log_request("/ambient", (time.time() - t0) * 1000, "error:no_audio", node=node_id)
            return

        # Pre-STT speech gate: a clip that is only noise never reaches
        # Whisper or the classifier. Fails open ("unavailable").
        gate_status, gate_voiced_ms, ms_vad = speech_gate(wav_bytes)
        gate_extra = {"speech_gate": gate_status, "vad_ms": ms_vad}
        if gate_voiced_ms is not None:
            gate_extra["voiced_ms"] = gate_voiced_ms
        if gate_status == "rejected":
            self._send_json(200, {
                "engage": False, "transcript": "", "score": 0.0,
                "classifier": "no-speech", "speaker": "", "speaker_score": 0.0,
                "ms": {"stt": 0.0, "classify": 0.0, "vad": ms_vad},
            })
            log_request(
                "/ambient", (time.time() - t0) * 1000, "ok:no_speech", node=node_id,
                stage_ms={"stt": 0.0, "classify": 0.0}, score=0.0, engaged=False,
                classifier="no-speech", chars=0, **gate_extra,
            )
            return

        sha = hashlib.sha256(wav_bytes).hexdigest()

        # Speaker ID runs in parallel with STT (same as the Go hub does for
        # an engaged turn) so it doesn't add its own latency on top —
        # identify_speaker() itself may take as long as the desktop embed
        # hop's timeout, but _handle_ambient only ever waits up to
        # AMBIENT_SPEAKER_WAIT_SEC for it AFTER STT is already done, below.
        speaker_result: dict = {}

        def _speaker_worker():
            # Written as a single tuple under one key — two separate keys
            # (speaker, then score) would let a join() that times out
            # between the two writes read a "" speaker next to a real score.
            speaker_result["r"] = identify_speaker(wav_bytes, sha)

        speaker_thread = threading.Thread(target=_speaker_worker, daemon=True)
        speaker_thread.start()

        t_stt0 = time.time()
        try:
            transcript = transcribe(wav_bytes)
        except Exception as exc:
            self._send_json(502, {"error": f"stt upstream failed: {exc}"})
            log_request("/ambient", (time.time() - t0) * 1000, "error:upstream", node=node_id)
            return
        ms_stt = (time.time() - t_stt0) * 1000

        speaker_thread.join(AMBIENT_SPEAKER_WAIT_SEC)
        speaker, speaker_score = speaker_result.get("r", ("", 0.0))

        resp = {
            "engage": False,
            "transcript": transcript,
            "score": 0.0,
            "classifier": "none",
            "speaker": speaker,
            "speaker_score": speaker_score,
            "ms": {"stt": round(ms_stt, 1), "classify": 0.0, "vad": ms_vad},
        }

        if is_hallucination_or_too_short(transcript):
            self._send_json(200, resp)
            log_request(
                "/ambient", (time.time() - t0) * 1000, "ok:skipped_short_or_hallucination",
                node=node_id, stage_ms=resp["ms"], score=0.0, engaged=False,
                classifier="none", chars=len(transcript or ""), **gate_extra,
            )
            return

        t_cls0 = time.time()
        detail = decide_addressed(node_id, transcript, speaker)
        score, classifier = detail["score"], detail["classifier"]
        laya_score = detail["laya_score"]
        ms_classify = (time.time() - t_cls0) * 1000
        engage = classifier != "none" and score >= AMBIENT_THRESHOLD

        # Rolling context always records what was heard, regardless of
        # engagement, so a later unnamed follow-up still has the prior turn
        # to reason about — but nothing is written to DISK unless engaged
        # (see _handle_ambient's logging below and the STT cache, which is
        # populated only on engage per spec).
        _record_ambient_transcript(node_id, transcript, speaker)
        if engage:
            _stt_cache_put(sha, transcript)
            _speaker_cache_put(sha, speaker, speaker_score)

        resp.update({
            "engage": engage,
            "score": round(score, 4),
            "classifier": classifier,
            **({"laya_score": round(laya_score, 4)} if laya_score is not None else {}),
            "ms": {"stt": round(ms_stt, 1), "classify": round(ms_classify, 1),
                   "vad": ms_vad},
        })
        self._send_json(200, resp)
        # Privacy: metadata only, never the transcript text, on either
        # branch — an engaged utterance's raw text still isn't logged here
        # (the edge client's own re-POST to /stt is what a human might see
        # in a transcript downstream, not this endpoint). The speaker label
        # is likewise NEVER logged, kept in-memory only (rolling context,
        # speaker cache) the same as the transcript text itself — a name is
        # still identifying information about who was in the room.
        log_request(
            "/ambient", (time.time() - t0) * 1000, "ok", node=node_id,
            stage_ms=resp["ms"], score=resp["score"], engaged=engage,
            classifier=classifier, chars=len(transcript or ""),
            **({"laya_score": round(laya_score, 4)} if laya_score is not None else {}),
            **gate_extra,
        )

    # -- /ask (SSE) --

    def _sse_event(self, event: str, data: dict) -> bytes:
        return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode("utf-8")

    def _handle_ask(self):
        t0 = time.time()
        body = self._read_body()
        try:
            req = json.loads(body.decode("utf-8")) if body else {}
            if not isinstance(req, dict):
                self._send_json(400, {"error": "request body must be a JSON object"})
                return
        except Exception:
            self._send_json(400, {"error": "invalid JSON body"})
            return

        node_id = (req.get("node_id") or self.headers.get("X-Node-Id") or "").strip()
        if not self._check_bearer_auth(node_id, "/ask", t0):
            return

        err_code, err_msg = validate_multimodal_payload(req)
        if err_code is not None:
            self._send_json(err_code, {"error": err_msg})
            return

        prompt = req.get("prompt") or ""
        session_id = req.get("session_id") or ""
        speaker = req.get("speaker") or ""
        speaker_score = req.get("speaker_score")
        _set_req_node(node_id)
        ctx = self._legacy_ctx(speaker, speaker_score, node_id,
                               self._req_tag(req, "device"), self._req_tag(req, "medium"),
                               image_b64=req.get("image_b64"), burst_images_b64=req.get("burst_images_b64"),
                               audio_b64=req.get("audio_b64"))

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        try:
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            return

        result_box: dict = {}
        done_event = threading.Event()

        def worker():
            _FRAME_CTX.ctx = ctx
            try:
                result_box["reply"] = BRAIN_CALL(
                    prompt, session_id, node_id, speaker, speaker_score, HIVE_TIMEOUT
                )
            except Exception as exc:
                result_box["error"] = str(exc)
            finally:
                done_event.set()

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

        try:
            self.wfile.write(self._sse_event("status", {"status": "thinking"}))
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            return

        outcome = "ok"
        while not done_event.wait(STATUS_INTERVAL_SEC):
            try:
                self.wfile.write(self._sse_event("status", {"status": "thinking"}))
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                log_request("/ask", (time.time() - t0) * 1000, "error:client_gone")
                return

        if "error" in result_box:
            outcome = "error:brain_exception"
            reply = APOLOGY
        else:
            raw = result_box.get("reply") or ""
            if raw.startswith("unanswered:"):
                outcome = f"unanswered:{raw.split(':', 1)[1].strip()[:40]}"
                reply = APOLOGY
            elif is_silent_sentinel(raw):
                # Silent sentinel: the brain chose to say nothing. Empty reply,
                # no record_reply, so no follow-up window either.
                outcome = "silent"
                reply = ""
            elif not raw.strip():
                outcome = "empty_reply"
                reply = APOLOGY
            else:
                outcome = "ok"
                reply = strip_markdown(raw)
                if os.environ.get("MVG_ARTIFACT_WEBHOOK_URL"):
                    reply, _ = forward_artifacts_to_webhook(raw, reply)
                if is_defer_enabled() and (has_defer_marker(raw) or has_defer_marker(reply)):
                    get_pending_store().store(
                        node_id=node_id,
                        prompt=prompt,
                        answer=strip_defer_marker(reply),
                    )
                reply = strip_defer_marker(reply)
                # Only a real answer opens the follow-up window for /ambient
                # — the apology isn't a conversational turn worth treating
                # as "the agent is mid-conversation" on this node.
                record_reply(node_id, reply)

        mic_reopen_on = os.environ.get("MVG_ENABLE_MIC_REOPEN", "0").strip().lower() in ("1", "true", "yes", "on")
        reply_dict: dict[str, Any] = {"reply": reply}
        done_dict: dict[str, Any] = {}
        if mic_reopen_on:
            expects = reply_expects_reply(reply)
            reply_dict["expects_reply"] = expects
            done_dict["expects_reply"] = expects
        try:
            self.wfile.write(self._sse_event("reply", reply_dict))
            self.wfile.write(self._sse_event("done", done_dict))
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        log_request("/ask", (time.time() - t0) * 1000, outcome, node_id=node_id)
        if os.environ.get("MVG_TRANSCRIPT_SINK_URL") or os.environ.get("MVG_TRANSCRIPT_LOG_PATH"):
            tr_rec: dict[str, Any] = {
                "kind": "ask",
                "endpoint": "/ask",
                "node_id": node_id or None,
                "prompt": prompt,
                "transcript": prompt,
                "reply": reply,
                "outcome": outcome,
                "total_ms": round((time.time() - t0) * 1000, 1),
                "timestamp": time.time(),
            }
            if mic_reopen_on:
                tr_rec["expects_reply"] = reply_expects_reply(reply)
            emit_turn_transcript(tr_rec)

    # -- /ask/stream (SSE, MANDOS) --
    #
    # specs/2026-09-28-mandos-streaming.md: cuts complete sentences out of
    # the brain call's in-progress answer (via STREAM_BRAIN_CALL's on_partial
    # callback) and synthesizes each one as soon as it's complete, so audio
    # starts well before the full reply is known. Pipelined: sentence N+1's
    # TTS submits to the executor while sentence N's audio is still being
    # emitted/played, and results are drained STRICTLY IN ORDER regardless
    # of which TTS call actually finishes first (Future.result() on index i
    # blocks until i is ready, even if i+1 finished sooner).
    #
    # Wire format is new (not hub.go's today) and additive: /ask is
    # completely unchanged. `sentence` events carry base64 WAV bytes
    # directly (SSE is text-only) so a caller that adopts this endpoint
    # doesn't need a second round trip to /v1/audio/speech per sentence.
    # The trailing `reply`+`done` pair matches /ask's shape for callers
    # that just want the transcript.

    STREAM_STATUS_POLL_SEC = 0.15

    def _synthesize_for_stream(self, text: str) -> tuple[bytes, str, str]:
        text = strip_defer_marker(text)
        wav_bytes, engine = synthesize(text)
        # MIRRORMERE-ONE-FORMAT: every sentence emitted on this stream is
        # 24 kHz mono 16-bit WAV, regardless of which engine spoke it — the
        # `engine` label still reflects which one actually did.
        wav_bytes = NORMALIZE_WAV(wav_bytes)
        return wav_bytes, engine, text

    def _handle_ask_stream(self):
        """Legacy /ask/stream: JSON {prompt, session_id, node_id, speaker,
        speaker_score} -> the shared brain+TTS stream (_stream_brain_turn)."""
        t0 = time.time()
        body = self._read_body()
        try:
            req = json.loads(body.decode("utf-8")) if body else {}
            if not isinstance(req, dict):
                self._send_json(400, {"error": "request body must be a JSON object"})
                return
        except Exception:
            self._send_json(400, {"error": "invalid JSON body"})
            return

        node_id = (req.get("node_id") or self.headers.get("X-Node-Id") or "").strip()
        if not self._check_bearer_auth(node_id, "/ask/stream", t0):
            return

        err_code, err_msg = validate_multimodal_payload(req)
        if err_code is not None:
            self._send_json(err_code, {"error": err_msg})
            return

        prompt = req.get("prompt") or ""
        session_id = req.get("session_id") or ""
        speaker = req.get("speaker") or ""
        speaker_score = req.get("speaker_score")
        _set_req_node(node_id)
        turn_id = str(uuid.uuid4())

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        if not self._sse("turn", {"turn_id": turn_id}):  # ears correlate their latency record with ours
            return
        self._stream_brain_turn(
            t0=t0, turn_id=turn_id, prompt=prompt, session_id=session_id, node_id=node_id,
            speaker=speaker, speaker_score=speaker_score,
            ctx=self._legacy_ctx(speaker, speaker_score, node_id,
                                 self._req_tag(req, "device"), self._req_tag(req, "medium"),
                                 image_b64=req.get("image_b64"), burst_images_b64=req.get("burst_images_b64"),
                                 audio_b64=req.get("audio_b64")),
            endpoint="/ask/stream", opts={})

    def _handle_intent(self):
        """POST /intent (LAMMAS), for the hub: JSON {text, speaker, node,
        turn_id}. 204 = no match (or kill switch / handler failure), caller
        uses its brain. 200 = matched: the SAME SSE stream /ask/stream returns
        (`turn`, `sentence`*, `reply`, `done`), via the shared
        _stream_brain_turn path, with the already-run match injected so the
        handler does not run twice."""
        t0 = time.time()
        body = self._read_body()
        try:
            req = json.loads(body.decode("utf-8")) if body else {}
            if not isinstance(req, dict):
                raise ValueError("body must be an object")
        except Exception:
            self._send_json(400, {"error": "invalid JSON body"})
            return

        node_id = (req.get("node") or req.get("node_id") or self.headers.get("X-Node-Id") or "").strip()
        if not self._check_bearer_auth(node_id, "/intent", t0):
            return
        text = req.get("text")
        if not isinstance(text, str) or not text.strip():
            self._send_json(400, {"error": "text required"})
            return
        node_id = req.get("node") or ""
        speaker = req.get("speaker") or ""
        _set_req_node(node_id)
        claimed, _ = self._trusted_claimed_speaker()
        auth_header = self.headers.get("Authorization", "")
        ha_auth = _is_ha_authorized(
            auth_header=auth_header,
            node_id=node_id,
            speaker=speaker,
            speaker_score=req.get("speaker_score"),
            claimed_speaker=claimed,
        )
        fast = _intent(text, ha_authorized=ha_auth)
        if not fast["hit"]:
            if fast.get("reason") == "ha_unauthorized":
                self._send_json(401, {"error": "unauthorized"})
                log_request("/intent", (time.time() - t0) * 1000, "error:unauthorized", node=node_id)
                return
            log_request("/intent", (time.time() - t0) * 1000, "ok:no_match", node=node_id,
                        intent=fast["intent"] or "miss", intent_match_ms=fast["match_ms"],
                        intent_handler_ms=fast["handler_ms"], reason=fast["reason"])
            self.send_response(204)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        tts = str(req.get("tts") or "").lower() or None
        if tts not in (None, "stream", "wav", "none"):
            self._send_json(400, {"error": "tts must be stream, wav or none"})
            log_request("/intent", (time.time() - t0) * 1000, "error:bad_tts", node=node_id)
            return
        turn_id = str(req.get("turn_id") or uuid.uuid4())
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        if not self._sse("turn", {"turn_id": turn_id}):
            return
        self._stream_brain_turn(
            t0=t0, turn_id=turn_id, prompt=text, session_id="", node_id=node_id,
            speaker=speaker, speaker_score=None,
            ctx=self._legacy_ctx(speaker, None, node_id,
                                 self._req_tag(req, "device"), self._req_tag(req, "medium")),
            endpoint="/intent", opts={"intent_result": fast, **({"tts": tts} if tts else {})})

    def _sse(self, event: str, data: dict) -> bool:
        """Write one SSE event. False = the client went away."""
        try:
            self.wfile.write(self._sse_event(event, data))
            self.wfile.flush()
            return True
        except (BrokenPipeError, ConnectionResetError, OSError):
            return False

    def _req_tag(self, req, name: str):
        """Raw device/medium from a JSON body, else the X-Voice-<Name> header."""
        v = req.get(name) if isinstance(req, dict) else None
        return v if v else self.headers.get(f"X-Voice-{name.capitalize()}")

    def _legacy_ctx(self, speaker: str, speaker_score, node_id: str,
                    device=None, medium=None, image_b64: str | None = None,
                    burst_images_b64: list[str] | None = None,
                    audio_b64: str | None = None) -> dict:
        """Framing context for the legacy endpoints: the hub's speaker (set
        only on a match) + speaker_score are the voice ID match. A trusted
        claimed_speaker header is honoured the same as on /voice/turn."""
        claimed, _ = self._trusted_claimed_speaker()
        dev, med = resolve_device_medium(node_id, device, medium)
        return make_frame_ctx(legacy_voice_id(speaker, speaker_score), claimed_speaker=claimed,
                              reply_cap_words=resolve_profile(node_id)["reply_cap_words"],
                              device=dev, medium=med,
                              image_b64=image_b64, burst_images_b64=burst_images_b64,
                              audio_b64=audio_b64)

    def _trusted_claimed_speaker(self) -> tuple[str, bool]:
        """(claimed_speaker, rejected). Honoured only with the shared secret:
        header X-Voice-Claimed-Speaker plus X-Voice-Gateway-Key equal to env
        MVG_TRUSTED_KEY. MVG_TRUSTED_KEY unset/empty -> always ignored. Never
        read from a request body or the query string."""
        claimed = (self.headers.get("X-Voice-Claimed-Speaker") or "").strip()
        if not claimed:
            return "", False
        key = os.environ.get("MVG_TRUSTED_KEY") or ""
        given = self.headers.get("X-Voice-Gateway-Key") or ""
        if key and hmac.compare_digest(key.encode(), given.encode()):
            return claimed[:64], False
        return "", True

    def _stream_brain_turn(self, *, node_id, ctx, **kw):
        """Registers the turn as the node's active task (for cancel_active_turn)
        and always removes it on the way out."""
        task = _register_task(node_id, ctx.get("target") or resolve_target(node_id))
        ctx["task_id"] = task["task_id"]
        try:
            return GatewayHandler._stream_brain_turn_inner(self, node_id=node_id, ctx=ctx, task=task, **kw)
        finally:
            _unregister_task(node_id, task)

    def _stream_brain_turn_inner(self, *, t0, turn_id, prompt, session_id, node_id, speaker,
                                 speaker_score, ctx, endpoint, opts, task, log_extra=None):
        """The shared brain -> sentence-TTS stream behind /ask/stream and
        /voice/turn: brain call, sentence cutting, TTS/gap-fill, `status`
        keepalives, then `reply` + `done`. The caller has already sent the
        response headers. opts (all optional): tts ("stream"|"wav"|"none"),
        gap_fill (bool), audio_chunk (bool, also emit the hub-dialect
        `audio_chunk` events), voice (bool: emit `state` events and the
        richer reply/done payloads). Returns (outcome, reply)."""
        voice = bool(opts.get("voice"))
        audio_chunk = bool(opts.get("audio_chunk"))
        log_extra = log_extra or {}
        mic_reopen_on = os.environ.get("MVG_ENABLE_MIC_REOPEN", "0").strip().lower() in ("1", "true", "yes", "on")

        segmenter = SentenceSegmenter(early_first=FIRST_CLAUSE_EARLY)
        wake = threading.Event()  # set when a sentence is queued or the brain finishes
        task["wake"] = wake
        cancel_event = task["cancel_event"]  # set by cancel_active_turn: stop speaking, drop the result
        spoken_text = ""  # stripped reply text already fed to the segmenter
        raw_seen = ""
        md_pending = ""
        spoken_tail_space = False
        seg_tail_space = False
        lock = threading.Lock()
        futures: list[Future] = []
        executor = ThreadPoolExecutor(max_workers=SENTENCE_CHUNK_MAX_WORKERS)
        # Sub-sentence PCM mode (MVG_TTS_STREAM): sentences are queued as
        # plain text and synthesized/streamed one at a time, in order, by
        # the drain loop below (the desktop serializes synthesis behind a
        # model lock, so parallel requests gained nothing). The mode is
        # fixed for the whole turn: every event is pcm, never wav.
        tts_opt = opts.get("tts")
        if tts_opt == "none":
            text_mode, stream_mode = True, False
        elif tts_opt == "wav":
            text_mode, stream_mode = False, False
        elif tts_opt == "stream":
            text_mode, stream_mode = False, True
        else:
            text_mode = bool(TEXT_SENTENCES)
            stream_mode = bool(TTS_STREAM) and not text_mode
        timing = {"first_submit": None, "first_audio_ms": None, "tts_first_ms": None,
                  "chunks": 0, "brain_first_text_ms": None, "first_sentence_ms": None,
                  "last_audio_ms": None, "sentences": 0, "audio_bytes": 0,
                  "engine": None, "play_end": None,
                  "first_text_event_ms": None}
        # Gap-fill (MVG_GAP_FILL): only in pcm stream mode, where the
        # turn's audio format is already pcm; a wav-mode turn never mixes.
        gap = {"used": [], "open": False, "mid": False, "slow": False,
               "ticks": 0, "first_ms": None}
        gap_enabled = GAP_FILL and stream_mode and opts.get("gap_fill", True)
        spoke = {"state": False, "synth": False, "chunk_idx": 0}
        target = ctx.get("target") or resolve_target(node_id)

        def _mark_speaking() -> bool:
            if voice and not spoke["state"]:
                spoke["state"] = True
                return self._sse("state", {"state": "speaking"})
            return True

        def _mark_synthesizing() -> bool:
            """`state: synthesizing`, once, when the first real sentence goes to
            TTS and nothing has been spoken yet. Text-only turns never synthesize."""
            if voice and not text_mode and not spoke["synth"] and not spoke["state"]:
                spoke["synth"] = True
                return self._sse("state", {"state": "synthesizing"})
            return True

        def _write_sentence(event: dict) -> bool:
            """One audio/text `sentence` event; with audio_chunk, also the
            hub-dialect `audio_chunk` the room ear plays from. False = client gone.
            A gap-fill clip (`filler`) is not the answer, so it does not move the
            turn to `speaking`."""
            if cancel_event.is_set():
                return True  # cancelled: nothing more reaches the client
            if not event.get("filler") and not _mark_speaking():
                return False
            if not self._sse("sentence", event):
                return False
            if audio_chunk and event.get("audio_b64") is not None:
                chunk = {"chunk_index": spoke["chunk_idx"], "format": event.get("format", "wav"),
                         "is_final": False, "data": event["audio_b64"],
                         "text": event.get("text") or "", "engine": event.get("engine") or ""}
                if event.get("sample_rate"):
                    chunk["sample_rate"] = event["sample_rate"]
                    chunk["channels"] = event.get("channels", 1)
                spoke["chunk_idx"] += 1
                return self._sse("audio_chunk", chunk)
            return True

        def _ms_since_start() -> float:
            return round((time.time() - t0) * 1000, 1)

        def _note_audio(nbytes: int) -> None:
            """Track when the audio sent so far finishes PLAYING (it is
            generated faster than real time), so a long silence is measured
            against playback, not against emission."""
            now = time.time()
            timing["last_audio_ms"] = round((now - t0) * 1000, 1)
            start = max(timing["play_end"] or now, now)
            timing["play_end"] = start + nbytes / (AUDIO_TARGET_RATE * 2.0)

        def _submit(sentence_text: str) -> None:
            if not sentence_text.strip() or cancel_event.is_set():
                return
            timing["sentences"] += 1
            if timing["first_sentence_ms"] is None:
                timing["first_sentence_ms"] = _ms_since_start()
            if stream_mode or text_mode:
                with lock:
                    if timing["first_submit"] is None:
                        timing["first_submit"] = time.time()
                    futures.append(sentence_text)
                wake.set()
                return
            fut = executor.submit(self._synthesize_for_stream, sentence_text)
            with lock:
                futures.append(fut)

        def _timing_extra() -> dict:
            # first_audio_ms: request start -> first audio event written.
            # tts_first_ms: first sentence text available -> first audio
            # written (the ~0.65 s the streaming endpoint is meant to hit).
            if not stream_mode:
                return {"tts_stream": False, "text_sentences": text_mode, "turn_id": turn_id}
            return {"tts_stream": True, "first_audio_ms": timing["first_audio_ms"],
                    "tts_first_ms": timing["tts_first_ms"], "audio_chunks": timing["chunks"],
                    "turn_id": turn_id, "gap_fill": list(gap["used"]),
                    "progress_lines": progress.lines if progress else 0}

        reply = ""
        stages: dict = {}
        # Spoken progress (MVG_SPOKEN_PROGRESS): pcm stream mode only, never on a
        # fixed-reply (intent) turn. Events come from stages["events"].
        progress = (ProgressNarrator(lambda: stages.get("events"))
                    if spoken_progress_enabled() and stream_mode and not opts.get("intent_result")
                    else None)
        # Context opener (MVG_CONTEXT_OPENER): one specific line from the first
        # stage event in place of the generic opener clip. pcm stream mode only.
        opener = {"on": context_opener_enabled() and stream_mode and not opts.get("intent_result"),
                  "kind": None, "ms": None}

        def _maybe_context_opener() -> None:
            if not opener["on"] or opener["kind"] is not None or cancel_event.is_set():
                return
            if raw_seen or timing["first_audio_ms"] is not None or timing["sentences"]:
                opener["on"] = False  # the real reply (or other speech) got there first
                return
            events = stages.get("events")
            if not events:
                return
            ev = normalize_stage_event(events[0])
            opener["on"] = False  # decided by the first event, once
            if not ev or not ev["detail"] or len(ev["detail"]) > STAGE_DETAIL_MAX:
                return
            gap["open"] = True  # the generic opener clip must not also play
            opener["kind"], opener["ms"] = "stage", _ms_since_start()
            if progress is not None:
                progress.seen, progress.last_spoken = 1, progress.clock()
            _submit("One moment, " + ev["detail"])

        # LAMMAS per-turn record: intent name or "miss", match_ms, handler_ms.
        intent_rec: dict = {"intent": "miss", "intent_match_ms": None, "intent_handler_ms": None}

        def _finish(outcome: str, with_node: bool = False) -> None:
            total_ms = round((time.time() - t0) * 1000, 1)
            extra = _timing_extra()
            if with_node:
                log_request(endpoint, total_ms, outcome, node_id=node_id, **extra, **intent_rec, **log_extra)
            else:
                log_request(endpoint, total_ms, outcome, **extra, **intent_rec, **log_extra)
            rec = {
                "kind": "turn", "node_id": node_id or None, "session_id": session_id or None,
                "turn_id": turn_id, "request_start": int(t0 * 1000),
                "brain_first_text_ms": timing["brain_first_text_ms"],
                "first_sentence_ms": timing["first_sentence_ms"],
                "gap_fill_first_ms": gap["first_ms"],
                "orchestrator": orchestrator_enabled(target),
                "tts_first_ms": timing["tts_first_ms"],
                "first_audio_ms": timing["first_audio_ms"],
                "last_audio_ms": timing["last_audio_ms"],
                "total_ms": total_ms, "sentences": timing["sentences"],
                "reply_chars": len(reply), "gap_fill": list(gap["used"]),
                "progress_lines": progress.lines if progress else 0,
                **({"opener_kind": opener["kind"] or ("gap_fill" if gap["used"] else "none"),
                    "opener_ms": opener["ms"] if opener["kind"] else gap["first_ms"]}
                   if context_opener_enabled() else {}),
                "audio_s": round(timing["audio_bytes"] / (AUDIO_TARGET_RATE * 2.0), 3),
                "tts_engine": timing["engine"] if stream_mode else ("text" if text_mode else "wav"),
                "text_sentences": text_mode,
                "first_text_event_ms": timing["first_text_event_ms"],
                "target": target, "outcome": outcome, "endpoint": endpoint,
                **intent_rec,
            }
            rec_stages = {k: v for k, v in stages.items() if k != "events"}
            if rec_stages:
                # Raw epoch ms per stage, each on the clock of the host that
                # stamped it (README: "Hand-off stages"). `_ms` keys ending
                # `_pi_ms`, inserted/first_partial_seen/final_seen are this
                # host (this box); claim_recv .. partial_write_start are the desktop.
                rec["stages"] = rec_stages
            rec.update(_correlated_stt(node_id, t0))
            _gap_turn_done(gap["used"])
            log_latency(rec)
            if os.environ.get("MVG_TRANSCRIPT_SINK_URL") or os.environ.get("MVG_TRANSCRIPT_LOG_PATH"):
                transcript_rec = dict(rec)
                transcript_rec["prompt"] = prompt
                transcript_rec["transcript"] = prompt
                transcript_rec["reply"] = reply
                if mic_reopen_on:
                    transcript_rec["expects_reply"] = reply_expects_reply(reply)
                emit_turn_transcript(transcript_rec)

        def _emit_gap_fill(categories, fallback=()) -> bool:
            """Emit one whole filler clip as pcm `sentence` events (text on
            the first chunk). Runs on the drain thread, the only writer, so
            a clip can never interleave with real audio. `fallback` is tried
            once when `categories` has no clip. False = client gone."""
            clip = pick_gap_clip(categories, gap["used"])
            if clip is None and fallback:
                clip = pick_gap_clip(fallback, gap["used"])
            if clip is None:
                return True
            pcm = clip["pcm"]
            # Only handoff clips keep their caption; vocal and machine
            # effects are silent on the wire.
            caption = clip["text"] if clip.get("category") == "handoff" else ""
            if gap["first_ms"] is None:
                gap["first_ms"] = _ms_since_start()
            gap["used"].append(clip["slug"])
            for i in range(0, len(pcm), GAP_FILL_CHUNK_BYTES):
                piece = pcm[i:i + GAP_FILL_CHUNK_BYTES]
                event = {
                    "text": caption if i == 0 else "",
                    "engine": "gap-fill", "format": "pcm",
                    "sample_rate": AUDIO_TARGET_RATE, "channels": AUDIO_TARGET_CHANNELS,
                    "audio_b64": base64.b64encode(piece).decode("ascii"),
                    "filler": True,
                }
                if not _write_sentence(event):
                    return False
                _note_audio(len(piece))
            return True

        def _maybe_gap_fill() -> bool:
            """Called when nothing real is queued and the brain is still
            working. At most one clip per call; each slot fires once per turn
            (its flag is set before emitting, so an empty category stays
            silent and is not retried). False = client gone."""
            if not gap_enabled:
                return True
            now = time.time()
            elapsed = (now - t0) * 1000
            play_end = timing["play_end"]
            # ms of playback silence; before anything has played, the whole elapsed time
            silence = (now - play_end) * 1000 if play_end is not None else elapsed
            if timing["first_audio_ms"] is None:
                # Before real audio: the vocal opener, then the timed slots.
                if not gap["open"] and elapsed >= GAP_FILL_AFTER_MS:
                    gap["open"] = True
                    return _emit_gap_fill(GAP_FILL_VOCAL_OPEN)
                if (gap["open"] and not gap["mid"] and elapsed >= GAP_FILL_MID_MS
                        and silence >= GAP_FILL_QUIET_MS):
                    gap["mid"] = True
                    return _emit_gap_fill(effect_categories(prompt), fallback=GAP_FILL_SFX_GENERIC)
                if (not gap["slow"] and elapsed >= GAP_FILL_SLOW_MS
                        and silence >= GAP_FILL_QUIET_MS):
                    gap["slow"] = True
                    return _emit_gap_fill(GAP_FILL_VOCAL_LONG)
                if (gap["slow"] and gap["ticks"] < GAP_FILL_MAX_TICKS
                        and silence >= GAP_FILL_TICK_MS):
                    gap["ticks"] += 1
                    return _emit_gap_fill(GAP_FILL_SFX_TICK)
                return True
            # After real audio has started: machine ticks only, never a vocal.
            if gap["ticks"] < GAP_FILL_MAX_TICKS and silence >= GAP_FILL_LONG_MS:
                gap["ticks"] += 1
                return _emit_gap_fill(GAP_FILL_SFX_TICK)
            return True

        def _stream_one(sentence_text: str) -> bool:
            """Stream one sentence as pcm `sentence` events. The sentence
            text rides on the first chunk only ("" after) so the hub's
            caption gets it exactly once. Returns False if the client went
            away. A sentence that yields no audio is skipped."""
            try:
                engine, chunks = synthesize_stream(sentence_text)
                timing["engine"] = timing["engine"] or engine
            except Exception as exc:
                log_fallback(f"stream_sentence_tts_failed:{exc}", tts_engine="stream-skip")
                return True
            first = True
            try:
                for pcm in chunks:
                    if not pcm:
                        continue
                    event = {
                        "text": sentence_text if first else "",
                        "engine": engine,
                        "format": "pcm",
                        "sample_rate": AUDIO_TARGET_RATE,
                        "channels": AUDIO_TARGET_CHANNELS,
                        "audio_b64": base64.b64encode(pcm).decode("ascii"),
                    }
                    if not _write_sentence(event):
                        return False
                    first = False
                    timing["chunks"] += 1
                    timing["audio_bytes"] += len(pcm)
                    _note_audio(len(pcm))
                    if timing["first_audio_ms"] is None:
                        now = time.time()
                        timing["first_audio_ms"] = round((now - t0) * 1000, 1)
                        if timing["first_submit"] is not None:
                            timing["tts_first_ms"] = round((now - timing["first_submit"]) * 1000, 1)
            except Exception as exc:
                # Mid-stream upstream failure: what was sent stays sent.
                log_fallback(f"stream_sentence_aborted:{str(exc)[:200]}", tts_engine="stream-abort")
            finally:
                close = getattr(chunks, "close", None)
                if close:
                    close()
            if first:
                log_fallback("stream_sentence_no_audio", tts_engine="stream-skip")
            return True

        def on_partial(chunk: str) -> None:
            nonlocal spoken_text, raw_seen, md_pending, spoken_tail_space, seg_tail_space
            if not chunk:
                return
            if (
                raw_seen
                and (len(chunk) > len(raw_seen) or (len(chunk) == len(raw_seen) and " " in chunk))
                and chunk.startswith(raw_seen)
            ):
                raw_delta = chunk[len(raw_seen):]
                raw_seen = chunk
            else:
                raw_delta = chunk
                raw_seen += chunk
            if not raw_delta:
                return
            work = md_pending + raw_delta if md_pending else raw_delta
            if len(raw_seen) <= 32 and is_silent_sentinel(raw_seen):
                # "Pass." would otherwise complete a sentence and be spoken
                # before the final-reply check can see it is the sentinel.
                md_pending = work
                return
            # Strip only the part whose markdown is settled: an unclosed link
            # or code span would otherwise collapse later and shift offsets.
            stable = stable_markdown_prefix(work)
            md_pending = work[len(stable):]
            if not stable:
                return
            has_lead = stable[0].isspace()
            has_trail = stable[-1].isspace()
            core = strip_defer_marker(strip_markdown(stable) or "")
            had_spoken = bool(spoken_text)
            if not core:
                if (has_lead or has_trail) and had_spoken:
                    spoken_tail_space = True
                    if not seg_tail_space:
                        seg_tail_space = True
                        for sentence in segmenter.feed(" "):
                            _submit(sentence)
                return
            need_sep = had_spoken and (spoken_tail_space or has_lead)
            if need_sep:
                spoken_text += " " + core
            else:
                spoken_text += core
            spoken_tail_space = has_trail
            seg_lead = " " if (need_sep and not seg_tail_space) else ""
            seg_trail = " " if has_trail else ""
            seg_delta = f"{seg_lead}{core}{seg_trail}"
            seg_tail_space = has_trail
            if timing["brain_first_text_ms"] is None:
                timing["brain_first_text_ms"] = _ms_since_start()
            for sentence in segmenter.feed(seg_delta):
                _submit(sentence)

        result_box: dict = {}
        done_event = threading.Event()

        def worker():
            _STAGE_SINK.stages = stages
            _FRAME_CTX.ctx = ctx
            try:
                # LAMMAS: deterministic fast path, after STT and the addressed
                # gate, before the brain. A miss / error / slow handler falls
                # through to the brain below, unchanged.
                has_vision = bool(ctx.get("image_b64") or ctx.get("burst_images_b64"))
                ha_auth = opts.get("ha_authorized")
                if ha_auth is None:
                    auth_header = (
                        self.headers.get("Authorization", "")
                        if hasattr(self, "headers") and self.headers
                        else ""
                    )
                    ha_auth = _is_ha_authorized(
                        auth_header=auth_header,
                        node_id=node_id,
                        speaker=speaker,
                        speaker_score=speaker_score,
                        voice_id=ctx.get("voice_id"),
                        claimed_speaker=ctx.get("claimed_speaker", ""),
                    )
                fast = opts.get("intent_result") or _intent(
                    prompt, has_images=has_vision, ha_authorized=ha_auth
                )
                intent_rec["intent_match_ms"] = fast["match_ms"]
                intent_rec["intent_handler_ms"] = fast["handler_ms"]
                if fast["hit"]:
                    intent_rec["intent"] = fast["intent"]
                    result_box["reply"] = fast["reply"]
                    return
                if fast["intent"]:  # matched but the handler failed: still a brain turn
                    intent_rec["intent_fallthrough"] = f"{fast['intent']}:{fast['reason']}"
                call_kwargs = {}
                if ctx.get("task_id"):
                    call_kwargs["task_id"] = ctx["task_id"]
                enable_vision = is_vision_enabled()
                if enable_vision:
                    if ctx.get("image_b64"):
                        call_kwargs["image_b64"] = ctx["image_b64"]
                    if ctx.get("burst_images_b64"):
                        call_kwargs["burst_images_b64"] = ctx["burst_images_b64"]
                if ctx.get("audio_clips"):
                    call_kwargs["audio_clips"] = ctx["audio_clips"]
                brain_direct_audio = bool(BRAIN_DIRECT_AUDIO) or (os.environ.get("MVG_BRAIN_DIRECT_AUDIO", "0") == "1")
                if brain_direct_audio:
                    if ctx.get("audio_b64"):
                        call_kwargs["audio_b64"] = ctx["audio_b64"]
                # Upfront inspect.signature check prevents TypeError retries from double-sending prompts
                try:
                    sig = inspect.signature(STREAM_BRAIN_CALL)
                    params = sig.parameters
                    has_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
                    filtered_kwargs = {
                        k: v for k, v in call_kwargs.items()
                        if has_var_keyword or k in params
                    }
                except (ValueError, TypeError):
                    filtered_kwargs = {}

                result_box["reply"] = STREAM_BRAIN_CALL(
                    prompt, session_id, node_id, speaker, speaker_score,
                    HIVE_TIMEOUT, on_partial, **filtered_kwargs,
                )
            except Exception as exc:
                result_box["error"] = str(exc)
            finally:
                done_event.set()
                wake.set()

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

        if voice and not self._sse("state", {"state": "thinking"}):
            executor.shutdown(wait=False, cancel_futures=True)
            return ("error:client_gone", "")
        if not self._sse("status", {"status": "thinking"}):
            executor.shutdown(wait=False, cancel_futures=True)
            return ("error:client_gone", "")

        def _drain_ready(next_idx: int) -> "int | None":
            """Emit every not-yet-sent future starting at next_idx, IN
            ORDER, blocking on each one's .result() (which is fine — by
            the time a later future exists, the earlier one has almost
            always already finished, since sentences are submitted and
            synthesized in the order they complete). Returns the new
            next_idx, or None if the client went away (caller must stop)."""
            while True:
                with lock:
                    available = len(futures)
                if cancel_event.is_set():
                    return available  # cancelled: skip whatever is still queued
                if next_idx >= available:
                    return next_idx
                fut = futures[next_idx]
                if not _mark_synthesizing():
                    return None
                if text_mode:
                    if not _write_sentence({"text": fut, "engine": "text"}):
                        return None
                    if timing["first_text_event_ms"] is None:
                        timing["first_text_event_ms"] = _ms_since_start()
                    next_idx += 1
                    continue
                if stream_mode:
                    if not _stream_one(fut):
                        return None
                    next_idx += 1
                    continue
                try:
                    wav_bytes, engine, sentence_text = fut.result()
                except Exception as exc:
                    log_fallback(f"stream_sentence_tts_failed:{exc}", tts_engine="stream-skip")
                    next_idx += 1
                    continue
                if not _write_sentence({
                    "text": sentence_text,
                    "engine": engine,
                    "audio_b64": base64.b64encode(wav_bytes).decode("ascii"),
                    **({"format": "wav"} if voice or audio_chunk else {}),
                }):
                    return None
                secs = _wav_seconds(wav_bytes)
                if secs:
                    now = time.time()
                    timing["play_end"] = max(timing["play_end"] or now, now) + secs
                next_idx += 1

        idx = 0
        client_gone = False
        last_status = time.time()
        while not done_event.is_set() and not cancel_event.is_set():
            wake.clear()
            idx = _drain_ready(idx)
            if idx is None:
                client_gone = True
                break
            if not done_event.is_set():
                with lock:
                    nothing_queued = idx >= len(futures)
                if nothing_queued:
                    _maybe_context_opener()
                if nothing_queued and not _maybe_gap_fill():
                    client_gone = True
                    break
                if nothing_queued and progress is not None and not cancel_event.is_set():
                    play_end = timing["play_end"]
                    line = progress.next_line(
                        playing=play_end is not None and play_end > time.time(),
                        reply_started=bool(raw_seen))
                    if line:
                        _submit(line)
            # Wake the moment a sentence is queued (was: a fixed 150 ms sleep
            # that delayed every first sentence by up to one poll interval).
            wake.wait(self.STREAM_STATUS_POLL_SEC)
            if not done_event.is_set() and time.time() - last_status >= self.STREAM_STATUS_POLL_SEC:
                last_status = time.time()
                try:
                    self.wfile.write(self._sse_event("status", {"status": "thinking"}))
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    client_gone = True
                    break

        if client_gone:
            executor.shutdown(wait=False, cancel_futures=True)
            _finish("error:client_gone")
            return ("error:client_gone", "")

        if cancel_event.is_set():
            outcome = "cancelled"
            reply = ""
        elif "error" in result_box:
            outcome = "error:brain_exception"
            reply = APOLOGY
            _submit(reply)
        else:
            raw = result_box.get("reply") or ""
            if raw.startswith("unanswered:"):
                outcome = f"unanswered:{raw.split(':', 1)[1].strip()[:40]}"
                reply = APOLOGY
                _submit(reply)
            elif is_silent_sentinel(raw):
                # Silent sentinel: nothing is submitted for TTS, so nothing is
                # spoken; reply stays "" and the window is not opened (only "ok" does).
                outcome = "silent"
                reply = ""
            elif not raw.strip():
                outcome = "empty_reply"
                reply = APOLOGY
                _submit(reply)
            else:
                outcome = "ok"
                reply = strip_markdown(raw)
                if os.environ.get("MVG_ARTIFACT_WEBHOOK_URL"):
                    reply, _ = forward_artifacts_to_webhook(raw, reply)
                if is_defer_enabled() and (has_defer_marker(raw) or has_defer_marker(reply)):
                    get_pending_store().store(
                        node_id=node_id,
                        prompt=prompt,
                        answer=strip_defer_marker(reply),
                    )
                reply = strip_defer_marker(reply)
                # Only a real answer opens the follow-up window for
                # /ambient — same rule as /ask (see its own comment above).
                # Without this, a hub that switches to /ask/stream loses
                # the follow-up window silently: /ambient would still work,
                # it would just never see a node as "mid-conversation".
                record_reply(node_id, reply)
                # Flush whatever the segmenter never saw a trailing
                # boundary for (the tail of the reply text the streamed
                # on_partial callback couldn't confirm as complete without
                # more text arriving — see SentenceSegmenter.flush).
                tail_delta = unspoken_delta(reply or "", spoken_text)
                if tail_delta.strip():
                    for sentence in segmenter.feed(tail_delta):
                        _submit(sentence)
                tail = segmenter.flush()
                if tail:
                    _submit(tail)

        idx = _drain_ready(idx)
        executor.shutdown(wait=True)
        if idx is None:
            _finish("error:client_gone")
            return ("error:client_gone", reply)
        if outcome == "ok":
            # Follow-up window counts from the end of playback, not from
            # when the reply text was ready (a barge-in only shortens what
            # was heard, so over-estimating is harmless).
            extend_reply_playback(node_id, timing["play_end"])

        if audio_chunk and stream_mode and spoke["chunk_idx"]:
            # The hub's end-of-audio marker: tells the ear's PCM player to drain.
            self._sse("audio_chunk", {"chunk_index": spoke["chunk_idx"], "format": "pcm",
                                      "is_final": True, "data": "",
                                      "sample_rate": AUDIO_TARGET_RATE,
                                      "channels": AUDIO_TARGET_CHANNELS})
        if voice:
            reply_ev = {"reply": reply, "text": reply,
                        "tts_engine": timing["engine"] or ("text" if text_mode else "wav")}
            done_ev = {"duration_ms": int((time.time() - t0) * 1000)}
        else:
            reply_ev = {"reply": reply}
            done_ev = {}
        if mic_reopen_on:
            expects = reply_expects_reply(reply)
            reply_ev["expects_reply"] = expects
            done_ev["expects_reply"] = expects
        self._sse("reply", reply_ev)
        self._sse("done", done_ev)
        _finish(outcome, with_node=True)
        return (outcome, reply)


    # -- /voice/turn (SSE): the whole pipeline, one request per utterance --

    def _handle_voice_turn(self):
        t0 = time.time()
        parsed = urlsplit(self.path)
        qs = parse_qs(parsed.query)
        content_type = self.headers.get("Content-Type", "")
        body = self._read_body()
        wav_bytes, fields = None, {}
        parts: dict = {}
        transcript: str = ""
        image_bytes: bytes | None = None
        burst_images: list[bytes] = []

        enable_adpcm = is_adpcm_enabled()
        enable_vision = is_vision_enabled()

        if content_type.startswith("multipart/form-data"):
            parts = parse_multipart_parts(body, content_type)
            if enable_vision and "burst_images_b64" in parts and isinstance(parts["burst_images_b64"], (bytes, bytearray)):
                try:
                    _parsed_burst = json.loads(parts["burst_images_b64"].decode("utf-8", "replace"))
                    if isinstance(_parsed_burst, list):
                        parts["burst_images_b64"] = _parsed_burst
                except Exception:
                    pass
            if enable_vision:
                image_bytes, burst_images = extract_vision_payloads(parts)
            wav_bytes = parts.pop("audio", None)
            fields = {k: v.decode("utf-8", "replace").strip() if isinstance(v, (bytes, bytearray)) else v for k, v in parts.items()}
            if enable_vision and not image_bytes and not burst_images:
                if "burst_images_b64" in fields and isinstance(fields["burst_images_b64"], str):
                    try:
                        _parsed_burst = json.loads(fields["burst_images_b64"])
                        if isinstance(_parsed_burst, list):
                            fields["burst_images_b64"] = _parsed_burst
                    except Exception:
                        pass
                image_bytes, burst_images = extract_vision_payloads(fields)
        elif content_type in ("audio/wav", "application/octet-stream", ""):
            wav_bytes = body

        def opt(name: str):
            """Multipart field first, then the query string; None if absent."""
            if name in fields:
                return fields[name]
            vals = qs.get(name)
            return vals[0].strip() if vals else None

        def truthy(v, default: bool) -> bool:
            if v is None or v == "":
                return default
            return str(v).strip().lower() in ("1", "true", "on", "yes")

        requested_node = (opt("node_id") or self.headers.get("X-Node-Id") or "").strip()
        forced_node = None
        if self.headers.get("X-Watch-Route") is not None:
            forced_node = watch_node_for(self.headers.get("Authorization", ""), requested_node)
            if forced_node is None:
                self._send_json(401, {"error": "unauthorized"})
                log_request("/voice/turn", (time.time() - t0) * 1000, "error:unauthorized")
                return
        node_id = forced_node or requested_node
        _set_req_node(node_id)

        if not self._check_bearer_auth(node_id, "/voice/turn", t0):
            return

        if enable_adpcm and wav_bytes:
            try:
                wav_bytes = decode_adpcm_to_wav(wav_bytes)
            except ValueError as exc:
                self._send_json(413, {"error": "payload_too_large", "detail": str(exc)})
                log_request("/voice/turn", (time.time() - t0) * 1000, "error:adpcm_too_large", node=node_id)
                return

        if enable_vision and not image_bytes and not burst_images and qs:
            image_bytes, burst_images = extract_vision_payloads({k: v[0] for k, v in qs.items() if v})

        if enable_vision:
            _bad_img = False
            if image_bytes and not is_valid_image_bytes(image_bytes):
                _bad_img = True
            elif burst_images and any(not is_valid_image_bytes(b) for b in burst_images):
                _bad_img = True
            elif content_type.startswith("multipart/form-data"):
                for _k in ("image", "image_0", "image_1", "image_2", "image_3", "image_4", "burst_images"):
                    if _k in parts and parts[_k]:
                        _pv = parts[_k]
                        if isinstance(_pv, (list, tuple)):
                            if any(not is_valid_image_bytes(x) for x in _pv if x):
                                _bad_img = True
                                break
                        elif not is_valid_image_bytes(_pv):
                            _bad_img = True
                            break

            if not _bad_img:
                _b64_val = parts.get("image_b64") or fields.get("image_b64") or (qs.get("image_b64", [None])[0] if qs else None)
                if _b64_val is not None:
                    _s = _b64_val.decode("utf-8", "replace").strip() if isinstance(_b64_val, (bytes, bytearray)) else str(_b64_val).strip()
                    if _s:
                        if not image_bytes:
                            _bad_img = True
                        else:
                            try:
                                _dec = base64.b64decode(_s, validate=True)
                                if not is_valid_image_bytes(_dec):
                                    _bad_img = True
                            except Exception:
                                _bad_img = True

                if not _bad_img:
                    _burst_b64_val = parts.get("burst_images_b64") or fields.get("burst_images_b64") or (qs.get("burst_images_b64", [None])[0] if qs else None)
                    if _burst_b64_val is not None:
                        if isinstance(_burst_b64_val, (list, tuple)):
                            _items = _burst_b64_val
                        else:
                            _s = _burst_b64_val.decode("utf-8", "replace").strip() if isinstance(_burst_b64_val, (bytes, bytearray)) else str(_burst_b64_val).strip()
                            if _s:
                                try:
                                    _parsed = json.loads(_s)
                                    _items = _parsed if isinstance(_parsed, list) else [_s]
                                except Exception:
                                    _items = [_s]
                            else:
                                _items = []
                        if _items:
                            if not burst_images:
                                _bad_img = True
                            else:
                                for _it in _items:
                                    _it_s = _it.decode("utf-8", "replace").strip() if isinstance(_it, (bytes, bytearray)) else str(_it).strip()
                                    try:
                                        _dec = base64.b64decode(_it_s, validate=True)
                                        if not is_valid_image_bytes(_dec):
                                            _bad_img = True
                                            break
                                    except Exception:
                                        _bad_img = True
                                        break
                        elif isinstance(_burst_b64_val, (bytes, bytearray, str)) and _burst_b64_val.strip():
                            _bad_img = True
            if _bad_img:
                self._send_json(400, {"error": "invalid image format (must be JPEG, PNG, or WEBP)"})
                log_request("/voice/turn", (time.time() - t0) * 1000, "error:invalid_image", node=node_id)
                return

        image_b64 = base64.b64encode(image_bytes).decode("ascii") if image_bytes else None
        burst_images_b64 = [base64.b64encode(b).decode("ascii") for b in burst_images] if burst_images else []

        if not node_id:
            self._send_json(400, {"error": "node_id required"})
            log_request("/voice/turn", (time.time() - t0) * 1000, "error:no_node")
            return
        if not wav_bytes:
            self._send_json(400, {"error": "no audio payload found"})
            log_request("/voice/turn", (time.time() - t0) * 1000, "error:no_audio", node=node_id)
            return
        tts = (opt("tts") or "stream").lower()
        if tts not in ("stream", "wav", "none"):
            self._send_json(400, {"error": "tts must be stream, wav or none"})
            log_request("/voice/turn", (time.time() - t0) * 1000, "error:bad_tts", node=node_id)
            return

        prof = resolve_profile(node_id)
        requested_target = opt("target")
        target = resolve_request_target(node_id, requested_target)
        open_room = truthy(opt("open_room"), bool(prof["open_room"]))
        audio_chunk = truthy(opt("audio_chunk"), bool(prof["audio_chunk"]))
        session_id = opt("session_id") or ""
        claimed, claim_rejected = self._trusted_claimed_speaker()
        turn_id = str(uuid.uuid4())
        sha = hashlib.sha256(wav_bytes).hexdigest()
        device, medium = resolve_device_medium(
            node_id, opt("device") or self.headers.get("X-Voice-Device"),
            opt("medium") or self.headers.get("X-Voice-Medium"))
        log_extra = {"node": node_id, "open_room": open_room, "tts": tts,
                     "claimed_speaker_rejected": claim_rejected}
        if device:
            log_extra["device"] = device
        if medium:
            log_extra["medium"] = medium

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        if not self._sse("turn", {"turn_id": turn_id, "node_id": node_id}):
            return

        stt_box: dict[str, Any] = {}
        transcript: str = ""
        spoken_reply: str = ""

        def end(outcome: str, reply: str = "", **extra) -> None:
            log_request("/voice/turn", (time.time() - t0) * 1000, outcome,
                        **{**log_extra, **extra})
            if os.environ.get("MVG_TRANSCRIPT_SINK_URL") or os.environ.get("MVG_TRANSCRIPT_LOG_PATH"):
                prompt_text = extra.pop("prompt", None) or extra.pop("transcript", None) or transcript or stt_box.get("text", "")
                final_reply = reply or extra.pop("reply", "") or spoken_reply
                mic_reopen_on = os.environ.get("MVG_ENABLE_MIC_REOPEN", "0").strip().lower() in ("1", "true", "yes", "on")
                tr_entry: dict[str, Any] = {
                    "kind": "turn",
                    "endpoint": "/voice/turn",
                    "node_id": node_id or None,
                    "session_id": session_id or None,
                    "turn_id": turn_id,
                    "prompt": prompt_text or "",
                    "transcript": prompt_text or "",
                    "reply": final_reply or "",
                    "outcome": outcome,
                    "total_ms": round((time.time() - t0) * 1000, 1),
                    "timestamp": time.time(),
                    **log_extra,
                    **extra,
                }
                if mic_reopen_on and final_reply:
                    tr_entry["expects_reply"] = reply_expects_reply(final_reply)
                emit_turn_transcript(tr_entry)

        # 1 + 2. Speech gate || transcribe || identify. All three start together so
        # the gate (~1.3 s on slow hardware) adds no latency; a rejected clip just
        # discards the transcript. Voice ID gets VOICE_ID_GRACE_SEC after STT.
        id_box: dict = {}
        id_thread = None
        if prof["identify"]:
            def _id_worker():
                id_box["r"] = identify_speaker_detail(wav_bytes, sha)
            id_thread = threading.Thread(target=_id_worker, daemon=True)
            id_thread.start()
        stt_box.clear()
        t_stt = time.time()

        def _stt_worker():
            try:
                stt_box["text"] = transcribe(wav_bytes)
            except Exception as exc:  # surfaced below
                stt_box["exc"] = exc
            stt_box["ms"] = (time.time() - t_stt) * 1000
        stt_thread = threading.Thread(target=_stt_worker, daemon=True)
        stt_thread.start()

        gate_extra = {}
        if prof["speech_gate"]:
            gate_status, voiced, vad_ms = speech_gate(wav_bytes)
            gate_extra = {"speech_gate": gate_status, "vad_ms": vad_ms}
            if voiced is not None:
                gate_extra["voiced_ms"] = voiced
            if gate_status == "rejected":
                self._sse("ignored", {"reason": "no-speech", "voiced_ms": voiced})
                end("ok:ignored:no-speech", **gate_extra)
                return

        if not self._sse("state", {"state": "transcribing"}):
            return
        stt_thread.join()
        if "exc" in stt_box:
            self._sse("error", {"error": "stt_error", "message": str(stt_box["exc"])})
            end("error:stt", **gate_extra)
            return
        transcript = stt_box.get("text", "")
        ms_stt = stt_box["ms"]
        _note_stt(node_id, _wav_seconds(wav_bytes), ms_stt, len(transcript or ""), False)
        if id_thread is not None:
            id_thread.join(VOICE_ID_GRACE_SEC)
        vid = id_box.get("r") or {"status": "unavailable", "id": "", "score": 0.0}
        verdict = voice_id_verdict(vid)
        speaker_id = vid["id"] if vid["status"] == "verified" else ""
        if not self._sse("transcript", {
            "text": transcript, "transcript": transcript,
            "speaker": claimed or speaker_id, "voice_id": verdict,
        }):
            return
        id_extra = {"voice_id_status": vid["status"],
                    "voice_id_score": round(float(vid.get("score") or 0.0), 3),
                    "voice_id_best": vid.get("best") or "",
                    "stt_ms": round(ms_stt, 1), **gate_extra}
        if prof.get("capture_audio") and gate_extra.get("speech_gate") == "passed":
            capture_turn_audio(node_id, turn_id, wav_bytes, transcript, vid)

        # 3. Hallucination filter
        if is_hallucination_or_too_short(transcript):
            self._sse("ignored", {"reason": "hallucination"})
            end("ok:ignored:hallucination", chars=len(transcript or ""), **id_extra)
            return

        # 3b. Echo filter: the mic heard the agent's own reply
        if is_reply_echo(node_id, transcript):
            self._sse("ignored", {"reason": "reply-echo"})
            end("ok:ignored:reply-echo", chars=len(transcript or ""), **id_extra)
            return

        # 4. Addressed gate (open room only)
        if open_room:
            if not self._sse("state", {"state": "classifying"}):
                return
            detail = decide_addressed(node_id, transcript, speaker_id)
            score, classifier = detail["score"], detail["classifier"]
            _record_ambient_transcript(node_id, transcript, speaker_id)
            if classifier == "none" or score < AMBIENT_THRESHOLD:
                self._sse("ignored", {"reason": "not-addressed", "score": round(score, 4),
                                      "classifier": classifier})
                end("ok:ignored:not-addressed", score=round(score, 4), classifier=classifier,
                    **id_extra)
                return
        else:
            _record_ambient_transcript(node_id, transcript, speaker_id)

        # 5 + 6. Brain, then speak
        brain_direct_audio = bool(BRAIN_DIRECT_AUDIO) or (os.environ.get("MVG_BRAIN_DIRECT_AUDIO", "0") == "1")
        audio_b64 = None
        if brain_direct_audio and wav_bytes:
            audio_b64 = base64.b64encode(wav_bytes).decode("ascii")
        ctx = make_frame_ctx(vid, claimed_speaker=claimed, open_room=open_room,
                             reply_cap_words=int(prof["reply_cap_words"]),
                             device=device, medium=medium, target=target,
                             image_b64=image_b64, burst_images_b64=burst_images_b64,
                             audio_b64=audio_b64, audio_clips=collect_audio_clips(wav_bytes))
        target_extra = {"target_rejected": True} if (
            requested_target and requested_target != target) else {}
        turn_opts = {"voice": True, "tts": tts, "gap_fill": bool(prof["gap_fill"]),
                     "audio_chunk": audio_chunk}
        if spoken_cancel_enabled() and is_cancel_phrase(transcript):
            # Cancel the node's running turn first; the acknowledgement then runs as
            # a fixed-reply turn. No running turn: fall through as an ordinary turn.
            cancel_result = cancel_active_turn(node_id)
            ack = _CANCEL_ACKS.get(cancel_result)
            if ack:
                turn_opts["intent_result"] = {"hit": True, "intent": "cancel", "match_ms": 0.0,
                                              "handler_ms": 0.0, "reply": ack, "reason": "spoken_cancel"}
                turn_opts["gap_fill"] = False
        self._stream_brain_turn(
            t0=t0, turn_id=turn_id, prompt=transcript, session_id=session_id, node_id=node_id,
            speaker=speaker_id, speaker_score=vid.get("score"), ctx=ctx, endpoint="/voice/turn",
            opts=turn_opts,
            log_extra={**log_extra, **id_extra, "target": target, **target_extra})

    # -- /voice/live (WebSocket, live_ws.py) --

    def _handle_voice_live(self):
        import asyncio
        import live_ws
        t0 = time.time()
        qs = parse_qs(urlsplit(self.path).query)
        node_id = ((qs.get("node_id") or [""])[0] or self.headers.get("X-Node-Id") or "").strip()
        if not self._check_bearer_auth(node_id, "/voice/live", t0):
            return
        profile = resolve_profile(node_id)
        voice_mode = str(profile.get("voice_mode") or "native").strip()
        if voice_mode not in ("native", "custom_tts"):
            self._send_json(400, {"error": f"invalid voice_mode: {voice_mode} (expected native or custom_tts)"})
            log_request("/voice/live", (time.time() - t0) * 1000, "error:bad_voice_mode", node=node_id)
            return
        backend_name = (profile.get("talker_backend")
                        or os.environ.get("MVG_TALKER_BACKEND", "")).strip()
        if not backend_name:
            self._send_json(501, {"error": "live voice is not configured: set MVG_TALKER_BACKEND "
                                           "(or talker_backend on the node)"})
            log_request("/voice/live", (time.time() - t0) * 1000, "error:not_configured", node=node_id)
            return
        if not live_ws.is_upgrade_request(self.headers):
            self._send_json(426, {"error": "WebSocket upgrade required"})
            return
        if not node_id:
            self._send_json(400, {"error": "node_id required"})
            return
        try:
            import talker_backend
            import live_harness
            backend_cls = talker_backend.load_backend(backend_name)
        except ValueError as exc:
            self._send_json(501, {"error": str(exc)})
            log_request("/voice/live", (time.time() - t0) * 1000, "error:bad_backend", node=node_id)
            return
        self.wfile.write(live_ws.handshake_response(self.headers["Sec-WebSocket-Key"]))
        self.wfile.flush()
        self.close_connection = True
        ws = live_ws.WSConn(self.connection, live_ws._env_num("MVG_LIVE_MAX_FRAME", "262144", int))
        target = resolve_request_target(node_id, None)
        session_id = f"live-{uuid.uuid4().hex[:8]}"
        instr = _resolve_talker_instructions(target, node_id)
        session_holder: dict = {}
        session_loop: list = [None]
        default_brain_call = http_brain_call if MVG_BRAIN == "http" else plugin_brain_call

        class _LiveEventList(list):
            def __init__(self, on_event):
                super().__init__()
                self._on_event = on_event

            def append(self, item):
                super().append(item)
                self._on_event(item)

            def extend(self, items):
                for item in items:
                    self.append(item)

        class _LiveStageSink(dict):
            def __init__(self, on_event):
                super().__init__()
                self._on_event = on_event
                super().__setitem__("events", _LiveEventList(on_event))

            def __setitem__(self, key, val):
                if key == "events" and isinstance(val, list) and not isinstance(val, _LiveEventList):
                    el = _LiveEventList(self._on_event)
                    for item in val:
                        el.append(item)
                    val = el
                super().__setitem__(key, val)

            def setdefault(self, key, default=None):
                if key not in self:
                    self[key] = default
                return self[key]

            def append(self, item):
                self["events"].append(item)

        def _forward_stage_event(ev):
            norm = normalize_stage_event(ev)
            sess = session_holder.get("session")
            loop = session_loop[0]
            if norm and sess is not None and loop is not None and not loop.is_closed():
                def _schedule(s=sess, n=norm):
                    if not s._closed:
                        asyncio.ensure_future(s.on_stage_event(n))
                try:
                    loop.call_soon_threadsafe(_schedule)
                except RuntimeError:
                    pass

        def brain(request, task_id):
            task = _register_task(node_id, target, task_id=task_id)
            tid = task["task_id"]
            stages_out = _LiveStageSink(_forward_stage_event)
            _STAGE_SINK.stages = stages_out
            try:
                use_stream = (
                    BRAIN_CALL is default_brain_call
                    and (
                        STREAM_BRAIN_CALL is not plugin_brain_call_stream
                        or _plugin_brain_call_stream is not None
                    )
                )
                if use_stream:
                    call_kwargs = {"task_id": tid, "stages_out": stages_out}
                    try:
                        params = inspect.signature(STREAM_BRAIN_CALL).parameters
                        has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
                        filtered = {k: v for k, v in call_kwargs.items() if has_var_kw or k in params}
                    except (ValueError, TypeError):
                        filtered = {}
                    return STREAM_BRAIN_CALL(
                        request, session_id, node_id, "", None, HIVE_TIMEOUT,
                        lambda _partial: None, **filtered,
                    )
                call_kwargs = {"task_id": tid, "stages_out": stages_out}
                try:
                    params = inspect.signature(BRAIN_CALL).parameters
                    has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
                    filtered = {k: v for k, v in call_kwargs.items() if has_var_kw or k in params}
                except (ValueError, TypeError):
                    filtered = {}
                return BRAIN_CALL(request, session_id, node_id, "", None, HIVE_TIMEOUT, **filtered)
            finally:
                _STAGE_SINK.stages = None
                _unregister_task(node_id, task)

        def factory(send_to_client):
            try:
                session_loop[0] = asyncio.get_running_loop()
            except RuntimeError:
                session_loop[0] = None
            cfg = {"node_id": node_id, "target": target, "voice_mode": voice_mode}
            if instr:
                cfg["instructions"] = instr
            sess = live_harness.LiveSession(
                backend_cls(), cfg, send_to_client, brain,
                getattr(sys.modules[__name__], "cancel_active_turn", None),
                lambda text: synthesize(text)[0])
            session_holder["session"] = sess
            return sess
        outcome = live_ws.run_live_session(ws, factory)
        log_request("/voice/live", (time.time() - t0) * 1000, outcome, node=node_id, backend=backend_name)

    # -- /v1/audio/speech --

    def _handle_speech(self):
        t0 = time.time()
        body = self._read_body()
        try:
            req = json.loads(body.decode("utf-8")) if body else {}
            if not isinstance(req, dict):
                req = {}
        except Exception:
            req = {}
        node_id = (req.get("node") or req.get("node_id") or self.headers.get("X-Node-Id") or "").strip()
        if not self._check_bearer_auth(node_id, "/v1/audio/speech", t0):
            return
        text = (req.get("input") or "").strip()
        text = strip_defer_marker(text)
        if not text:
            self._send_json(400, {"error": "input is required"})
            log_request("/v1/audio/speech", (time.time() - t0) * 1000, "error:no_input")
            return
        if TEXT_SENTENCES and req.get("stream") not in (False, 0, "false", "0"):
            self._handle_speech_stream(text, t0)
            return
        try:
            wav_bytes, engine = synthesize_chunked(text)
        except Exception as exc:
            self._send_json(502, {"error": f"tts failed: {exc}"})
            log_request("/v1/audio/speech", (time.time() - t0) * 1000, "error:both_engines_failed")
            return
        # MIRRORMERE-ONE-FORMAT: always hand back 24 kHz mono 16-bit WAV.
        wav_bytes = NORMALIZE_WAV(wav_bytes)
        self._send_bytes(200, wav_bytes, "audio/wav", {"X-TTS-Engine": engine})
        log_request("/v1/audio/speech", (time.time() - t0) * 1000, "ok", engine=engine, chars=len(text))

    def _handle_speech_stream(self, text: str, t0: float) -> None:
        """Streamed /v1/audio/speech (MVG_TEXT_SENTENCES): a 44-byte WAV
        header with unknown (0xFFFFFFFF) sizes, then 24 kHz mono s16le PCM as
        it is synthesized. This is the shape the hub's streaming
        TTS client wants (standard RIFF/WAVE/fmt /data header, PCM format 1,
        rate and channels read from the header, the rest forwarded as PCM).
        synthesize_stream() reads the first audio chunk before returning, so
        every fallback (desktop stream -> desktop /tts -> local Piper) still
        happens before any byte is sent, and a total failure is
        still a clean 502."""
        try:
            engine, chunks = synthesize_stream(text)
        except Exception as exc:
            self._send_json(502, {"error": f"tts failed: {exc}"})
            log_request("/v1/audio/speech", (time.time() - t0) * 1000, "error:both_engines_failed")
            return
        outcome = "ok"
        nbytes = 0
        try:
            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("X-TTS-Engine", engine)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(streaming_wav_header(AUDIO_TARGET_RATE, AUDIO_TARGET_CHANNELS))
            self.wfile.flush()
            for pcm in chunks:
                if not pcm:
                    continue
                self.wfile.write(pcm)
                self.wfile.flush()
                nbytes += len(pcm)
        except (BrokenPipeError, ConnectionResetError, OSError):
            outcome = "error:client_gone"
        except Exception as exc:
            # Mid-stream upstream failure: what was sent stays sent.
            log_fallback(f"speech_stream_aborted:{str(exc)[:200]}", tts_engine="stream-abort")
            outcome = "error:stream_aborted"
        finally:
            close = getattr(chunks, "close", None)
            if close:
                close()
        log_request("/v1/audio/speech", (time.time() - t0) * 1000, outcome,
                    engine=engine, chars=len(text), streamed=True, audio_bytes=nbytes)

    # -- /embed (SPEC-011 speaker identification) --

    def _handle_embed(self):
        t0 = time.time()
        node_id = (self.headers.get("X-Node-Id") or
                   (parse_qs(urlsplit(self.path).query).get("node_id") or [""])[0]).strip()
        if not self._check_bearer_auth(node_id, "/embed", t0):
            return
        wav_bytes = self._read_body()
        if not wav_bytes:
            self._send_json(400, {"error": "no audio payload found"})
            log_request("/embed", (time.time() - t0) * 1000, "error:no_audio")
            return
        status, body = proxy_embed(wav_bytes)
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass
        outcome = "ok" if status == 200 else f"error:upstream_{status}"
        log_request("/embed", (time.time() - t0) * 1000, outcome, bytes=len(wav_bytes))

    # -- /v1/systemone (Jev wire protocol -> desktop Laya classifier) --

    def _handle_systemone(self):
        t0 = time.time()
        body = self._read_body()
        if not body:
            self._send_json(400, {"error": "no body"})
            log_request("/v1/systemone", (time.time() - t0) * 1000, "error:no_body")
            return
        status, out = proxy_systemone(body)
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        try:
            self.wfile.write(out)
        except (BrokenPipeError, ConnectionResetError):
            pass
        outcome = "ok" if status == 200 else f"error:upstream_{status}"
        log_request("/v1/systemone", (time.time() - t0) * 1000, outcome, bytes=len(body))


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def _probe(url: str) -> str:
    """GET with a 1 s timeout. Any HTTP answer, even an error status such as
    405 from a POST-only endpoint, means the server is up."""
    try:
        with urlopen(url, timeout=1):
            return "reachable"
    except HTTPError:
        return "reachable"
    except Exception:
        return "unreachable"


_TTS_REQUIRED_ENV = {
    "elevenlabs": ("ELEVENLABS_API_KEY", "MVG_ELEVENLABS_VOICE_ID"),
    "openai": ("MVG_OPENAI_TTS_URL",),
    "http": ("MVG_HTTP_TTS_URL",),
}


def _tts_entry_status(name: str) -> str:
    if name == "desktop":
        return _probe(DESKTOP_TTS_URL)
    if name == "piper":
        return "present" if os.path.exists(_piper_binary_path()) else "missing"
    required = _TTS_REQUIRED_ENV.get(name)
    if required is None:
        return "unknown"
    return "configured" if all(os.environ.get(k) for k in required) else "missing config"


def startup_lines() -> list[str]:
    """One status line per backend, printed by main() before serving. Probes
    use a 1 s timeout and never raise; nothing here calls a paid API."""
    def safe(fn):
        try:
            return fn()
        except Exception as exc:
            return f"probe error ({type(exc).__name__})"

    speaker_on = os.environ.get("MVG_SPEAKER_ID", "1") != "0"
    return [
        f"stt: {WHISPER_URL} " + safe(lambda: _probe(WHISPER_HEALTH_URL)),
        "tts: chain=" + ",".join(TTS_CHAIN) + "".join(
            f" {n}:" + safe(lambda n=n: _tts_entry_status(n)) for n in TTS_CHAIN),
        f"speaker-id: {'on' if speaker_on else 'off'} embed={DESKTOP_EMBED_URL} "
        + safe(lambda: _probe(DESKTOP_EMBED_URL)),
        "classifier: " + ("present" if _jev_classifier is not None else "absent (name-only)"),
        f"brain: {MVG_BRAIN} {MVG_BRAIN_URL or DEFAULT_TARGET}",
    ]


def main():
    server = Server((BIND, PORT), GatewayHandler)
    print(f"[lomin] listening on {BIND}:{PORT}", flush=True)
    for line in startup_lines():
        print(line, flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
