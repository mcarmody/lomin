"""Gemini Live Talker backend (stdlib only).

Implements talker_backend.TalkerBackend over the Gemini Live WebSocket API
(BidiGenerateContent) with a small asyncio RFC 6455 client, so no vendor SDK or
`websockets` package is needed. Register with
`MVG_TALKER_BACKENDS=gemini=gemini_live:GeminiLiveBackend`, then
`MVG_TALKER_BACKEND=gemini`.

Env: MVG_GEMINI_API_KEY (required), MVG_GEMINI_MODEL (default below),
MVG_GEMINI_VOICE (optional prebuilt voice name), MVG_GEMINI_URL (override the
endpoint, used by tests and proxies), MVG_GEMINI_INSTRUCTIONS (system prompt).

The wire format follows Google's public Live API reference. The audio mime type
(`audio/pcm;rate=16000`), 24 kHz PCM16 output and the FunctionCall/
FunctionResponse field names come from that reference's generate-content types;
treat them as unverified against the live service until a key-backed run.
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import ssl
import struct
from typing import AsyncIterator
from urllib.parse import urlsplit

from talker_backend import TalkerBackend

DEFAULT_URL = ("wss://generativelanguage.googleapis.com/ws/"
               "google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent")
DEFAULT_MODEL = "gemini-3.8-live"
MAX_FRAME = 16 * 1024 * 1024

TOOL_DECLARATIONS = [
    {"name": "call_thinker", "description": "Hand a request to the slower thinking agent. Returns at once; the answer arrives later as context.",
     "parameters": {"type": "OBJECT", "properties": {"request": {"type": "STRING"}}, "required": ["request"]}},
    {"name": "cancel_thinker", "description": "Cancel the running thinking-agent task.",
     "parameters": {"type": "OBJECT", "properties": {}}},
    {"name": "answer_prompt", "description": "Answer a pending approval prompt.",
     "parameters": {"type": "OBJECT", "properties": {"prompt_id": {"type": "STRING"}, "decision": {"type": "STRING"}},
                    "required": ["prompt_id", "decision"]}},
    {"name": "set_progress_updates", "description": "Turn spoken progress updates on or off.",
     "parameters": {"type": "OBJECT", "properties": {"enabled": {"type": "BOOLEAN"}}, "required": ["enabled"]}},
]


class WSClient:
    """Minimal asyncio WebSocket client: text/binary messages, ping/pong, close."""

    def __init__(self, reader, writer):
        self.r, self.w = reader, writer

    @classmethod
    async def connect(cls, url: str, timeout: float = 15.0) -> "WSClient":
        u = urlsplit(url)
        secure = u.scheme == "wss"
        port = u.port or (443 if secure else 80)
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(u.hostname, port, ssl=ssl.create_default_context() if secure else None), timeout)
        key = base64.b64encode(os.urandom(16)).decode()
        path = (u.path or "/") + (f"?{u.query}" if u.query else "")
        writer.write((f"GET {path} HTTP/1.1\r\nHost: {u.netloc}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
                      f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n").encode())
        await writer.drain()
        head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), timeout)
        if b" 101 " not in head.split(b"\r\n", 1)[0]:
            writer.close()
            raise ConnectionError("websocket upgrade refused: " + head.split(b"\r\n", 1)[0].decode("latin-1")[:80])
        return cls(reader, writer)

    async def _send(self, op: int, payload: bytes) -> None:
        n = len(payload)
        hdr = bytes([0x80 | op])
        if n < 126:
            hdr += bytes([0x80 | n])
        elif n < 65536:
            hdr += bytes([0x80 | 126]) + struct.pack(">H", n)
        else:
            hdr += bytes([0x80 | 127]) + struct.pack(">Q", n)
        mask = os.urandom(4)
        masked = bytes(b ^ mask[i % 4] for i, b in enumerate(payload)) if n < 4096 else _mask_big(payload, mask)
        self.w.write(hdr + mask + masked)
        await self.w.drain()

    async def send_json(self, obj: dict) -> None:
        await self._send(0x1, json.dumps(obj).encode())

    async def recv(self) -> str | bytes | None:
        """Next data message (str for text, bytes for binary); None when closed."""
        buf, first_op = b"", None
        while True:
            try:
                b1, b2 = await self.r.readexactly(2)
                n = b2 & 0x7F
                if n == 126:
                    n = struct.unpack(">H", await self.r.readexactly(2))[0]
                elif n == 127:
                    n = struct.unpack(">Q", await self.r.readexactly(8))[0]
                if n > MAX_FRAME:
                    return None
                mask = await self.r.readexactly(4) if b2 & 0x80 else None
                data = await self.r.readexactly(n)
            except (asyncio.IncompleteReadError, ConnectionError):
                return None
            if mask:
                data = _mask_big(data, mask)
            op = b1 & 0x0F
            if op == 0x8:
                return None
            if op == 0x9:
                await self._send(0xA, data)
                continue
            if op == 0xA:
                continue
            if op != 0x0:
                first_op = op
            buf += data
            if b1 & 0x80:
                return buf.decode("utf-8", "replace") if first_op == 0x1 else buf

    async def close(self) -> None:
        try:
            await self._send(0x8, struct.pack(">H", 1000))
        except Exception:  # noqa: BLE001
            pass
        try:
            self.w.close()
        except Exception:  # noqa: BLE001
            pass


def _mask_big(data: bytes, mask: bytes) -> bytes:
    n = len(data)
    if not n:
        return data
    key = (mask * (n // 4 + 1))[:n]
    return (int.from_bytes(data, "big") ^ int.from_bytes(key, "big")).to_bytes(n, "big")


def server_message_events(msg: dict, call_names: dict) -> list[dict]:
    """Translate one decoded Gemini server message into Talker events."""
    out: list[dict] = []
    sc = msg.get("serverContent")
    if isinstance(sc, dict):
        if sc.get("interrupted"):
            out.append({"type": "interrupted"})
        for part in ((sc.get("modelTurn") or {}).get("parts") or []):
            inline = part.get("inlineData")
            if inline and str(inline.get("mimeType", "")).startswith("audio/"):
                try:
                    out.append({"type": "audio", "pcm": base64.b64decode(inline.get("data") or "")})
                except ValueError:
                    pass
            elif isinstance(part.get("text"), str) and part["text"]:
                out.append({"type": "text", "text": part["text"], "spoken": False})
        if sc.get("turnComplete"):
            out.append({"type": "turn_end"})
    tc = msg.get("toolCall")
    if isinstance(tc, dict):
        for fc in tc.get("functionCalls") or []:
            cid = str(fc.get("id") or "")
            call_names[cid] = fc.get("name") or ""
            out.append({"type": "tool_call", "call_id": cid, "name": fc.get("name"), "args": fc.get("args") or {}})
    sru = msg.get("sessionResumptionUpdate")
    if isinstance(sru, dict) and sru.get("resumable") and sru.get("newHandle"):
        out.append({"type": "resume_handle", "handle": str(sru["newHandle"])})
    if "goAway" in msg:
        out.append({"type": "reconnect", "time_left": str((msg.get("goAway") or {}).get("timeLeft", ""))})
    return out


class GeminiLiveBackend(TalkerBackend):
    capabilities = frozenset({"audio_in", "audio_out", "text_out", "full_duplex", "tool_calls", "silent_context"})

    def __init__(self, connect=None):
        self._connect = connect or WSClient.connect  # injectable for tests
        self._ws = None
        self._call_names: dict = {}
        self._closed = False
        self.resume_handle: str | None = None
        self._config: dict = {}

    async def open(self, config: dict) -> None:
        self._config = dict(config or {})
        await self._handshake()

    async def _handshake(self) -> None:
        key = os.environ.get("MVG_GEMINI_API_KEY", "")
        url = os.environ.get("MVG_GEMINI_URL") or DEFAULT_URL
        if not key and "MVG_GEMINI_URL" not in os.environ:
            raise ValueError("MVG_GEMINI_API_KEY is not set")
        if key:
            url += ("&" if "?" in url else "?") + "key=" + key
        model = os.environ.get("MVG_GEMINI_MODEL", DEFAULT_MODEL)
        voice_mode = self._config.get("voice_mode") or "native"
        if voice_mode == "custom_tts":
            gen = {"responseModalities": ["TEXT"]}
        else:
            gen = {"responseModalities": ["AUDIO"]}
            voice = os.environ.get("MVG_GEMINI_VOICE") or self._config.get("live_voice")
            if voice:
                gen["speechConfig"] = {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": voice}}}
        setup = {"model": model if model.startswith("models/") else f"models/{model}",
                 "generationConfig": gen,
                 "tools": [{"functionDeclarations": TOOL_DECLARATIONS}],
                 "contextWindowCompression": {"slidingWindow": {}},
                 "sessionResumption": {"handle": self.resume_handle} if self.resume_handle else {}}
        instr = self._config.get("instructions") or os.environ.get("MVG_GEMINI_INSTRUCTIONS")
        if instr:
            setup["systemInstruction"] = {"parts": [{"text": str(instr)}]}
        self._ws = await self._connect(url)
        await self._ws.send_json({"setup": setup})
        first = await self._ws.recv()
        if first is None or "setupComplete" not in _loads(first):
            raise ConnectionError("gemini setup did not complete")

    async def reconnect(self) -> None:
        if self._ws is not None:
            await self._ws.close()
            self._ws = None
        await self._handshake()

    async def send_audio(self, pcm16_16k: bytes) -> None:
        await self._ws.send_json({"realtimeInput": {"audio": {
            "mimeType": "audio/pcm;rate=16000", "data": base64.b64encode(pcm16_16k).decode()}}})

    async def send_text(self, text: str, *, silent: bool = False) -> None:
        await self._ws.send_json({"clientContent": {
            "turns": [{"role": "user", "parts": [{"text": text}]}], "turnComplete": not silent}})

    async def send_tool_result(self, call_id: str, result: dict) -> None:
        await self._ws.send_json({"toolResponse": {"functionResponses": [{
            "id": call_id, "name": self._call_names.pop(call_id, ""), "response": result}]}})

    async def events(self) -> AsyncIterator[dict]:
        while not self._closed:
            raw = await self._ws.recv()
            if raw is None:
                if not self._closed:
                    yield {"type": "error", "message": "gemini connection closed"}
                return
            for ev in server_message_events(_loads(raw), self._call_names):
                if ev.get("type") == "resume_handle":
                    self.resume_handle = ev["handle"]
                yield ev

    async def cancel(self) -> None:
        # Gemini has no explicit cancel frame; a new realtime input interrupts generation.
        return None

    async def close(self) -> None:
        self._closed = True
        if self._ws is not None:
            await self._ws.close()


def _loads(raw) -> dict:
    try:
        d = json.loads(raw if isinstance(raw, str) else raw.decode("utf-8", "replace"))
        return d if isinstance(d, dict) else {}
    except ValueError:
        return {}
