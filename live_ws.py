"""Stdlib WebSocket (RFC 6455, server side) and the `/voice/live` bridge.

Decision (issue C5): a small stdlib implementation, not the optional
`websockets` package. The gateway is a blocking ThreadingHTTPServer, so the
handler thread can own the socket directly; `websockets` is asyncio-only and
would need a second server or a hijacked socket. Server side needs only the
handshake, masked-frame reading, ping/pong, close and unfragmented writes.

Frames on /voice/live: binary client->server = 16 kHz mono PCM16; binary
server->client = 24 kHz mono PCM16 from the Talker; text frames = JSON
`{"event": ...}` control messages.

Close codes: 1000 normal, 1009 frame too large, 1011 internal/backend error,
4000 idle timeout (MVG_LIVE_IDLE_SEC), 4001 max duration (MVG_LIVE_MAX_SEC).
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import socket
import struct
import threading
import time

GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
OP_CONT, OP_TEXT, OP_BIN, OP_CLOSE, OP_PING, OP_PONG = 0x0, 0x1, 0x2, 0x8, 0x9, 0xA

CLOSE_NORMAL, CLOSE_TOO_BIG, CLOSE_ERROR, CLOSE_IDLE, CLOSE_MAX_SEC = 1000, 1009, 1011, 4000, 4001


def _env_num(name, default, cast):
    try:
        return cast(os.environ.get(name, "").strip() or default)
    except ValueError:
        return cast(default)


def accept_key(client_key: str) -> str:
    return base64.b64encode(hashlib.sha1((client_key + GUID).encode()).digest()).decode()


def is_upgrade_request(headers) -> bool:
    return ("websocket" in (headers.get("Upgrade") or "").lower()
            and "upgrade" in (headers.get("Connection") or "").lower()
            and bool(headers.get("Sec-WebSocket-Key"))
            and (headers.get("Sec-WebSocket-Version") or "") == "13")


class FrameTooLarge(Exception):
    pass


class WSConn:
    """Blocking server-side WebSocket over a socket."""

    def __init__(self, sock: socket.socket, max_frame: int):
        self.sock = sock
        self.max_frame = max_frame
        self._wlock = threading.Lock()
        self.closed = False

    def _read(self, n: int) -> bytes:
        buf = b""
        while len(buf) < n:
            chunk = self.sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("peer closed")
            buf += chunk
        return buf

    def recv(self):
        """Next data message as (opcode, payload); None on close. Answers pings.
        Raises socket.timeout on idle, FrameTooLarge over max_frame."""
        msg, opcode = b"", None
        while True:
            b1, b2 = self._read(2)
            fin, op = bool(b1 & 0x80), b1 & 0x0F
            masked, ln = bool(b2 & 0x80), b2 & 0x7F
            if ln == 126:
                ln = struct.unpack(">H", self._read(2))[0]
            elif ln == 127:
                ln = struct.unpack(">Q", self._read(8))[0]
            if ln > self.max_frame or len(msg) + ln > self.max_frame:
                raise FrameTooLarge(ln)
            if not masked:
                raise ConnectionError("unmasked client frame")
            mask = self._read(4)
            data = bytearray(self._read(ln))
            for i in range(ln):
                data[i] ^= mask[i % 4]
            data = bytes(data)
            if op == OP_PING:
                self._send(OP_PONG, data)
                continue
            if op == OP_PONG:
                continue
            if op == OP_CLOSE:
                return None
            if op in (OP_TEXT, OP_BIN):
                opcode, msg = op, data
            elif op == OP_CONT and opcode is not None:
                msg += data
            else:
                raise ConnectionError("bad frame sequence")
            if fin:
                return opcode, msg

    def _send(self, op: int, payload: bytes) -> None:
        n = len(payload)
        head = bytes([0x80 | op])
        if n < 126:
            head += bytes([n])
        elif n < 65536:
            head += bytes([126]) + struct.pack(">H", n)
        else:
            head += bytes([127]) + struct.pack(">Q", n)
        with self._wlock:
            if not self.closed:
                self.sock.sendall(head + payload)

    def send_binary(self, data: bytes) -> None:
        self._send(OP_BIN, data)

    def send_json(self, obj: dict) -> None:
        self._send(OP_TEXT, json.dumps(obj).encode())

    def close(self, code: int = CLOSE_NORMAL, reason: str = "") -> None:
        try:
            self._send(OP_CLOSE, struct.pack(">H", code) + reason.encode()[:100])
        except OSError:
            pass
        self.closed = True


def handshake_response(client_key: str) -> bytes:
    return ("HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
            f"Sec-WebSocket-Accept: {accept_key(client_key)}\r\n\r\n").encode()


def _fallback(ws, reason: str) -> None:
    """Tell the client to continue on the request/response pipeline. Best effort."""
    try:
        ws.send_json({"event": "fallback", "endpoint": "/voice/turn", "reason": str(reason)[:200]})
    except Exception:  # noqa: BLE001
        pass


def run_live_session(ws: WSConn, session_factory, *, max_sec=None, idle_sec=None) -> str:
    """Bridge a WSConn to a LiveSession until it ends. session_factory(send_to_client)
    returns an unopened LiveSession. Returns the outcome string for the log."""
    max_sec = _env_num("MVG_LIVE_MAX_SEC", "600", float) if max_sec is None else max_sec
    idle_sec = _env_num("MVG_LIVE_IDLE_SEC", "30", float) if idle_sec is None else idle_sec
    outcome = {"v": "ok", "code": CLOSE_NORMAL}

    def to_client(frame: dict) -> None:
        if frame.get("type") == "audio":
            ws.send_binary(frame.get("pcm") or b"")
        else:
            ws.send_json({"event": frame.get("type"), **{k: v for k, v in frame.items() if k not in ("type", "pcm")}})

    async def main():
        loop = asyncio.get_running_loop()
        inbox: asyncio.Queue = asyncio.Queue()
        session = session_factory(to_client)
        try:
            await session.open()
        except Exception as exc:  # noqa: BLE001
            _fallback(ws, f"open_failed: {type(exc).__name__}")
            outcome.update(v="error:open_failed", code=CLOSE_ERROR)
            await session.close()
            return
        ws.sock.settimeout(idle_sec)

        def post(m):
            try:
                loop.call_soon_threadsafe(inbox.put_nowait, m)
            except RuntimeError:  # loop already closed: the session is over
                pass

        def reader():
            try:
                while True:
                    m = ws.recv()
                    post(m)
                    if m is None:
                        return
            except socket.timeout:
                post(("idle", None))
            except FrameTooLarge:
                post(("big", None))
            except (ConnectionError, OSError):
                post(None)

        threading.Thread(target=reader, daemon=True, name="live-ws-reader").start()
        events = asyncio.ensure_future(session.run())
        deadline = time.monotonic() + max_sec
        try:
            while True:
                getter = asyncio.ensure_future(inbox.get())
                done, _ = await asyncio.wait({getter, events}, timeout=max(0.0, deadline - time.monotonic()),
                                             return_when=asyncio.FIRST_COMPLETED)
                if not done:
                    getter.cancel()
                    outcome.update(v="max_duration", code=CLOSE_MAX_SEC)
                    return
                if events in done and getter not in done:
                    getter.cancel()
                    if session.error:
                        _fallback(ws, session.error)
                        outcome.update(v="error:backend", code=CLOSE_ERROR)
                    return
                m = getter.result()
                if m is None:
                    outcome.update(v="client_closed")
                    return
                op, payload = m
                if op == "idle":
                    outcome.update(v="idle_timeout", code=CLOSE_IDLE)
                    return
                if op == "big":
                    outcome.update(v="frame_too_large", code=CLOSE_TOO_BIG)
                    return
                if op == OP_BIN:
                    await session.feed_audio(payload)
                elif op == OP_TEXT:
                    try:
                        msg = json.loads(payload.decode("utf-8"))
                    except ValueError:
                        continue
                    ev = msg.get("event") if isinstance(msg, dict) else None
                    if ev == "status":
                        await session.status_request()
                    elif ev == "text":
                        if isinstance(msg.get("text"), str) and msg["text"].strip() and hasattr(session, "feed_text"):
                            await session.feed_text(msg["text"], silent=bool(msg.get("silent", False)))
                    elif ev == "audio_end":
                        if hasattr(session, "flush_audio"):
                            await session.flush_audio()
                    elif ev == "end":
                        outcome.update(v="client_ended")
                        return
        finally:
            events.cancel()
            await session.close()

    try:
        asyncio.run(main())
    except Exception as exc:
        outcome.update(v=f"error:{type(exc).__name__}", code=CLOSE_ERROR)
    ws.close(outcome["code"], outcome["v"])
    return outcome["v"]
