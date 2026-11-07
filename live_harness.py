"""Live voice session harness (stdlib only).

`LiveSession` drives one Talker backend (see talker_backend.py): it forwards
client audio in, dispatches backend events out, and owns the four coordination
tools so backends stay thin:

  call_thinker(request)              start a brain call, return {"status": "started", "handoff_id": n}
                                     at once; the brain result is sent to the backend as text
                                     tagged with the handoff id. Calls queue one at a time.
  cancel_thinker()                   cancel the running turn (cancel_turn callback), drop queued calls
  answer_prompt(prompt_id, decision) stub: {"status": "unsupported"}
  set_progress_updates(enabled)      stores `progress_updates` for the progress relay

Callbacks supplied by the gateway: send_to_client(frame) (sync or async),
brain_call(request, task_id) -> result (blocking, run in an executor) and
optional cancel_turn(node_id) -> "cancelled" | "stopped_speech_only" | "no_active_turn".
Malformed tool calls return {"status": "error", "message": ...}; they never raise.
"""
from __future__ import annotations

import asyncio
import inspect
import os
import time
import uuid
from typing import Any, Callable

MAX_REQUEST_CHARS = 4000
TOOL_NAMES = ("call_thinker", "cancel_thinker", "answer_prompt", "set_progress_updates")


def _error(message: str) -> dict:
    return {"status": "error", "message": message}


def validate_tool_args(name: str, args: Any) -> str | None:
    """None when valid, else a one-line reason."""
    if not isinstance(args, dict):
        return "args must be an object"
    keys = set(args)
    if name == "call_thinker":
        if keys != {"request"}:
            return "call_thinker takes exactly: request"
        r = args["request"]
        if not isinstance(r, str) or not r.strip():
            return "request must be a non-empty string"
        if len(r) > MAX_REQUEST_CHARS:
            return f"request is longer than {MAX_REQUEST_CHARS} characters"
    elif name == "cancel_thinker":
        if keys:
            return "cancel_thinker takes no arguments"
    elif name == "answer_prompt":
        if keys != {"prompt_id", "decision"}:
            return "answer_prompt takes exactly: prompt_id, decision"
        if not isinstance(args["prompt_id"], str) or not args["prompt_id"]:
            return "prompt_id must be a non-empty string"
        if args["decision"] not in ("approve", "deny"):
            return "decision must be approve or deny"
    elif name == "set_progress_updates":
        if keys != {"enabled"} or not isinstance(args["enabled"], bool):
            return "set_progress_updates takes exactly: enabled (boolean)"
    else:
        return f"unknown tool {name!r}"
    return None


def _normalize(raw):
    """gateway.normalize_stage_event when the gateway has it, else a minimal local reading."""
    try:
        from gateway import normalize_stage_event
        return normalize_stage_event(raw)
    except Exception:
        if isinstance(raw, dict) and isinstance(raw.get("stage"), str) and raw["stage"].strip():
            d = raw.get("detail")
            return {"stage": raw["stage"].strip(), "detail": " ".join(d.split())[:120] if isinstance(d, str) else "", "t": 0.0}
        return None


class ProgressRelay:
    """Turns stage events into `[task progress] <detail>` lines for the Talker.

    Backends with `silent_context` get each line through send_text(silent=True),
    rate limited to one per MVG_PROGRESS_MIN_GAP_SEC with the latest always kept.
    Other backends get nothing unprompted: the latest line is held and sent only
    when the user asks (status()) or after set_progress_updates(true). finish()
    delivers the held latest line at task end."""

    def __init__(self, session, clock=time.monotonic, min_gap: float | None = None):
        self.session = session
        self.clock = clock
        try:
            self.min_gap = float(os.environ.get("MVG_PROGRESS_MIN_GAP_SEC", "12")) if min_gap is None else min_gap
        except ValueError:
            self.min_gap = 12.0
        self.latest: str | None = None
        self.pending: str | None = None
        self.last_sent: float | None = None
        self._timer = None

    @property
    def silent(self) -> bool:
        return "silent_context" in getattr(self.session.backend, "capabilities", frozenset())

    @property
    def push_enabled(self) -> bool:
        return self.silent or self.session.progress_updates

    async def push(self, raw) -> None:
        ev = _normalize(raw)
        if not ev or not ev["detail"]:
            return
        self.latest = f"[task progress] {ev['detail']}"
        if not self.push_enabled:
            return
        self.pending = self.latest
        await self._maybe_send()

    async def _maybe_send(self) -> None:
        if self.pending is None or self.session._closed:
            return
        now = self.clock()
        wait = 0.0 if self.last_sent is None else self.min_gap - (now - self.last_sent)
        if wait > 0:
            if self._timer is None:
                self._timer = asyncio.get_running_loop().call_later(
                    wait, lambda: asyncio.ensure_future(self._timer_fire()))
            return
        await self._send()

    async def _timer_fire(self) -> None:
        self._timer = None
        await self._maybe_send()

    async def _send(self) -> None:
        line, self.pending = self.pending, None
        self.last_sent = self.clock()
        await self.session.backend.send_text(line, silent=self.silent)

    async def finish(self) -> None:
        """Task end: deliver the latest unsent line now, ignoring the gap."""
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        if self.pending is not None:
            await self._send()

    async def status(self) -> str | None:
        """The user asked how it is going: send the latest line, spoken."""
        if self.latest is None:
            return None
        await self.session.backend.send_text(self.latest, silent=False)
        self.pending = None
        return self.latest

    def close(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None


class LiveSession:
    def __init__(self, backend, config: dict, send_to_client: Callable, brain_call: Callable,
                 cancel_turn: Callable | None = None, speak: Callable | None = None):
        self.backend = backend
        # speak(text) -> WAV bytes (the gateway TTS chain). Used for text-out replies.
        self._speak = speak
        self.config = dict(config or {})
        self.node_id = str(self.config.get("node_id") or "")
        self._send_to_client = send_to_client
        self._brain_call = brain_call
        self._cancel_turn = cancel_turn
        self.progress_updates = False
        self.handoff_counter = 0
        self.ended = asyncio.Event()
        self.error: str | None = None
        self._queue: asyncio.Queue = asyncio.Queue()
        self._worker: asyncio.Task | None = None
        self._running: dict | None = None
        self._closed = False
        self.relay = ProgressRelay(self)

    # -- progress --

    async def on_stage_event(self, raw) -> None:
        """Gateway hook: a brain stage event for the running handoff."""
        await self.relay.push(raw)

    async def status_request(self) -> str | None:
        return await self.relay.status()

    # -- lifecycle --

    async def open(self) -> None:
        await self.backend.open(self.config)
        self._worker = asyncio.ensure_future(self._work())

    async def feed_audio(self, pcm16_16k: bytes) -> None:
        if not self._closed:
            await self.backend.send_audio(pcm16_16k)

    async def run(self) -> None:
        """Consume backend events until they end, an error event arrives, or close()."""
        try:
            async for ev in self.backend.events():
                if self._closed:
                    break
                if not await self._dispatch(ev):
                    break
        finally:
            self.ended.set()

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.relay.close()
        if self._worker is not None:
            self._worker.cancel()
            try:
                await self._worker
            except (asyncio.CancelledError, Exception):
                pass
        try:
            await self.backend.close()
        finally:
            self.ended.set()

    # -- events --

    async def _client(self, frame: dict) -> None:
        res = self._send_to_client(frame)
        if inspect.isawaitable(res):
            await res

    async def _dispatch(self, ev: dict) -> bool:
        """False ends the session."""
        kind = ev.get("type") if isinstance(ev, dict) else None
        if kind == "tool_call":
            result = await self._handle_tool(ev.get("name"), ev.get("args"))
            await self.backend.send_tool_result(str(ev.get("call_id") or ""), result)
        elif kind == "text" and self._speak is not None and not ev.get("spoken"):
            await self._client(ev)
            await self._speak_text(str(ev.get("text") or ""))
        elif kind in ("audio", "text", "interrupted", "turn_end"):
            await self._client(ev)
        elif kind == "resume_handle":
            return True
        elif kind == "reconnect":
            try:
                await self.backend.reconnect()
            except Exception as exc:  # noqa: BLE001
                self.error = str(exc) or "reconnect failed"
                await self._client({"type": "error", "message": self.error})
                return False
        elif kind == "error":
            self.error = str(ev.get("message") or "backend error")
            await self._client({"type": "error", "message": self.error})
            return False
        return True

    async def _speak_text(self, text: str) -> None:
        """Synthesize a text-out reply and send it as audio. TTS failure is reported to
        the client as a non-fatal event; the text frame was already delivered."""
        if not text.strip():
            return
        try:
            loop = asyncio.get_running_loop()
            wav = await loop.run_in_executor(None, self._speak, text)
            import audio_utils
            import tts_backends
            wav = audio_utils.normalize_wav(wav, target_rate=24000)
            pcm = tts_backends.extract_pcm_from_wav_or_raw(wav)
        except Exception as exc:  # noqa: BLE001
            await self._client({"type": "tts_error", "message": str(exc)[:200]})
            return
        if pcm and not self._closed:
            await self._client({"type": "audio", "pcm": pcm, "rate": 24000, "synthesized": True})

    # -- tools --

    async def _handle_tool(self, name: Any, args: Any) -> dict:
        name = name if isinstance(name, str) else ""
        reason = validate_tool_args(name, args)
        if reason:
            return _error(reason)
        try:
            if name == "call_thinker":
                self.handoff_counter += 1
                hid = self.handoff_counter
                self._queue.put_nowait({"id": hid, "request": args["request"], "cancelled": False})
                return {"status": "started", "handoff_id": hid}
            if name == "cancel_thinker":
                return await self._cancel()
            if name == "answer_prompt":
                return {"status": "unsupported"}
            self.progress_updates = args["enabled"]
            return {"status": "ok", "enabled": self.progress_updates}
        except Exception as exc:
            return _error(f"{type(exc).__name__}: {exc}"[:200])

    async def _cancel(self) -> dict:
        dropped = 0
        while not self._queue.empty():
            self._queue.get_nowait()
            dropped += 1
        running = self._running
        if running is not None:
            running["cancelled"] = True
        outcome = "no_active_turn"
        if self._cancel_turn is not None:
            res = self._cancel_turn(self.node_id)
            outcome = await res if inspect.isawaitable(res) else res
        elif running is not None:
            outcome = "stopped_speech_only"
        return {"status": outcome, "dropped_queued": dropped}

    async def _work(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            job = await self._queue.get()
            self._running = job
            task_id = uuid.uuid4().hex[:12]
            try:
                result = await loop.run_in_executor(None, self._brain_call, job["request"], task_id)
                text = self._result_text(result)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                text = f"The brain call failed: {type(exc).__name__}"
            finally:
                self._running = None
            if job["cancelled"] or self._closed:
                self.relay.pending = None
                continue
            await self.relay.finish()
            await self.backend.send_text(f"[thinker result, handoff {job['id']}] {text}")

    @staticmethod
    def _result_text(result: Any) -> str:
        if isinstance(result, dict):
            for k in ("reply", "text", "result"):
                if isinstance(result.get(k), str):
                    return result[k]
        return result if isinstance(result, str) else str(result)
