"""Talker backend contract for the live voice session (stdlib only).

A Talker is the fast conversational half of a live session. Any class that
meets this contract can drive `/voice/live`; the gateway never imports a vendor
SDK itself. Backends are looked up by name with `load_backend`.

Capabilities (a frozenset of strings on the class):
  audio_in        accepts 16 kHz mono PCM16 through send_audio
  audio_out       emits {"type": "audio"} events
  text_out        emits {"type": "text"} events
  full_duplex     can be spoken over (emits "interrupted")
  tool_calls      emits {"type": "tool_call"} events
  silent_context  send_text(..., silent=True) adds context without a spoken reply

Event dicts yielded by `events()`:
  {"type": "audio", "pcm": bytes}
  {"type": "text", "text": str}
  {"type": "tool_call", "call_id": str, "name": str, "args": dict}
  {"type": "interrupted"}
  {"type": "turn_end"}
  {"type": "error", "message": str}

Plugin backends: set `MVG_TALKER_BACKENDS` to comma-separated `name=module:Class`
entries; they are imported lazily on first use.
"""
from __future__ import annotations

import asyncio
import importlib
import json
import os
from typing import Any, AsyncIterator
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

CAPABILITIES = frozenset({"audio_in", "audio_out", "text_out", "full_duplex", "tool_calls", "silent_context"})
EVENT_TYPES = frozenset({"audio", "text", "tool_call", "interrupted", "turn_end", "error", "resume_handle", "reconnect"})


class TalkerBackend:
    capabilities: frozenset = frozenset()

    async def open(self, config: dict) -> None:
        raise NotImplementedError

    async def send_audio(self, pcm16_16k: bytes) -> None:
        raise NotImplementedError

    async def send_text(self, text: str, *, silent: bool = False) -> None:
        raise NotImplementedError

    async def send_tool_result(self, call_id: str, result: dict) -> None:
        raise NotImplementedError

    def events(self) -> AsyncIterator[dict]:
        raise NotImplementedError

    async def reconnect(self) -> None:
        raise NotImplementedError

    async def cancel(self) -> None:
        raise NotImplementedError

    async def close(self) -> None:
        raise NotImplementedError


def requires(backend, *caps: str) -> None:
    """Raise ValueError naming every capability the backend lacks."""
    have = getattr(backend, "capabilities", frozenset())
    missing = [c for c in caps if c not in have]
    if missing:
        raise ValueError(f"backend {type(backend).__name__} lacks capability: {', '.join(missing)}")


class MockBackend(TalkerBackend):
    """Replays a scripted list of events and records everything sent to it."""
    capabilities = frozenset({"audio_in", "audio_out", "text_out", "full_duplex", "tool_calls", "silent_context"})

    def __init__(self, script: list | None = None):
        self.script = list(script or [])
        self.sent: list = []
        self.opened = False
        self.closed = False
        self.reconnected = 0
        self.reconnect_error: Exception | None = None

    async def open(self, config: dict) -> None:
        self.opened = True
        self.sent.append(("open", dict(config or {})))

    async def send_audio(self, pcm16_16k: bytes) -> None:
        self.sent.append(("audio", bytes(pcm16_16k)))

    async def send_text(self, text: str, *, silent: bool = False) -> None:
        self.sent.append(("text", text, silent))

    async def send_tool_result(self, call_id: str, result: dict) -> None:
        self.sent.append(("tool_result", call_id, result))

    async def events(self) -> AsyncIterator[dict]:
        for ev in self.script:
            await asyncio.sleep(0)
            yield ev

    async def reconnect(self) -> None:
        self.reconnected += 1
        self.sent.append(("reconnect",))
        if self.reconnect_error is not None:
            raise self.reconnect_error

    async def cancel(self) -> None:
        self.sent.append(("cancel",))

    async def close(self) -> None:
        self.closed = True
        self.sent.append(("close",))


DEFAULT_CASCADED_PROMPT = (
    "You are the fast conversational Talker in a Talker + Thinker voice assistant. "
    "Speak one short, natural sentence acknowledging the user, then call `call_thinker` "
    "for any task or question that needs deeper reasoning or tools. Keep replies short "
    "and plain-spoken, with no markdown or code fences."
)

CASCADED_TOOLS = [
    {"type": "function", "function": {
        "name": "call_thinker",
        "description": "Hand a request to the slower thinking agent. Returns at once; the answer arrives later as context.",
        "parameters": {"type": "object", "properties": {"request": {"type": "string"}}, "required": ["request"]},
    }},
    {"type": "function", "function": {
        "name": "cancel_thinker",
        "description": "Cancel the running thinking-agent task.",
        "parameters": {"type": "object", "properties": {}},
    }},
    {"type": "function", "function": {
        "name": "answer_prompt",
        "description": "Answer a pending approval prompt.",
        "parameters": {"type": "object", "properties": {"prompt_id": {"type": "string"}, "decision": {"type": "string"}},
                       "required": ["prompt_id", "decision"]},
    }},
    {"type": "function", "function": {
        "name": "set_progress_updates",
        "description": "Turn spoken progress updates on or off.",
        "parameters": {"type": "object", "properties": {"enabled": {"type": "boolean"}}, "required": ["enabled"]},
    }},
]


def _clean_spoken_text(text: str) -> str:
    try:
        from text_utils import strip_markdown
        return strip_markdown(text)
    except Exception:
        return text.replace("```", "").replace("**", "").replace("`", "").strip()


class CascadedTextBackend(TalkerBackend):
    """Text-in, text-out cascaded Talker backend (Mode 2: STT -> fast LLM -> Lomin TTS).

    Accepts transcript or context updates via `send_text` and emits `text`,
    `tool_call`, `turn_end`, or `error` events. Supports either an injected
    callable (`responder(messages, config)`), an OpenAI-compatible HTTP chat
    completions endpoint (`MVG_TALKER_URL`, `MVG_TALKER_MODEL`, `MVG_TALKER_KEY`),
    or a deterministic built-in fast acknowledgment + `call_thinker` handoff when
    no external URL is configured.
    """
    capabilities = frozenset({"text_out", "tool_calls", "silent_context"})

    def __init__(self, responder=None):
        self._responder = responder
        self._config: dict = {}
        self.messages: list[dict] = []
        self._queue: asyncio.Queue = asyncio.Queue()
        self.opened = False
        self.closed = False
        self.reconnected = 0

    async def open(self, config: dict) -> None:
        self._config = dict(config or {})
        self.opened = True
        self.closed = False
        instr = (self._config.get("instructions")
                 or os.environ.get("MVG_TALKER_INSTRUCTIONS")
                 or DEFAULT_CASCADED_PROMPT)
        self.messages = [{"role": "system", "content": str(instr)}]

    async def send_audio(self, pcm16_16k: bytes) -> None:
        # Cascaded mode takes transcribed text via send_text; raw audio is ignored or unsupported.
        return None

    async def send_text(self, text: str, *, silent: bool = False) -> None:
        self.messages.append({"role": "user", "content": str(text)})
        if silent:
            return
        await self._generate_reply(str(text))

    async def send_tool_result(self, call_id: str, result: dict) -> None:
        self.messages.append({
            "role": "tool",
            "tool_call_id": str(call_id),
            "content": json.dumps(result),
        })

    async def _generate_reply(self, latest_text: str) -> None:
        try:
            if self._responder is not None:
                res = self._responder(list(self.messages), dict(self._config))
                if asyncio.iscoroutine(res):
                    res = await res
                await self._emit_response(res)
                return
            url = (self._config.get("talker_url") or os.environ.get("MVG_TALKER_URL") or "").strip()
            if url:
                loop = asyncio.get_running_loop()
                res = await loop.run_in_executor(None, self._http_chat, url)
                await self._emit_response(res)
                return
            # Built-in local cascaded behavior when no external HTTP endpoint is configured:
            if latest_text.startswith("[thinker result,"):
                _, _, body = latest_text.partition("]")
                cleaned = _clean_spoken_text(body.strip() or latest_text)
                if cleaned:
                    self.messages.append({"role": "assistant", "content": cleaned})
                    await self._queue.put({"type": "text", "text": cleaned, "spoken": False})
                await self._queue.put({"type": "turn_end"})
            else:
                ack = "On it."
                self.messages.append({"role": "assistant", "content": ack})
                await self._queue.put({"type": "text", "text": ack, "spoken": False})
                await self._queue.put({
                    "type": "tool_call",
                    "call_id": f"call_{len(self.messages)}",
                    "name": "call_thinker",
                    "args": {"request": latest_text},
                })
                await self._queue.put({"type": "turn_end"})
        except Exception as exc:  # noqa: BLE001
            await self._queue.put({"type": "error", "message": str(exc)})

    async def _emit_response(self, res: Any) -> None:
        if isinstance(res, str):
            cleaned = _clean_spoken_text(res)
            if cleaned:
                self.messages.append({"role": "assistant", "content": cleaned})
                await self._queue.put({"type": "text", "text": cleaned, "spoken": False})
            await self._queue.put({"type": "turn_end"})
            return
        if isinstance(res, list):
            for ev in res:
                if isinstance(ev, dict):
                    if ev.get("type") == "text":
                        ev = {**ev, "text": _clean_spoken_text(str(ev.get("text") or ""))}
                    await self._queue.put(ev)
            return
        if isinstance(res, dict):
            text = _clean_spoken_text(str(res.get("text") or res.get("content") or ""))
            if text:
                self.messages.append({"role": "assistant", "content": text})
                await self._queue.put({"type": "text", "text": text, "spoken": False})
            for idx, tc in enumerate(res.get("tool_calls") or []):
                if "function" in tc:
                    fn = tc["function"] or {}
                    raw_args = fn.get("arguments") or "{}"
                    try:
                        args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
                    except ValueError:
                        args = {}
                    await self._queue.put({
                        "type": "tool_call",
                        "call_id": str(tc.get("id") or f"call_{idx + 1}"),
                        "name": fn.get("name") or "",
                        "args": args,
                    })
                else:
                    await self._queue.put({
                        "type": "tool_call",
                        "call_id": str(tc.get("call_id") or tc.get("id") or f"call_{idx + 1}"),
                        "name": tc.get("name") or "",
                        "args": tc.get("args") or {},
                    })
            await self._queue.put({"type": "turn_end"})

    def _http_chat(self, url: str) -> dict:
        model = (self._config.get("talker_model") or os.environ.get("MVG_TALKER_MODEL") or "gpt-4o-mini").strip()
        key = (self._config.get("talker_key") or os.environ.get("MVG_TALKER_KEY") or "").strip()
        timeout = float(os.environ.get("MVG_TALKER_TIMEOUT", "15"))
        payload = json.dumps({
            "model": model,
            "messages": self.messages,
            "tools": CASCADED_TOOLS,
        }).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        last_err: Exception | None = None
        for attempt in range(2):
            try:
                req = Request(url, data=payload, headers=headers, method="POST")
                with urlopen(req, timeout=timeout) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                choice = ((body.get("choices") or [{}])[0]).get("message") or {}
                return {
                    "text": choice.get("content") or "",
                    "tool_calls": choice.get("tool_calls") or [],
                }
            except HTTPError as exc:
                last_err = exc
                if exc.code >= 500 and attempt == 0:
                    continue
                raise
            except (URLError, TimeoutError, ValueError) as exc:
                last_err = exc
                raise
        raise RuntimeError(str(last_err or "cascaded talker request failed"))

    async def events(self) -> AsyncIterator[dict]:
        while not self.closed:
            ev = await self._queue.get()
            if ev is None:
                return
            yield ev

    async def reconnect(self) -> None:
        self.reconnected += 1

    async def cancel(self) -> None:
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def close(self) -> None:
        self.closed = True
        await self._queue.put(None)


_REGISTRY: dict = {"mock": MockBackend, "cascaded": CascadedTextBackend}


def _plugin_entries() -> dict:
    out = {}
    for item in os.environ.get("MVG_TALKER_BACKENDS", "").split(","):
        item = item.strip()
        if not item:
            continue
        name, sep, target = item.partition("=")
        if sep and ":" in target:
            out[name.strip()] = target.strip()
    return out


def load_backend(name: str) -> type:
    """Return the backend class registered under name (builtin first, then
    MVG_TALKER_BACKENDS). Unknown names raise ValueError listing what exists."""
    if name in _REGISTRY:
        return _REGISTRY[name]
    if name == "gemini":
        from gemini_live import GeminiLiveBackend
        return GeminiLiveBackend
    plugins = _plugin_entries()
    if name in plugins:
        module, _, cls = plugins[name].partition(":")
        try:
            return getattr(importlib.import_module(module), cls)
        except (ImportError, AttributeError) as exc:
            raise ValueError(f"talker backend {name!r} ({plugins[name]}) failed to load: {exc}") from exc
    known = sorted(set(_REGISTRY) | set(plugins) | {"gemini"})
    raise ValueError(f"unknown talker backend {name!r}; known: {', '.join(known) or 'none'}")
