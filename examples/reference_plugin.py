"""Reference Lomin plugin (stdlib only, no site dependencies).

Demonstrates the full plugin contract with zero external dependencies:
- `apply_defaults()`
- `brain_call(target, prompt, timeout)`
- `brain_call_stream(target, prompt, timeout, on_partial, stages_out, *, task_id=None, audio_clips=None)`
- `brain_cancel(target, task_id)`
- `talker_instructions(target, node_id)`
- `try_intent(text)`

Can be pointed to directly via `MVG_PLUGIN=examples.reference_plugin` or used
as a starting template for custom agent transports.
"""
from __future__ import annotations

import base64
import json
import os
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Callable

# Active in-flight tasks keyed by task_id -> {"target": str, "cancel": threading.Event, "resp": Any}
_IN_FLIGHT: dict[str, dict[str, Any]] = {}
_LOCK = threading.Lock()

# Configurable step delay for testing cancellation and streaming behaviour.
STEP_DELAY_SEC: float = 0.01

DEFAULT_TALKER_INSTRUCTIONS: dict[str, str] = {
    "agent": (
        "You are a concise, natural-sounding voice Talker. Acknowledge the user "
        "in one short sentence and call `call_thinker` for questions or tasks."
    ),
}


def apply_defaults() -> None:
    """Populate environment defaults using `os.environ.setdefault` only."""
    os.environ.setdefault("MVG_AGENT_NAME", "the assistant")
    os.environ.setdefault("MVG_DEFAULT_TARGET", "agent")


def register_task(target: str, task_id: str | None) -> threading.Event:
    """Register an in-flight task by `task_id` and return its cancellation event."""
    ev = threading.Event()
    if task_id:
        with _LOCK:
            _IN_FLIGHT[str(task_id)] = {"target": str(target or ""), "cancel": ev, "resp": None}
    return ev


def unregister_task(task_id: str | None) -> None:
    """Remove a completed or cancelled task from the in-flight registry."""
    if task_id:
        with _LOCK:
            _IN_FLIGHT.pop(str(task_id), None)


def active_task_ids() -> list[str]:
    """Return a snapshot of currently registered in-flight task IDs."""
    with _LOCK:
        return list(_IN_FLIGHT.keys())


def brain_cancel(target: str, task_id: str) -> bool:
    """Abort an in-flight brain call matching `task_id` (and `target` when given).

    Returns True if a running task was found and signaled to stop, else False.
    """
    if not task_id:
        return False
    with _LOCK:
        entry = _IN_FLIGHT.get(str(task_id))
        if entry is None:
            return False
        if target and entry["target"] and entry["target"] != target:
            return False
        entry["cancel"].set()
        resp = entry.get("resp")
        if resp is not None:
            try:
                resp.close()
            except Exception:
                pass
        return True


def _extract_user_text(prompt: str) -> str:
    """Extract the core user utterance from a framed gateway prompt."""
    text = (prompt or "").strip()
    if text.startswith("[VOICE ") and "]" in text:
        text = text.split("]", 1)[1].strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    filtered = [
        line for line in lines
        if not line.startswith("This reply will be spoken aloud:")
        and not line.startswith("A spoken request, verified or not,")
    ]
    return " ".join(filtered) if filtered else text


def brain_call(target: str, prompt: str, timeout: int = 45) -> str:
    """Synchronous non-streaming brain hook."""
    url = os.environ.get("REF_PLUGIN_BRAIN_URL", "").strip()
    if not url:
        utterance = _extract_user_text(prompt)
        return f"[{target}] Heard: {utterance}"

    payload = {
        "target": target,
        "prompt": prompt,
        "stream": False,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    token = os.environ.get("REF_PLUGIN_BRAIN_TOKEN", "").strip()
    if token:
        req.add_header("Authorization", f"Bearer {token}")

    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
        try:
            parsed = json.loads(body)
            if isinstance(parsed, dict):
                reply = parsed.get("reply")
                if reply is None:
                    reply = parsed.get("text", body)
                return str(reply)
            return str(parsed)
        except Exception:
            return body


def brain_call_stream(
    target: str,
    prompt: str,
    timeout: int,
    on_partial: Callable[[str], None] | None = None,
    stages_out: dict[str, Any] | None = None,
    *,
    task_id: str | None = None,
    audio_clips: list[bytes] | None = None,
) -> str:
    """Streaming brain hook supporting partial updates, stage events, audio clips, and cancellation."""
    t0 = time.monotonic()
    cancel_ev = register_task(target, task_id)
    resp: Any = None
    try:
        if isinstance(stages_out, dict):
            stages_out["call_start_ms"] = int(time.time() * 1000)
            events = stages_out.setdefault("events", [])
            events.append({
                "stage": "tool_start",
                "detail": f"routing request to {target or 'agent'}",
                "t": round(time.monotonic() - t0, 3),
            })
            if audio_clips:
                total_bytes = sum(len(c) for c in audio_clips if isinstance(c, (bytes, bytearray)))
                events.append({
                    "stage": "milestone",
                    "detail": f"received {len(audio_clips)} audio clip(s) ({total_bytes} bytes)",
                    "t": round(time.monotonic() - t0, 3),
                })

        url = os.environ.get("REF_PLUGIN_BRAIN_URL", "").strip()
        if not url:
            utterance = _extract_user_text(prompt)
            clip_note = f" (with {len(audio_clips)} audio clip)" if audio_clips else ""
            clauses = [
                f"First sentence from {target}{clip_note}.",
                f"You asked: {utterance}.",
            ]

            accumulated = ""
            for idx, clause in enumerate(clauses):
                if cancel_ev.wait(timeout=max(0.0, STEP_DELAY_SEC)):
                    if isinstance(stages_out, dict):
                        stages_out["cancelled"] = True
                    return accumulated
                accumulated = f"{accumulated} {clause}".strip() if accumulated else clause
                if on_partial is not None:
                    on_partial(accumulated)
                if idx == 0 and isinstance(stages_out, dict):
                    stages_out.setdefault("events", []).append({
                        "stage": "milestone",
                        "detail": "streamed opening sentence",
                        "t": round(time.monotonic() - t0, 3),
                    })

            if isinstance(stages_out, dict):
                stages_out["final_seen_ms"] = int(time.time() * 1000)
            return accumulated

        # HTTP streaming mode
        payload = {
            "target": target,
            "prompt": prompt,
            "task_id": task_id,
            "stream": True,
            "audio_clips_b64": [
                base64.b64encode(c).decode("ascii")
                for c in (audio_clips or [])
                if isinstance(c, (bytes, bytearray))
            ],
        }
        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream, application/json",
        }
        token = os.environ.get("REF_PLUGIN_BRAIN_TOKEN", "").strip()
        if token:
            headers["Authorization"] = f"Bearer {token}"

        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        resp = urllib.request.urlopen(req, timeout=timeout)
        if task_id:
            with _LOCK:
                entry = _IN_FLIGHT.get(str(task_id))
                if entry is not None:
                    entry["resp"] = resp

        content_type = (resp.headers.get("Content-Type") or "").lower()
        if "text/event-stream" in content_type:
            accumulated = ""
            try:
                for raw_line in resp:
                    if cancel_ev.is_set():
                        if isinstance(stages_out, dict):
                            stages_out["cancelled"] = True
                        return accumulated
                    line = raw_line.decode("utf-8").strip() if isinstance(raw_line, (bytes, bytearray)) else str(raw_line).strip()
                    if not line or line.startswith(":"):
                        continue
                    if line.startswith("data:"):
                        payload_str = line[5:].strip()
                        if payload_str == "[DONE]":
                            break
                        try:
                            obj = json.loads(payload_str)
                        except Exception:
                            continue
                        if isinstance(obj, dict):
                            if "stage" in obj and isinstance(stages_out, dict):
                                stages_out.setdefault("events", []).append({
                                    "stage": str(obj["stage"]),
                                    "detail": str(obj.get("detail") or ""),
                                    "t": round(time.monotonic() - t0, 3),
                                })
                            if "delta" in obj or "text" in obj:
                                delta = str(obj.get("delta") or obj.get("text") or "")
                                accumulated += delta
                                if on_partial is not None:
                                    on_partial(accumulated)
                            if "reply" in obj:
                                accumulated = str(obj["reply"])
                                if on_partial is not None:
                                    on_partial(accumulated)
            except Exception:
                if cancel_ev.is_set():
                    if isinstance(stages_out, dict):
                        stages_out["cancelled"] = True
                    return accumulated
                raise

            if cancel_ev.is_set():
                if isinstance(stages_out, dict):
                    stages_out["cancelled"] = True
                return accumulated

            if isinstance(stages_out, dict):
                stages_out["final_seen_ms"] = int(time.time() * 1000)
            return accumulated
        else:
            body = resp.read().decode("utf-8")
            obj = json.loads(body)
            if isinstance(stages_out, dict):
                for ev_item in (obj.get("events") or []):
                    stages_out.setdefault("events", []).append(ev_item)
            reply = str(obj.get("reply") or obj.get("text") or "")
            if on_partial is not None and reply:
                on_partial(reply)
            if isinstance(stages_out, dict):
                stages_out["final_seen_ms"] = int(time.time() * 1000)
            return reply
    finally:
        if resp is not None:
            try:
                resp.close()
            except Exception:
                pass
        unregister_task(task_id)


def talker_instructions(target: str, node_id: str) -> str | None:
    """Return per-target or per-node instructions for the `/voice/live` Talker."""
    target_key = (target or "").strip() or "agent"
    target_env = os.environ.get(
        f"REF_PLUGIN_TALKER_INSTRUCTIONS_{target_key.upper().replace('-', '_')}", ""
    ).strip()
    global_env = os.environ.get("REF_PLUGIN_TALKER_INSTRUCTIONS", "").strip()

    if target_env:
        base = target_env
    elif global_env:
        base = global_env
    else:
        base = DEFAULT_TALKER_INSTRUCTIONS.get(
            target_key,
            f"You are the live voice Talker for {target_key}. Keep responses brief and natural.",
        )

    if node_id:
        if f"(node: {node_id})" not in base:
            return f"{base} (node: {node_id})"
    return base


def try_intent(text: str) -> dict[str, Any] | None:
    """Deterministic fast-path intent hook for simple built-in phrases."""
    cleaned = " ".join((text or "").strip().lower().split()).rstrip(".!?")
    if cleaned in ("ping", "health check"):
        return {
            "hit": True,
            "intent": "ping",
            "reply": "Pong. The voice gateway is online.",
            "match_ms": 0.1,
            "handler_ms": 0.1,
        }
    return None
