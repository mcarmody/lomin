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

import os
import threading
import time
from typing import Any, Callable

# Active in-flight tasks keyed by task_id -> {"target": str, "cancel": threading.Event}
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
            _IN_FLIGHT[str(task_id)] = {"target": str(target or ""), "cancel": ev}
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
    utterance = _extract_user_text(prompt)
    return f"[{target}] Heard: {utterance}"


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
    finally:
        unregister_task(task_id)


def talker_instructions(target: str, node_id: str) -> str | None:
    """Return per-target or per-node instructions for the `/voice/live` Talker."""
    target_key = (target or "").strip() or "agent"
    base = DEFAULT_TALKER_INSTRUCTIONS.get(
        target_key,
        f"You are the live voice Talker for {target_key}. Keep responses brief and natural.",
    )
    if node_id:
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
