"""Neutral reference plugin for Lomin Agent Voice Gateway (stdlib only)."""

from __future__ import annotations

import json
import os
import threading
import urllib.request
from typing import Any

_INFLIGHT: dict[str, Any] = {}
_LOCK = threading.Lock()


def brain_call_stream(
    target: str,
    prompt: str,
    timeout: float,
    on_partial,
    stages_out,
    audio_clips=None,
    task_id: str | None = None,
) -> str:
    url = os.environ["REFERENCE_BRAIN_URL"]
    body = json.dumps({"target": target, "prompt": prompt}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    resp = urllib.request.urlopen(req, timeout=timeout)
    if task_id is not None:
        with _LOCK:
            _INFLIGHT[task_id] = resp
    try:
        text = resp.read().decode("utf-8")
    finally:
        if task_id is not None:
            with _LOCK:
                _INFLIGHT.pop(task_id, None)
        try:
            resp.close()
        except Exception:
            pass
    if on_partial is not None:
        on_partial(text)
    return text


def brain_cancel(target: str, task_id: str) -> bool:
    with _LOCK:
        resp = _INFLIGHT.pop(task_id, None)
    if resp is None:
        return False
    resp.close()
    return True


def talker_instructions(target: str, node_id: str) -> str:
    return os.environ.get("REFERENCE_TALKER_INSTRUCTIONS", "")


def try_intent(text: str):
    return None
