#!/usr/bin/env python3
"""Tests for the idle-unload path in ./desktop_embed.py (MDE_IDLE_UNLOAD_MIN).

No network, no GPU, no real speechbrain model: the model is a fake encoder
and the loader is a fake that counts calls. Tests that exercise torch
tensors skip when torch is not importable; the rest run under plain python3
with fastapi installed.
Run: python3 -m pytest ./tests -v
"""
import asyncio
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import desktop_embed  # noqa: E402


class _FakeEncoder:
    """Stands in for the speechbrain EncoderClassifier."""

    def encode_batch(self, tensor):
        import torch
        return torch.zeros((1, 1, 192))


@pytest.fixture
def idle_state(monkeypatch):
    """Pins every idle-related global so no test leaks state into another."""
    monkeypatch.setattr(desktop_embed, "_model", None)
    monkeypatch.setattr(desktop_embed, "_last_used", 0.0)
    monkeypatch.setattr(desktop_embed, "_idle_unloaded", False)
    monkeypatch.setattr(desktop_embed, "_load_error", None)
    return monkeypatch


def test_unload_disabled_when_idle_sec_is_zero(idle_state):
    model = object()
    idle_state.setattr(desktop_embed, "_model", model)
    idle_state.setattr(desktop_embed, "_last_used", 0.0)

    assert desktop_embed._unload_if_idle(1e9, 0) is False
    assert desktop_embed._model is model
    assert desktop_embed._idle_unloaded is False


def test_recent_use_keeps_model(idle_state):
    model = object()
    now = 1000.0
    idle_state.setattr(desktop_embed, "_model", model)
    idle_state.setattr(desktop_embed, "_last_used", now - 10.0)

    assert desktop_embed._unload_if_idle(now, 60.0) is False
    assert desktop_embed._model is model
    assert desktop_embed._idle_unloaded is False


def test_idle_past_limit_unloads(idle_state):
    now = 1000.0
    idle_state.setattr(desktop_embed, "_model", object())
    idle_state.setattr(desktop_embed, "_last_used", now - 120.0)

    assert desktop_embed._unload_if_idle(now, 60.0) is True
    assert desktop_embed._model is None
    assert desktop_embed._idle_unloaded is True


def test_health_while_unloaded_is_idle_200(idle_state):
    idle_state.setattr(desktop_embed, "_idle_unloaded", True)

    resp = asyncio.run(desktop_embed.health())
    body = json.loads(resp.body)

    assert resp.status_code == 200
    assert body["status"] == "idle"
    assert body["loaded"] is False
    assert body["load_error"] is None


def test_health_startup_failure_is_503(idle_state):
    idle_state.setattr(desktop_embed, "_load_error", "x")

    resp = asyncio.run(desktop_embed.health())
    body = json.loads(resp.body)

    assert resp.status_code == 503
    assert body["status"] == "error"
    assert body["load_error"] == "x"


def test_reload_on_demand_after_unload(idle_state):
    pytest.importorskip("torch")
    np = pytest.importorskip("numpy")

    calls = []

    def fake_load():
        calls.append(1)
        desktop_embed._model = _FakeEncoder()
        desktop_embed._idle_unloaded = False

    idle_state.setattr(desktop_embed, "_load_model", fake_load)
    idle_state.setattr(desktop_embed, "_decode_and_resample",
                       lambda body: np.zeros(8000, dtype=np.float32))
    idle_state.setattr(desktop_embed, "_model", None)
    idle_state.setattr(desktop_embed, "_idle_unloaded", True)

    result = desktop_embed._embed_sync(b"x")

    assert len(calls) == 1
    assert "embedding" in result
    assert len(result["embedding"]) == 192
    assert desktop_embed._idle_unloaded is False
    assert desktop_embed._last_used > 0.0


def test_bad_body_does_not_reload(idle_state):
    pytest.importorskip("torch")

    calls = []

    def fake_load():
        calls.append(1)

    def bad_decode(body):
        raise ValueError("could not decode audio")

    idle_state.setattr(desktop_embed, "_load_model", fake_load)
    idle_state.setattr(desktop_embed, "_decode_and_resample", bad_decode)
    idle_state.setattr(desktop_embed, "_model", None)
    idle_state.setattr(desktop_embed, "_idle_unloaded", True)

    with pytest.raises(ValueError):
        desktop_embed._embed_sync(b"junk")
    assert calls == []
    assert desktop_embed._idle_unloaded is True


def test_failed_reload_raises_and_retries_next_time(idle_state):
    pytest.importorskip("torch")
    np = pytest.importorskip("numpy")

    calls = []

    def failing_load():
        calls.append(1)
        desktop_embed._load_error = "boom"
        # _model stays None, _idle_unloaded stays True

    idle_state.setattr(desktop_embed, "_load_model", failing_load)
    idle_state.setattr(desktop_embed, "_decode_and_resample",
                       lambda body: np.zeros(8000, dtype=np.float32))
    idle_state.setattr(desktop_embed, "_model", None)
    idle_state.setattr(desktop_embed, "_idle_unloaded", True)

    with pytest.raises(RuntimeError, match="boom"):
        desktop_embed._embed_sync(b"x")
    assert len(calls) == 1
    assert desktop_embed._model is None
    assert desktop_embed._idle_unloaded is True

    # The next request tries the load again.
    with pytest.raises(RuntimeError, match="boom"):
        desktop_embed._embed_sync(b"x")
    assert len(calls) == 2
