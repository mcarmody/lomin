"""Spoken cancel on /voice/turn (MVG_SPOKEN_CANCEL): phrase matching and the
cancel-then-acknowledge flow. Upstreams are monkeypatched; no network."""
import json
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gateway  # noqa: E402
from test_voice_turn import server, stubs, _isolate, turn, first, names  # noqa: E402,F401


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv("MVG_CANCEL_PHRASES", raising=False)
    monkeypatch.delenv("MVG_SPOKEN_CANCEL", raising=False)
    gateway._ACTIVE_TASKS.clear()
    yield
    gateway._ACTIVE_TASKS.clear()


@pytest.mark.parametrize("text,expected", [
    ("stop", True), ("Stop.", True), ("cancel that", True), ("Cancel that!", True),
    ("never mind", True), ("nevermind", True), ("stop that", True),
    ("cancel", True), ("stop it now", True), ("cancel the task", True),
    ("stop the deploy script from restarting", False),
    ("please stop", False), ("don't stop", False), ("what does stop mean", False),
    ("", False), ("   ", False), ("...", False),
])
def test_phrase_table(text, expected):
    assert gateway.is_cancel_phrase(text) is expected


def test_phrase_override(monkeypatch):
    monkeypatch.setenv("MVG_CANCEL_PHRASES", "abort, hold on")
    assert gateway.is_cancel_phrase("Hold on!")
    assert gateway.is_cancel_phrase("abort")
    assert not gateway.is_cancel_phrase("never mind")
    # the "stop" / "cancel" short-form rule still applies
    assert gateway.is_cancel_phrase("stop")


def _records(tmp_path):
    p = tmp_path / "voice-latency.jsonl"
    return [r for r in map(json.loads, p.read_text().splitlines()) if r["kind"] == "turn"] if p.exists() else []


def _run_long_then(server, stubs, monkeypatch, second_text):
    texts = ["do the long thing", second_text]
    monkeypatch.setattr(gateway, "transcribe", lambda wav: texts.pop(0))
    prompts = []

    def brain(prompt, session_id, node_id, speaker, speaker_score, timeout, on_partial, task_id=None):
        prompts.append(prompt)
        on_partial("Working on it now. ")
        time.sleep(1.0)
        return "Working on it now. All done."
    monkeypatch.setattr(gateway, "STREAM_BRAIN_CALL", brain)
    box = {}
    t = threading.Thread(target=lambda: box.setdefault("first", turn(server)))
    t.start()
    for _ in range(100):
        if "lammen" in gateway._ACTIVE_TASKS:
            break
        time.sleep(0.02)
    time.sleep(0.3)
    second = turn(server)
    t.join(15)
    return box["first"], second, prompts


def test_cancel_with_active_turn(server, stubs, monkeypatch):
    monkeypatch.setenv("MVG_SPOKEN_CANCEL", "1")
    calls = []
    monkeypatch.setattr(gateway, "_plugin_brain_cancel", lambda target, tid: calls.append(tid) or True)
    first_ev, second, prompts = _run_long_then(server, stubs, monkeypatch, "stop")
    assert len(calls) == 1
    assert prompts == ["do the long thing"]  # the brain never saw "stop"
    assert first(second, "reply")["reply"] == "Cancelled."
    assert names(first_ev)[-1] == "done"
    assert "All done" not in json.dumps(first_ev)


def test_cancel_without_hook_acknowledges_limit(server, stubs, monkeypatch):
    monkeypatch.setenv("MVG_SPOKEN_CANCEL", "1")
    monkeypatch.setattr(gateway, "_plugin_brain_cancel", None)
    _f, second, _p = _run_long_then(server, stubs, monkeypatch, "cancel that")
    assert "may still be running" in first(second, "reply")["reply"]


def test_cancel_without_active_turn_is_ordinary(server, stubs, monkeypatch):
    monkeypatch.setenv("MVG_SPOKEN_CANCEL", "1")
    stubs.transcript = "stop"
    ev = turn(server)
    assert stubs.prompts == ["stop"]
    assert first(ev, "reply")["reply"] == "It is sunny."


def test_off_by_default(server, stubs, monkeypatch):
    calls = []
    monkeypatch.setattr(gateway, "_plugin_brain_cancel", lambda target, tid: calls.append(tid) or True)
    _f, second, prompts = _run_long_then(server, stubs, monkeypatch, "stop")
    assert calls == []
    assert prompts == ["do the long thing", "stop"]
