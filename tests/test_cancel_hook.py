"""brain_cancel hook, per-call task id and cancel_active_turn, with a fake slow brain."""
import http.client
import json
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gateway  # noqa: E402
from test_ask_stream_pcm import fake_tts, gw, _events  # noqa: E402,F401


@pytest.fixture(autouse=True)
def _clean_registry():
    gateway._ACTIVE_TASKS.clear()
    yield
    assert gateway._ACTIVE_TASKS == {}


def _slow(monkeypatch, delay=1.2, seen=None):
    def brain(prompt, session_id, node_id, speaker, score, timeout, on_partial, task_id=None):
        if seen is not None:
            seen["task_id"] = task_id
        on_partial("First sentence is here. Second sent")
        time.sleep(delay)
        on_partial("First sentence is here. Second sentence is here.")
        return "First sentence is here. Second sentence is here."
    monkeypatch.setattr(gateway, "STREAM_BRAIN_CALL", brain)


def _turn(port, results, node="n1"):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=20)
    conn.request("POST", "/ask/stream", body=json.dumps({"prompt": "hi", "session_id": "s", "node_id": node}),
                 headers={"Content-Type": "application/json"})
    results["events"] = _events(conn.getresponse())
    conn.close()


def _cancel_during_turn(port, monkeypatch, delay=1.2, seen=None):
    _slow(monkeypatch, delay, seen)
    results = {}
    t = threading.Thread(target=_turn, args=(port, results))
    t.start()
    for _ in range(100):
        if "n1" in gateway._ACTIVE_TASKS:
            break
        time.sleep(0.02)
    time.sleep(0.3)
    first = gateway.cancel_active_turn("n1")
    second = gateway.cancel_active_turn("n1")
    t.join(15)
    return first, second, results["events"]


def _turn_records(tmp_path):
    lines = (tmp_path / "voice-latency.jsonl").read_text().splitlines()
    return [r for r in map(json.loads, lines) if r["kind"] == "turn"]


def test_cancel_with_hook(gw, fake_tts, monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(gateway, "_plugin_brain_cancel", lambda target, task_id: calls.append((target, task_id)) or True)
    seen = {}
    first, second, events = _cancel_during_turn(gw, monkeypatch, seen=seen)
    assert first == "cancelled" and second == "no_active_turn"
    assert len(calls) == 1 and calls[0][1] == seen["task_id"] and len(seen["task_id"]) == 12
    assert _turn_records(tmp_path)[-1]["outcome"] == "cancelled"
    assert events[-1][0] == "done"


def test_cancel_without_hook_stops_speech_only(gw, fake_tts, monkeypatch, tmp_path):
    monkeypatch.setattr(gateway, "_plugin_brain_cancel", None)
    first, second, events = _cancel_during_turn(gw, monkeypatch)
    assert (first, second) == ("stopped_speech_only", "no_active_turn")
    assert _turn_records(tmp_path)[-1]["outcome"] == "cancelled"
    reply = [d for e, d in events if e == "reply"][0]
    assert reply["reply"] == ""


def test_raising_hook_counts_as_speech_only(gw, fake_tts, monkeypatch):
    def boom(target, task_id):
        raise RuntimeError("nope")
    monkeypatch.setattr(gateway, "_plugin_brain_cancel", boom)
    first, _second, _events_ = _cancel_during_turn(gw, monkeypatch)
    assert first == "stopped_speech_only"


def test_falsy_hook_counts_as_speech_only(gw, fake_tts, monkeypatch):
    monkeypatch.setattr(gateway, "_plugin_brain_cancel", lambda target, task_id: False)
    first, _s, _e = _cancel_during_turn(gw, monkeypatch)
    assert first == "stopped_speech_only"


def test_no_audio_after_cancel(gw, fake_tts, monkeypatch):
    monkeypatch.setattr(gateway, "_plugin_brain_cancel", None)
    _first, _second, events = _cancel_during_turn(gw, monkeypatch, delay=1.5)
    texts = " ".join(d.get("text", "") for e, d in events if e == "sentence")
    assert "Second sentence is here" not in texts


def test_cancel_when_idle():
    assert gateway.cancel_active_turn("n1") == "no_active_turn"
    assert gateway.cancel_active_turn("") == "no_active_turn"


def test_registry_empty_after_success(gw, fake_tts, monkeypatch):
    _slow(monkeypatch, delay=0.05)
    results = {}
    _turn(gw, results)
    assert any(e == "done" for e, _ in results["events"])


def test_registry_empty_after_brain_error(gw, fake_tts, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("brain down")
    monkeypatch.setattr(gateway, "STREAM_BRAIN_CALL", boom)
    results = {}
    _turn(gw, results)
    assert any(e == "done" for e, _ in results["events"])


def test_plugin_gets_task_id_only_if_accepted(monkeypatch):
    got = {}

    def old(target, prompt, timeout, on_partial, stages_out):
        got["old"] = True
        return "ok"

    def new(target, prompt, timeout, on_partial, stages_out, task_id=None):
        got["task_id"] = task_id
        return "ok"

    monkeypatch.setattr(gateway, "_plugin_brain_call_stream", old)
    gateway.plugin_brain_call_stream("hi", "s", "n1", "", None, 5, lambda t: None, task_id="abc")
    monkeypatch.setattr(gateway, "_plugin_brain_call_stream", new)
    gateway.plugin_brain_call_stream("hi", "s", "n1", "", None, 5, lambda t: None, task_id="abc")
    assert got == {"old": True, "task_id": "abc"}
