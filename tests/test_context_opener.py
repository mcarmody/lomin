"""Context opener (MVG_CONTEXT_OPENER): one specific line from the first stage
event replaces the generic opener clip. Fake brain, fake TTS."""
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gateway  # noqa: E402
from test_ask_stream_pcm import FakeTTS, _env, fake_tts, gw, _last_log  # noqa: E402,F401
from test_gap_fill_latency import _gap, _run, _sentences, _latency  # noqa: E402,F401


def _brain(monkeypatch, delay_before_event=0.05, event=None, delay_after=0.6, text="All finished here."):
    ev = event if event is not None else {"stage": "tool_start", "detail": "checking the build logs", "t": 0.1}

    def brain(prompt, session_id, node_id, speaker, score, timeout, on_partial):
        time.sleep(delay_before_event)
        if ev:
            gateway._STAGE_SINK.stages.setdefault("events", []).append(ev)
        time.sleep(delay_after)
        on_partial(text)
        return text
    monkeypatch.setattr(gateway, "STREAM_BRAIN_CALL", brain)


def _engines(events):
    return [d["engine"] for d in _sentences(events)]


def test_stage_opener_replaces_generic_clip(gw, fake_tts, monkeypatch, tmp_path):
    monkeypatch.setenv("MVG_CONTEXT_OPENER", "1")
    _brain(monkeypatch)
    events = _run(gw)
    texts = [d["text"] for d in _sentences(events) if d["text"]]
    assert texts[0] == "One moment, checking the build logs"
    assert sum("One moment" in t for t in texts) == 1
    rec = _last_log(tmp_path)
    assert not any(slug.startswith("vocal_open/") for slug in rec["gap_fill"])
    lat = _latency(tmp_path)[-1]
    assert lat["opener_kind"] == "stage" and lat["opener_ms"] is not None


def test_no_opener_when_reply_starts_first(gw, fake_tts, monkeypatch, tmp_path):
    monkeypatch.setenv("MVG_CONTEXT_OPENER", "1")

    def brain(prompt, session_id, node_id, speaker, score, timeout, on_partial):
        on_partial("Quick answer here. ")
        time.sleep(0.2)
        gateway._STAGE_SINK.stages.setdefault("events", []).append(
            {"stage": "tool_start", "detail": "too late", "t": 1.0})
        time.sleep(0.2)
        return "Quick answer here."
    monkeypatch.setattr(gateway, "STREAM_BRAIN_CALL", brain)
    events = _run(gw)
    assert not any("One moment" in d["text"] for d in _sentences(events))
    assert _latency(tmp_path)[-1]["opener_kind"] == "none"


def test_empty_detail_skips_opener(gw, fake_tts, monkeypatch, tmp_path):
    monkeypatch.setenv("MVG_CONTEXT_OPENER", "1")
    _brain(monkeypatch, event={"stage": "tool_start", "detail": "", "t": 0.1})
    events = _run(gw)
    assert not any("One moment" in d["text"] for d in _sentences(events))


def test_silent_brain_keeps_generic_slots(gw, fake_tts, monkeypatch, tmp_path):
    monkeypatch.setenv("MVG_CONTEXT_OPENER", "1")
    _brain(monkeypatch, event=False, delay_before_event=0.0, delay_after=1.6)
    events = _run(gw)
    rec = _last_log(tmp_path)
    assert any(slug.startswith("vocal_open/") for slug in rec["gap_fill"])
    assert len(rec["gap_fill"]) >= 2  # later slots still fire
    assert _latency(tmp_path)[-1]["opener_kind"] == "gap_fill"


def test_off_by_default(gw, fake_tts, monkeypatch, tmp_path):
    monkeypatch.delenv("MVG_CONTEXT_OPENER", raising=False)
    _brain(monkeypatch)
    events = _run(gw)
    assert not any("One moment" in d["text"] for d in _sentences(events))
    assert "opener_kind" not in _latency(tmp_path)[-1]


def test_cancelled_turn_suppresses_context_opener(gw, fake_tts, monkeypatch, tmp_path):
    monkeypatch.setenv("MVG_CONTEXT_OPENER", "1")

    def brain(prompt, session_id, node_id, speaker, score, timeout, on_partial):
        assert gateway.cancel_active_turn(node_id) in ("cancelled", "stopped_speech_only")
        gateway._STAGE_SINK.stages.setdefault("events", []).append(
            {"stage": "tool_start", "detail": "checking the build logs", "t": 0.05}
        )
        time.sleep(0.2)
        return ""

    monkeypatch.setattr(gateway, "STREAM_BRAIN_CALL", brain)
    events = _run(gw)
    assert not any("One moment" in d.get("text", "") for d in _sentences(events))
    assert _latency(tmp_path)[-1].get("opener_kind") != "stage"
