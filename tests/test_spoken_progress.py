"""Spoken progress (MVG_SPOKEN_PROGRESS): the narrator's rules and the stream
integration, against a fake brain that reports stage events."""
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
from test_ask_stream_pcm import fake_tts, gw, _events, _env  # noqa: E402,F401


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def _ev(stage, detail="doing a thing", t=0.0):
    return {"stage": stage, "detail": detail, "t": t}


def test_ten_events_over_sixty_seconds_respects_cap_and_gap(monkeypatch):
    monkeypatch.delenv("MVG_PROGRESS_STAGES", raising=False)
    clock, events = Clock(), []
    n = gateway.ProgressNarrator(lambda: events, clock=clock, min_gap=12, max_lines=3)
    spoken = []
    for i in range(10):
        events.append(_ev("tool_start", f"step {i}"))
        clock.t += 6.0
        line = n.next_line(playing=False, reply_started=False)
        if line:
            spoken.append((clock.t, line))
    assert 1 <= len(spoken) <= 3
    assert all(b[0] - a[0] >= 12 for a, b in zip(spoken, spoken[1:]))


def test_unlisted_stage_never_spoken():
    events = [_ev("thinking"), _ev("token")]
    n = gateway.ProgressNarrator(lambda: events, clock=Clock(), min_gap=0, max_lines=3,
                                 allow=["tool_start", "milestone"])
    assert n.next_line(False, False) is None


def test_not_while_playing_or_after_reply_start():
    events = [_ev("milestone", "halfway")]
    n = gateway.ProgressNarrator(lambda: events, clock=Clock(), min_gap=0, max_lines=3, allow=["milestone"])
    assert n.next_line(playing=True, reply_started=False) is None
    events.append(_ev("milestone", "later"))
    assert n.next_line(playing=False, reply_started=True) is None
    assert n.lines == 0


def test_newest_eligible_wins_and_empty_detail_skipped():
    events = [_ev("milestone", "old"), _ev("milestone", "new"), _ev("milestone", "")]
    n = gateway.ProgressNarrator(lambda: events, clock=Clock(), min_gap=0, max_lines=3, allow=["milestone"])
    assert n.next_line(False, False) == "new"


def test_bad_source_is_quiet():
    def boom():
        raise RuntimeError("x")
    n = gateway.ProgressNarrator(boom, clock=Clock(), min_gap=0, max_lines=3, allow=["milestone"])
    assert n.next_line(False, False) is None


def _turn(port):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    conn.request("POST", "/ask/stream", body=json.dumps({"prompt": "hi", "session_id": "s", "node_id": "n1"}),
                 headers={"Content-Type": "application/json"})
    ev = _events(conn.getresponse())
    conn.close()
    return ev


def _brain(monkeypatch, stage="tool_start"):
    def brain(prompt, session_id, node_id, speaker, speaker_score, timeout, on_partial):
        sink = getattr(gateway._STAGE_SINK, "stages", None)
        sink.setdefault("events", []).append({"stage": stage, "detail": "checking the logs", "t": 1.0})
        time.sleep(1.5)
        on_partial("All finished here.")
        return "All finished here."
    monkeypatch.setattr(gateway, "STREAM_BRAIN_CALL", brain)


def _records(tmp_path):
    lines = (tmp_path / "voice-latency.jsonl").read_text().splitlines()
    return [r for r in map(json.loads, lines) if r["kind"] == "turn"]


def test_progress_line_spoken_and_counted(gw, fake_tts, monkeypatch, tmp_path):
    monkeypatch.setenv("MVG_SPOKEN_PROGRESS", "1")
    _brain(monkeypatch)
    ev = _turn(gw)
    texts = [d.get("text", "") for e, d in ev if e == "sentence"]
    assert "checking the logs" in " ".join(texts)
    rec = _records(tmp_path)[-1]
    assert rec["progress_lines"] == 1
    assert "events" not in rec.get("stages", {})


def test_off_by_default(gw, fake_tts, monkeypatch, tmp_path):
    monkeypatch.delenv("MVG_SPOKEN_PROGRESS", raising=False)
    _brain(monkeypatch)
    ev = _turn(gw)
    assert "checking the logs" not in " ".join(d.get("text", "") for e, d in ev if e == "sentence")
    assert _records(tmp_path)[-1]["progress_lines"] == 0


def test_unlisted_stage_not_spoken_in_stream(gw, fake_tts, monkeypatch):
    monkeypatch.setenv("MVG_SPOKEN_PROGRESS", "1")
    _brain(monkeypatch, stage="token")
    ev = _turn(gw)
    assert "checking the logs" not in " ".join(d.get("text", "") for e, d in ev if e == "sentence")


def test_no_thread_left_after_turn(gw, fake_tts, monkeypatch):
    monkeypatch.setenv("MVG_SPOKEN_PROGRESS", "1")
    _brain(monkeypatch)
    before = {t.ident for t in threading.enumerate()}
    _turn(gw)
    time.sleep(0.3)
    extra = [t for t in threading.enumerate() if t.ident not in before and "progress" in t.name.lower()]
    assert extra == []


def test_cancelled_turn_suppresses_spoken_progress(gw, fake_tts, monkeypatch, tmp_path):
    monkeypatch.setenv("MVG_SPOKEN_PROGRESS", "1")

    def brain(prompt, session_id, node_id, speaker, speaker_score, timeout, on_partial):
        gateway._STAGE_SINK.stages.setdefault("events", []).append(
            {"stage": "tool_start", "detail": "checking the logs", "t": 0.1}
        )
        assert gateway.cancel_active_turn("n1") in ("cancelled", "stopped_speech_only")
        time.sleep(0.2)
        return ""

    monkeypatch.setattr(gateway, "STREAM_BRAIN_CALL", brain)
    ev = _turn(gw)
    assert "checking the logs" not in " ".join(d.get("text", "") for e, d in ev if e == "sentence")
    assert _records(tmp_path)[-1]["progress_lines"] == 0
