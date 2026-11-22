"""LiveSession state frames: {"type": "state", "state": name} for the turn strip."""
import asyncio
import gc
import struct
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import audio_utils  # noqa: E402
import live_harness as lh  # noqa: E402
from talker_backend import MockBackend  # noqa: E402

CASCADED_CAPS = frozenset({"text_out", "tool_calls", "silent_context"})
END = {"type": "turn_end"}
AUDIO = {"type": "audio", "pcm": b"\x00\x00"}


def loud(ms):
    return struct.pack("<h", 3000) * (16 * ms)


def quiet(ms):
    return b"\x00\x00" * (16 * ms)


def states(frames):
    return [f["state"] for f in frames if f.get("type") == "state"]


def kinds(frames):
    return [f.get("state") or f["type"] for f in frames]


def call_thinker(cid="c1"):
    return {"type": "tool_call", "call_id": cid, "name": "call_thinker", "args": {"request": "x"}}


class QueueBackend(MockBackend):
    """Events are pushed by the test, so they interleave with fed audio in a known order."""

    def __init__(self, audio_in=True):
        super().__init__()
        if not audio_in:
            self.capabilities = CASCADED_CAPS
        self.q = asyncio.Queue()

    async def emit(self, *events):
        for ev in events:
            await self.q.put(ev)
        await asyncio.sleep(0.02)

    async def events(self):
        while True:
            ev = await self.q.get()
            if ev is None:
                return
            yield ev

    async def close(self):
        await super().close()
        await self.q.put(None)


class ReplyingBackend(QueueBackend):
    """Like a cascaded backend: send_text emits its reply events and only returns later."""

    def __init__(self):
        super().__init__(audio_in=False)

    async def send_text(self, text, *, silent=False):
        await super().send_text(text, silent=silent)
        await self.q.put(AUDIO)
        await asyncio.sleep(0.05)
        await self.q.put(END)


class SlowReplyingBackend(ReplyingBackend):
    """The reply request takes 0.4 s and no backend event comes before it returns."""

    async def send_text(self, text, *, silent=False):
        await asyncio.sleep(0.4)
        await super().send_text(text, silent=silent)


def go(test, backend=None, **kw):
    """Run `test(session, backend, frames)` on an open session with run() going."""
    async def main():
        b = backend or QueueBackend()
        frames = []
        stale = kw.pop("stale_sec", 30.0)
        brain = kw.pop("brain", lambda r, t: "")
        s = lh.LiveSession(b, {"node_id": "n1"}, frames.append, brain, **kw)
        s.stale_sec = stale
        await s.open()
        task = asyncio.ensure_future(s.run())
        try:
            await test(s, b, frames)
            await asyncio.sleep(0.02)
        finally:
            await s.close()
            await asyncio.gather(task, return_exceptions=True)
        return s, frames
    return asyncio.run(main())


async def feed(s, pcm, step_ms=40):
    size = 32 * step_ms
    for i in range(0, len(pcm), size):
        await s.feed_audio(pcm[i:i + size])


def test_audio_in_vad_received_then_thinking():
    async def t(s, b, frames):
        await feed(s, loud(240))
        await feed(s, quiet(700))
    _, frames = go(t)
    assert states(frames) == ["received", "thinking"]


def test_speech_under_200_ms_lights_nothing():
    async def t(s, b, frames):
        await feed(s, loud(100))
        await feed(s, quiet(700))
    _, frames = go(t)
    assert states(frames) == []


def test_received_sent_once_per_utterance():
    async def t(s, b, frames):
        await feed(s, loud(400))
    _, frames = go(t)
    assert states(frames) == ["received"]


def test_cascaded_flush_sequence():
    async def t(s, b, frames):
        await feed(s, loud(300))
        await feed(s, quiet(700))
    _, frames = go(t, QueueBackend(audio_in=False), transcribe=lambda wav: "hello")
    assert states(frames) == ["received", "transcribing", "thinking"]


def test_cascaded_empty_transcript_returns_to_listening():
    async def t(s, b, frames):
        await feed(s, loud(300))
        await feed(s, quiet(700))
    _, frames = go(t, QueueBackend(audio_in=False), transcribe=lambda wav: "")
    assert states(frames) == ["received", "transcribing", "listening"]


def test_cascaded_stt_failure_returns_to_listening():
    def boom(wav):
        raise RuntimeError("down")

    async def t(s, b, frames):
        await feed(s, loud(300))
        await feed(s, quiet(700))
    _, frames = go(t, QueueBackend(audio_in=False), transcribe=boom)
    assert states(frames) == ["received", "transcribing", "listening"]
    assert any(f.get("type") == "tts_error" for f in frames)


def test_cascaded_thinking_goes_out_before_the_reply_it_waits_for():
    async def t(s, b, frames):
        await feed(s, loud(300))
        await feed(s, quiet(700))
        await asyncio.sleep(0.2)
    _, frames = go(t, ReplyingBackend(), transcribe=lambda wav: "hello")
    assert states(frames) == ["received", "transcribing", "thinking", "speaking", "listening"]


def test_input_transcript_lights_transcribing_and_is_not_forwarded():
    async def t(s, b, frames):
        await feed(s, loud(240))
        await b.emit({"type": "input_transcript", "text": "zebra"})
    _, frames = go(t)
    assert states(frames) == ["received", "transcribing"]
    assert not any(f.get("type") == "input_transcript" for f in frames)
    assert "zebra" not in repr(frames)


def test_native_turn_with_input_transcription_lights_text_then_think():
    async def t(s, b, frames):
        await feed(s, loud(240))
        await b.emit({"type": "input_transcript", "text": "zebra"})
        await feed(s, loud(400))
        # Speech while transcribing does not count as new voiced time, and it resets the silence count.
        assert s._vad_voiced_ms == 240.0
        await feed(s, quiet(700))
    _, frames = go(t)
    assert states(frames) == ["received", "transcribing", "thinking"]


def test_input_transcript_outside_the_users_turn_is_dropped():
    late = {"type": "input_transcript", "text": "zebra"}

    async def idle(s, b, frames):
        await b.emit(late)
    assert states(go(idle)[1]) == []

    async def after_turn(s, b, frames):
        await b.emit(AUDIO, END, late)
    assert states(go(after_turn)[1]) == ["speaking", "listening"]

    async def while_thinking(s, b, frames):
        await feed(s, loud(240))
        await b.emit(call_thinker(), late)
    assert states(go(while_thinking)[1]) == ["received", "thinking"]


def test_call_thinker_marks_thinking_and_an_agent_turn_never_lights_heard():
    async def t(s, b, frames):
        await b.emit(call_thinker())
        await asyncio.sleep(0.1)  # the brain call returns at once, so no job is pending at turn_end
        await b.emit(AUDIO, END)
    _, frames = go(t)
    assert states(frames) == ["thinking", "speaking", "listening"]


def test_other_tools_do_not_mark_thinking():
    call = {"type": "tool_call", "call_id": "c1", "name": "set_progress_updates", "args": {"enabled": True}}

    async def t(s, b, frames):
        await b.emit(call)
    _, frames = go(t)
    assert states(frames) == []


def test_audio_marks_speaking_before_the_audio_frame():
    async def t(s, b, frames):
        await b.emit(AUDIO, AUDIO, END)
    _, frames = go(t)
    assert kinds(frames) == ["speaking", "audio", "audio", "turn_end", "listening"]


def test_interrupted_returns_to_listening():
    async def t(s, b, frames):
        await b.emit(AUDIO, {"type": "interrupted"})
    _, frames = go(t)
    assert kinds(frames) == ["speaking", "audio", "interrupted", "listening"]


def test_reconnect_returns_to_listening():
    async def t(s, b, frames):
        await b.emit(AUDIO, {"type": "reconnect"})
    _, frames = go(t)
    assert states(frames) == ["speaking", "listening"]


def test_never_moves_backwards():
    async def t(s, b, frames):
        await s._set_phase("speaking")
        await s._set_phase("thinking")
        await s._set_phase("speaking")
    _, frames = go(t)
    assert states(frames) == ["speaking"]


def test_unknown_state_name_is_logged_and_skipped(capsys):
    async def t(s, b, frames):
        await s._set_phase("bogus")
        assert s._phase == "listening"
    _, frames = go(t)
    assert states(frames) == []
    assert "unknown state 'bogus'" in capsys.readouterr().err


def _wav():
    return audio_utils.pcm_to_wav(b"\x00\x00" * 2400, 24000)


def test_custom_tts_sends_synthesizing_then_speaking():
    text = {"type": "text", "text": "Hello there.", "spoken": False}

    async def t(s, b, frames):
        await b.emit(text, END)
    _, frames = go(t, QueueBackend(audio_in=False), speak=lambda t: _wav())
    assert kinds(frames) == ["text", "synthesizing", "speaking", "audio", "turn_end", "listening"]


def test_custom_tts_failure_sends_no_speaking():
    def boom(t):
        raise RuntimeError("tts down")

    async def t(s, b, frames):
        await b.emit({"type": "text", "text": "Hello there.", "spoken": False}, END)
    _, frames = go(t, QueueBackend(audio_in=False), speak=boom)
    assert states(frames) == ["synthesizing", "listening"]
    assert any(f.get("type") == "tts_error" for f in frames)


@pytest.mark.parametrize("value", ["0", "false", "FALSE", "no", "off", " Off "])
def test_state_events_off(monkeypatch, value):
    monkeypatch.setenv("MVG_LIVE_STATE_EVENTS", value)

    async def t(s, b, frames):
        await feed(s, loud(240))
        await b.emit(AUDIO, END)
    _, frames = go(t)
    assert states(frames) == []
    assert [f["type"] for f in frames] == ["audio", "turn_end"]


@pytest.mark.parametrize("value", [None, "", "1", "true", "yes"])
def test_state_events_on_by_default(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("MVG_LIVE_STATE_EVENTS", raising=False)
    else:
        monkeypatch.setenv("MVG_LIVE_STATE_EVENTS", value)

    async def t(s, b, frames):
        await b.emit(AUDIO, END)
    _, frames = go(t)
    assert states(frames) == ["speaking", "listening"]


def test_detector_ignores_the_users_audio_while_the_agent_speaks():
    async def t(s, b, frames):
        await b.emit(AUDIO)
        await feed(s, loud(400))
        assert s._vad_voiced_ms == 0.0
    _, frames = go(t)
    assert states(frames) == ["speaking"]


def test_detector_counters_reset_when_the_turn_returns_to_listening():
    async def t(s, b, frames):
        await feed(s, loud(240))
        await b.emit(AUDIO, END)
        await feed(s, loud(40))
    _, frames = go(t)
    assert states(frames) == ["received", "speaking", "listening"]


def test_stale_turn_returns_to_listening():
    async def t(s, b, frames):
        await feed(s, loud(240))
        await asyncio.sleep(0.3)  # 6x the 0.05 s timeout
    _, frames = go(t, stale_sec=0.05)
    assert states(frames) == ["received", "listening"]


def test_stale_wait_is_not_armed_for_speaking():
    async def t(s, b, frames):
        await b.emit(AUDIO)
        assert s._stale_task is None
        await asyncio.sleep(0.3)  # 6x the 0.05 s timeout
    _, frames = go(t, stale_sec=0.05)
    assert states(frames) == ["speaking"]


def test_a_backend_event_restarts_the_stale_wait():
    async def t(s, b, frames):
        await feed(s, loud(240))
        for _ in range(6):  # about 0.42 s of events, an event every 0.07 s against a 0.3 s timeout
            await asyncio.sleep(0.05)
            await b.emit({"type": "resume_handle", "handle": "h"})
        assert states(frames) == ["received"]
        await asyncio.sleep(1.5)  # 5x the 0.3 s timeout after the last event
    _, frames = go(t, stale_sec=0.3)
    assert states(frames) == ["received", "listening"]


def test_voiced_frames_keep_the_stale_wait_from_firing():
    async def t(s, b, frames):
        for _ in range(14):  # about 0.5 s of speech, one 40 ms frame every 30 ms or so
            await s.feed_audio(loud(40))
            await asyncio.sleep(0.03)
        assert states(frames) == ["received"]
        await asyncio.sleep(1.0)  # 5x the 0.2 s timeout after the last voiced frame
    _, frames = go(t, stale_sec=0.2)
    assert states(frames) == ["received", "listening"]


def test_a_thinker_job_that_outlasts_the_timeout_keeps_thinking():
    def slow_brain(request, task_id):
        time.sleep(0.4)
        return "done"

    async def t(s, b, frames):
        await b.emit(call_thinker())
        await asyncio.sleep(0.25)  # 5x the 0.05 s timeout; the brain is still running
        assert states(frames) == ["thinking"]
        assert s._stale_task is None
        await asyncio.sleep(0.4)  # the result is sent and the timer re-arms
    _, frames = go(t, brain=slow_brain, stale_sec=0.05)
    assert states(frames) == ["thinking", "listening"]


def test_a_thinker_job_still_running_at_turn_end_keeps_thinking_lit():
    def slow_brain(request, task_id):
        time.sleep(0.8)  # still running when the checks below run
        return "done"

    async def t(s, b, frames):
        await b.emit(call_thinker(), AUDIO, END)  # the Talker answers, then turn_end: listening, then thinking
        await feed(s, loud(240))                  # the user speaks; THINK ranks above HEARD, so nothing goes out
        await asyncio.sleep(0.25)                 # 5x the 0.05 s timeout
        assert s._stale_task is None
    _, frames = go(t, brain=slow_brain, stale_sec=0.05)
    assert states(frames) == ["thinking", "speaking", "listening", "thinking"]


def test_a_cancelled_thinker_job_lets_the_strip_return_to_listening():
    def slow_brain(request, task_id):
        time.sleep(0.4)
        return "done"

    cancel = {"type": "tool_call", "call_id": "c2", "name": "cancel_thinker", "args": {}}

    async def t(s, b, frames):
        await b.emit(call_thinker(), AUDIO, END)
        await b.emit(cancel)  # the job is running by now
        await asyncio.sleep(0.8)  # the cancelled job ends at 0.4 s; 5x the 0.05 s timeout is left
    _, frames = go(t, brain=slow_brain, stale_sec=0.05)
    assert states(frames) == ["thinking", "speaking", "listening", "thinking", "listening"]


def test_close_cancels_the_stale_wait():
    async def main():
        b = QueueBackend()
        frames = []
        s = lh.LiveSession(b, {}, frames.append, lambda r, t: "")
        s.stale_sec = 0.05
        await s.open()
        await feed(s, loud(240))
        task = s._stale_task
        assert task is not None
        await s.close()
        await asyncio.sleep(0.3)  # 6x the 0.05 s timeout
        return task, frames
    task, frames = asyncio.run(main())
    assert task.cancelled()
    assert states(frames) == ["received"]


def test_scattered_noise_bursts_do_not_light_heard():
    async def t(s, b, frames):
        for _ in range(6):  # 240 ms of voice in all, but 400 ms of silence after each burst
            await feed(s, loud(40))
            await feed(s, quiet(400))
    _, frames = go(t)
    assert states(frames) == []


def test_speech_with_a_short_gap_still_counts():
    async def t(s, b, frames):
        await feed(s, loud(100))
        await feed(s, quiet(200))  # under the 300 ms decay
        await feed(s, loud(120))
    _, frames = go(t)
    assert states(frames) == ["received"]


@pytest.mark.parametrize("transcribe", [lambda wav: "", lambda wav: 1 / 0], ids=["empty", "fails"])
def test_an_empty_or_failed_transcript_leaves_a_speaking_turn_alone(transcribe):
    async def t(s, b, frames):
        await b.emit(AUDIO)  # the agent is speaking when the user's noise reaches the cascaded buffer
        await feed(s, loud(300))
        await feed(s, quiet(700))
    _, frames = go(t, QueueBackend(audio_in=False), transcribe=transcribe)
    assert states(frames) == ["speaking"]


def test_a_slow_cascaded_reply_keeps_thinking_lit():
    async def t(s, b, frames):
        await feed(s, loud(300))
        await feed(s, quiet(700))
        await asyncio.sleep(0.8)  # the reply returns at 0.4 s; 8x the 0.05 s timeout
    _, frames = go(t, SlowReplyingBackend(), transcribe=lambda wav: "hello", stale_sec=0.05)
    assert states(frames) == ["received", "transcribing", "thinking", "speaking", "listening"]


def test_a_pause_then_speech_then_a_pause_waits_for_the_last_pause():
    async def t(s, b, frames):
        await feed(s, loud(240))
        await b.emit({"type": "input_transcript", "text": "zebra"})
        await feed(s, quiet(300))
        await feed(s, loud(40))   # speech in transcribing resets the silence count
        await feed(s, quiet(300))
        assert states(frames) == ["received", "transcribing"]
        await feed(s, quiet(400))
    _, frames = go(t)
    assert states(frames) == ["received", "transcribing", "thinking"]


def test_a_late_turn_end_while_the_user_speaks_keeps_heard():
    async def t(s, b, frames):
        await feed(s, loud(240))
        await b.emit(END)
        await feed(s, loud(240))
    _, frames = go(t)
    assert states(frames) == ["received"]


def test_the_detector_does_not_run_with_state_events_off(monkeypatch):
    monkeypatch.setenv("MVG_LIVE_STATE_EVENTS", "0")

    async def t(s, b, frames):
        await feed(s, loud(240))
        assert s._vad_voiced_ms == 0.0
    _, frames = go(t)
    assert states(frames) == []


def test_voiced_frames_reuse_the_stale_task():
    async def t(s, b, frames):
        await feed(s, loud(240))
        first = s._stale_task
        assert first is not None
        await feed(s, loud(400))
        assert s._stale_task is first and not first.done()
    go(t, stale_sec=30.0)


def test_a_stale_wait_whose_send_fails_leaves_no_task_error():
    async def main():
        errors, dead = [], []
        asyncio.get_running_loop().set_exception_handler(lambda loop, ctx: errors.append(ctx))

        def client(frame):
            if dead:
                raise ConnectionError("closed")

        b = QueueBackend()
        s = lh.LiveSession(b, {}, client, lambda r, t: "")
        s.stale_sec = 0.05
        await s.open()
        await feed(s, loud(240))  # received goes out
        dead.append(True)
        await asyncio.sleep(0.3)  # 6x the 0.05 s timeout; the stale wait's send raises
        gc.collect()
        await s.close()
        return errors
    assert asyncio.run(main()) == []
