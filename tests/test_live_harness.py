"""LiveSession with MockBackend: tools, queueing, cancel, errors, shutdown."""
import asyncio
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import live_harness as lh  # noqa: E402
from talker_backend import CascadedTextBackend, MockBackend  # noqa: E402


def call(cid, name, args):
    return {"type": "tool_call", "call_id": cid, "name": name, "args": args}


async def _drive(script, brain, cancel_turn=None, wait=0.05, after=None):
    backend = MockBackend(script)
    frames = []
    s = lh.LiveSession(backend, {"node_id": "n1"}, frames.append, brain, cancel_turn)
    await s.open()
    await s.run()
    if after:
        await after(s)
    await asyncio.sleep(wait)
    await s.close()
    return s, backend, frames


def results(backend):
    return {x[1]: x[2] for x in backend.sent if x[0] == "tool_result"}


def texts(backend):
    return [x[1] for x in backend.sent if x[0] == "text"]


def test_call_thinker_returns_started_then_result_matched():
    brain = lambda req, tid: f"answer to {req}"
    s, b, _ = asyncio.run(_drive([call("c1", "call_thinker", {"request": "do x"})], brain))
    assert results(b)["c1"] == {"status": "started", "handoff_id": 1}
    assert texts(b) == ["[thinker result, handoff 1] answer to do x"]


def test_two_calls_queue_in_order():
    order = []
    def brain(req, tid):
        order.append(req)
        return req
    script = [call("c1", "call_thinker", {"request": "first"}), call("c2", "call_thinker", {"request": "second"})]
    s, b, _ = asyncio.run(_drive(script, brain, wait=0.1))
    assert results(b)["c1"]["handoff_id"] == 1 and results(b)["c2"]["handoff_id"] == 2
    assert order == ["first", "second"]
    assert texts(b) == ["[thinker result, handoff 1] first", "[thinker result, handoff 2] second"]


def test_cancel_mid_call_drops_result_and_queue():
    release = threading.Event()
    started = threading.Event()
    def brain(req, tid):
        started.set()
        release.wait(2)
        return "late"
    seen = []
    def cancel_turn(node):
        seen.append(node)
        return "cancelled"

    async def go():
        backend = MockBackend([])
        frames = []
        s = lh.LiveSession(backend, {"node_id": "n1"}, frames.append, brain, cancel_turn)
        await s.open()
        await s._handle_tool("call_thinker", {"request": "long"})
        await s._handle_tool("call_thinker", {"request": "queued"})
        await asyncio.get_running_loop().run_in_executor(None, started.wait, 2)
        res = await s._handle_tool("cancel_thinker", {})
        release.set()
        await asyncio.sleep(0.1)
        await s.close()
        return res, backend
    res, backend = asyncio.run(go())
    assert res == {"status": "cancelled", "dropped_queued": 1}
    assert seen == ["n1"]
    assert texts(backend) == []


@pytest.mark.parametrize("name,args", [
    ("call_thinker", {}), ("call_thinker", {"request": ""}), ("call_thinker", {"request": 5}),
    ("call_thinker", {"request": "x", "extra": 1}), ("call_thinker", "nope"),
    ("cancel_thinker", {"x": 1}), ("answer_prompt", {"prompt_id": "p"}),
    ("answer_prompt", {"prompt_id": "p", "decision": "maybe"}),
    ("set_progress_updates", {"enabled": "yes"}), ("nonsense", {}), (None, {}),
])
def test_malformed_calls_return_error(name, args):
    s, b, _ = asyncio.run(_drive([call("c1", name, args)], lambda r, t: ""))
    assert results(b)["c1"]["status"] == "error"


def test_answer_prompt_unsupported_and_progress_flag():
    script = [call("a", "answer_prompt", {"prompt_id": "p", "decision": "approve"}),
              call("b", "set_progress_updates", {"enabled": True})]
    s, b, _ = asyncio.run(_drive(script, lambda r, t: ""))
    assert results(b)["a"] == {"status": "unsupported"}
    assert results(b)["b"]["status"] == "ok" and s.progress_updates is True


def test_backend_error_ends_session_cleanly():
    script = [{"type": "text", "text": "hi"}, {"type": "error", "message": "boom"},
              {"type": "text", "text": "never forwarded"}]
    s, b, frames = asyncio.run(_drive(script, lambda r, t: ""))
    assert s.error == "boom" and s.ended.is_set() and b.closed
    assert [f["type"] for f in frames] == ["text", "error"]


def test_audio_forwarded_both_ways_and_no_leaked_tasks():
    async def go():
        backend = MockBackend([{"type": "audio", "pcm": b"\x01\x02"}, {"type": "turn_end"}])
        frames = []
        s = lh.LiveSession(backend, {}, frames.append, lambda r, t: "")
        await s.open()
        await s.feed_audio(b"\x09\x09")
        await s.run()
        await s.close()
        left = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        return backend, frames, left
    backend, frames, left = asyncio.run(go())
    assert ("audio", b"\x09\x09") in backend.sent
    assert [f["type"] for f in frames] == ["audio", "turn_end"]
    assert left == []


def test_brain_exception_is_reported_not_raised():
    def brain(r, t):
        raise RuntimeError("down")
    s, b, _ = asyncio.run(_drive([call("c1", "call_thinker", {"request": "x"})], brain))
    assert "failed" in texts(b)[0]


def _wav(n=2400, rate=24000):
    import io, wave
    b = io.BytesIO()
    with wave.open(b, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate)
        w.writeframes(b"\x01\x00" * n)
    return b.getvalue()


def _run_speak(script, speak):
    async def go():
        backend = MockBackend(script)
        frames = []
        s = lh.LiveSession(backend, {"node_id": "n1"}, frames.append, lambda r, t: "", None, speak)
        await s.open()
        await s.run()
        await s.close()
        return frames
    return asyncio.run(go())


def test_text_reply_is_spoken():
    frames = _run_speak([{"type": "text", "text": "hello"}], lambda t: _wav())
    assert [f["type"] for f in frames] == ["text", "audio"]
    assert frames[1]["synthesized"] and len(frames[1]["pcm"]) == 4800


def test_spoken_text_not_resynthesized():
    frames = _run_speak([{"type": "text", "text": "hi", "spoken": True}], lambda t: _wav())
    assert [f["type"] for f in frames] == ["text"]


def test_tts_failure_is_non_fatal():
    def boom(t):
        raise RuntimeError("no tts")
    frames = _run_speak([{"type": "text", "text": "hello"}, {"type": "turn_end"}], boom)
    assert [f["type"] for f in frames] == ["text", "tts_error", "turn_end"]


def test_no_speak_callable_leaves_text_alone():
    frames = _run_speak([{"type": "text", "text": "hello"}], None)
    assert [f["type"] for f in frames] == ["text"]


def test_resume_handle_and_reconnect_dispatch():
    script = [
        {"type": "resume_handle", "handle": "h-99"},
        {"type": "reconnect", "time_left": "3s"},
        {"type": "text", "text": "still here"},
        {"type": "turn_end"},
    ]
    s, b, frames = asyncio.run(_drive(script, lambda r, t: ""))
    assert b.reconnected == 1
    assert ("reconnect",) in b.sent
    assert s.error is None
    assert [f["type"] for f in frames] == ["text", "turn_end"]


def test_reconnect_failure_ends_session_with_error():
    async def go():
        backend = MockBackend([{"type": "reconnect", "time_left": "1s"}, {"type": "text", "text": "unreached"}])
        backend.reconnect_error = ConnectionError("socket dead")
        frames = []
        s = lh.LiveSession(backend, {"node_id": "n1"}, frames.append, lambda r, t: "")
        await s.open()
        await s.run()
        await s.close()
        return s, backend, frames

    s, b, frames = asyncio.run(go())
    assert b.reconnected == 1
    assert s.error == "socket dead"
    assert s.ended.is_set()
    assert frames == [{"type": "error", "message": "socket dead"}]


def test_streaming_text_deltas_buffered_by_sentence_segmenter():
    spoken = []

    def speak(text):
        spoken.append(text)
        return _wav()

    script = [
        {"type": "text", "text": "Hello "},
        {"type": "text", "text": "world. "},
        {"type": "text", "text": "Next "},
        {"type": "text", "text": "line."},
        {"type": "turn_end"},
    ]
    frames = _run_speak(script, speak)

    received_texts = [f["text"] for f in frames if f["type"] == "text"]
    assert received_texts == ["Hello ", "world. ", "Next ", "line."]
    assert spoken == ["Hello world.", "Next line."]
    assert [f["type"] for f in frames] == ["text", "text", "audio", "text", "text", "audio", "turn_end"]


def test_interrupted_discards_buffered_unspoken_text():
    spoken = []

    def speak(text):
        spoken.append(text)
        return _wav()

    script = [
        {"type": "text", "text": "Partial unfin"},
        {"type": "interrupted"},
        {"type": "text", "text": "Fresh reply."},
        {"type": "turn_end"},
    ]
    frames = _run_speak(script, speak)
    assert spoken == ["Fresh reply."]
    assert [f["type"] for f in frames] == ["text", "interrupted", "text", "audio", "turn_end"]


def test_cascaded_session_audio_stt_and_feed_text():
    transcribed_wavs = []

    def fake_transcribe(wav):
        transcribed_wavs.append(wav)
        return "check status"

    def brain(req, tid):
        return f"status normal: {req}"

    backend = CascadedTextBackend()
    frames = []

    async def go():
        s = lh.LiveSession(
            backend,
            {"node_id": "n1"},
            frames.append,
            brain,
            speak=lambda t: _wav(),
            transcribe=fake_transcribe,
        )
        await s.open()
        run_task = asyncio.create_task(s.run())

        # Voiced PCM16 16kHz frames (4000 samples = 250 ms)
        voiced = b"\x00\x10" * 4000
        await s.feed_audio(voiced)

        # Silence (11200 samples = 700 ms, triggers silence >= 600ms)
        silence = b"\x00\x00" * 11200
        await s.feed_audio(silence)

        # Wait until thinker result arrives and is synthesized
        for _ in range(50):
            if any(f.get("type") == "text" and "status normal" in f.get("text", "") for f in frames):
                break
            await asyncio.sleep(0.05)

        # Verify feed_text directly
        await s.feed_text("hello")
        await asyncio.sleep(0.05)

        await s.close()
        await run_task

    asyncio.run(go())

    assert len(transcribed_wavs) == 1
    assert transcribed_wavs[0].startswith(b"RIFF")

    texts_seen = [f.get("text") for f in frames if f.get("type") == "text"]
    assert "On it." in texts_seen
    assert any("status normal: check status" in (t or "") for t in texts_seen)
    assert any(m.get("content") == "hello" for m in backend.messages)
