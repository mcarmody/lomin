"""LiveSession.send_context: harness lines wait for the Talker's turn to end."""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import live_harness as lh  # noqa: E402
from talker_backend import MockBackend  # noqa: E402

AUDIO = {"type": "audio", "pcm": b"\x00\x00"}
END = {"type": "turn_end"}


def mk(backend=None, brain=lambda r, t: "", hold_max=None):
    b = backend or MockBackend()
    s = lh.LiveSession(b, {"node_id": "n"}, lambda f: None, brain)
    if hold_max is not None:
        s.hold_max = hold_max
    return s, b


def sent(b):
    return [(x[1], x[2]) for x in b.sent if x[0] == "text"]


def test_idle_talker_gets_the_line_at_once():
    async def go():
        s, b = mk()
        await s.send_context("[thinker result, handoff 1] x")
        return sent(b)
    assert asyncio.run(go()) == [("[thinker result, handoff 1] x", False)]


def test_mid_turn_line_waits_for_turn_end():
    async def go():
        s, b = mk()
        await s._dispatch(AUDIO)
        await s.send_context("[thinker result, handoff 1] x")
        during = sent(b)
        await s._dispatch(END)
        await s.close()
        return during, sent(b)
    during, after = asyncio.run(go())
    assert during == []
    assert after == [("[thinker result, handoff 1] x", False)]


def test_text_marks_the_turn_busy_and_a_bare_tool_call_does_not():
    async def go(first):
        s, b = mk()
        await s._dispatch(first)
        await s.send_context("line")
        during = sent(b)
        await s._dispatch(END)
        await s.close()
        return during, sent(b)
    during, after = asyncio.run(go({"type": "text", "text": "Checking.", "spoken": True}))
    assert during == [] and after == [("line", False)]
    # A backend that calls a tool without speaking may never send turn_end: do not hold on it.
    call = {"type": "tool_call", "call_id": "c1", "name": "set_progress_updates", "args": {"enabled": True}}
    during, after = asyncio.run(go(call))
    assert during == [("line", False)] and after == [("line", False)]


def test_held_lines_keep_order_and_newest_progress_wins():
    async def go():
        s, b = mk()
        await s._dispatch(AUDIO)
        await s.send_context("[task progress] a", silent=True, key="progress")
        await s.send_context("[task progress] b", silent=True, key="progress")
        await s.send_context("[thinker result, handoff 1] done")
        await s._dispatch(END)
        await s.close()
        return sent(b)
    assert asyncio.run(go()) == [("[task progress] b", True), ("[thinker result, handoff 1] done", False)]


def test_interrupted_turn_still_waits_for_its_turn_end():
    async def go():
        s, b = mk()
        await s._dispatch(AUDIO)
        await s.send_context("line")
        await s._dispatch({"type": "interrupted"})
        during = sent(b)
        await s._dispatch(END)
        await s.close()
        return during, sent(b)
    during, after = asyncio.run(go())
    assert during == [] and after == [("line", False)]


def test_hold_gives_up_after_hold_max():
    async def go():
        s, b = mk(hold_max=0.03)
        await s._dispatch(AUDIO)
        await s.send_context("line")
        during = sent(b)
        await asyncio.sleep(0.1)
        after = sent(b)
        await s.send_context("next")     # the timer cleared the busy flag
        await s.close()
        return during, after, sent(b)
    during, after, final = asyncio.run(go())
    assert during == []
    assert after == [("line", False)]
    assert final == [("line", False), ("next", False)]


def test_reconnect_releases_held_lines():
    async def go():
        s, b = mk()
        await s._dispatch(AUDIO)
        await s.send_context("line")
        await s._dispatch({"type": "reconnect"})
        await s.close()
        return sent(b)
    assert asyncio.run(go()) == [("line", False)]


def test_close_drops_held_lines_and_timer():
    async def go():
        s, b = mk(hold_max=0.03)
        await s._dispatch(AUDIO)
        await s.send_context("line")
        await s.close()
        await asyncio.sleep(0.08)
        return sent(b), s._hold_timer
    texts, timer = asyncio.run(go())
    assert texts == [] and timer is None


class Gated(MockBackend):
    """Speaks, hands off, keeps speaking until the gate opens, then ends the turn."""
    def __init__(self):
        super().__init__()
        self.gate = asyncio.Event()

    async def events(self):
        yield AUDIO
        yield {"type": "tool_call", "call_id": "c1", "name": "call_thinker", "args": {"request": "do x"}}
        yield AUDIO
        await self.gate.wait()
        yield END


def test_thinker_result_does_not_cut_off_the_spoken_turn():
    async def go():
        b = Gated()
        s, _ = mk(b, brain=lambda r, t: f"answer to {r}")
        await s.open()
        run = asyncio.ensure_future(s.run())
        for _ in range(200):               # the brain answers while the Talker still speaks
            if s._held:
                break
            await asyncio.sleep(0.005)
        held, during = list(s._held), sent(b)
        b.gate.set()
        await run
        await s.close()
        return held, during, sent(b)
    held, during, after = asyncio.run(go())
    assert [h[0] for h in held] == ["[thinker result, handoff 1] answer to do x"]
    assert during == []
    assert after == [("[thinker result, handoff 1] answer to do x", False)]


def test_progress_relay_lines_go_through_the_hold():
    async def go():
        s, b = mk()
        await s._dispatch(AUDIO)
        await s.on_stage_event({"stage": "tool_start", "detail": "reading the calendar", "t": 1.0})
        during = sent(b)
        await s._dispatch(END)
        await s.close()
        return during, sent(b)
    during, after = asyncio.run(go())
    assert during == [] and after == [("[task progress] reading the calendar", True)]
