"""ProgressRelay: silent context, rate limit, hold-until-asked, final line."""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import live_harness as lh  # noqa: E402
from talker_backend import MockBackend  # noqa: E402


class NoSilent(MockBackend):
    capabilities = frozenset({"audio_in", "audio_out"})


class Clock:
    t = 100.0
    def __call__(self):
        return self.t


def ev(detail):
    return {"stage": "tool_start", "detail": detail, "t": 1.0}


def mk(backend, gap=12.0):
    s = lh.LiveSession(backend, {"node_id": "n"}, lambda f: None, lambda r, t: "")
    clk = Clock()
    s.relay = lh.ProgressRelay(s, clock=clk, min_gap=gap)
    return s, clk


def sent(b):
    return [(x[1], x[2]) for x in b.sent if x[0] == "text"]


def test_silent_lines_in_order_rate_limited_latest_kept():
    async def go():
        b = MockBackend()
        s, clk = mk(b)
        await s.on_stage_event(ev("one"))
        clk.t += 3
        await s.on_stage_event(ev("two"))      # inside the gap: held
        clk.t += 3
        await s.on_stage_event(ev("three"))    # replaces two
        before = sent(b)
        await s.relay.finish()                 # task end flushes the latest
        s.relay.close()
        return before, sent(b)
    before, after = asyncio.run(go())
    assert before == [("[task progress] one", True)]
    assert after == [("[task progress] one", True), ("[task progress] three", True)]


def test_gap_timer_delivers_held_line():
    async def go():
        b = MockBackend()
        s, clk = mk(b, gap=0.05)
        await s.on_stage_event(ev("a"))
        await s.on_stage_event(ev("b"))
        clk.t += 1.0
        await asyncio.sleep(0.12)
        return sent(b)
    out = asyncio.run(go())
    assert out == [("[task progress] a", True), ("[task progress] b", True)]


def test_no_silent_context_holds_until_asked():
    async def go():
        b = NoSilent()
        s, clk = mk(b)
        await s.on_stage_event(ev("working"))
        await s.relay.finish()
        none_sent = sent(b)
        got = await s.status_request()
        return none_sent, got, sent(b)
    none_sent, got, after = asyncio.run(go())
    assert none_sent == []
    assert got == "[task progress] working" and after == [("[task progress] working", False)]


def test_no_silent_context_pushes_when_enabled():
    async def go():
        b = NoSilent()
        s, clk = mk(b)
        s.progress_updates = True
        await s.on_stage_event(ev("pushed"))
        return sent(b)
    assert asyncio.run(go()) == [("[task progress] pushed", False)]


def test_empty_detail_and_bad_events_ignored():
    async def go():
        b = MockBackend()
        s, _ = mk(b)
        for raw in (ev(""), "junk", None, {"detail": "no stage"}):
            await s.on_stage_event(raw)
        return sent(b), await s.status_request()
    assert asyncio.run(go()) == ([], None)


def test_last_line_delivered_at_task_end():
    async def go():
        b = MockBackend()
        s, clk = mk(b)
        await s.open()
        await s.on_stage_event(ev("first"))
        clk.t += 1
        await s.on_stage_event(ev("last"))
        await s._handle_tool("call_thinker", {"request": "x"})
        await asyncio.sleep(0.05)
        await s.close()
        return sent(b)
    out = asyncio.run(go())
    assert [t for t, _ in out][:2] == ["[task progress] first", "[task progress] last"]
