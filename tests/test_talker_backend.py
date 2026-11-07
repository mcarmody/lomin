"""TalkerBackend contract, MockBackend and load_backend."""
import asyncio
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import talker_backend as tb  # noqa: E402


def run(coro):
    return asyncio.run(coro)


def test_mock_round_trip():
    script = [{"type": "audio", "pcm": b"\x00\x01"}, {"type": "text", "text": "hi"},
              {"type": "tool_call", "call_id": "c1", "name": "delegate", "args": {"q": 1}},
              {"type": "turn_end"}]
    b = tb.MockBackend(script)

    async def go():
        await b.open({"voice": "x"})
        await b.send_audio(b"\x02\x03")
        await b.send_text("ctx", silent=True)
        await b.send_tool_result("c1", {"ok": True})
        got = [e async for e in b.events()]
        await b.cancel()
        await b.close()
        return got
    got = run(go())
    assert got == script
    assert b.sent == [("open", {"voice": "x"}), ("audio", b"\x02\x03"), ("text", "ctx", True),
                      ("tool_result", "c1", {"ok": True}), ("cancel",), ("close",)]
    assert b.opened and b.closed
    assert {e["type"] for e in got} <= tb.EVENT_TYPES


def test_requires():
    b = tb.MockBackend()
    tb.requires(b, "audio_in", "tool_calls")
    class Bare(tb.TalkerBackend):
        capabilities = frozenset({"text_out"})
    with pytest.raises(ValueError, match="audio_in"):
        tb.requires(Bare(), "text_out", "audio_in")


def test_load_builtin():
    assert tb.load_backend("mock") is tb.MockBackend
    assert tb.load_backend("cascaded") is tb.CascadedTextBackend


def test_mock_reconnect():
    b = tb.MockBackend()
    run(b.reconnect())
    assert b.reconnected == 1
    assert ("reconnect",) in b.sent
    b.reconnect_error = RuntimeError("fail")
    with pytest.raises(RuntimeError, match="fail"):
        run(b.reconnect())
    assert b.reconnected == 2


def test_cascaded_backend_default_and_responder():
    b = tb.CascadedTextBackend()
    tb.requires(b, "text_out", "tool_calls", "silent_context")

    async def go_default():
        await b.open({"instructions": "Be brief."})
        await b.send_text("quiet progress", silent=True)
        await b.send_text("what is the weather?")
        evs = []
        async for ev in b.events():
            evs.append(ev)
            if ev["type"] == "turn_end":
                break
        await b.send_tool_result("call_1", {"status": "started", "handoff_id": 1})
        await b.send_text("[thinker result, handoff 1] **Sunny** and `72F`.")
        async for ev in b.events():
            evs.append(ev)
            if ev["type"] == "turn_end":
                break
        await b.reconnect()
        await b.close()
        return evs

    evs = run(go_default())
    assert [e["type"] for e in evs] == ["text", "tool_call", "turn_end", "text", "turn_end"]
    assert evs[1]["name"] == "call_thinker"
    assert evs[1]["args"] == {"request": "what is the weather?"}
    assert "**" not in evs[3]["text"] and "`" not in evs[3]["text"]
    assert "Sunny" in evs[3]["text"]
    assert b.reconnected == 1

    # Custom responder
    b2 = tb.CascadedTextBackend(responder=lambda msgs, cfg: {
        "text": "```code``` Checking now.",
        "tool_calls": [{"id": "tc1", "function": {"name": "call_thinker", "arguments": '{"request": "check"}'}}],
    })

    async def go_custom():
        await b2.open({})
        await b2.send_text("check please")
        out = []
        async for ev in b2.events():
            out.append(ev)
            if ev["type"] == "turn_end":
                break
        await b2.close()
        return out

    out = run(go_custom())
    assert [e["type"] for e in out] == ["text", "tool_call", "turn_end"]
    assert "```" not in out[0]["text"]
    assert out[1] == {"type": "tool_call", "call_id": "tc1", "name": "call_thinker", "args": {"request": "check"}}


def test_load_plugin_entry(monkeypatch):
    mod = types.ModuleType("fake_talker_mod")
    class Plug(tb.MockBackend):
        pass
    mod.Plug = Plug
    monkeypatch.setitem(sys.modules, "fake_talker_mod", mod)
    monkeypatch.setenv("MVG_TALKER_BACKENDS", "plug=fake_talker_mod:Plug, junk")
    assert tb.load_backend("plug") is Plug


def test_unknown_and_broken_plugin(monkeypatch):
    monkeypatch.setenv("MVG_TALKER_BACKENDS", "bad=no_such_mod_xyz:Cls")
    with pytest.raises(ValueError, match="unknown talker backend 'nope'"):
        tb.load_backend("nope")
    with pytest.raises(ValueError, match="failed to load"):
        tb.load_backend("bad")


def test_base_class_methods_not_implemented():
    with pytest.raises(NotImplementedError):
        run(tb.TalkerBackend().open({}))
