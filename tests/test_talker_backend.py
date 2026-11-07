"""TalkerBackend contract, MockBackend and load_backend."""
import asyncio
import json
import sys
import threading
import time
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
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


def test_cascaded_http_sse_streaming():
    requests_received = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length).decode("utf-8")
            requests_received.append(json.loads(body))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            chunks = [
                'data: {"choices": [{"delta": {"content": "Hello "}}]}\n\n',
                'data: {"choices": [{"delta": {"content": "world!"}}]}\n\n',
                'data: [DONE]\n\n',
            ]
            for c in chunks:
                self.wfile.write(c.encode("utf-8"))
                self.wfile.flush()

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    th = threading.Thread(target=server.serve_forever, daemon=True)
    th.start()
    try:
        b = tb.CascadedTextBackend()

        async def go():
            await b.open({"talker_url": f"http://127.0.0.1:{port}/chat"})
            await b.send_text("Greetings")
            evs = []
            async for ev in b.events():
                evs.append(ev)
                if ev["type"] == "turn_end":
                    break
            await b.close()
            return evs

        evs = run(go())
        assert [e["type"] for e in evs] == ["text", "text", "turn_end"]
        assert [e["text"] for e in evs if e["type"] == "text"] == ["Hello ", "world!"]
        assert b.messages[-1] == {"role": "assistant", "content": "Hello world!"}
        assert len(requests_received) == 1
        assert requests_received[0]["messages"][-1] == {"role": "user", "content": "Greetings"}
    finally:
        server.shutdown()
        server.server_close()


def test_cascaded_http_json_tool_call_and_followup():
    requests_received = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length).decode("utf-8")
            data = json.loads(body)
            requests_received.append(data)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            if len(requests_received) == 1:
                resp = {
                    "choices": [{
                        "message": {
                            "content": "Checking temperature.",
                            "tool_calls": [{
                                "id": "tc_1",
                                "type": "function",
                                "function": {
                                    "name": "call_thinker",
                                    "arguments": json.dumps({"request": "get temp"}),
                                },
                            }],
                        }
                    }]
                }
            else:
                resp = {
                    "choices": [{
                        "message": {
                            "content": "It is currently 72 degrees.",
                        }
                    }]
                }
            self.wfile.write(json.dumps(resp).encode("utf-8"))

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    th = threading.Thread(target=server.serve_forever, daemon=True)
    th.start()
    try:
        b = tb.CascadedTextBackend()

        async def go():
            await b.open({"talker_url": f"http://127.0.0.1:{port}/chat"})
            await b.send_text("What is the temperature?")
            evs_1 = []
            async for ev in b.events():
                evs_1.append(ev)
                if ev["type"] == "turn_end":
                    break
            # Caller invokes send_tool_result and follow-up send_text
            await b.send_tool_result("tc_1", {"status": "started", "handoff_id": 1})
            await b.send_text("[thinker result, handoff #1] It is 72 degrees.")
            evs_2 = []
            async for ev in b.events():
                evs_2.append(ev)
                if ev["type"] == "turn_end":
                    break
            await b.close()
            return evs_1, evs_2

        evs_1, evs_2 = run(go())
        assert [e["type"] for e in evs_1] == ["text", "tool_call", "turn_end"]
        assert evs_1[0]["text"] == "Checking temperature."
        assert evs_1[1]["name"] == "call_thinker"
        assert evs_1[1]["args"] == {"request": "get temp"}

        assert [e["type"] for e in evs_2] == ["text", "turn_end"]
        assert evs_2[0]["text"] == "It is currently 72 degrees."

        assert len(requests_received) == 2
        req2_msgs = requests_received[1]["messages"]
        assert any(m.get("role") == "tool" and m.get("tool_call_id") == "tc_1" for m in req2_msgs)
        assert any("[thinker result, handoff #1] It is 72 degrees." in m.get("content", "") for m in req2_msgs)
    finally:
        server.shutdown()
        server.server_close()


def test_cascaded_http_5xx_retry_and_4xx_error(monkeypatch):
    req_count = 0

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            nonlocal req_count
            length = int(self.headers.get("Content-Length", 0))
            self.rfile.read(length)
            if self.path == "/retry":
                req_count += 1
                if req_count == 1:
                    self.send_response(502)
                    self.end_headers()
                    self.wfile.write(b"Bad Gateway")
                else:
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"choices": [{"message": {"content": "Recovered"}}]}).encode("utf-8"))
            elif self.path == "/auth_fail":
                self.send_response(401)
                self.end_headers()
                self.wfile.write(b"Unauthorized")
            elif self.path == "/slow":
                time.sleep(0.5)
                self.send_response(200)
                self.end_headers()

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    th = threading.Thread(target=server.serve_forever, daemon=True)
    th.start()
    try:
        # 1. 502 on request 1, 200 on request 2 (succeeds on retry)
        b = tb.CascadedTextBackend()

        async def go_retry():
            await b.open({"talker_url": f"http://127.0.0.1:{port}/retry"})
            await b.send_text("retry test")
            evs = []
            async for ev in b.events():
                evs.append(ev)
                if ev["type"] == "turn_end":
                    break
            await b.close()
            return evs

        evs = run(go_retry())
        assert req_count == 2
        assert [e["type"] for e in evs] == ["text", "turn_end"]
        assert evs[0]["text"] == "Recovered"

        # 2. 401 error
        b401 = tb.CascadedTextBackend()

        async def go_401():
            await b401.open({"talker_url": f"http://127.0.0.1:{port}/auth_fail"})
            await b401.send_text("401 test")
            evs = []
            async for ev in b401.events():
                evs.append(ev)
                if ev["type"] == "error":
                    break
            await b401.close()
            return evs

        evs_401 = run(go_401())
        assert len(evs_401) == 1 and evs_401[0]["type"] == "error"
        assert "401" in evs_401[0]["message"]

        # 3. Connection timeout
        b_slow = tb.CascadedTextBackend()
        monkeypatch.setenv("MVG_TALKER_TIMEOUT", "0.2")

        async def go_slow():
            await b_slow.open({"talker_url": f"http://127.0.0.1:{port}/slow"})
            await b_slow.send_text("timeout test")
            evs = []
            async for ev in b_slow.events():
                evs.append(ev)
                if ev["type"] == "error":
                    break
            await b_slow.close()
            return evs

        evs_slow = run(go_slow())
        assert len(evs_slow) == 1 and evs_slow[0]["type"] == "error"
    finally:
        server.shutdown()
        server.server_close()
