"""/voice/live over the stdlib WebSocket layer, with a mock backend and a raw-socket client."""
import base64
import json
import os
import socket
import struct
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gateway  # noqa: E402
import live_ws  # noqa: E402
import talker_backend as tb  # noqa: E402


@pytest.fixture()
def server():
    srv = gateway.Server(("127.0.0.1", 0), gateway.GatewayHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv.server_address[1]
    srv.shutdown()


class EchoBackend(tb.MockBackend):
    """Echoes each audio frame back as an audio event, then ends on cancel/close."""
    def __init__(self):
        super().__init__()
        import asyncio
        self.q = asyncio.Queue()

    async def send_audio(self, pcm):
        await super().send_audio(pcm)
        await self.q.put({"type": "audio", "pcm": pcm[::-1]})

    async def events(self):
        while True:
            ev = await self.q.get()
            if ev is None:
                return
            yield ev

    async def close(self):
        await super().close()
        await self.q.put(None)


class FailBackend(tb.MockBackend):
    def __init__(self):
        super().__init__([{"type": "error", "message": "boom"}])


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.delenv("MVG_DEVICE_TOKENS", raising=False)
    monkeypatch.setenv("MVG_TALKER_BACKEND", "echo")
    monkeypatch.setenv("MVG_TALKER_BACKENDS", "echo=test_live_ws:EchoBackend,fail=test_live_ws:FailBackend")
    sys.modules.setdefault("test_live_ws", sys.modules[__name__])


@pytest.fixture(autouse=True)
def _clear_auth_failures():
    """Failed-auth counts are per client IP and every test here is 127.0.0.1."""
    verifier = gateway.get_device_token_verifier()
    with verifier._lock:
        verifier._failures.clear()
    yield
    with verifier._lock:
        verifier._failures.clear()


class Client:
    def __init__(self, port, path="/voice/live?node_id=n1", headers=None, upgrade=True):
        self.s = socket.create_connection(("127.0.0.1", port), timeout=5)
        key = base64.b64encode(os.urandom(16)).decode()
        h = {"Host": "x"}
        if upgrade:
            h.update({"Upgrade": "websocket", "Connection": "Upgrade", "Sec-WebSocket-Key": key,
                      "Sec-WebSocket-Version": "13"})
        h.update(headers or {})
        req = f"GET {path} HTTP/1.1\r\n" + "".join(f"{k}: {v}\r\n" for k, v in h.items()) + "\r\n"
        self.s.sendall(req.encode())
        self.key = key
        self.head = self._read_head()

    def _read_head(self):
        buf = b""
        while b"\r\n\r\n" not in buf:
            c = self.s.recv(1)
            if not c:
                break
            buf += c
        return buf.decode()

    @property
    def status(self):
        return int(self.head.split()[1])

    def send(self, op, payload=b""):
        mask = os.urandom(4)
        n = len(payload)
        head = bytes([0x80 | op])
        if n < 126:
            head += bytes([0x80 | n])
        elif n < 65536:
            head += bytes([0x80 | 126]) + struct.pack(">H", n)
        else:
            head += bytes([0x80 | 127]) + struct.pack(">Q", n)
        self.s.sendall(head + mask + bytes(b ^ mask[i % 4] for i, b in enumerate(payload)))

    def _read(self, n):
        buf = b""
        while len(buf) < n:
            c = self.s.recv(n - len(buf))
            if not c:
                raise ConnectionError
            buf += c
        return buf

    def recv(self):
        b1, b2 = self._read(2)
        ln = b2 & 0x7F
        if ln == 126:
            ln = struct.unpack(">H", self._read(2))[0]
        elif ln == 127:
            ln = struct.unpack(">Q", self._read(8))[0]
        return b1 & 0x0F, self._read(ln)


def test_accept_key_rfc_example():
    assert live_ws.accept_key("dGhlIHNhbXBsZSBub25jZQ==") == "s3pPLMBiTxaQ9kYGzzhZRbK+xOo="


def test_happy_path_audio_round_trip(server):
    c = Client(server)
    assert c.status == 101
    assert live_ws.accept_key(c.key) in c.head
    c.send(live_ws.OP_BIN, b"\x01\x02\x03\x04")
    op, payload = c.recv()
    assert (op, payload) == (live_ws.OP_BIN, b"\x04\x03\x02\x01")
    c.send(live_ws.OP_PING, b"hi")
    assert c.recv() == (live_ws.OP_PONG, b"hi")
    c.send(live_ws.OP_TEXT, json.dumps({"event": "end"}).encode())
    op, payload = c.recv()
    assert op == live_ws.OP_CLOSE and struct.unpack(">H", payload[:2])[0] == 1000


def test_auth_failure_is_401(server, monkeypatch):
    monkeypatch.setenv("MVG_DEVICE_TOKENS", "n1:secret")
    assert Client(server).status == 401
    ok = Client(server, headers={"Authorization": "Bearer secret"})
    assert ok.status == 101


def test_not_configured_is_501(server, monkeypatch):
    monkeypatch.delenv("MVG_TALKER_BACKEND")
    assert Client(server).status == 501


def test_unknown_backend_is_501(server, monkeypatch):
    monkeypatch.setenv("MVG_TALKER_BACKEND", "nope")
    assert Client(server).status == 501


def test_plain_get_without_upgrade_is_426(server):
    assert Client(server, upgrade=False).status == 426


def test_oversized_frame_closes_1009(server, monkeypatch):
    monkeypatch.setenv("MVG_LIVE_MAX_FRAME", "1000")
    c = Client(server)
    c.send(live_ws.OP_BIN, b"\x00" * 2000)
    op, payload = c.recv()
    assert op == live_ws.OP_CLOSE and struct.unpack(">H", payload[:2])[0] == 1009


def test_idle_timeout_closes_4000(server, monkeypatch):
    monkeypatch.setenv("MVG_LIVE_IDLE_SEC", "0.2")
    c = Client(server)
    op, payload = c.recv()
    assert op == live_ws.OP_CLOSE and struct.unpack(">H", payload[:2])[0] == 4000


def test_max_duration_closes_4001(server, monkeypatch):
    monkeypatch.setenv("MVG_LIVE_MAX_SEC", "0.2")
    c = Client(server)
    op, payload = c.recv()
    assert op == live_ws.OP_CLOSE and struct.unpack(">H", payload[:2])[0] == 4001


def test_backend_error_closes_1011_with_error_event(server, monkeypatch):
    monkeypatch.setenv("MVG_TALKER_BACKEND", "fail")
    c = Client(server)
    op, payload = c.recv()
    assert op == live_ws.OP_TEXT and json.loads(payload)["event"] == "error"
    op, payload = c.recv()
    fb = json.loads(payload)
    assert op == live_ws.OP_TEXT and fb["event"] == "fallback" and fb["endpoint"] == "/voice/turn"
    op, payload = c.recv()
    assert op == live_ws.OP_CLOSE and struct.unpack(">H", payload[:2])[0] == 1011


def test_unmasked_frame_rejected():
    class S:
        def __init__(self): self.data = bytes([0x82, 0x01, 0x00])
        def recv(self, n):
            d, self.data = self.data[:n], self.data[n:]
            return d
    with pytest.raises(ConnectionError):
        live_ws.WSConn(S(), 100).recv()


class OpenFailBackend(tb.MockBackend):
    async def open(self, config):
        raise ConnectionError("no route")


def test_open_failure_sends_fallback_then_1011(server, monkeypatch):
    monkeypatch.setenv("MVG_TALKER_BACKENDS", "openfail=test_live_ws:OpenFailBackend")
    monkeypatch.setenv("MVG_TALKER_BACKEND", "openfail")
    c = Client(server)
    op, payload = c.recv()
    fb = json.loads(payload)
    assert fb["event"] == "fallback" and "open_failed" in fb["reason"]
    op, payload = c.recv()
    assert op == live_ws.OP_CLOSE and struct.unpack(">H", payload[:2])[0] == 1011


class InspectBackend(EchoBackend):
    last_instance = None

    def __init__(self):
        super().__init__()
        self.opened_with = None
        InspectBackend.last_instance = self

    async def open(self, config):
        await super().open(config)
        self.opened_with = dict(config or {})


def test_voice_mode_validation(server, monkeypatch):
    monkeypatch.setenv("MVG_TALKER_BACKENDS", "inspect=test_live_ws:InspectBackend")
    monkeypatch.setenv("MVG_TALKER_BACKEND", "inspect")

    # 1. Invalid voice_mode -> 400
    monkeypatch.setattr(gateway, "resolve_profile", lambda nid: {"target": "agent", "voice_mode": "invalid_mode"})
    c = Client(server)
    assert c.status == 400
    assert "invalid voice_mode: invalid_mode" in c.s.recv(4096).decode()
    c.s.close()

    # 2. Valid voice_mode: native -> 101 and config carries voice_mode="native"
    monkeypatch.setattr(gateway, "resolve_profile", lambda nid: {"target": "agent", "voice_mode": "native"})
    c_nat = Client(server)
    assert c_nat.status == 101
    c_nat.send(live_ws.OP_TEXT, json.dumps({"event": "end"}).encode())
    c_nat.recv()
    assert InspectBackend.last_instance.opened_with["voice_mode"] == "native"

    # 3. Valid voice_mode: custom_tts -> 101 and config carries voice_mode="custom_tts"
    monkeypatch.setattr(gateway, "resolve_profile", lambda nid: {"target": "agent", "voice_mode": "custom_tts"})
    c_tts = Client(server)
    assert c_tts.status == 101
    c_tts.send(live_ws.OP_TEXT, json.dumps({"event": "end"}).encode())
    c_tts.recv()
    assert InspectBackend.last_instance.opened_with["voice_mode"] == "custom_tts"


def test_talker_instructions_hook_and_persona_fallback(server, monkeypatch, tmp_path):
    monkeypatch.setenv("MVG_TALKER_BACKENDS", "inspect=test_live_ws:InspectBackend")
    monkeypatch.setenv("MVG_TALKER_BACKEND", "inspect")
    monkeypatch.setattr(gateway, "resolve_profile", lambda nid: {"target": "myagent"})

    persona_dir = tmp_path / "personas"
    agent_dir = persona_dir / "myagent"
    agent_dir.mkdir(parents=True)
    monkeypatch.setenv("MVG_PERSONA_DIR", str(persona_dir))

    # 1. voice.md fallback when voice_talker.md does not exist
    (agent_dir / "voice.md").write_text("  General voice persona  \n", encoding="utf-8")
    c = Client(server)
    assert c.status == 101
    c.send(live_ws.OP_TEXT, json.dumps({"event": "end"}).encode())
    c.recv()
    assert InspectBackend.last_instance.opened_with.get("instructions") == "General voice persona"

    # 2. voice_talker.md takes precedence over voice.md
    (agent_dir / "voice_talker.md").write_text("  Talker specific persona  \n", encoding="utf-8")
    c2 = Client(server)
    assert c2.status == 101
    c2.send(live_ws.OP_TEXT, json.dumps({"event": "end"}).encode())
    c2.recv()
    assert InspectBackend.last_instance.opened_with.get("instructions") == "Talker specific persona"

    # 3. _PLUGIN_MOD.talker_instructions hook takes precedence over files
    class FakePlugin:
        @staticmethod
        def talker_instructions(target, node_id):
            return f"Hook instructions for {target}@{node_id}"

    monkeypatch.setattr(gateway, "_PLUGIN_MOD", FakePlugin)
    c3 = Client(server)
    assert c3.status == 101
    c3.send(live_ws.OP_TEXT, json.dumps({"event": "end"}).encode())
    c3.recv()
    assert InspectBackend.last_instance.opened_with.get("instructions") == "Hook instructions for myagent@n1"

    # 4. Raising hook falls back to persona file
    class RaisingPlugin:
        @staticmethod
        def talker_instructions(target, node_id):
            raise RuntimeError("hook failed")

    monkeypatch.setattr(gateway, "_PLUGIN_MOD", RaisingPlugin)
    c4 = Client(server)
    assert c4.status == 101
    c4.send(live_ws.OP_TEXT, json.dumps({"event": "end"}).encode())
    c4.recv()
    assert InspectBackend.last_instance.opened_with.get("instructions") == "Talker specific persona"


class CancelFlowBackend(EchoBackend):
    brain_started = threading.Event()
    release_brain = threading.Event()
    tool_results = []

    def __init__(self):
        super().__init__()
        CancelFlowBackend.brain_started.clear()
        CancelFlowBackend.release_brain.clear()
        CancelFlowBackend.tool_results = []

    async def open(self, config):
        await super().open(config)
        await self.q.put({"type": "tool_call", "call_id": "c1", "name": "call_thinker", "args": {"request": "slow job"}})

    async def send_tool_result(self, call_id, result):
        await super().send_tool_result(call_id, result)
        CancelFlowBackend.tool_results.append((call_id, result))
        if call_id == "c1":
            import asyncio
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, CancelFlowBackend.brain_started.wait, 5.0)
            await self.q.put({"type": "tool_call", "call_id": "c2", "name": "cancel_thinker", "args": {}})
        elif call_id == "c2":
            CancelFlowBackend.release_brain.set()
            await self.q.put({"type": "text", "text": "done_cancel", "spoken": True})


def test_live_cancel_thinker_reaches_brain_cancel(server, monkeypatch):
    monkeypatch.setenv("MVG_TALKER_BACKENDS", "cancelflow=test_live_ws:CancelFlowBackend")
    monkeypatch.setenv("MVG_TALKER_BACKEND", "cancelflow")

    cancelled_calls = []
    seen_task_ids = []

    def fake_cancel(target, task_id):
        cancelled_calls.append((target, task_id))
        return True

    def slow_brain(prompt, session_id, node_id, speaker, score, timeout, on_partial=None, task_id=None, **kw):
        seen_task_ids.append(task_id)
        CancelFlowBackend.brain_started.set()
        CancelFlowBackend.release_brain.wait(5.0)
        return "finished"

    monkeypatch.setattr(gateway, "_plugin_brain_cancel", fake_cancel)
    monkeypatch.setattr(gateway, "STREAM_BRAIN_CALL", slow_brain)

    c = Client(server)
    assert c.status == 101
    op, payload = c.recv()
    assert op == live_ws.OP_TEXT and json.loads(payload)["text"] == "done_cancel"
    c.send(live_ws.OP_TEXT, json.dumps({"event": "end"}).encode())
    c.recv()

    assert len(cancelled_calls) == 1
    assert cancelled_calls[0][1] == seen_task_ids[0]
    assert dict(CancelFlowBackend.tool_results)["c2"]["status"] == "cancelled"
    assert gateway._ACTIVE_TASKS == {}


class StageRelayBackend(EchoBackend):
    last_instance = None

    def __init__(self):
        super().__init__()
        StageRelayBackend.last_instance = self

    async def open(self, config):
        await super().open(config)
        await self.q.put({"type": "tool_call", "call_id": "s1", "name": "call_thinker", "args": {"request": "run stages"}})

    async def send_text(self, text, *, silent=False):
        await super().send_text(text, silent=silent)
        if text.startswith("[thinker result"):
            await self.q.put({"type": "text", "text": "all_done", "spoken": True})


def test_live_stage_events_produce_two_relay_lines(server, monkeypatch):
    monkeypatch.setenv("MVG_TALKER_BACKENDS", "stagerelay=test_live_ws:StageRelayBackend")
    monkeypatch.setenv("MVG_TALKER_BACKEND", "stagerelay")
    monkeypatch.setenv("MVG_PROGRESS_MIN_GAP_SEC", "0")

    import time

    def stage_brain(prompt, session_id, node_id, speaker, score, timeout, on_partial=None, stages_out=None, **kw):
        sink = stages_out if stages_out is not None else getattr(gateway._STAGE_SINK, "stages", None)
        sink.setdefault("events", []).append({"stage": "tool_start", "detail": "step one", "t": 1.0})
        time.sleep(0.05)
        sink.setdefault("events", []).append({"stage": "milestone", "detail": "step two", "t": 2.0})
        time.sleep(0.05)
        return "final answer"

    monkeypatch.setattr(gateway, "STREAM_BRAIN_CALL", stage_brain)

    c = Client(server)
    assert c.status == 101
    op, payload = c.recv()
    assert op == live_ws.OP_TEXT and json.loads(payload)["text"] == "all_done"
    c.send(live_ws.OP_TEXT, json.dumps({"event": "end"}).encode())
    c.recv()

    sent_texts = [(x[1], x[2]) for x in StageRelayBackend.last_instance.sent if x[0] == "text"]
    progress_lines = [(t, silent) for t, silent in sent_texts if t.startswith("[task progress]")]
    assert progress_lines == [
        ("[task progress] step one", True),
        ("[task progress] step two", True),
    ]


def test_stage_event_during_session_close_leaves_no_unawaited_coroutine(server, monkeypatch):
    import gc
    import time
    import warnings

    monkeypatch.setenv("MVG_TALKER_BACKENDS", "stagerelay=test_live_ws:StageRelayBackend")
    monkeypatch.setenv("MVG_TALKER_BACKEND", "stagerelay")

    brain_started = threading.Event()
    client_closed_event = threading.Event()
    captured_sink = []

    def late_brain(prompt, session_id, node_id, speaker, score, timeout, on_partial=None, stages_out=None, **kw):
        sink = stages_out if stages_out is not None else getattr(gateway._STAGE_SINK, "stages", None)
        captured_sink.append(sink)
        brain_started.set()
        client_closed_event.wait(5.0)
        sink["events"].append({"stage": "tool_start", "detail": "late stage event", "t": 0.2})
        return "final answer"

    monkeypatch.setattr(gateway, "STREAM_BRAIN_CALL", late_brain)

    with warnings.catch_warnings(record=True) as recorded_warnings:
        warnings.simplefilter("always")

        c = Client(server)
        assert c.status == 101
        assert brain_started.wait(5.0)

        c.send(live_ws.OP_TEXT, json.dumps({"event": "end"}).encode())
        client_closed_event.set()
        try:
            c.recv()
        except Exception:
            pass
        c.s.close()

        time.sleep(0.1)

        sink = captured_sink[0]
        closure = dict(zip(sink._on_event.__code__.co_freevars, [cell.cell_contents for cell in sink._on_event.__closure__]))
        session_loop = closure["session_loop"]
        assert session_loop[0] is not None
        assert session_loop[0].is_closed()

        sink["events"].append({"stage": "tool_start", "detail": "closed loop event", "t": 0.3})

        class DummyLoop:
            def is_closed(self):
                return False

            def call_soon_threadsafe(self, callback, *args):
                raise RuntimeError("Event loop is closed")

        session_loop[0] = DummyLoop()
        gateway._STAGE_SINK.stages = sink
        gateway._STAGE_SINK.stages.append({"stage": "tool_start", "detail": "race stage event", "t": 0.4})

        gc.collect()

        unawaited = [
            w for w in recorded_warnings
            if issubclass(w.category, RuntimeWarning) and "was never awaited" in str(w.message)
        ]
        assert len(unawaited) == 0


def _watch_tokens(monkeypatch, tmp_path, value):
    f = tmp_path / "watch-tokens.json"
    f.write_text(json.dumps({"tok": value}))
    monkeypatch.setattr(gateway, "WATCH_TOKENS_PATH", f)


def test_live_watch_route_bad_token_is_401(server, monkeypatch, tmp_path):
    _watch_tokens(monkeypatch, tmp_path, {"nodes": ["watch"], "speaker": "alice"})
    monkeypatch.setenv("MVG_DEVICE_TOKENS", "n1:secret")
    for headers in ({"X-Watch-Route": "1"},
                    {"X-Watch-Route": "1", "Authorization": "Bearer wrong"},
                    # the device token is not a watch token
                    {"X-Watch-Route": "1", "Authorization": "Bearer secret"}):
        c = Client(server, headers=headers)
        assert c.status == 401
        assert '"unauthorized"' in c.s.recv(4096).decode()
        c.s.close()


def test_live_watch_route_failures_count_toward_lockout(server, monkeypatch, tmp_path):
    _watch_tokens(monkeypatch, tmp_path, "watch")
    monkeypatch.setenv("MVG_DEVICE_TOKENS", "n1:secret")
    monkeypatch.setenv("MVG_AUTH_MAX_FAILURES", "2")
    statuses = []
    for _ in range(3):
        c = Client(server, headers={"X-Watch-Route": "1", "Authorization": "Bearer wrong"})
        statuses.append(c.status)
        c.s.close()
    assert statuses == [401, 401, 429]


def test_live_without_watch_route_ignores_watch_tokens(server, monkeypatch, tmp_path):
    _watch_tokens(monkeypatch, tmp_path, "watch")
    monkeypatch.setenv("MVG_DEVICE_TOKENS", "n1:secret")
    assert Client(server, headers={"Authorization": "Bearer tok"}).status == 401
    assert Client(server, headers={"Authorization": "Bearer secret"}).status == 101


def test_live_watch_route_good_token_sets_node_and_speaker(server, monkeypatch, tmp_path):
    monkeypatch.setenv("MVG_TALKER_BACKENDS", "inspect=test_live_ws:InspectBackend")
    monkeypatch.setenv("MVG_TALKER_BACKEND", "inspect")
    monkeypatch.setenv("MVG_TRUSTED_SPEAKERS", "alice")
    _watch_tokens(monkeypatch, tmp_path, {"nodes": ["watch-node"], "speaker": "alice"})

    c = Client(server, path="/voice/live?node_id=other",
               headers={"X-Watch-Route": "1", "Authorization": "Bearer tok"})
    assert c.status == 101
    c.send(live_ws.OP_TEXT, json.dumps({"event": "end"}).encode())
    c.recv()
    assert InspectBackend.last_instance.opened_with["node_id"] == "watch-node"


class ThinkOnceBackend(EchoBackend):
    async def open(self, config):
        await super().open(config)
        await self.q.put({"type": "tool_call", "call_id": "t1", "name": "call_thinker",
                          "args": {"request": "do it"}})

    async def send_tool_result(self, call_id, result):
        await super().send_tool_result(call_id, result)

    async def send_text(self, text, *, silent=False):
        await super().send_text(text, silent=silent)
        if text.startswith("[thinker result"):
            await self.q.put({"type": "text", "text": "thought", "spoken": True})


def test_live_thinker_frame_carries_device_speaker_and_trust(server, monkeypatch, tmp_path):
    monkeypatch.setenv("MVG_TALKER_BACKENDS", "thinkonce=test_live_ws:ThinkOnceBackend")
    monkeypatch.setenv("MVG_TALKER_BACKEND", "thinkonce")
    monkeypatch.setenv("MVG_TRUSTED_SPEAKERS", "alice")
    _watch_tokens(monkeypatch, tmp_path, {"nodes": ["watch-node"], "speaker": "alice"})
    seen = []

    def real_brain(prompt, session_id, node_id, speaker, score, timeout, on_partial=None, **kw):
        seen.append((node_id, gateway._frame_for_call(prompt, node_id, speaker, score)))
        return "done"

    monkeypatch.setattr(gateway, "STREAM_BRAIN_CALL", real_brain)

    c = Client(server, path="/voice/live?node_id=other",
               headers={"X-Watch-Route": "1", "Authorization": "Bearer tok"})
    assert c.status == 101
    op, payload = c.recv()
    assert json.loads(payload)["text"] == "thought"
    c.send(live_ws.OP_TEXT, json.dumps({"event": "end"}).encode())
    c.recv()
    assert seen and seen[0][0] == "watch-node"
    assert "speaker: Alice (paired device)" in seen[0][1]
    assert "trusted: paired device" in seen[0][1]
    assert gateway.SECURITY_SUFFIX not in seen[0][1]


def test_client_error_event_closes_with_reason(server):
    c = Client(server)
    c.send(live_ws.OP_TEXT, json.dumps({"event": "client_error", "reason": "audio: engine\nstopped"}).encode())
    while True:
        op, payload = c.recv()
        if op == live_ws.OP_CLOSE:
            break
    assert struct.unpack(">H", payload[:2])[0] == 1000
    assert payload[2:].decode() == "client_error:audio: engine stopped"
