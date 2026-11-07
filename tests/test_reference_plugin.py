"""Tests for examples/reference_plugin.py (#71)."""
from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from examples import reference_plugin as rp  # noqa: E402
import gateway  # noqa: E402


def test_apply_defaults_uses_setdefault(monkeypatch):
    monkeypatch.delenv("MVG_AGENT_NAME", raising=False)
    monkeypatch.setenv("MVG_DEFAULT_TARGET", "custom_target")
    rp.apply_defaults()
    assert rp.os.environ["MVG_AGENT_NAME"] == "the assistant"
    assert rp.os.environ["MVG_DEFAULT_TARGET"] == "custom_target"


def test_brain_call_and_try_intent():
    reply = rp.brain_call("helper", "[VOICE via node kitchen] What is the weather?\n\nThis reply will be spoken aloud: plain sentences.")
    assert reply == "[helper] Heard: What is the weather?"
    hit = rp.try_intent("Ping!")
    assert hit is not None and hit["hit"] is True and hit["intent"] == "ping"
    assert rp.try_intent("tell me a story") is None


def test_brain_call_stream_partials_stages_and_audio_clips(monkeypatch):
    monkeypatch.setattr(rp, "STEP_DELAY_SEC", 0.001)
    partials: list[str] = []
    stages: dict = {}
    clips = [b"RIFF\x00\x00\x00\x00WAVEfmt ", b"pcm_clip_2"]

    reply = rp.brain_call_stream(
        "agent",
        "[VOICE via node desk] Check system status",
        timeout=10,
        on_partial=partials.append,
        stages_out=stages,
        task_id="task-stream-1",
        audio_clips=clips,
    )

    assert len(partials) == 2
    assert partials[0].startswith("First sentence from agent (with 2 audio clip).")
    assert partials[-1] == reply
    assert "You asked: Check system status." in reply
    assert "call_start_ms" in stages and "final_seen_ms" in stages
    assert len(stages["events"]) >= 3
    for raw_ev in stages["events"]:
        norm = gateway.normalize_stage_event(raw_ev)
        assert norm is not None
        assert norm["stage"] in ("tool_start", "milestone")
        assert norm["detail"]
    assert rp.active_task_ids() == []


def test_brain_cancel_aborts_in_flight_stream(monkeypatch):
    monkeypatch.setattr(rp, "STEP_DELAY_SEC", 2.0)
    stages: dict = {}
    partials: list[str] = []
    result_box: dict = {}

    def worker():
        result_box["reply"] = rp.brain_call_stream(
            "agent",
            "Run a long job",
            timeout=10,
            on_partial=partials.append,
            stages_out=stages,
            task_id="cancel-me-123",
        )

    t = threading.Thread(target=worker, daemon=True)
    t.start()

    deadline = time.monotonic() + 2.0
    while "cancel-me-123" not in rp.active_task_ids() and time.monotonic() < deadline:
        time.sleep(0.005)

    assert "cancel-me-123" in rp.active_task_ids()
    # Mismatched target does not cancel
    assert rp.brain_cancel("other_target", "cancel-me-123") is False
    # Matching target cancels immediately
    assert rp.brain_cancel("agent", "cancel-me-123") is True

    t.join(timeout=1.0)
    assert not t.is_alive()
    assert stages.get("cancelled") is True
    assert rp.active_task_ids() == []
    # Unknown task returns False
    assert rp.brain_cancel("agent", "nonexistent") is False


def test_talker_instructions_and_gateway_integration(monkeypatch):
    instr = rp.talker_instructions("agent", "kitchen")
    assert instr is not None
    assert "call_thinker" in instr
    assert "node: kitchen" in instr

    custom = rp.talker_instructions("researcher", "")
    assert custom is not None
    assert "researcher" in custom

    monkeypatch.setattr(gateway, "_PLUGIN_MOD", rp)
    monkeypatch.setattr(gateway, "_plugin_talker_instructions", rp.talker_instructions)
    resolved = gateway._resolve_talker_instructions("agent", "living-room")
    assert resolved == rp.talker_instructions("agent", "living-room")


def test_gemini_live_smoke_tool_and_silent_progress_check(monkeypatch):
    import asyncio
    import base64
    import json
    import struct
    from tools import gemini_live_smoke as smoke

    # Missing key/URL returns exit code 2
    monkeypatch.delenv("MVG_GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("MVG_GEMINI_URL", raising=False)
    assert smoke.main([]) == 2

    async def _ws_read(reader):
        b1, b2 = await reader.readexactly(2)
        n = b2 & 0x7F
        if n == 126:
            n = struct.unpack(">H", await reader.readexactly(2))[0]
        mask = await reader.readexactly(4)
        data = await reader.readexactly(n)
        return b1 & 0xF, bytes(b ^ mask[i % 4] for i, b in enumerate(data))

    def _frame(obj):
        p = json.dumps(obj).encode()
        hdr = bytes([0x81]) + (bytes([len(p)]) if len(p) < 126 else bytes([126]) + struct.pack(">H", len(p)))
        return hdr + p

    received = []

    async def handle(reader, writer):
        await reader.readuntil(b"\r\n\r\n")
        writer.write(b"HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n\r\n")
        # 1. Setup frame
        _, data = await _ws_read(reader)
        received.append(json.loads(data))
        writer.write(_frame({"setupComplete": {}}))
        await writer.drain()
        # 2. Initial prompt frame
        _, data = await _ws_read(reader)
        received.append(json.loads(data))
        # Emit first audio chunk so --test-silent-progress injects silent send_text
        writer.write(_frame({
            "serverContent": {"modelTurn": {"parts": [{
                "inlineData": {"mimeType": "audio/pcm;rate=24000", "data": base64.b64encode(b"\x01\x02\x03\x04").decode()}
            }]}}
        }))
        await writer.drain()
        # 3. Silent progress frame
        _, data = await _ws_read(reader)
        received.append(json.loads(data))
        # Continue speaking without interruption and complete turn
        writer.write(_frame({
            "serverContent": {"modelTurn": {"parts": [{
                "inlineData": {"mimeType": "audio/pcm;rate=24000", "data": base64.b64encode(b"\x05\x06").decode()}
            }]}}
        }))
        writer.write(_frame({"serverContent": {"turnComplete": True}}))
        await writer.drain()
        writer.close()

    async def _run():
        server = await asyncio.start_server(handle, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        monkeypatch.setenv("MVG_GEMINI_URL", f"ws://127.0.0.1:{port}/live")
        try:
            return await smoke.run_smoke("Count to two", test_silent_progress=True, timeout=5.0)
        finally:
            server.close()

    summary = asyncio.run(_run())
    assert summary["turn_completed"] is True
    assert summary["interrupted"] is False
    assert summary["silent_injected"] is True
    assert summary["audio_chunks_after_silent"] == 1
    assert summary["silent_progress_verdict"] == "CONTINUES_CLEANLY"
    assert received[1]["clientContent"]["turnComplete"] is True
    assert received[2]["clientContent"]["turns"][0]["parts"][0]["text"] == "[task progress] step 1"
    assert received[2]["clientContent"]["turnComplete"] is False


def test_http_brain_stream_and_cancel_against_local_server(monkeypatch):
    class StubHandler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def do_POST(self):
            content_len = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_len)
            data = json.loads(body.decode("utf-8"))
            self.server.last_payload = data

            if not data.get("stream"):
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"reply": f"echo: {data.get('prompt')}"}).encode("utf-8"))
                return

            task_id = data.get("task_id")
            if task_id == "stall-task":
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                self.wfile.write(b"data: {\"stage\": \"stalled\", \"detail\": \"waiting\"}\n\n")
                self.wfile.flush()
                self.server.stall_entered.set()
                try:
                    while not self.server.stop_stall.is_set():
                        time.sleep(0.01)
                        self.wfile.write(b": ping\n\n")
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass
                return

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            events = [
                'data: {"stage": "remote_tool", "detail": "executing query"}\n\n',
                'data: {"delta": "Hello "}\n\n',
                'data: {"delta": "world!"}\n\n',
                'data: [DONE]\n\n',
            ]
            for ev in events:
                self.wfile.write(ev.encode("utf-8"))
                self.wfile.flush()

    server = ThreadingHTTPServer(("127.0.0.1", 0), StubHandler)
    server.last_payload = None
    server.stop_stall = threading.Event()
    server.stall_entered = threading.Event()
    port = server.server_address[1]
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    try:
        monkeypatch.setenv("REF_PLUGIN_BRAIN_URL", f"http://127.0.0.1:{port}/brain")
        monkeypatch.setenv("REF_PLUGIN_BRAIN_TOKEN", "test-token-123")

        # 1. Test rp.brain_call (JSON reply)
        reply = rp.brain_call("agent", "What is the date?", timeout=5)
        assert reply == "echo: What is the date?"
        assert server.last_payload["target"] == "agent"
        assert server.last_payload["stream"] is False

        # 2. Test rp.brain_call_stream (SSE stage + delta + [DONE])
        partials: list[str] = []
        stages: dict = {}
        clips = [b"wav_chunk_1", b"wav_chunk_2"]
        stream_reply = rp.brain_call_stream(
            "agent",
            "Stream me a greeting",
            timeout=5,
            on_partial=partials.append,
            stages_out=stages,
            task_id="stream-task-1",
            audio_clips=clips,
        )
        assert stream_reply == "Hello world!"
        assert partials == ["Hello ", "Hello world!"]
        assert server.last_payload["stream"] is True
        assert server.last_payload["audio_clips_b64"] == [
            base64.b64encode(b"wav_chunk_1").decode("ascii"),
            base64.b64encode(b"wav_chunk_2").decode("ascii"),
        ]
        stage_names = [e["stage"] for e in stages.get("events", [])]
        assert "tool_start" in stage_names
        assert "milestone" in stage_names
        assert "remote_tool" in stage_names
        assert "call_start_ms" in stages
        assert "final_seen_ms" in stages
        assert rp.active_task_ids() == []

        # 3. Test rp.brain_cancel aborting a stalled SSE stream immediately
        cancel_partials: list[str] = []
        cancel_stages: dict = {}
        cancel_result: dict = {}

        def worker():
            cancel_result["reply"] = rp.brain_call_stream(
                "agent",
                "Stall me",
                timeout=10,
                on_partial=cancel_partials.append,
                stages_out=cancel_stages,
                task_id="stall-task",
            )

        t = threading.Thread(target=worker, daemon=True)
        t.start()

        server.stall_entered.wait(timeout=2.0)
        assert "stall-task" in rp.active_task_ids()

        t0_cancel = time.monotonic()
        assert rp.brain_cancel("agent", "stall-task") is True

        t.join(timeout=2.0)
        cancel_duration = time.monotonic() - t0_cancel
        assert not t.is_alive()
        assert cancel_duration < 1.0, f"Cancellation took too long: {cancel_duration}s"
        assert cancel_stages.get("cancelled") is True
        assert rp.active_task_ids() == []
    finally:
        server.stop_stall.set()
        server.shutdown()
        server.server_close()


def test_talker_instructions_env_override(monkeypatch):
    monkeypatch.delenv("REF_PLUGIN_TALKER_INSTRUCTIONS", raising=False)
    monkeypatch.delenv("REF_PLUGIN_TALKER_INSTRUCTIONS_RESEARCHER", raising=False)

    default_agent = rp.talker_instructions("agent", "desk")
    assert "call_thinker" in default_agent
    assert "(node: desk)" in default_agent

    monkeypatch.setenv("REF_PLUGIN_TALKER_INSTRUCTIONS", "Global talker instructions")
    assert rp.talker_instructions("agent", "desk") == "Global talker instructions (node: desk)"
    assert rp.talker_instructions("researcher", "") == "Global talker instructions"

    monkeypatch.setenv("REF_PLUGIN_TALKER_INSTRUCTIONS_RESEARCHER", "Researcher specific instructions")
    assert rp.talker_instructions("researcher", "lab") == "Researcher specific instructions (node: lab)"
    assert rp.talker_instructions("agent", "desk") == "Global talker instructions (node: desk)"

    monkeypatch.setenv("REF_PLUGIN_TALKER_INSTRUCTIONS_RESEARCHER", "Researcher at (node: lab)")
    assert rp.talker_instructions("researcher", "lab") == "Researcher at (node: lab)"


def test_gateway_loads_reference_plugin_end_to_end(tmp_path):
    class BrainHandler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def do_POST(self):
            content_len = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_len)
            data = json.loads(body.decode("utf-8"))
            self.server.received.append(data)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"reply": "Brain reply from stub"}).encode("utf-8"))

    server = ThreadingHTTPServer(("127.0.0.1", 0), BrainHandler)
    server.received = []
    port = server.server_address[1]
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    gateway_dir = Path(__file__).resolve().parents[1]
    code = (
        "import gateway\n"
        "assert gateway._PLUGIN_MOD is not None, 'Plugin module should be loaded'\n"
        "reply = gateway.plugin_brain_call('Hello', 's1', 'desk', '', None, 5)\n"
        "assert reply == 'Brain reply from stub', f'Unexpected brain reply: {reply}'\n"
        "instr = gateway._resolve_talker_instructions('agent', 'desk')\n"
        "assert instr == 'Custom env talker (node: desk)', f'Unexpected instructions: {instr}'\n"
        "print('END_TO_END_OK')\n"
    )

    env = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": str(gateway_dir),
        "MVG_LOG_DIR": str(tmp_path / "logs"),
        "MVG_HOME": str(tmp_path / "home"),
        "MVG_NODES_FILE": str(tmp_path / "absent-nodes.json"),
        "MVG_PLUGIN": "examples.reference_plugin",
        "REF_PLUGIN_BRAIN_URL": f"http://127.0.0.1:{port}/brain",
        "REF_PLUGIN_TALKER_INSTRUCTIONS": "Custom env talker",
    }

    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            cwd=gateway_dir,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert proc.returncode == 0, f"Subprocess failed:\nstdout: {proc.stdout}\nstderr: {proc.stderr}"
        assert "END_TO_END_OK" in proc.stdout
        assert len(server.received) >= 1
    finally:
        server.shutdown()
        server.server_close()


def test_examples_readme_has_no_site_specific_queue_internals():
    repo_root = Path(__file__).resolve().parents[1]
    examples_readme = (repo_root / "examples" / "README.md").read_text(encoding="utf-8")
    banned_keywords = (
        "claimed_pi_ms",
        "partial_stored_pi_ms",
        "brain_call_stages",
        "#brain",
        "MVG_HIVE_POLL_SEC",
        "CLAIM_LONGPOLL",
    )
    for kw in banned_keywords:
        assert kw not in examples_readme, f"Found banned keyword {kw!r} in examples/README.md"

    readme = (repo_root / "README.md").read_text(encoding="utf-8")
    plugin_section = readme.split("## Plugin", 1)[1].split("##", 1)[0]
    plugin_table_lines = [
        line for line in plugin_section.splitlines()
        if line.strip().startswith("|")
    ]
    stream_hook_rows = [
        line for line in plugin_table_lines
        if line.strip().startswith("| `brain_call_stream")
    ]
    assert len(stream_hook_rows) == 1, f"Expected exactly 1 brain_call_stream hook row, got {len(stream_hook_rows)}"


