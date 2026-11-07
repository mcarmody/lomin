"""Tests for examples/reference_plugin.py (#71)."""
from __future__ import annotations

import sys
import threading
import time
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

