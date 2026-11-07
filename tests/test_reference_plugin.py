"""Tests for examples/reference_plugin.py."""

from __future__ import annotations

import http.server
import json
import sys
import threading
import time
from pathlib import Path

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

import reference_plugin  # noqa: E402


def _start_server(sleep_sec: float):
    received: list[dict] = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            received.append(json.loads(raw.decode("utf-8")))
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.flush()
            time.sleep(sleep_sec)
            try:
                self.wfile.write(b"hello")
                self.wfile.flush()
            except OSError:
                pass

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, received


def test_brain_call_stream_and_cancel_after(monkeypatch):
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    server, received = _start_server(0.2)
    try:
        monkeypatch.setenv("REFERENCE_BRAIN_URL", f"http://127.0.0.1:{server.server_port}/brain")
        got: list[str] = []
        res = reference_plugin.brain_call_stream("t", "p", 5, got.append, {}, task_id="a")
        assert res == "hello"
        assert got == ["hello"]
        assert received == [{"target": "t", "prompt": "p"}]
        assert reference_plugin.brain_cancel("t", "a") is False
    finally:
        server.shutdown()
        server.server_close()


def test_brain_cancel_inflight(monkeypatch):
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    server, _ = _start_server(1.0)
    try:
        monkeypatch.setenv("REFERENCE_BRAIN_URL", f"http://127.0.0.1:{server.server_port}/brain")
        got: list[str] = []

        def _worker():
            try:
                reference_plugin.brain_call_stream("t", "p", 5, got.append, {}, task_id="b")
            except Exception:
                pass

        th = threading.Thread(target=_worker, daemon=True)
        th.start()
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            with reference_plugin._LOCK:
                if "b" in reference_plugin._INFLIGHT:
                    break
            time.sleep(0.01)
        assert reference_plugin.brain_cancel("t", "b") is True
        th.join(timeout=2.0)
        assert "b" not in reference_plugin._INFLIGHT
    finally:
        server.shutdown()
        server.server_close()


def test_talker_instructions_and_try_intent(monkeypatch):
    monkeypatch.delenv("REFERENCE_TALKER_INSTRUCTIONS", raising=False)
    assert reference_plugin.talker_instructions("t", "node-1") == ""
    monkeypatch.setenv("REFERENCE_TALKER_INSTRUCTIONS", "Be brief.")
    assert reference_plugin.talker_instructions("t", "node-1") == "Be brief."
    assert reference_plugin.try_intent("turn on the lights") is None
