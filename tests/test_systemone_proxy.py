#!/usr/bin/env python3
"""Tests for the POST /v1/systemone proxy route (gateway.py -> desktop Laya).

No network: the Laya classifier is a fake HTTP server on 127.0.0.1 port 0.
Run: python3 -m pytest -q tests/test_systemone_proxy.py
"""
import http.client
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gateway  # noqa: E402


class _FakeLayaHandler(BaseHTTPRequestHandler):
    received = []  # (content_type, body) per request
    response_body = b'{"option": "addressed_to_agent", "confidence": 0.91}'
    response_status = 200

    def log_message(self, *_):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        _FakeLayaHandler.received.append((self.headers.get("Content-Type"), body))
        self.send_response(_FakeLayaHandler.response_status)
        self.send_header("Content-Length", str(len(_FakeLayaHandler.response_body)))
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(_FakeLayaHandler.response_body)


@pytest.fixture()
def gateway_server():
    server = gateway.Server(("127.0.0.1", 0), gateway.GatewayHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield port
    server.shutdown()


@pytest.fixture()
def fake_laya(monkeypatch):
    _FakeLayaHandler.received = []
    _FakeLayaHandler.response_body = b'{"option": "addressed_to_agent", "confidence": 0.91}'
    _FakeLayaHandler.response_status = 200
    server = HTTPServer(("127.0.0.1", 0), _FakeLayaHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setattr(gateway, "DESKTOP_LAYA_URL", f"http://127.0.0.1:{port}/v1/systemone")
    yield server
    server.shutdown()


def _post(port, body: bytes):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    conn.request("POST", "/v1/systemone", body=body,
                 headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    data = resp.read()
    conn.close()
    return resp, data


def test_systemone_proxies_200_exact_body_and_content_type(gateway_server, fake_laya):
    req = json.dumps({"text": "hey laya, what's up", "node": "kitchen"}).encode("utf-8")
    resp, data = _post(gateway_server, req)
    assert resp.status == 200
    assert resp.getheader("Content-Type") == "application/json"
    assert data == _FakeLayaHandler.response_body
    assert len(_FakeLayaHandler.received) == 1
    ctype, got = _FakeLayaHandler.received[-1]
    assert got == req
    assert ctype == "application/json"


def test_systemone_passes_upstream_400_through(gateway_server, fake_laya):
    _FakeLayaHandler.response_status = 400
    _FakeLayaHandler.response_body = b'{"error": "missing field: text"}'
    resp, data = _post(gateway_server, b'{"node": "kitchen"}')
    assert resp.status == 400
    assert data == _FakeLayaHandler.response_body


def test_systemone_unreachable_upstream_is_502(gateway_server, monkeypatch):
    monkeypatch.setattr(gateway, "DESKTOP_LAYA_URL", "http://127.0.0.1:1/v1/systemone")
    resp, data = _post(gateway_server, b'{"text": "hi"}')
    assert resp.status == 502
    assert b"laya upstream unreachable" in data


def test_systemone_empty_body_is_400(gateway_server, fake_laya):
    resp, data = _post(gateway_server, b"")
    assert resp.status == 400
    assert json.loads(data) == {"error": "no body"}
    # The upstream must not be called for an empty body.
    assert _FakeLayaHandler.received == []
