#!/usr/bin/env python3
"""tests/test_auth_trusted_peers.py: MVG_AUTH_TRUSTED_NETS, MVG_AUTH_MONITOR and
the watch-route skip for the MVG_DEVICE_TOKENS check."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

import pytest

import gateway


@pytest.fixture()
def server():
    srv = gateway.Server(("127.0.0.1", 0), gateway.GatewayHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv.server_address[1]
    srv.shutdown()


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    gateway._ambient_reset_state()
    monkeypatch.setattr(gateway, "NODES_PATH", tmp_path / "nodes.json")
    monkeypatch.setattr(gateway, "NODE_TARGETS_PATH", tmp_path / "node-targets.json")
    monkeypatch.setattr(gateway, "WATCH_TOKENS_PATH", tmp_path / "watch-tokens.json")
    for name in ("MVG_AUTH_TRUSTED_NETS", "MVG_AUTH_MONITOR", "MVG_AUTH_LOG",
                 "MVG_HA_INTENT_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MVG_DEVICE_TOKENS", "echo-deck:tok-deck-1")
    gateway.get_device_token_verifier().reset_failures()
    yield
    gateway.get_device_token_verifier().reset_failures()


def post(port, path, headers=None, body=b""):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}", data=body, method="POST",
        headers={"Content-Type": "audio/wav", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status
    except urllib.error.HTTPError as exc:
        return exc.code


def test_default_nobody_exempt(server):
    assert post(server, "/stt") == 401


def test_trusted_loopback_skips_token(server, monkeypatch):
    monkeypatch.setenv("MVG_AUTH_TRUSTED_NETS", "127.0.0.0/8")
    assert post(server, "/stt") == 400  # past auth, no audio


def test_untrusted_net_still_401(server, monkeypatch):
    monkeypatch.setenv("MVG_AUTH_TRUSTED_NETS", "172.18.0.0/16")
    assert post(server, "/stt") == 401


def test_bad_net_entry_exempts_nobody(server, monkeypatch):
    monkeypatch.setenv("MVG_AUTH_TRUSTED_NETS", "not-a-net,")
    assert post(server, "/stt") == 401


@pytest.mark.parametrize("hdr", ["X-Forwarded-For", "X-Real-IP"])
def test_proxied_request_is_never_exempt(server, monkeypatch, hdr):
    monkeypatch.setenv("MVG_AUTH_TRUSTED_NETS", "127.0.0.0/8")
    assert post(server, "/stt", headers={hdr: "203.0.113.9"}) == 401


def test_device_token_with_node_header(server):
    ok = {"X-Node-Id": "echo-deck", "Authorization": "Bearer tok-deck-1"}
    assert post(server, "/stt", headers=ok) == 400
    bad = {"X-Node-Id": "echo-deck", "Authorization": "Bearer wrong"}
    assert post(server, "/stt", headers=bad) == 401
    assert post(server, "/stt", headers={"Authorization": "Bearer tok-deck-1"}) == 401


def test_monitor_lets_through_and_logs(server, monkeypatch, tmp_path):
    log = tmp_path / "auth.jsonl"
    monkeypatch.setenv("MVG_AUTH_MONITOR", "1")
    monkeypatch.setenv("MVG_AUTH_LOG", str(log))
    assert post(server, "/stt") == 400
    recs = [json.loads(line) for line in log.read_text().splitlines()]
    assert recs[-1]["outcome"] == "would_reject"
    assert recs[-1]["peer"] == "127.0.0.1"
    assert recs[-1]["endpoint"] == "/stt"
    ok = {"X-Node-Id": "echo-deck", "Authorization": "Bearer tok-deck-1"}
    assert post(server, "/stt", headers=ok) == 400
    last = json.loads(log.read_text().splitlines()[-1])
    assert last["outcome"] == "ok_token" and last["node"] == "echo-deck"


def test_enforcing_logs_rejection(server, monkeypatch, tmp_path):
    log = tmp_path / "auth.jsonl"
    monkeypatch.setenv("MVG_AUTH_LOG", str(log))
    assert post(server, "/stt") == 401
    assert json.loads(log.read_text().splitlines()[-1])["outcome"] == "rejected"


def test_watch_route_not_blocked_by_device_tokens(server, tmp_path):
    (tmp_path / "watch-tokens.json").write_text(json.dumps({"watch-tok": "apple-watch"}))
    h = {"X-Watch-Route": "1", "Authorization": "Bearer watch-tok"}
    assert post(server, "/voice/turn", headers=h) not in (401, 429)
    h_bad = {"X-Watch-Route": "1", "Authorization": "Bearer nope"}
    assert post(server, "/voice/turn", headers=h_bad) == 401


def test_direct_voice_turn_still_needs_token(server):
    assert post(server, "/voice/turn") == 401


def test_ha_authorized_for_trusted_peer():
    assert gateway._is_ha_authorized(auth_header=None, node_id="x") is False
    assert gateway._is_ha_authorized(auth_header=None, node_id="x", peer_trusted=True) is True


def test_ha_monitor_mode_does_not_gate(monkeypatch):
    monkeypatch.setenv("MVG_AUTH_MONITOR", "1")
    assert gateway._is_ha_authorized(auth_header=None, node_id="x") is True


@pytest.mark.parametrize("ip,expect", [
    ("127.0.0.1", True), ("::ffff:127.0.0.1", True), ("172.18.0.7", True),
    ("172.17.0.2", False), ("203.0.113.9", False), ("garbage", False),
])
def test_ip_is_trusted(monkeypatch, ip, expect):
    monkeypatch.setenv("MVG_AUTH_TRUSTED_NETS", "127.0.0.0/8, 172.18.0.0/16")
    assert gateway._ip_is_trusted(ip) is expect
