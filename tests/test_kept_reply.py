#!/usr/bin/env python3
"""A voice reply that finishes after the client left is kept and can be spoken on request.
Every upstream is a monkeypatched module function: no network."""
import http.client
import json
import threading
import time
import uuid

import pytest

import gateway
from pending_and_auth import PendingVoiceAnswerStore

WAV = gateway.pcm_to_wav(b"\x01\x00" * 2400, 24000)
TEXT = "It is sunny. Bring a hat."


# --------------------------------------------------------------------------
# harness
# --------------------------------------------------------------------------

@pytest.fixture()
def server():
    srv = gateway.Server(("127.0.0.1", 0), gateway.GatewayHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv.server_address[1]
    srv.shutdown()


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    gateway._ambient_reset_state()
    monkeypatch.setattr(gateway, "NODES_PATH", tmp_path / "nodes.json")
    monkeypatch.setattr(gateway, "NODE_TARGETS_PATH", tmp_path / "node-targets.json")
    monkeypatch.setattr(gateway, "WATCH_TOKENS_PATH", tmp_path / "watch-tokens.json")
    monkeypatch.delenv("MVG_NODE_TARGETS", raising=False)
    monkeypatch.delenv("MVG_TRUSTED_KEY", raising=False)
    monkeypatch.setattr(gateway, "VOICE_ID_GRACE_SEC", 0.5)
    monkeypatch.setattr(gateway, "synthesize", lambda text: (WAV, "fake"))
    yield
    gateway._ambient_reset_state()


class Kept:
    def __init__(self):
        self.calls = []
        self.event = threading.Event()


@pytest.fixture(autouse=True)
def kept(monkeypatch, tmp_path):
    k = Kept()

    def hook(node_id, reply_id, target, text):
        k.calls.append((node_id, reply_id, target, text))
        k.event.set()

    monkeypatch.setenv("MVG_KEEP_REPLY_NODES", "node-a")
    monkeypatch.setattr(gateway, "LOG_DIR", tmp_path)
    monkeypatch.setattr(gateway, "speech_gate", lambda wav: ("passed", 900.0, 1.0))
    monkeypatch.setattr(gateway, "transcribe", lambda wav: "what is the weather today")
    monkeypatch.setattr(gateway, "identify_speaker", lambda wav, sha: ("", 0.0))
    gateway.get_kept_store().clear()
    monkeypatch.setattr(gateway, "_plugin_on_reply_kept", hook)
    yield k
    gateway.get_kept_store().clear()


def parse_sse(raw: bytes):
    events, ev = [], None
    for line in raw.decode().splitlines():
        if line.startswith("event:"):
            ev = line.split(":", 1)[1].strip()
        elif line.startswith("data:"):
            events.append((ev, json.loads(line.split(":", 1)[1])))
    return events


def multipart(fields: dict, audio: bytes | None = WAV):
    boundary = "----t" + uuid.uuid4().hex
    body = bytearray()
    for k, v in fields.items():
        body += f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
    if audio is not None:
        body += (f'--{boundary}\r\nContent-Disposition: form-data; name="audio"; '
                 f'filename="u.wav"\r\nContent-Type: audio/wav\r\n\r\n').encode()
        body += audio + b"\r\n"
    body += f"--{boundary}--\r\n".encode()
    return bytes(body), f"multipart/form-data; boundary={boundary}"


def post(port, fields=None, audio=WAV):
    body, ctype = multipart(fields or {}, audio)
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=20)
    conn.request("POST", "/voice/turn", body=body, headers={"Content-Type": ctype})
    resp = conn.getresponse()
    data = resp.read()
    conn.close()
    return resp.status, data


def drop_sentences(monkeypatch):
    """A client that left: every `sentence` write fails, the rest goes through."""
    orig = gateway.GatewayHandler._sse

    def sse(self, event, data):
        if event == "sentence":
            return False
        return orig(self, event, data)

    monkeypatch.setattr(gateway.GatewayHandler, "_sse", sse)


def brain(monkeypatch, text, release):
    """Calls on_partial(text) at once, then holds the brain call until `release` is set."""
    def call(prompt, session_id, node_id, speaker, score, timeout, on_partial):
        on_partial(text)
        release.wait(5)
        return text
    monkeypatch.setattr(gateway, "STREAM_BRAIN_CALL", call)


def instant_brain(monkeypatch, text=TEXT, calls=None):
    def call(prompt, session_id, node_id, speaker, score, timeout, on_partial):
        if calls is not None:
            calls.append(prompt)
        return text
    monkeypatch.setattr(gateway, "STREAM_BRAIN_CALL", call)


def log_has(tmp_path, outcome):
    p = tmp_path / "lomin.jsonl"
    return p.exists() and outcome in p.read_text()


def wait_for(cond, seconds=5):
    deadline = time.time() + seconds
    val = cond()
    while not val and time.time() < deadline:
        time.sleep(0.05)
        val = cond()
    return val


def keep_thread_done():
    return wait_for(lambda: not any(t.name == "kept-reply" and t.is_alive() for t in threading.enumerate()))


def turn(server, node="node-a"):
    """A voice turn whose client is, in effect, gone: post and ignore the body."""
    return post(server, {"node_id": node, "tts": "wav"})


def drop_flow(server, monkeypatch, tmp_path, text=TEXT):
    release = threading.Event()
    drop_sentences(monkeypatch)
    brain(monkeypatch, text, release)
    turn(server)
    assert wait_for(lambda: log_has(tmp_path, "error:client_gone"))
    release.set()


def seed(node="node-a", answer=TEXT):
    return gateway.get_kept_store().store(node_id=node, prompt="p", answer=answer)["id"]


# --------------------------------------------------------------------------
# keeping
# --------------------------------------------------------------------------

def test_client_gone_keeps_the_reply_and_calls_the_hook(server, kept, monkeypatch, tmp_path):
    drop_flow(server, monkeypatch, tmp_path)
    assert kept.event.wait(5)
    assert len(kept.calls) == 1
    node, rid, target, text = kept.calls[0]
    assert node == "node-a" and len(rid) == 36 and text == TEXT
    assert target == gateway.resolve_request_target("node-a", None)
    assert gateway.get_kept_store().get("node-a", rid)["answer"] == TEXT


def test_a_client_that_got_the_whole_reply_keeps_nothing(server, kept, monkeypatch):
    instant_brain(monkeypatch)
    status, raw = turn(server)
    assert status == 200
    assert "done" in [e for e, _ in parse_sse(raw)]
    assert not kept.event.wait(0.5)
    assert gateway.get_kept_store().count_for_node("node-a") == 0


def test_a_node_that_is_not_listed_keeps_nothing(server, kept, monkeypatch, tmp_path):
    monkeypatch.setenv("MVG_KEEP_REPLY_NODES", "node-b")
    drop_flow(server, monkeypatch, tmp_path)
    assert keep_thread_done()
    assert kept.calls == []
    assert gateway.get_kept_store().count_for_node("node-a") == 0


def test_switch_unset_keeps_nothing(server, kept, monkeypatch, tmp_path):
    monkeypatch.delenv("MVG_KEEP_REPLY_NODES")
    drop_flow(server, monkeypatch, tmp_path)
    assert keep_thread_done()
    assert kept.calls == []
    assert gateway.get_kept_store().count_for_node("node-a") == 0


@pytest.mark.parametrize("mode", ["empty", "unanswered", "raises"])
def test_nothing_to_keep_is_not_kept(server, kept, monkeypatch, tmp_path, mode):
    release = threading.Event()
    drop_sentences(monkeypatch)
    if mode == "raises":
        def call(prompt, session_id, node_id, speaker, score, timeout, on_partial):
            on_partial("It is sunny. Bring")
            release.wait(5)
            raise RuntimeError("x")
    else:
        reply = "" if mode == "empty" else "unanswered:timeout"

        def call(prompt, session_id, node_id, speaker, score, timeout, on_partial):
            on_partial("It is sunny. Bring")
            release.wait(5)
            return reply
    monkeypatch.setattr(gateway, "STREAM_BRAIN_CALL", call)
    turn(server)
    assert wait_for(lambda: log_has(tmp_path, "error:client_gone"))
    release.set()
    assert keep_thread_done()
    assert kept.calls == []
    assert gateway.get_kept_store().count_for_node("node-a") == 0


def test_a_brain_that_never_finishes_keeps_nothing(server, kept, monkeypatch, tmp_path):
    monkeypatch.setenv("MVG_KEPT_REPLY_WAIT_SEC", "0.2")
    release = threading.Event()
    drop_sentences(monkeypatch)
    brain(monkeypatch, TEXT, release)
    turn(server)
    assert wait_for(lambda: log_has(tmp_path, "error:client_gone"))
    assert wait_for(lambda: log_has(tmp_path, "kept_reply:timeout"))
    release.set()
    time.sleep(0.5)
    assert kept.calls == []
    assert gateway.get_kept_store().count_for_node("node-a") == 0


def test_a_hook_that_raises_does_not_lose_the_reply(server, kept, monkeypatch, tmp_path):
    def raiser(node_id, reply_id, target, text):
        raise RuntimeError("boom")

    fallbacks = []
    monkeypatch.setattr(gateway, "_plugin_on_reply_kept", raiser)
    monkeypatch.setattr(gateway, "log_fallback", lambda reason, **kw: fallbacks.append(reason))
    drop_flow(server, monkeypatch, tmp_path)
    assert keep_thread_done()
    assert gateway.get_kept_store().count_for_node("node-a") == 1
    assert fallbacks == ["kept_reply_failed:boom"]


def test_no_plugin_hook_still_keeps(server, kept, monkeypatch, tmp_path):
    fallbacks = []
    monkeypatch.setattr(gateway, "_plugin_on_reply_kept", None)
    monkeypatch.setattr(gateway, "log_fallback", lambda reason, **kw: fallbacks.append(reason))
    drop_flow(server, monkeypatch, tmp_path)
    assert keep_thread_done()
    assert gateway.get_kept_store().count_for_node("node-a") == 1
    assert fallbacks == []


def test_a_kept_reply_keeps_its_markdown_and_the_hook_gets_plain_text(server, kept, monkeypatch, tmp_path):
    drop_flow(server, monkeypatch, tmp_path, text="**It is sunny.** Bring a hat.")
    assert kept.event.wait(5)
    _, rid, _, text = kept.calls[0]
    assert text == TEXT
    assert gateway.get_kept_store().get("node-a", rid)["answer"] == "**It is sunny.** Bring a hat."


def test_a_new_keep_drops_expired_records(server, kept, monkeypatch, tmp_path):
    seed()
    monkeypatch.setenv("MVG_KEPT_REPLY_TTL_SEC", "0.05")
    time.sleep(0.1)
    drop_flow(server, monkeypatch, tmp_path)
    assert kept.event.wait(5)
    assert keep_thread_done()
    assert gateway.get_kept_store().count_for_node("node-a") == 1
    assert gateway.get_kept_store().peek_for_node("node-a")["id"] == kept.calls[0][1]


def test_a_cancel_acknowledgement_is_not_kept(server, kept, monkeypatch, tmp_path):
    monkeypatch.setenv("MVG_SPOKEN_CANCEL", "1")
    monkeypatch.setattr(gateway, "transcribe", lambda wav: "cancel that")
    monkeypatch.setattr(gateway, "cancel_active_turn", lambda node_id: "cancelled")
    drop_sentences(monkeypatch)
    turn(server)
    assert wait_for(lambda: log_has(tmp_path, "error:client_gone"))
    assert keep_thread_done()
    assert kept.calls == []
    assert gateway.get_kept_store().count_for_node("node-a") == 0


def test_a_fast_path_reply_is_not_kept(server, kept, monkeypatch, tmp_path):
    monkeypatch.setattr(gateway, "_intent", lambda *a, **k: {
        "hit": True, "intent": "demo", "match_ms": 0.0, "handler_ms": 0.0, "reply": "Done.", "reason": "ok"})
    drop_sentences(monkeypatch)
    turn(server)
    assert wait_for(lambda: log_has(tmp_path, "error:client_gone"))
    assert keep_thread_done()
    assert kept.calls == []
    assert gateway.get_kept_store().count_for_node("node-a") == 0


# --------------------------------------------------------------------------
# replay
# --------------------------------------------------------------------------

def test_replay_speaks_the_kept_reply_without_audio_or_brain(server, kept, monkeypatch):
    rid = seed()
    brain_calls, transcribed = [], []
    instant_brain(monkeypatch, calls=brain_calls)
    monkeypatch.setattr(gateway, "transcribe", lambda wav: transcribed.append(1) or "x")
    status, raw = post(server, {"node_id": "node-a", "tts": "wav", "reply_id": rid}, audio=None)
    assert status == 200
    events = parse_sse(raw)
    names = [e for e, _ in events]
    assert names[0] == "turn"
    first = next(d for e, d in events if e == "sentence")
    assert first["text"] == "It is sunny." and first["audio_b64"]
    assert ("reply" in names) and names[-1] == "done"
    assert next(d for e, d in events if e == "reply")["reply"] == TEXT
    assert brain_calls == [] and transcribed == []


def _expect_404(status, raw):
    assert status == 404
    assert json.loads(raw) == {"error": "reply_expired"}


def test_replay_of_an_expired_reply_is_404(server, kept, monkeypatch):
    rid = seed()
    monkeypatch.setenv("MVG_KEPT_REPLY_TTL_SEC", "0")
    _expect_404(*post(server, {"node_id": "node-a", "tts": "wav", "reply_id": rid}, audio=None))


def test_replay_of_another_nodes_reply_is_404(server, kept, monkeypatch):
    rid = seed("node-a")
    monkeypatch.setenv("MVG_KEEP_REPLY_NODES", "node-a,node-b")
    _expect_404(*post(server, {"node_id": "node-b", "tts": "wav", "reply_id": rid}, audio=None))


def test_replay_of_an_unknown_id_is_404(server, kept):
    _expect_404(*post(server, {"node_id": "node-a", "tts": "wav",
                               "reply_id": "00000000-0000-4000-8000-000000000000"}, audio=None))


def test_replay_with_the_switch_unset_is_404(server, kept, monkeypatch):
    rid = seed()
    monkeypatch.delenv("MVG_KEEP_REPLY_NODES")
    _expect_404(*post(server, {"node_id": "node-a", "tts": "wav", "reply_id": rid}, audio=None))


def test_a_replay_that_loses_its_client_keeps_nothing_again(server, kept, monkeypatch, tmp_path):
    rid = seed()
    drop_sentences(monkeypatch)
    post(server, {"node_id": "node-a", "tts": "wav", "reply_id": rid}, audio=None)
    assert wait_for(lambda: log_has(tmp_path, "error:client_gone"))
    assert keep_thread_done()
    assert kept.calls == []
    assert gateway.get_kept_store().count_for_node("node-a") == 1


def test_a_request_with_both_reply_id_and_audio_is_400(server, kept, monkeypatch):
    rid = seed()
    brain_calls = []
    instant_brain(monkeypatch, calls=brain_calls)
    status, raw = post(server, {"node_id": "node-a", "tts": "wav", "reply_id": rid})
    assert status == 400
    assert json.loads(raw) == {"error": "reply_id and audio are exclusive"}
    assert brain_calls == []


def test_a_replay_writes_no_second_transcript(server, kept, monkeypatch, tmp_path):
    path = tmp_path / "t.jsonl"
    monkeypatch.setenv("MVG_TRANSCRIPT_LOG_PATH", str(path))
    instant_brain(monkeypatch)

    def lines():
        return len(path.read_text().splitlines()) if path.exists() else 0

    status, raw = turn(server)
    assert status == 200
    assert wait_for(lambda: lines() == 1)
    rid = seed()
    status, raw = post(server, {"node_id": "node-a", "tts": "wav", "reply_id": rid}, audio=None)
    assert status == 200 and parse_sse(raw)[-1][0] == "done"
    time.sleep(0.5)
    assert lines() == 1


# --------------------------------------------------------------------------
# store
# --------------------------------------------------------------------------

def test_get_does_not_remove_and_checks_the_node():
    store = PendingVoiceAnswerStore()
    rid = store.store(node_id="node-a", prompt="p", answer="a")["id"]
    assert store.get("node-a", rid)["id"] == rid
    assert store.get("node-a", rid)["id"] == rid
    assert store.get("node-b", rid) is None
    assert store.get("node-a", "") is None
    other = store.store(node_id="node-a", prompt="p", answer="b")["id"]
    assert store.get("node-a", other)["answer"] == "b"
    assert store.get("node-a", "00000000-0000-4000-8000-000000000000") is None


def test_purge_drops_only_expired_records():
    store = PendingVoiceAnswerStore()
    store.store(node_id="node-a", prompt="p", answer="a")
    time.sleep(0.1)
    store.store(node_id="node-b", prompt="p", answer="b")
    assert store.purge(0.05) == 1
    assert store.count_for_node("node-a") == 0
    assert store.count_for_node("node-b") == 1
