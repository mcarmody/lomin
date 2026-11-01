#!/usr/bin/env python3
"""Tests for POST /voice/turn, nodes.json, the framing helper and the
legacy-endpoint log line. Every upstream (speech gate, Whisper, voice ID,
classifier, brain, TTS) is a monkeypatched module function: no network."""
import base64
import http.client
import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

import gateway


# --------------------------------------------------------------------------
# harness
# --------------------------------------------------------------------------

WAV = gateway.pcm_to_wav(b"\x01\x00" * 2400, 24000)


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


class Calls:
    def __init__(self):
        self.order = []
        self.prompts = []


@pytest.fixture()
def stubs(monkeypatch):
    """Stage stubs that record the order they ran in."""
    c = Calls()
    c.gate = ("passed", 900.0, 1.0)
    c.transcript = "what's the weather, Amos"
    c.identify = ("", 0.0)
    c.classify = {"score": 0.9, "classifier": "jev", "laya_score": None}
    c.reply = "It is sunny."

    def gate(wav):
        c.order.append("gate")
        return c.gate

    def transcribe(wav):
        c.order.append("transcribe")
        return c.transcript

    def identify(wav, sha):
        c.order.append("identify")
        return c.identify

    def classify(node_id, transcript, speaker=""):
        c.order.append("classify")
        return dict(c.classify)

    def brain(prompt, session_id, node_id, speaker, speaker_score, timeout, on_partial):
        c.order.append("brain")
        c.prompts.append(prompt)
        return c.reply

    monkeypatch.setattr(gateway, "speech_gate", gate)
    monkeypatch.setattr(gateway, "transcribe", transcribe)
    monkeypatch.setattr(gateway, "identify_speaker", identify)
    monkeypatch.setattr(gateway, "decide_addressed", classify)
    monkeypatch.setattr(gateway, "STREAM_BRAIN_CALL", brain)
    return c


def parse_sse(raw: bytes):
    events, ev = [], None
    for line in raw.decode().splitlines():
        if line.startswith("event:"):
            ev = line.split(":", 1)[1].strip()
        elif line.startswith("data:"):
            events.append((ev, json.loads(line.split(":", 1)[1])))
    return events


def multipart(fields: dict, audio: bytes = WAV):
    boundary = "----t" + uuid.uuid4().hex
    body = bytearray()
    for k, v in fields.items():
        body += f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
    body += (f'--{boundary}\r\nContent-Disposition: form-data; name="audio"; '
             f'filename="u.wav"\r\nContent-Type: audio/wav\r\n\r\n').encode()
    body += audio + b"\r\n" + f"--{boundary}--\r\n".encode()
    return bytes(body), f"multipart/form-data; boundary={boundary}"


def post(port, path="/voice/turn", fields=None, headers=None, raw=False, audio=WAV):
    if raw:
        body, ctype = audio, "audio/wav"
    else:
        body, ctype = multipart(fields or {}, audio)
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=20)
    conn.request("POST", path, body=body, headers={"Content-Type": ctype, **(headers or {})})
    resp = conn.getresponse()
    data = resp.read()
    conn.close()
    return resp.status, data


def turn(port, fields=None, **kw):
    fields = {"node_id": "lammen", "tts": "wav", **(fields or {})}
    status, data = post(port, fields=fields, **kw)
    assert status == 200, data
    return parse_sse(data)


def names(events):
    return [e for e, _ in events]


def first(events, name):
    return next(d for e, d in events if e == name)


# --------------------------------------------------------------------------
# stage order, events
# --------------------------------------------------------------------------

def test_stage_order_and_events(server, stubs):
    events = turn(server, {"open_room": "1"})
    # gate, transcribe and identify start together; all precede classify and brain
    assert set(stubs.order[:3]) >= {"gate", "transcribe"}
    assert stubs.order.index("transcribe") < stubs.order.index("classify") < stubs.order.index("brain")
    ns = names(events)
    assert ns[0] == "turn"
    states = [d["state"] for e, d in events if e == "state"]
    assert states == ["transcribing", "classifying", "thinking", "speaking"]
    # Fixed order of the stages on the wire.
    pos = lambda n: ns.index(n)  # noqa: E731
    assert pos("turn") < pos("transcript") < pos("sentence") < pos("reply") < pos("done")
    assert ns[-1] == "done" and ns.count("reply") == 1
    tr = first(events, "transcript")
    assert tr["text"] == tr["transcript"] == "what's the weather, Amos"
    sent = first(events, "sentence")
    assert sent["text"] and base64.b64decode(sent["audio_b64"]).startswith(b"RIFF")
    assert first(events, "reply")["text"] == "It is sunny."
    assert "duration_ms" in first(events, "done")


def test_raw_wav_body_and_query_flags(server, stubs, brain):
    status, data = post(server, path="/voice/turn?node_id=lammen&tts=none&open_room=1",
                        raw=True)
    assert status == 200
    ev = parse_sse(data)
    assert "classify" in stubs.order            # open_room came from the query
    assert first(ev, "sentence")["engine"] == "text"   # tts=none: text only
    assert "audio_b64" not in first(ev, "sentence")
    assert "[VOICE via node lammen" in brain[0]["prompt"] and "open room" in brain[0]["prompt"]


def test_node_id_required_and_audio_required(server, stubs):
    status, _ = post(server, fields={"tts": "wav"})
    assert status == 400
    status, _ = post(server, fields={"node_id": "x"}, audio=b"")
    assert status == 400


# --------------------------------------------------------------------------
# parallel voice ID
# --------------------------------------------------------------------------

def test_transcribe_and_identify_run_in_parallel(server, stubs, monkeypatch):
    starts = {}

    def slow_transcribe(wav):
        starts["transcribe"] = time.time()
        time.sleep(0.4)
        return "hello Amos"

    def slow_identify(wav, sha):
        starts["identify"] = time.time()
        time.sleep(0.3)
        return ("mike", 0.82)

    monkeypatch.setattr(gateway, "transcribe", slow_transcribe)
    monkeypatch.setattr(gateway, "identify_speaker", slow_identify)
    t0 = time.time()
    events = turn(server)
    assert abs(starts["transcribe"] - starts["identify"]) < 0.15
    assert time.time() - t0 < 0.4 + 0.3 + 0.3   # well under sequential 0.7 + stages
    assert first(events, "transcript")["voice_id"] == "verified mike 0.82"


def test_voice_id_grace_is_bounded(server, stubs, monkeypatch):
    monkeypatch.setattr(gateway, "VOICE_ID_GRACE_SEC", 0.1)
    monkeypatch.setattr(gateway, "identify_speaker", lambda w, s: (time.sleep(1.5), ("mike", 0.9))[1])
    t0 = time.time()
    events = turn(server)
    assert time.time() - t0 < 1.2               # did not wait for the slow ID
    assert first(events, "transcript")["voice_id"] == "unavailable"


def test_identify_off_in_profile_is_unavailable_and_never_called(server, stubs, tmp_path):
    (tmp_path / "nodes.json").write_text(json.dumps({"lammen": {"identify": False}}))
    events = turn(server)
    assert "identify" not in stubs.order
    assert first(events, "transcript")["voice_id"] == "unavailable"


# --------------------------------------------------------------------------
# ignored paths
# --------------------------------------------------------------------------

def test_open_room_not_addressed_is_ignored(server, stubs):
    stubs.classify = {"score": 0.12, "classifier": "jev", "laya_score": None}
    events = turn(server, {"open_room": "1"})
    assert names(events)[-1] == "ignored"
    ig = first(events, "ignored")
    assert ig["reason"] == "not-addressed" and ig["score"] == 0.12
    assert "brain" not in stubs.order
    assert "done" not in names(events) and "sentence" not in names(events)


def test_closed_room_never_runs_addressed_gate(server, stubs, brain):
    events = turn(server)           # lammen profile: open_room off
    assert "classify" not in stubs.order
    assert "done" in names(events)
    assert "open room" not in brain[0]["prompt"]


def test_profile_open_room_default_and_request_override(server, stubs, tmp_path):
    (tmp_path / "nodes.json").write_text(json.dumps({"lammen": {"open_room": True}}))
    turn(server)
    assert "classify" in stubs.order
    stubs.order.clear()
    turn(server, {"open_room": "0"})
    assert "classify" not in stubs.order


def test_no_speech_path(server, stubs):
    stubs.gate = ("rejected", 80.0, 2.0)
    events = turn(server)
    assert names(events) == ["turn", "ignored"]
    assert first(events, "ignored")["reason"] == "no-speech"
    # STT/ID run in parallel with the gate and are discarded; nothing downstream runs
    assert "gate" in stubs.order
    assert not {"classify", "brain"} & set(stubs.order)


def test_speech_gate_off_in_profile_skips_gate(server, stubs, tmp_path):
    (tmp_path / "nodes.json").write_text(json.dumps({"lammen": {"speech_gate": False}}))
    stubs.gate = ("rejected", 0.0, 0.0)
    events = turn(server)
    assert "gate" not in stubs.order and "done" in names(events)


def test_hallucination_is_ignored_after_transcript(server, stubs):
    stubs.transcript = "Thank you."
    events = turn(server)
    assert names(events)[-1] == "ignored"
    assert "transcript" in names(events)
    assert first(events, "ignored")["reason"] == "hallucination"
    assert "brain" not in stubs.order


def test_stt_failure_is_an_error_event(server, stubs, monkeypatch):
    def boom(wav):
        raise RuntimeError("whisper down")
    monkeypatch.setattr(gateway, "transcribe", boom)
    events = turn(server)
    assert first(events, "error")["error"] == "stt_error"
    assert "brain" not in stubs.order


# --------------------------------------------------------------------------
# voice ID verdict strings (through the real framing path)
# --------------------------------------------------------------------------

@pytest.fixture()
def brain(monkeypatch):
    seen = []

    def fake_stream(target, wrapped, timeout=None, on_partial=None, stages_out=None):
        seen.append({"target": target, "prompt": wrapped})
        return "Fine."

    monkeypatch.setattr(gateway, "_plugin_brain_call_stream", fake_stream)
    monkeypatch.setattr(gateway, "STREAM_BRAIN_CALL", gateway.plugin_brain_call_stream)
    return seen


@pytest.mark.parametrize("ident,verdict,speaker", [
    (("mike", 0.82), "verified mike 0.82", "speaker: Mike ·"),
    (("", 0.41), "not verified (best 0.41)", "speaker: unknown"),
    (("", 0.0), "unavailable", "speaker: unknown"),
])
def test_voice_id_verdict_strings(server, stubs, brain, ident, verdict, speaker):
    stubs.identify = ident
    events = turn(server)
    assert first(events, "transcript")["voice_id"] == verdict
    p = brain[0]["prompt"]
    assert p.startswith(f"[VOICE via node lammen · {speaker}")
    assert f"voice ID: {verdict}" in p
    assert "what's the weather, Amos" in p
    assert "under 60 words" in p
    assert brain[0]["target"] == gateway.DEFAULT_TARGET


def test_match_speaker_exposes_best_score_but_never_a_sub_threshold_name():
    emb = [1.0, 0.0]
    sps = [{"id": "mike", "embedding": [0.6, 0.8]}]          # cosine 0.6
    assert gateway._match_speaker(emb, sps, 0.7) == ("", 0.6)
    assert gateway._match_speaker(emb, sps, 0.6) == ("mike", 0.6)   # >= threshold
    assert gateway._match_speaker(emb, [], 0.7) == ("", 0.0)


def test_identify_speaker_detail_statuses(monkeypatch):
    monkeypatch.setattr(gateway, "identify_speaker", lambda w, s: ("bob", 0.9))
    assert gateway.identify_speaker_detail(b"x", "s")["status"] == "verified"
    monkeypatch.setattr(gateway, "identify_speaker", lambda w, s: ("", 0.3))
    assert gateway.identify_speaker_detail(b"x", "s") == {"status": "not_verified", "id": "", "score": 0.3, "best": ""}
    monkeypatch.setattr(gateway, "identify_speaker", lambda w, s: ("", 0.0))
    assert gateway.identify_speaker_detail(b"x", "s")["status"] == "unavailable"


def test_reply_cap_words_comes_from_profile(server, stubs, brain, tmp_path):
    (tmp_path / "nodes.json").write_text(json.dumps({"lammen": {"reply_cap_words": 25}}))
    turn(server)
    assert "under 25 words" in brain[0]["prompt"]


# --------------------------------------------------------------------------
# claimed_speaker trust rule
# --------------------------------------------------------------------------

CLAIM = {"X-Voice-Claimed-Speaker": "mike", "X-Voice-Gateway-Key": "s3cret"}


def test_claimed_speaker_trusted_with_key(server, stubs, brain, monkeypatch):
    monkeypatch.setenv("MVG_TRUSTED_KEY", "s3cret")
    stubs.identify = ("", 0.4)
    events = turn(server, headers=CLAIM)
    assert "speaker: Mike (logged in)" in brain[0]["prompt"]
    # A login never upgrades the voice ID verdict.
    assert "voice ID: not verified (best 0.40)" in brain[0]["prompt"]
    assert first(events, "transcript")["voice_id"] == "not verified (best 0.40)"


@pytest.mark.parametrize("headers,env", [
    ({"X-Voice-Claimed-Speaker": "mike", "X-Voice-Gateway-Key": "wrong"}, "s3cret"),
    ({"X-Voice-Claimed-Speaker": "mike"}, "s3cret"),            # no key header
    (CLAIM, None),                                              # env unset: ignore entirely
    (CLAIM, ""),                                                # env empty: ignore entirely
    ({"X-Voice-Claimed-Speaker": "mike", "X-Voice-Gateway-Key": ""}, ""),  # empty == empty
])
def test_claimed_speaker_ignored_without_valid_key(server, stubs, brain, monkeypatch, headers, env):
    if env is not None:
        monkeypatch.setenv("MVG_TRUSTED_KEY", env)
    turn(server, headers=headers)
    assert "logged in" not in brain[0]["prompt"]
    assert "speaker: unknown" in brain[0]["prompt"]


def test_claimed_speaker_never_taken_from_body_or_query(server, stubs, brain, monkeypatch):
    monkeypatch.setenv("MVG_TRUSTED_KEY", "s3cret")
    status, data = post(server, path="/voice/turn?claimed_speaker=mike",
                        fields={"node_id": "lammen", "tts": "wav", "claimed_speaker": "mike"},
                        headers={"X-Voice-Gateway-Key": "s3cret"})
    assert status == 200
    assert "logged in" not in brain[0]["prompt"]


# --------------------------------------------------------------------------
# watch route (X-Watch-Route + bearer token)
# --------------------------------------------------------------------------

WATCH = {"X-Watch-Route": "1"}


def test_watch_route_without_token_is_401(server, stubs):
    status, data = post(server, fields={"node_id": "lammen", "tts": "wav"}, headers=WATCH)
    assert status == 401
    assert json.loads(data) == {"error": "unauthorized"}
    assert "brain" not in stubs.order


def test_watch_route_bad_token_is_401(server, stubs, tmp_path):
    (tmp_path / "watch-tokens.json").write_text(json.dumps({"test-token-abc": "apple-watch"}))
    status, data = post(server, fields={"node_id": "lammen", "tts": "wav"},
                        headers={**WATCH, "Authorization": "Bearer wrong"})
    assert status == 401
    assert json.loads(data) == {"error": "unauthorized"}


def test_watch_route_good_token_forces_node(server, stubs, brain, tmp_path):
    (tmp_path / "watch-tokens.json").write_text(json.dumps({"test-token-abc": "apple-watch"}))
    events = turn(server, {"node_id": "lammen", "open_room": "1"},
                  headers={**WATCH, "Authorization": "Bearer test-token-abc"})
    assert names(events)[-1] == "done"
    assert "[VOICE via node apple-watch" in brain[0]["prompt"]
    assert "node lammen" not in brain[0]["prompt"]


def test_lan_request_needs_no_token(server, stubs):
    events = turn(server)
    assert names(events)[-1] == "done"


# --------------------------------------------------------------------------
# nodes.json
# --------------------------------------------------------------------------

def test_nodes_json_profiles_and_defaults(tmp_path):
    (tmp_path / "nodes.json").write_text(json.dumps({
        "deck-node": {"target": "deck-node", "gap_fill": False},
        "_note": "ignored",
    }))
    p = gateway.resolve_profile("deck-node")
    assert p["target"] == "deck-node" and p["gap_fill"] is False
    assert p["identify"] is True and p["speech_gate"] is True and p["reply_cap_words"] == 60
    assert p["open_room"] is False
    unknown = gateway.resolve_profile("nope")
    assert unknown["target"] == gateway.DEFAULT_TARGET
    assert gateway.resolve_target("deck-node") == "deck-node"


def test_nodes_json_env_override_wins(monkeypatch, tmp_path):
    (tmp_path / "nodes.json").write_text(json.dumps({"lammen": {"target": "voice-node"}}))
    monkeypatch.setenv("MVG_NODE_TARGETS", json.dumps({"lammen": "mac-node"}))
    assert gateway.resolve_target("lammen") == "mac-node"


def test_nodes_json_is_read_per_request(tmp_path):
    f = tmp_path / "nodes.json"
    f.write_text(json.dumps({"lammen": {"target": "a"}}))
    assert gateway.resolve_target("lammen") == "a"
    f.write_text(json.dumps({"lammen": {"target": "b"}}))
    assert gateway.resolve_target("lammen") == "b"


def test_node_targets_json_fallback_only_when_nodes_json_absent(tmp_path):
    (tmp_path / "node-targets.json").write_text(json.dumps({"deck-node": "deck-node"}))
    assert gateway.resolve_target("deck-node") == "deck-node"
    (tmp_path / "nodes.json").write_text(json.dumps({"other": {"target": "x"}}))
    assert gateway.resolve_target("deck-node") == gateway.DEFAULT_TARGET


def test_example_nodes_json_has_the_starting_profiles(monkeypatch):
    monkeypatch.setattr(gateway, "NODES_PATH", gateway.Path(gateway.__file__).parent / "nodes.example.json")
    assert gateway.resolve_target("kitchen-speaker") == "agent"
    assert gateway.resolve_target("phone") == "agent"
    assert gateway.resolve_target("unlisted") == gateway.DEFAULT_TARGET
    assert gateway.resolve_profile("kitchen-speaker")["open_room"] is False
    assert gateway.resolve_profile("phone")["open_room"] is False


# --------------------------------------------------------------------------
# legacy framing + legacy_endpoint log line
# --------------------------------------------------------------------------

def post_json(port, path, payload):
    body = json.dumps(payload).encode()
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=20)
    conn.request("POST", path, body=body, headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    data = resp.read()
    conn.close()
    return resp.status, data


@pytest.mark.parametrize("payload,verdict,speaker", [
    ({"speaker": "mike", "speaker_score": 0.91}, "verified mike 0.91", "speaker: Mike"),
    ({"speaker": "", "speaker_score": 0.33}, "not verified (best 0.33)", "speaker: unknown"),
    ({"speaker": "", "speaker_score": 0}, "unavailable", "speaker: unknown"),
    ({}, "unavailable", "speaker: unknown"),
])
def test_legacy_ask_stream_framing(server, brain, monkeypatch, payload, verdict, speaker):
    post_json(server, "/ask/stream", {"prompt": "turn on the lights", "node_id": "lammen", **payload})
    p = brain[0]["prompt"]
    assert p.startswith("[VOICE via node lammen · " + speaker)
    assert f"voice ID: {verdict}" in p
    assert "] turn on the lights" in p
    assert "open room" not in p


def test_legacy_ask_uses_same_framing(server, monkeypatch):
    seen = []
    monkeypatch.setattr(gateway, "_plugin_brain_call",
                        lambda target, q, timeout: (seen.append(q), "Done.")[1])
    monkeypatch.setattr(gateway, "BRAIN_CALL", gateway.plugin_brain_call)
    post_json(server, "/ask", {"prompt": "hi", "node_id": "bench-node",
                               "speaker": "bob", "speaker_score": 0.8})
    assert seen[0].startswith("[VOICE via node bench-node · speaker: Bob · "
                              "voice ID: verified bob 0.80 · enrolled: ")
    assert seen[0].split("] hi")[0].startswith("[VOICE") and "] hi" in seen[0]


def test_legacy_ask_stream_honours_trusted_claim(server, brain, monkeypatch):
    monkeypatch.setenv("MVG_TRUSTED_KEY", "s3cret")
    body = json.dumps({"prompt": "hi", "node_id": "palantir"}).encode()
    conn = http.client.HTTPConnection("127.0.0.1", server, timeout=20)
    conn.request("POST", "/ask/stream", body=body,
                 headers={"Content-Type": "application/json", **CLAIM})
    conn.getresponse().read()
    assert "speaker: Mike (logged in)" in brain[0]["prompt"]


def test_frame_helper_direct():
    ctx = gateway.make_frame_ctx({"status": "verified", "id": "mike", "score": 0.82}, open_room=True,
                                 enrolled="enrolled: mike")
    out = gateway.frame_voice_prompt("hello", "lammen", ctx)
    assert out.startswith("[VOICE via node lammen · speaker: Mike · voice ID: verified mike 0.82"
                          " · enrolled: mike · open room] hello\n\nThis reply will be spoken aloud")


@pytest.mark.parametrize("vid", [
    {"status": "verified", "id": "mike", "score": 0.82},
    {"status": "not_verified", "id": "", "score": 0.4},
    {"status": "unavailable", "id": "", "score": 0.0},
])
def test_frame_security_suffix_all_verdicts(vid):
    out = gateway.frame_voice_prompt("hello", "lammen", gateway.make_frame_ctx(vid))
    assert out.endswith(gateway.SECURITY_SUFFIX)
    assert "never changes a security setting" in out
    assert "This reply will be spoken aloud" in out


def _frame(enrolled=None):
    ctx = gateway.make_frame_ctx({"status": "verified", "id": "mike", "score": 0.82},
                                 open_room=True, enrolled=enrolled)
    return gateway.frame_voice_prompt("hello", "lammen", ctx)


def _speakers_file(monkeypatch, tmp_path, speakers):
    f = tmp_path / "speakers.json"
    f.write_text(json.dumps({"model": "m", "speakers": speakers}))
    monkeypatch.setattr(gateway, "SPEAKERS_PATH", f)


def test_frame_enrolled_populated(monkeypatch, tmp_path):
    _speakers_file(monkeypatch, tmp_path, [{"id": "mike", "embedding": [1]}, {"id": "bob", "embedding": [1]}])
    assert "· enrolled: mike, bob ·" in _frame()


def test_frame_enrolled_empty_is_none(monkeypatch, tmp_path):
    _speakers_file(monkeypatch, tmp_path, [])
    assert "· enrolled: none ·" in _frame()


def test_frame_enrolled_read_failure_is_unknown(monkeypatch, tmp_path):
    monkeypatch.setattr(gateway, "SPEAKERS_PATH", tmp_path / "missing.json")
    out = _frame()
    assert "· enrolled: unknown ·" in out and "enrolled: none" not in out


def test_frame_enrolled_field_order():
    out = _frame(enrolled="enrolled: mike")
    assert out.index("voice ID:") < out.index("enrolled: mike") < out.index("open room")


def _log_lines(tmp_path):
    f = tmp_path / "lomin.jsonl"
    return [json.loads(l) for l in f.read_text().splitlines()] if f.exists() else []


def test_legacy_endpoints_log_legacy_endpoint_and_node(server, stubs, brain, monkeypatch, tmp_path):
    monkeypatch.setattr(gateway, "BRAIN_CALL", lambda *a, **k: "ok")
    post_json(server, "/ask", {"prompt": "x", "node_id": "node-ask"})
    post_json(server, "/ask/stream", {"prompt": "x", "node_id": "node-stream"})
    post(server, path="/stt", raw=True, headers={"X-Node-Id": "node-stt"})
    post(server, path="/ambient?node_id=node-amb", raw=True)
    for _ in range(40):   # /ambient and /stt log just after replying
        lines = {l["endpoint"]: l for l in _log_lines(tmp_path) if l.get("legacy_endpoint")}
        if len(lines) == 4:
            break
        time.sleep(0.05)
    for ep, node in (("/ask", "node-ask"), ("/ask/stream", "node-stream"),
                     ("/stt", "node-stt"), ("/ambient", "node-amb")):
        assert lines[ep]["legacy_endpoint"] == ep and lines[ep]["node"] == node, ep


def test_voice_turn_is_not_flagged_legacy_and_logs_no_text(server, stubs, tmp_path):
    turn(server)
    time.sleep(0.2)
    recs = [l for l in _log_lines(tmp_path) if l["endpoint"] == "/voice/turn"]
    assert recs and "legacy_endpoint" not in recs[0] and recs[0]["node"] == "lammen"
    assert "weather" not in json.dumps(recs)    # transcript never logged


# --------------------------------------------------------------------------
# ear (hub dialect) compatibility
# --------------------------------------------------------------------------

def test_audio_chunk_events_for_the_room_ear(server, stubs, monkeypatch, tmp_path):
    (tmp_path / "nodes.json").write_text(json.dumps({"bench-node": {"audio_chunk": True}}))
    pcm = b"\x02\x00" * 4800
    monkeypatch.setattr(gateway, "synthesize_stream", lambda text: ("fake", iter([pcm, pcm])))
    events = turn(server, {"node_id": "bench-node", "tts": "stream"})
    chunks = [d for e, d in events if e == "audio_chunk"]
    real = [c for c in chunks if not c["is_final"]]
    assert real and all(c["format"] == "pcm" and c["sample_rate"] == 24000 and c["channels"] == 1
                        for c in real)
    assert base64.b64decode(real[0]["data"]) == pcm
    assert chunks[-1]["is_final"] is True and chunks[-1]["data"] == ""
    assert names(events).index("audio_chunk") < names(events).index("reply")
    assert first(events, "transcript")["transcript"]      # the ear reads this key
    assert first(events, "reply")["reply"]                # and this one


def test_no_audio_chunk_events_by_default(server, stubs):
    assert "audio_chunk" not in names(turn(server))


def test_heartbeat_alias_returns_204(server):
    status, _ = post(server, path="/api/voice/heartbeat", raw=True)
    assert status == 204


# --------------------------------------------------------------------------
# voice-ID score logging + opt-in audio capture
# --------------------------------------------------------------------------

def _voice_turn_rec(tmp_path):
    for _ in range(40):
        recs = [l for l in _log_lines(tmp_path) if l["endpoint"] == "/voice/turn"]
        if recs:
            return recs[-1]
        time.sleep(0.05)
    raise AssertionError("no /voice/turn log line")


def _wait_files(d, suffix, n=1):
    for _ in range(60):
        got = sorted(d.glob(f"*{suffix}")) if d.exists() else []
        if len(got) >= n:
            return got
        time.sleep(0.05)
    return []


def test_log_carries_voice_id_score_and_best_below_threshold(server, stubs, monkeypatch, tmp_path):
    def ident(wav, sha):
        gateway._note_best_id("bob")
        return ("", 0.41234)
    monkeypatch.setattr(gateway, "identify_speaker", ident)
    turn(server)
    rec = _voice_turn_rec(tmp_path)
    assert rec["voice_id_status"] == "not_verified"
    assert rec["voice_id_score"] == 0.412 and rec["voice_id_best"] == "bob"


def test_log_voice_id_unavailable_has_empty_best(server, stubs, tmp_path):
    turn(server)
    rec = _voice_turn_rec(tmp_path)
    assert rec["voice_id_score"] == 0.0 and rec["voice_id_best"] == ""


def test_capture_on_saves_wav_and_sidecar(server, stubs, monkeypatch, tmp_path):
    cap = tmp_path / "cap"
    monkeypatch.setattr(gateway, "CAPTURE_DIR", cap)
    (tmp_path / "nodes.json").write_text(json.dumps({"lammen": {"capture_audio": True}}))
    stubs.identify = ("mike", 0.9123)
    ev = turn(server)
    turn_id = ev[0][1]["turn_id"]
    wavs = _wait_files(cap / "lammen", ".wav")
    assert len(wavs) == 1 and wavs[0].read_bytes() == WAV
    assert f"_{turn_id}_mike_0.912.wav" in wavs[0].name
    side = json.loads(wavs[0].with_suffix(".json").read_text())
    assert side["transcript"] == stubs.transcript and side["best_id"] == "mike"
    assert side["voice_id"].startswith("verified mike") and side["score"] == 0.912


def test_capture_off_by_default_and_skipped_when_gate_rejects(server, stubs, monkeypatch, tmp_path):
    cap = tmp_path / "cap"
    monkeypatch.setattr(gateway, "CAPTURE_DIR", cap)
    turn(server)                                   # no flag
    (tmp_path / "nodes.json").write_text(json.dumps({"lammen": {"capture_audio": True}}))
    stubs.gate = ("rejected", 10.0, 1.0)
    turn(server)                                   # flag on, gate rejects
    time.sleep(0.3)
    assert not cap.exists() or not list(cap.rglob("*"))


def test_capture_failure_never_fails_turn(server, stubs, monkeypatch, tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    monkeypatch.setattr(gateway, "CAPTURE_DIR", blocker)   # mkdir under a file fails
    (tmp_path / "nodes.json").write_text(json.dumps({"lammen": {"capture_audio": True}}))
    ev = turn(server)
    assert ev[-1][0] == "done"


def test_capture_prunes_old_files(tmp_path):
    import os
    d = tmp_path / "n"
    d.mkdir()
    old, new = d / "old.wav", d / "new.wav"
    old.write_bytes(b"a"); old.with_suffix(".json").write_text("{}"); new.write_bytes(b"b")
    t = time.time() - 49 * 3600
    os.utime(old, (t, t)); os.utime(old.with_suffix(".json"), (t, t))
    gateway._prune_capture_dir(d, 48)
    assert not old.exists() and not old.with_suffix(".json").exists() and new.exists()


# --------------------------------------------------------------------------
# device / medium in the framing; a watch token may cover several nodes
# --------------------------------------------------------------------------

def test_device_medium_from_request_in_framing(server, stubs, brain):
    turn(server, {"node_id": "apple-phone", "device": "iphone", "medium": "phone-app"})
    assert "[VOICE via node apple-phone · device: iphone · medium: phone-app · speaker: " in brain[0]["prompt"]


def test_device_medium_from_headers_in_framing(server, stubs, brain):
    turn(server, headers={"X-Voice-Device": "iphone", "X-Voice-Medium": "phone-app"})
    assert "[VOICE via node lammen · device: iphone · medium: phone-app · speaker: " in brain[0]["prompt"]


def test_device_medium_default_from_nodes_json(server, stubs, brain, tmp_path):
    (tmp_path / "nodes.json").write_text(json.dumps(
        {"lammen": {"device": "dashboard", "medium": "browser"}}))
    turn(server)
    assert "[VOICE via node lammen · device: dashboard · medium: browser · speaker: " in brain[0]["prompt"]
    # a request value beats the profile value
    turn(server, {"device": "ipad"})
    assert "device: ipad · medium: browser · speaker: " in brain[1]["prompt"]


def test_device_medium_invalid_dropped(server, stubs, brain):
    turn(server, {"device": "Bad Value!", "medium": "x" * 40})
    assert "device:" not in brain[0]["prompt"] and "medium:" not in brain[0]["prompt"]


def test_clean_tag():
    assert gateway.clean_tag("  IPhone ") == "iphone"
    assert gateway.clean_tag("phone-app") == "phone-app"
    for bad in ("", "-x", "a b", "a_b", "x" * 33, None, 5):
        assert gateway.clean_tag(bad) == ""


def test_framing_unchanged_without_device(server, stubs, brain):
    turn(server)
    assert brain[0]["prompt"].startswith("[VOICE via node lammen · speaker: ")
    assert "device:" not in brain[0]["prompt"] and "medium:" not in brain[0]["prompt"]
    ctx = gateway.make_frame_ctx({"status": "verified", "id": "mike", "score": 0.82}, open_room=True,
                                 enrolled="enrolled: mike")
    assert gateway.frame_voice_prompt("hello", "lammen", ctx).startswith(
        "[VOICE via node lammen · speaker: Mike · voice ID: verified mike 0.82"
        " · enrolled: mike · open room] hello")


def test_device_medium_logged(server, stubs, brain, monkeypatch):
    logged = []
    monkeypatch.setattr(gateway, "log_request",
                        lambda endpoint, ms, outcome, **extra: logged.append((endpoint, outcome, extra)))
    turn(server, {"device": "iphone", "medium": "phone-app"})
    ok = [x for x in logged if x[0] == "/voice/turn" and x[1] == "ok"]
    assert ok and ok[-1][2]["device"] == "iphone" and ok[-1][2]["medium"] == "phone-app"


def test_ask_device_medium_in_framing(server, brain):
    post_json(server, "/ask/stream", {"prompt": "hi", "node_id": "lammen",
                                      "device": "iphone", "medium": "phone-app"})
    assert brain[0]["prompt"].startswith("[VOICE via node lammen · device: iphone · medium: phone-app · speaker: ")


def test_ask_device_medium_default_from_profile(server, brain, tmp_path, monkeypatch):
    (tmp_path / "nodes.json").write_text(json.dumps({"lammen": {"device": "dashboard", "medium": "browser"}}))
    seen = []
    monkeypatch.setattr(gateway, "_plugin_brain_call",
                        lambda target, q, timeout: (seen.append(q), "Done.")[1])
    monkeypatch.setattr(gateway, "BRAIN_CALL", gateway.plugin_brain_call)
    post_json(server, "/ask", {"prompt": "hi", "node_id": "lammen", "medium": "Voice-Widget"})
    assert seen[0].startswith("[VOICE via node lammen · device: dashboard · medium: voice-widget · speaker: ")


def test_watch_token_list_allows_requested_node(server, stubs, brain, tmp_path):
    (tmp_path / "watch-tokens.json").write_text(
        json.dumps({"test-token-abc": ["apple-watch", "apple-phone"]}))
    events = turn(server, {"node_id": "apple-phone"},
                  headers={**WATCH, "Authorization": "Bearer test-token-abc"})
    assert names(events)[-1] == "done"
    assert first(events, "turn")["node_id"] == "apple-phone"
    assert "[VOICE via node apple-phone" in brain[0]["prompt"]


def test_watch_token_list_unknown_node_falls_back_to_first(server, stubs, brain, tmp_path):
    (tmp_path / "watch-tokens.json").write_text(
        json.dumps({"test-token-abc": ["apple-watch", "apple-phone"]}))
    events = turn(server, {"node_id": "lammen"},
                  headers={**WATCH, "Authorization": "Bearer test-token-abc"})
    assert first(events, "turn")["node_id"] == "apple-watch"
    assert "[VOICE via node apple-watch" in brain[0]["prompt"]


def test_watch_token_string_still_forces_node(server, stubs, brain, tmp_path):
    (tmp_path / "watch-tokens.json").write_text(json.dumps({"test-token-abc": "apple-watch"}))
    events = turn(server, {"node_id": "apple-phone"},
                  headers={**WATCH, "Authorization": "Bearer test-token-abc"})
    assert first(events, "turn")["node_id"] == "apple-watch"


@pytest.mark.parametrize("value", [[], 5, None, {"a": 1}, [1, 2]])
def test_watch_token_bad_value_is_401(server, stubs, tmp_path, value):
    (tmp_path / "watch-tokens.json").write_text(json.dumps({"test-token-abc": value}))
    status, _ = post(server, fields={"node_id": "lammen", "tts": "wav"},
                     headers={**WATCH, "Authorization": "Bearer test-token-abc"})
    assert status == 401


def test_example_nodes_json_has_device_medium(monkeypatch):
    monkeypatch.setattr(gateway, "NODES_PATH", gateway.Path(gateway.__file__).parent / "nodes.example.json")
    assert gateway.resolve_device_medium("phone") == ("iphone", "phone-app")
    assert gateway.resolve_device_medium("kitchen-speaker") == ("pi", "room-speaker")
    assert gateway.resolve_device_medium("nope") == ("", "")


# --------------------------------------------------------------------------
# per-request target: a node lists the agents a request may pick
# --------------------------------------------------------------------------

PHONE = {"phone": {"target": "agent", "targets": ["agent", "helper"]}}


def _write_nodes(tmp_path, nodes):
    (tmp_path / "nodes.json").write_text(json.dumps(nodes))


def _voice_turn_log(tmp_path, name="lomin.jsonl"):
    """The /voice/turn log line in `name` (request log or latency log). Written as
    the turn ends, so poll briefly."""
    for _ in range(40):
        f = tmp_path / name
        recs = [json.loads(l) for l in f.read_text().splitlines()
                if '"/voice/turn"' in l] if f.exists() else []
        if recs:
            return recs[-1]
        time.sleep(0.05)
    raise AssertionError(f"no /voice/turn line in {name}")


@pytest.fixture()
def targets(monkeypatch, stubs):
    """The brain leg each brain call sees. Read in the worker thread, where
    _FRAME_CTX.ctx carries the request's target."""
    seen = []

    def brain_target(prompt, session_id, node_id, speaker, speaker_score, timeout, on_partial):
        seen.append(gateway._call_target(node_id))
        return "Fine."

    monkeypatch.setattr(gateway, "STREAM_BRAIN_CALL", brain_target)
    return seen


@pytest.fixture()
def frame_ctx():
    """This thread's frame context, cleared before and after the test."""
    def clear():
        try:
            del gateway._FRAME_CTX.ctx
        except AttributeError:
            pass
    clear()
    yield gateway._FRAME_CTX
    clear()


def test_listed_target_reaches_the_brain_and_is_logged(server, targets, tmp_path):
    _write_nodes(tmp_path, PHONE)
    turn(server, {"node_id": "phone", "target": "helper"})
    assert targets == ["helper"]
    rec = _voice_turn_log(tmp_path)
    assert rec["target"] == "helper" and "target_rejected" not in rec
    assert _voice_turn_log(tmp_path, "voice-latency.jsonl")["target"] == "helper"


def test_unlisted_target_falls_back_to_node_target(server, targets, tmp_path):
    _write_nodes(tmp_path, PHONE)
    turn(server, {"node_id": "phone", "target": "intruder"})
    assert targets == ["agent"]
    rec = _voice_turn_log(tmp_path)
    assert rec["target"] == "agent" and rec["target_rejected"] is True


def test_absent_target_uses_node_target(server, targets, tmp_path):
    _write_nodes(tmp_path, PHONE)
    turn(server, {"node_id": "phone"})
    assert targets == ["agent"]
    rec = _voice_turn_log(tmp_path)
    assert rec["target"] == "agent" and "target_rejected" not in rec


def test_node_without_targets_key_ignores_request_target(server, targets, tmp_path):
    _write_nodes(tmp_path, {"phone": {"target": "agent"}})
    turn(server, {"node_id": "phone", "target": "helper"})
    assert targets == ["agent"]


def test_string_targets_is_not_a_list(server, targets, tmp_path):
    _write_nodes(tmp_path, {"phone": {"target": "agent", "targets": "helper"}})
    turn(server, {"node_id": "phone", "target": "helper"})
    assert targets == ["agent"]


@pytest.mark.parametrize("nodes,requested,expected", [
    ({"phone": {"target": "node-b", "targets": ["agent", "helper"]}}, "helper", "helper"),
    ({"phone": {"target": "node-b", "targets": ["agent", "helper"]}}, " helper ", "helper"),
    ({"phone": {"target": "node-b", "targets": ["agent", "helper"]}}, "intruder", "node-b"),
    ({"phone": {"target": "node-b", "targets": ["agent", "helper"]}}, None, "node-b"),
    ({"phone": {"target": "node-b"}}, "helper", "node-b"),
    ({"phone": {"target": "node-b", "targets": "helper"}}, "helper", "node-b"),
], ids=["listed", "whitespace-padded", "unlisted", "absent", "no-targets-key", "string-targets"])
def test_resolve_request_target(tmp_path, nodes, requested, expected):
    # node-b is not DEFAULT_TARGET, so a fallback to the default is visible
    _write_nodes(tmp_path, nodes)
    assert gateway.resolve_request_target("phone", requested) == expected


def test_call_target_without_frame_context_is_node_target(tmp_path, frame_ctx):
    _write_nodes(tmp_path, {"phone": {"target": "node-b"}})
    assert gateway._call_target("phone") == gateway.resolve_target("phone") == "node-b"


def test_plugin_brain_call_uses_request_target(frame_ctx, monkeypatch):
    seen = []
    monkeypatch.setattr(gateway, "_plugin_brain_call",
                        lambda target, prompt, timeout: (seen.append(target), "ok")[1])
    frame_ctx.ctx = gateway.make_frame_ctx(None, target="helper")
    assert gateway.plugin_brain_call("hi", "s", "phone", "", None, 5) == "ok"
    assert seen == ["helper"]


def test_plugin_brain_call_stream_uses_request_target(frame_ctx, monkeypatch):
    seen = []

    def stream(target, prompt, timeout, on_partial, stages_out):
        seen.append(target)
        on_partial("ok")
        return "ok"

    monkeypatch.setattr(gateway, "_plugin_brain_call_stream", stream)
    frame_ctx.ctx = gateway.make_frame_ctx(None, target="helper")
    assert gateway.plugin_brain_call_stream("hi", "s", "phone", "", None, 5, lambda t: None) == "ok"
    assert seen == ["helper"]


def test_http_brain_call_sends_request_target(frame_ctx, monkeypatch):
    bodies = []

    class Brain(BaseHTTPRequestHandler):
        def do_POST(self):
            bodies.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            out = json.dumps({"reply": "ok"}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *args):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), Brain)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        monkeypatch.setattr(gateway, "MVG_BRAIN_URL", f"http://127.0.0.1:{srv.server_address[1]}/brain")
        frame_ctx.ctx = gateway.make_frame_ctx(None, target="helper")
        assert gateway.http_brain_call("hi", "s", "phone", "", None, 5) == "ok"
    finally:
        srv.shutdown()
        srv.server_close()
    assert bodies[0]["target"] == "helper"
