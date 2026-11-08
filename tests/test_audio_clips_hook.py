"""audio_clips: optional raw-audio argument to the plugin's brain_call_stream."""
import gateway


def _run(monkeypatch, fake):
    monkeypatch.setattr(gateway, "_plugin_brain_call_stream", fake)
    seen = []
    gateway.plugin_brain_call_stream("hi", "s", "n1", "", None, 5, seen.append, audio_clips=[b"RIFFwav"])
    return seen


def test_plugin_without_audio_clips_gets_the_old_call(monkeypatch):
    calls = []

    def old(target, prompt, timeout, on_partial, stages_out):
        calls.append((target, timeout, stages_out))
        return "ok"

    _run(monkeypatch, old)
    assert len(calls) == 1


def test_plugin_with_audio_clips_receives_them(monkeypatch):
    got = {}

    def new(target, prompt, timeout, on_partial, stages_out, audio_clips=None):
        got["clips"] = audio_clips
        return "ok"

    _run(monkeypatch, new)
    assert got["clips"] == [b"RIFFwav"]


def test_plugin_with_kwargs_receives_them(monkeypatch):
    got = {}

    def kw(target, prompt, timeout, on_partial, stages_out, **kwargs):
        got.update(kwargs)
        return "ok"

    _run(monkeypatch, kw)
    assert got == {"audio_clips": [b"RIFFwav"]}


def test_none_clips_are_not_passed(monkeypatch):
    got = {}

    def new(target, prompt, timeout, on_partial, stages_out, audio_clips="unset"):
        got["clips"] = audio_clips
        return "ok"

    monkeypatch.setattr(gateway, "_plugin_brain_call_stream", new)
    gateway.plugin_brain_call_stream("hi", "s", "n1", "", None, 5, lambda t: None)
    assert got["clips"] == "unset"


def test_collect_off_by_default(monkeypatch):
    monkeypatch.delenv("MVG_SPLICE_RAW_AUDIO", raising=False)
    assert gateway.collect_audio_clips(b"wav") is None


def test_collect_on_returns_one_clip(monkeypatch):
    monkeypatch.setenv("MVG_SPLICE_RAW_AUDIO", "1")
    assert gateway.collect_audio_clips(b"wav") == [b"wav"]
    assert gateway.collect_audio_clips(b"") is None


def test_over_cap_is_dropped_and_logged(monkeypatch):
    monkeypatch.setenv("MVG_SPLICE_RAW_AUDIO", "1")
    monkeypatch.setenv("MVG_AUDIO_CLIPS_MAX_BYTES", "4")
    logged = []
    monkeypatch.setattr(gateway, "log_fallback", lambda reason, **kw: logged.append(reason))
    assert gateway.collect_audio_clips(b"12345") is None
    assert logged == ["audio_clips_over_cap"]


def test_ctx_carries_clips():
    ctx = gateway.make_frame_ctx(None, audio_clips=[b"x"])
    assert ctx["audio_clips"] == [b"x"]
    assert "audio_clips" not in gateway.make_frame_ctx(None)
