"""normalize_stage_event: the stable shape for brain stage events."""
import gateway


def test_valid_dict_passes_through():
    ev = {"stage": "tool_start", "detail": "checking the logs", "t": 2.4}
    assert gateway.normalize_stage_event(ev) == ev


def test_missing_detail_and_t_get_defaults():
    assert gateway.normalize_stage_event({"stage": "milestone"}) == {"stage": "milestone", "detail": "", "t": 0.0}


def test_overlong_detail_is_truncated():
    out = gateway.normalize_stage_event({"stage": "x", "detail": "a" * 500})
    assert len(out["detail"]) == gateway.STAGE_DETAIL_MAX == 120


def test_non_dict_is_none():
    for raw in (None, "tool_start", 3, ["stage"], ("stage", 1)):
        assert gateway.normalize_stage_event(raw) is None


def test_empty_stage_name_is_none():
    for raw in ({}, {"stage": ""}, {"stage": "   "}, {"stage": 5}, {"detail": "no stage"}):
        assert gateway.normalize_stage_event(raw) is None


def test_alternate_keys_and_ms():
    out = gateway.normalize_stage_event({"name": "claimed", "message": "  picked   up ", "t_ms": 1500})
    assert out == {"stage": "claimed", "detail": "picked up", "t": 1.5}


def test_bad_time_falls_back_to_zero():
    for t in (-1, "soon", True, None):
        assert gateway.normalize_stage_event({"stage": "s", "t": t})["t"] == 0.0
