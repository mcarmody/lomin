"""tools/latency_summary.py: nearest-rank percentiles over voice-latency.jsonl."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import latency_summary as ls  # noqa: E402


def _write(path, recs, raw=()):
    path.write_text("\n".join([json.dumps(r) for r in recs] + list(raw)) + "\n")


def test_empty_file_prints_zero_turns(tmp_path):
    p = tmp_path / "l.jsonl"
    p.write_text("")
    out = subprocess.run([sys.executable, str(ROOT / "tools" / "latency_summary.py"), str(p)],
                         capture_output=True, text=True)
    assert out.returncode == 0 and out.stdout.strip() == "0 turns"


def test_percentiles_nearest_rank():
    vals = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    assert ls.percentile(vals, 50) == 500   # rank ceil(0.5*10) = 5
    assert ls.percentile(vals, 90) == 900   # rank 9
    assert ls.percentile([7], 90) == 7
    assert ls.percentile([], 50) is None


def test_report_counts_and_skips_bad_records(tmp_path):
    p = tmp_path / "l.jsonl"
    recs = [{"kind": "turn", "first_audio_ms": 1000 * i, "total_ms": 2000 * i} for i in range(1, 11)]
    recs.append({"kind": "stt", "stt_ms": 5})
    recs.append({"kind": "turn", "first_audio_ms": None, "total_ms": "n/a"})
    _write(p, recs, raw=["not json"])
    text = ls.report(ls.load_turns(str(p)))
    assert text.splitlines()[0] == "11 turns"
    assert "first_audio_ms: n=10 p50=5000 p90=9000" in text
    assert "total_ms: n=10 p50=10000 p90=18000" in text


def test_split_by_opener_kind(tmp_path):
    p = tmp_path / "l.jsonl"
    _write(p, [{"kind": "turn", "first_audio_ms": 100, "total_ms": 900, "opener_kind": "stage"},
               {"kind": "turn", "first_audio_ms": 300, "total_ms": 1100, "opener_kind": "stage"},
               {"kind": "turn", "first_audio_ms": 1200, "total_ms": 4000, "opener_kind": "gap_fill"},
               {"kind": "turn", "first_audio_ms": 1500, "total_ms": 5000}])
    text = ls.report(ls.load_turns(str(p)))
    assert "opener_kind=stage\n2 turns" in text
    assert "opener_kind=gap_fill\n1 turns" in text
    assert text.splitlines()[0] == "4 turns"
