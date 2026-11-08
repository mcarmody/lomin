#!/usr/bin/env python3
"""Summarise logs/voice-latency.jsonl: count and p50/p90 of first_audio_ms and
total_ms for `kind == "turn"` records, split by `opener_kind` when present.

Usage: python3 tools/latency_summary.py PATH
Stdlib only. Nearest-rank percentiles. Records missing a field are skipped for
that field, never fatal."""
import json
import math
import sys

FIELDS = ("first_audio_ms", "total_ms")


def percentile(sorted_vals: list, p: float):
    """Nearest-rank percentile of an ascending list; None if empty."""
    if not sorted_vals:
        return None
    rank = max(1, math.ceil(p / 100.0 * len(sorted_vals)))
    return sorted_vals[rank - 1]


def load_turns(path: str) -> list:
    turns = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if isinstance(rec, dict) and rec.get("kind") == "turn":
                turns.append(rec)
    return turns


def _num(v):
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def summarize(turns: list) -> list:
    """Lines of text for a set of turn records."""
    out = [f"{len(turns)} turns"]
    if not turns:
        return out
    for field in FIELDS:
        vals = sorted(v for v in (_num(t.get(field)) for t in turns) if v is not None)
        if not vals:
            out.append(f"  {field}: no data")
            continue
        out.append(f"  {field}: n={len(vals)} p50={percentile(vals, 50):g} p90={percentile(vals, 90):g}")
    return out


def report(turns: list) -> str:
    lines = summarize(turns)
    kinds = sorted({str(t["opener_kind"]) for t in turns if t.get("opener_kind") is not None})
    for kind in kinds:
        lines.append(f"opener_kind={kind}")
        lines += summarize([t for t in turns if str(t.get("opener_kind")) == kind])
    return "\n".join(lines)


def main(argv: list) -> int:
    if len(argv) != 2:
        print(__doc__.strip().splitlines()[2], file=sys.stderr)
        return 2
    print(report(load_turns(argv[1])))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
