#!/usr/bin/env python3
"""Runnable smoke test for GeminiLiveBackend (`gemini_live.py`, issue #69).

Connects to the Gemini Live WebSocket API (using `MVG_GEMINI_API_KEY` or a local
test server at `MVG_GEMINI_URL`), opens `GeminiLiveBackend`, sends a text prompt,
and prints received Talker events (`audio`, `text`, `tool_call`, `interrupted`,
`resume_handle`, `reconnect`, `turn_end`, `error`).

Includes `--test-silent-progress` to verify whether sending
`send_text("[task progress] step 1", silent=True)` (`turnComplete: false`)
while the model is speaking interrupts generation (`interrupted` event or
premature cutoff) or is absorbed silently as context.

Usage:
    MVG_GEMINI_API_KEY=... python3 tools/gemini_live_smoke.py
    MVG_GEMINI_API_KEY=... python3 tools/gemini_live_smoke.py --test-silent-progress
    MVG_GEMINI_API_KEY=... python3 tools/gemini_live_smoke.py --voice-mode custom_tts --prompt "Say hello."
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gemini_live import DEFAULT_MODEL, GeminiLiveBackend  # noqa: E402


def _format_event(ev: dict, t0: float) -> str:
    elapsed = time.monotonic() - t0
    kind = ev.get("type", "unknown")
    if kind == "audio":
        pcm = ev.get("pcm") or b""
        return f"[{elapsed:6.3f}s] event=audio bytes={len(pcm)}"
    if kind == "text":
        return f"[{elapsed:6.3f}s] event=text text={ev.get('text')!r} spoken={ev.get('spoken')}"
    if kind == "tool_call":
        return f"[{elapsed:6.3f}s] event=tool_call id={ev.get('call_id')} name={ev.get('name')} args={json.dumps(ev.get('args') or {})}"
    if kind == "resume_handle":
        handle = str(ev.get("handle") or "")
        preview = handle[:16] + "..." if len(handle) > 16 else handle
        return f"[{elapsed:6.3f}s] event=resume_handle handle={preview!r}"
    if kind == "reconnect":
        return f"[{elapsed:6.3f}s] event=reconnect time_left={ev.get('time_left')!r}"
    if kind == "error":
        return f"[{elapsed:6.3f}s] event=error message={ev.get('message')!r}"
    return f"[{elapsed:6.3f}s] event={kind} raw={json.dumps(ev, default=str)}"


async def run_smoke(
    prompt: str,
    *,
    voice_mode: str = "native",
    instructions: str | None = None,
    timeout: float = 20.0,
    test_silent_progress: bool = False,
    silent_text: str = "[task progress] step 1",
    connect=None,
) -> dict:
    """Execute one smoke session against GeminiLiveBackend and return a summary dict."""
    backend = GeminiLiveBackend(connect=connect)
    cfg: dict = {"voice_mode": voice_mode}
    if instructions:
        cfg["instructions"] = instructions

    t0 = time.monotonic()
    events: list[dict] = []
    audio_bytes = 0
    interrupted = False
    silent_injected = False
    audio_chunks_after_silent = 0
    text_chunks_after_silent = 0

    await backend.open(cfg)
    try:
        await backend.send_text(prompt, silent=False)

        async def _consume():
            nonlocal audio_bytes, interrupted, silent_injected
            nonlocal audio_chunks_after_silent, text_chunks_after_silent
            async for ev in backend.events():
                events.append(ev)
                print(_format_event(ev, t0), flush=True)
                kind = ev.get("type")
                if kind == "audio":
                    audio_bytes += len(ev.get("pcm") or b"")
                    if silent_injected:
                        audio_chunks_after_silent += 1
                elif kind == "text":
                    if silent_injected:
                        text_chunks_after_silent += 1
                elif kind == "interrupted":
                    interrupted = True

                # Inject silent progress right after the model begins speaking (first audio/text chunk)
                if test_silent_progress and not silent_injected and kind in ("audio", "text"):
                    silent_injected = True
                    elapsed = time.monotonic() - t0
                    print(
                        f"[{elapsed:6.3f}s] >>> injecting send_text({silent_text!r}, silent=True) "
                        "(turnComplete: false) while model is generating",
                        flush=True,
                    )
                    await backend.send_text(silent_text, silent=True)

                if kind in ("turn_end", "error"):
                    break

        await asyncio.wait_for(_consume(), timeout=timeout)
    finally:
        await backend.close()

    summary = {
        "model": os.environ.get("MVG_GEMINI_MODEL", DEFAULT_MODEL),
        "voice_mode": voice_mode,
        "event_count": len(events),
        "event_types": [e.get("type") for e in events],
        "audio_bytes": audio_bytes,
        "interrupted": interrupted,
        "test_silent_progress": test_silent_progress,
        "silent_injected": silent_injected,
        "audio_chunks_after_silent": audio_chunks_after_silent,
        "text_chunks_after_silent": text_chunks_after_silent,
        "turn_completed": any(e.get("type") == "turn_end" for e in events),
        "errors": [e.get("message") for e in events if e.get("type") == "error"],
    }
    if test_silent_progress:
        verdict = (
            "INTERRUPTED" if interrupted
            else ("NO_GENERATION_SEEN" if not silent_injected else "CONTINUES_CLEANLY")
        )
        summary["silent_progress_verdict"] = verdict
        print(
            f"[smoke] silent-progress check verdict={verdict} "
            f"(silent_injected={silent_injected}, interrupted={interrupted}, "
            f"post_silent_audio_chunks={audio_chunks_after_silent}, "
            f"post_silent_text_chunks={text_chunks_after_silent})",
            flush=True,
        )
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Smoke test for GeminiLiveBackend (requires MVG_GEMINI_API_KEY or MVG_GEMINI_URL)."
    )
    p.add_argument(
        "--prompt",
        default="Hello! Please count slowly from one to five.",
        help="Prompt text sent via send_text(prompt, silent=False).",
    )
    p.add_argument(
        "--voice-mode",
        choices=("native", "custom_tts"),
        default="native",
        help="Talker voice_mode passed to backend.open(config).",
    )
    p.add_argument(
        "--instructions",
        default=None,
        help="Optional system instructions override.",
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=20.0,
        help="Timeout in seconds waiting for turn_end (default: 20).",
    )
    p.add_argument(
        "--test-silent-progress",
        action="store_true",
        help=(
            "Send send_text('[task progress] step 1', silent=True) as soon as the model "
            "emits its first audio/text chunk and verify whether generation is interrupted."
        ),
    )
    p.add_argument(
        "--silent-text",
        default="[task progress] step 1",
        help="Text sent for --test-silent-progress (default: '[task progress] step 1').",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not os.environ.get("MVG_GEMINI_API_KEY") and not os.environ.get("MVG_GEMINI_URL"):
        print(
            "error: MVG_GEMINI_API_KEY (or MVG_GEMINI_URL) must be set to run gemini_live_smoke.py",
            file=sys.stderr,
        )
        return 2
    try:
        summary = asyncio.run(
            run_smoke(
                args.prompt,
                voice_mode=args.voice_mode,
                instructions=args.instructions,
                timeout=args.timeout,
                test_silent_progress=args.test_silent_progress,
                silent_text=args.silent_text,
            )
        )
    except Exception as exc:  # noqa: BLE001
        print(f"error: smoke test failed: {exc}", file=sys.stderr)
        return 1

    print(f"[smoke] summary: {json.dumps(summary)}", flush=True)
    if summary["errors"] or not summary["turn_completed"]:
        return 1
    if args.test_silent_progress and summary.get("silent_progress_verdict") != "CONTINUES_CLEANLY":
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
