"""MVG_ORCHESTRATOR_PROMPT adds a generic orchestrator block to the framed prompt (off by default)."""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SNIPPET = ("import gateway as g; "
           "c = g.make_frame_ctx({'status': 'not_verified', 'id': '', 'score': 0.2}, target='t1'); "
           "print(g.frame_voice_prompt('hi', 'n1', c))")


def _frame(extra_env):
    env = {k: v for k, v in os.environ.items() if not k.startswith("MVG_")}
    env.update({"MVG_LOG_DIR": "/tmp", **extra_env})
    return subprocess.run([sys.executable, "-c", SNIPPET], cwd=ROOT, env=env,
                          capture_output=True, text=True, check=True).stdout


def test_off_by_default():
    assert "voice orchestrator" not in _frame({})


def test_on_adds_generic_block():
    f = _frame({"MVG_ORCHESTRATOR_PROMPT": "1"})
    assert "You are the voice orchestrator" in f
    assert f.index("under 60 words") < f.index("voice orchestrator")


def test_site_text_replaces_default():
    f = _frame({"MVG_ORCHESTRATOR_PROMPT": "1", "MVG_ORCHESTRATOR_TEXT": "Delegate everything."})
    assert "Delegate everything." in f and "voice orchestrator" not in f


def test_target_filter():
    assert "voice orchestrator" in _frame({"MVG_ORCHESTRATOR_PROMPT": "1", "MVG_ORCHESTRATOR_TARGETS": "t1,t2"})
    assert "voice orchestrator" not in _frame({"MVG_ORCHESTRATOR_PROMPT": "1", "MVG_ORCHESTRATOR_TARGETS": "other"})
