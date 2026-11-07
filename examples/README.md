# Reference Plugin

Minimal neutral plugin (`examples/reference_plugin.py`) using Python stdlib only.

## Running

```sh
REFERENCE_BRAIN_URL=http://127.0.0.1:8000/brain \
REFERENCE_TALKER_INSTRUCTIONS="Speak concisely." \
MVG_PLUGIN=reference_plugin PYTHONPATH=examples python3 gateway.py
```

## Hooks

- `brain_call_stream(target, prompt, timeout, on_partial, stages_out, audio_clips=None, task_id=None)` — POSTs `{"target": target, "prompt": prompt}` to `REFERENCE_BRAIN_URL`, tracks open responses by `task_id` in `_INFLIGHT`, calls `on_partial(text)` once, and returns the response text.
- `brain_cancel(target, task_id)` — closes the open response in `_INFLIGHT[task_id]` if present and returns `True`, else `False`.
- `talker_instructions(target, node_id)` — returns the `REFERENCE_TALKER_INSTRUCTIONS` environment variable (or `""` if unset).
- `try_intent(text)` — deterministic fast-path stub that always returns `None`.
