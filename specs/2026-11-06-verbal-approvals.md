# Verbal approvals for agent prompts

Status: draft for review. No code is written until this is reviewed and merged.
Context: section 6.5 of the Talker + Thinker live architecture spec defers this. `answer_prompt` in the live harness is a stub that returns `{"status": "unsupported"}`.

## 1. Goal and non-goals

Goal: let a person approve or deny a pending agent prompt by voice, safely, for low-risk prompts only.

Non-goals: approving anything that changes money, access, deletion or external sends; replacing out-of-band confirmation; wake-word or speaker enrolment design (this spec only states what it requires from them).

## 2. Allowlist (the main control)

- The brain marks every prompt with `risk`: `"low"`, `"medium"` or `"high"`. Missing or unknown means `"high"`.
- Only `risk == "low"` can be approved by voice. Anything else is answered with a spoken instruction to confirm out of band (the existing approval channel), and the voice session records `voice_refused: out_of_band_required`.
- Denial by voice is always allowed, for every risk level. Denying is never the dangerous direction.
- A pattern check over `command` (destructive verbs, `rm`, `drop`, force flags, pipes to a shell, writes outside the working area, network egress) runs as a second layer. It can only downgrade a prompt to "needs out-of-band"; it can never upgrade one. The brain's `risk` is the allowlist; the regex is a net.
- The allowlist and the pattern list are configuration (`MVG_VOICE_APPROVAL=off|on`, default `off`), reviewed like code.

## 3. Prompt identity and read-back

- Each approval binds to one `prompt_id`. A decision without a matching live `prompt_id` is rejected.
- Before a decision counts, the system speaks a read-back of `summary` (never the raw command) and waits for an explicit decision phrase that was spoken after the read-back finished. Audio captured during the read-back does not count.
- A prompt expires after `MVG_VOICE_APPROVAL_TTL` seconds (default 60). An expired prompt cannot be approved by voice.
- One decision per prompt. A second decision for the same `prompt_id` is ignored and logged.
- Decision vocabulary is closed ("approve", "deny", "repeat"). "Repeat" re-reads the summary and restarts the clock. Anything else is not a decision.

## 4. Speaker verification

- Required, not optional: a voice approval counts only when the speaker-verification score for the deciding utterance meets `MVG_VOICE_APPROVAL_MIN_SCORE` and the speaker is on the approver list for that prompt's agent.
- If speaker verification is not configured, voice approval is unavailable and the feature stays off. There is no degraded mode.
- Ambient and echoed speech: the deciding utterance must start after read-back audio ended (a guard interval, default 500 ms), must come from the node that played the read-back, and must not match the node's own recent playback (echo check against the played audio). Utterances with low speech confidence or multiple overlapping speakers are discarded.

## 5. Event schema and delivery

Brain to gateway, one event per pending prompt:

```json
{"prompt_id": "p-123", "summary": "Restart the build worker", "command": "systemctl restart build-worker", "risk": "low"}
```

- Delivery is a new plugin hook, `pending_prompt(event) -> None`, called by the gateway when the brain stream emits a prompt event. Decisions go back through a second hook, `prompt_decision(prompt_id, decision, meta) -> None`, where `meta` carries node, speaker, score and timestamps.
- This is a new subsystem. It does not extend `pending_and_auth.py`, which handles pending messages and device auth; the two share no state.
- The live harness `answer_prompt(prompt_id, decision)` tool calls `prompt_decision` only after sections 2 to 4 pass; otherwise it returns `{"status": "refused", "reason": ...}`.
- Every prompt, read-back, decision and refusal is written to the request log (ids and outcomes, no audio, no command text).

## 6. Abuse cases

| Case | Defence |
|---|---|
| A television or radio says "yes" | Speaker verification fails; echo check; read-back guard interval; closed vocabulary |
| A guest in the room | Not on the approver list; verification fails; denial still allowed |
| Replay of a recorded approval | Decision must follow a fresh read-back with a unique `prompt_id` and an expiry; liveness is not guaranteed by audio alone, so only `risk: low` prompts are eligible |
| Synthetic voice of the approver | Same limit: the allowlist bounds the damage to low-risk actions; the feature is off by default |
| Approver speaks while another prompt is live | Decision binds only to the prompt read back last; others stay pending |
| Brain mislabels a risky prompt as low | Pattern layer can downgrade; this residual risk is why high-impact actions are never `low` by policy |
| Replay of an old `prompt_id` | Expired or already decided prompts are rejected |

## 7. Test plan

Positive: low-risk prompt, read-back, verified approver says "approve" after read-back: `prompt_decision` called once.

Negative (each must not call `prompt_decision` with an approval):
- `risk` missing, `medium`, `high` or unknown.
- Pattern layer hits on a prompt marked `low`.
- Verification off, score below threshold, speaker not on the approver list.
- Utterance starts during read-back or inside the guard interval.
- Utterance matches played audio (echo).
- Wrong or expired `prompt_id`; second decision on the same prompt.
- Phrase outside the closed vocabulary; "yes" without "approve".
- Utterance from a different node than the read-back.
- `MVG_VOICE_APPROVAL=off`.

Also: denial works for every risk level; "repeat" restarts the TTL; hooks are patched in tests; no network and no keys.

## 8. Rollout

1. Spec reviewed and merged (this document).
2. Hooks and gateway plumbing behind `MVG_VOICE_APPROVAL=off`, with the negative tests.
3. Enable per node for one approver in a supervised trial; review logged refusals before widening.

## 9. Open questions

- Where the approver list lives (per agent in config versus supplied by the plugin).
- Whether a stricter second factor (a spoken code word) should be required for the first rollout.
- How speaker-verification scores are exposed to the gateway today.
