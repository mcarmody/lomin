"""pending_and_auth.py — Deferred Async Answers & Per-Device Bearer Auth.

Provides:
- BearerTokenVerifier:
    Verifies HTTP 'Authorization: Bearer <token>' headers using constant-time
    comparison (hmac.compare_digest). Supports single shared token or
    per-device mapping (node1:tok1,node2:tok2[,shared_fallback]) from MVG_DEVICE_TOKENS.
    Supports shared tokens containing ':' via 'shared:tok' or '*:tok' prefixes or quotes.
    Enforces strict 'Bearer ' prefix in Authorization headers.
    Provides failed-authentication rate limiting and lockout per client IP.
    Disabled when MVG_DEVICE_TOKENS is empty or unset.
- is_defer_enabled() -> bool:
    Checks if MVG_ENABLE_DEFER is enabled.
- has_defer_marker(text: str) -> bool:
    Detects if '[[defer]]' marker is present in text.
- strip_defer_marker(text: str) -> str:
    Strips '[[defer]]' marker(s) without collapsing internal whitespace or newlines.
- PendingVoiceAnswerStore:
    Thread-safe in-memory store for deferred asynchronous voice replies.
    Supports store(node_id, prompt, answer, topic=''), pop_for_node(node_id),
    and peek_for_node(node_id). Bounded by max_nodes and max_answer_chars.
"""

from __future__ import annotations

import collections
import hmac
import os
import re
import threading
import time
import uuid
from typing import Any

# Regex to match [[defer]] case-insensitively with optional internal whitespace
_DEFER_RE = re.compile(r"\[\[\s*defer\s*\]\]", re.IGNORECASE)


def is_defer_enabled() -> bool:
    """Return True if deferred async voice answers are enabled."""
    return os.environ.get("MVG_ENABLE_DEFER", "0").strip().lower() in ("1", "true", "yes", "on")


def has_defer_marker(text: str | None) -> bool:
    """Return True if text contains the [[defer]] marker."""
    if not text or not isinstance(text, str):
        return False
    return bool(_DEFER_RE.search(text))


def strip_defer_marker(text: str | None) -> str:
    """Strip [[defer]] marker(s) from text without collapsing internal whitespace or newlines."""
    if not text or not isinstance(text, str):
        return ""
    if not has_defer_marker(text):
        return text

    def _repl(m: re.Match[str]) -> str:
        start, end = m.start(), m.end()
        left_char = text[start - 1] if start > 0 else ""
        right_char = text[end] if end < len(text) else ""
        if left_char and left_char not in " \t\r\n" and right_char and right_char not in " \t\r\n":
            return " "
        return ""

    res = re.sub(r"[ \t]*\[\[\s*defer\s*\]\][ \t]*", _repl, text, flags=re.IGNORECASE)
    if not res.strip():
        return ""
    return res


class BearerTokenVerifier:
    """Verifies Bearer tokens using constant-time hmac.compare_digest.

    Configuration can be passed directly as a tokens_str or read dynamically from
    MVG_DEVICE_TOKENS environment variable.
    Format: 'node1:tok1,node2:tok2' or single shared token 'shared_secret',
    or mixed 'node1:tok1,shared:tok_with:colons', or '*:tok_with:colons',
    or quoted '"tok:with:colons"'.
    """

    def __init__(
        self,
        tokens_str: str | None = None,
        max_failures: int | None = None,
        lockout_sec: float | None = None,
    ) -> None:
        self._tokens_str = tokens_str
        self._max_failures = max_failures
        self._lockout_sec = lockout_sec
        self._failures: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def _get_max_failures(self) -> int:
        if self._max_failures is not None:
            return self._max_failures
        try:
            return int(os.environ.get("MVG_AUTH_MAX_FAILURES", "5"))
        except (ValueError, TypeError):
            return 5

    def _get_lockout_sec(self) -> float:
        if self._lockout_sec is not None:
            return self._lockout_sec
        try:
            return float(os.environ.get("MVG_AUTH_LOCKOUT_SEC", "60.0"))
        except (ValueError, TypeError):
            return 60.0

    def is_locked_out(self, client_id: str) -> bool:
        """Return True if client_id is currently locked out due to excessive failed attempts."""
        if not client_id:
            return False
        with self._lock:
            now = time.time()
            lockout = self._get_lockout_sec()
            timestamps = [t for t in self._failures.get(client_id, []) if now - t < lockout]
            self._failures[client_id] = timestamps
            return len(timestamps) >= self._get_max_failures()

    def record_failure(self, client_id: str) -> None:
        """Record a failed auth attempt for client_id."""
        if not client_id:
            return
        with self._lock:
            now = time.time()
            lockout = self._get_lockout_sec()
            timestamps = [t for t in self._failures.get(client_id, []) if now - t < lockout]
            timestamps.append(now)
            self._failures[client_id] = timestamps

    def record_success(self, client_id: str) -> None:
        """Record a successful auth attempt, clearing previous failures for client_id."""
        if not client_id:
            return
        with self._lock:
            self._failures.pop(client_id, None)

    def reset_failures(self) -> None:
        """Clear all failure records."""
        with self._lock:
            self._failures.clear()

    def _get_raw(self) -> str:
        if self._tokens_str is not None:
            return self._tokens_str.strip()
        return os.environ.get("MVG_DEVICE_TOKENS", "").strip()

    @property
    def is_enabled(self) -> bool:
        """Returns True if tokens are configured, False if empty/disabled."""
        return bool(self._get_raw())

    def _parse_tokens(self) -> tuple[dict[str, str], str | None]:
        raw = self._get_raw()
        if not raw:
            return {}, None
        per_node: dict[str, str] = {}
        shared_fallback: str | None = None
        for part in raw.split(","):
            part = part.strip()
            if not part:
                continue
            # Handle quoted tokens: "my:token:with:colons"
            if (part.startswith('"') and part.endswith('"')) or (part.startswith("'") and part.endswith("'")):
                shared_fallback = part[1:-1].strip()
                continue
            if part.lower().startswith("shared="):
                shared_fallback = part[7:].strip()
                continue
            if ":" in part:
                node, tok = part.split(":", 1)
                node_clean = node.strip()
                tok_clean = tok.strip()
                if node_clean.lower() in ("shared", "*", "default", "_"):
                    shared_fallback = tok_clean
                elif node_clean and tok_clean:
                    per_node[node_clean] = tok_clean
            else:
                shared_fallback = part.strip()
        return per_node, shared_fallback

    def get_expected_token(self, node_id: str = "") -> str | None:
        """Resolve expected token for the given node_id, or shared fallback."""
        per_node, shared_fallback = self._parse_tokens()
        if node_id and node_id in per_node:
            return per_node[node_id]
        return shared_fallback

    @staticmethod
    def _extract_token(auth_header: str | None) -> str | None:
        if not auth_header or not isinstance(auth_header, str):
            return None
        h = auth_header.strip()
        if not h:
            return None
        if h.lower().startswith("bearer "):
            tok = h[7:].strip()
            return tok if tok else None
        return None

    def verify(self, auth_header: str | None, node_id: str = "") -> bool:
        """Verify an Authorization header in constant time.

        If disabled (no tokens configured), always returns True.
        If enabled, extracts Bearer token and verifies using hmac.compare_digest.
        """
        if not self.is_enabled:
            return True

        actual_token = self._extract_token(auth_header)
        expected_token = self.get_expected_token(node_id)

        # Constant-time comparison to prevent timing leak
        if actual_token is None or expected_token is None:
            dummy_a = actual_token.encode("utf-8") if actual_token else b"dummy_pad_token_value_a"
            dummy_b = b"dummy_pad_token_value_b"
            hmac.compare_digest(dummy_a, dummy_b)
            return False

        return hmac.compare_digest(actual_token.encode("utf-8"), expected_token.encode("utf-8"))

    def verify_token(self, token: str | None, node_id: str = "") -> bool:
        """Verify a raw token string against expected token for node_id."""
        return self.verify(f"Bearer {token}" if token else None, node_id)


class PendingVoiceAnswerStore:
    """Thread-safe store for deferred voice answers.

    Holds pending responses per node_id until popped or peeked by client.
    Bounds maximum entries per node, maximum nodes, and maximum answer size.
    """

    def __init__(
        self,
        max_per_node: int = 50,
        max_nodes: int = 1000,
        max_answer_chars: int = 65536,
    ) -> None:
        self._lock = threading.Lock()
        self._store: dict[str, collections.deque[dict[str, Any]]] = {}
        self._max_per_node = max(1, max_per_node)
        self._max_nodes = max(1, max_nodes)
        self._max_answer_chars = max(1, max_answer_chars)

    def store(self, node_id: str, prompt: str, answer: str, topic: str = "") -> dict[str, Any]:
        """Store a pending answer for node_id and return the stored record."""
        clean_node = (node_id or "").strip()
        clean_answer = str(answer or "")[:self._max_answer_chars]
        record: dict[str, Any] = {
            "id": str(uuid.uuid4()),
            "node_id": clean_node,
            "prompt": str(prompt or ""),
            "answer": clean_answer,
            "topic": str(topic or ""),
            "created_at": time.time(),
        }
        with self._lock:
            if clean_node not in self._store and len(self._store) >= self._max_nodes:
                oldest_node = next(iter(self._store))
                self._store.pop(oldest_node, None)
            q = self._store.setdefault(clean_node, collections.deque(maxlen=self._max_per_node))
            q.append(record)
        return dict(record)

    def pop_for_node(self, node_id: str) -> dict[str, Any] | None:
        """Pop and return the oldest pending answer for node_id, or None if empty."""
        clean_node = (node_id or "").strip()
        with self._lock:
            q = self._store.get(clean_node)
            if not q:
                return None
            item = q.popleft()
            if not q:
                self._store.pop(clean_node, None)
            return dict(item)

    def peek_for_node(self, node_id: str) -> dict[str, Any] | None:
        """Return the oldest pending answer for node_id without removing it, or None."""
        clean_node = (node_id or "").strip()
        with self._lock:
            q = self._store.get(clean_node)
            if not q:
                return None
            return dict(q[0])

    def get(self, node_id: str, record_id: str, max_age_sec: float | None = None) -> dict[str, Any] | None:
        """Return the record with this id for node_id without removing it, or None.
        None too when it is older than max_age_sec (when given)."""
        clean_node = (node_id or "").strip()
        clean_id = (record_id or "").strip()
        if not clean_node or not clean_id:
            return None
        with self._lock:
            for item in self._store.get(clean_node, ()):
                if item["id"] == clean_id:
                    if max_age_sec is not None and time.time() - item["created_at"] > max_age_sec:
                        return None
                    return dict(item)
        return None

    def purge(self, max_age_sec: float) -> int:
        """Drop every record older than max_age_sec, for every node. Returns how many were dropped."""
        dropped = 0
        with self._lock:
            for node in list(self._store):
                q = self._store[node]
                fresh = [i for i in q if time.time() - i["created_at"] <= max_age_sec]
                dropped += len(q) - len(fresh)
                if fresh:
                    self._store[node] = collections.deque(fresh, maxlen=self._max_per_node)
                else:
                    del self._store[node]
        return dropped

    def count_for_node(self, node_id: str) -> int:
        """Return the count of pending answers for node_id."""
        clean_node = (node_id or "").strip()
        with self._lock:
            return len(self._store.get(clean_node, ()))

    def clear(self, node_id: str = "") -> None:
        """Clear store for node_id or entire store if node_id is empty."""
        clean_node = (node_id or "").strip()
        with self._lock:
            if clean_node:
                self._store.pop(clean_node, None)
            else:
                self._store.clear()


# Default process-wide singletons
_DEFAULT_STORE = PendingVoiceAnswerStore()
_DEFAULT_VERIFIER = BearerTokenVerifier()


def get_pending_store() -> PendingVoiceAnswerStore:
    """Return the global PendingVoiceAnswerStore singleton."""
    return _DEFAULT_STORE


# Replies kept for a client that went away (gateway.py keep_reply_when_done). A second instance, so the
# default store, which /voice/pending pops by node, never hands one out.
_KEPT_STORE = PendingVoiceAnswerStore(max_per_node=20)


def get_kept_store() -> PendingVoiceAnswerStore:
    """Return the store of replies kept for a client that went away."""
    return _KEPT_STORE


def get_device_token_verifier() -> BearerTokenVerifier:
    """Return the global BearerTokenVerifier singleton."""
    return _DEFAULT_VERIFIER
