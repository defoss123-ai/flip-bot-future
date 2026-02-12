"""Helpers for signing MEXC Futures API requests."""

from __future__ import annotations

import hashlib
import hmac
import time
from urllib.parse import urlencode


def build_query(params: dict[str, object]) -> str:
    """Build deterministic query string without None values."""
    filtered = {k: v for k, v in params.items() if v is not None}
    return urlencode(sorted(filtered.items()), doseq=True)


def sign_payload(secret: str, payload: str) -> str:
    """Return HMAC SHA256 hex digest for payload."""
    return hmac.new(secret.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).hexdigest()


def signed_query(params: dict[str, object], api_secret: str, use_millis: bool = True) -> tuple[str, str]:
    """Create query string + signature with timestamp field included."""
    ts = int(time.time() * 1000) if use_millis else int(time.time())
    all_params = {**params, "timestamp": ts}
    query = build_query(all_params)
    signature = sign_payload(api_secret, query)
    return query, signature
