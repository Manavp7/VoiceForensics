"""Signed webhook delivery with retries."""

from __future__ import annotations

import hashlib
import hmac
import json
import time

import httpx

from voiceforensics.config import Settings, get_settings
from voiceforensics.service.security import is_safe_url


def sign_payload(body: bytes, secret: str) -> str:
    return "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def deliver(url: str, payload: dict, *, settings: Settings | None = None) -> bool:
    """POST ``payload`` (JSON) to ``url`` with an HMAC signature header and retries.

    Returns True on a 2xx response. The webhook URL is SSRF-checked first.
    """
    settings = settings or get_settings()
    if not is_safe_url(url, settings=settings):
        return False

    body = json.dumps(payload, default=str).encode()
    headers = {"Content-Type": "application/json"}
    if settings.webhook_secret:
        headers["X-VoiceForensics-Signature"] = sign_payload(body, settings.webhook_secret)

    delay = 0.5
    for attempt in range(1, settings.webhook_max_retries + 1):
        try:
            with httpx.Client(timeout=settings.webhook_timeout_s) as client:
                resp = client.post(url, content=body, headers=headers)
            if 200 <= resp.status_code < 300:
                return True
        except httpx.HTTPError:
            pass
        if attempt < settings.webhook_max_retries:
            time.sleep(delay)
            delay *= 2
    return False
