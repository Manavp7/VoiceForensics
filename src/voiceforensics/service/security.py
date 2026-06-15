"""Security primitives: SSRF guard, API-key auth, and rate limiting."""

from __future__ import annotations

import ipaddress
import socket
import threading
import time
from urllib.parse import urlparse

from voiceforensics.config import Settings, get_settings


class SSRFError(Exception):
    """Raised when a URL resolves to a disallowed (private/internal) address."""


def _ip_is_disallowed(ip: str) -> bool:
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return True
    return (
        addr.is_private
        or addr.is_loopback
        or addr.is_link_local
        or addr.is_reserved
        or addr.is_multicast
        or addr.is_unspecified
    )


def is_safe_url(url: str, *, settings: Settings | None = None) -> bool:
    """Return True if ``url`` is an http(s) URL that does not resolve to a private/
    internal address (mitigates SSRF). Honours ``allow_private_url_fetch`` (dev only).
    """
    settings = settings or get_settings()
    if settings.allow_private_url_fetch:
        return True
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return False
    host = parsed.hostname
    if not host:
        return False
    # Block obvious metadata/loopback hostnames quickly.
    if host in {"localhost", "metadata.google.internal"}:
        return False
    try:
        infos = socket.getaddrinfo(host, parsed.port or 80, proto=socket.IPPROTO_TCP)
    except socket.gaierror:
        return False
    for info in infos:
        ip = info[4][0]
        if _ip_is_disallowed(ip):
            return False
    return True


def assert_safe_url(url: str, *, settings: Settings | None = None) -> None:
    if not is_safe_url(url, settings=settings):
        raise SSRFError(f"refusing to fetch disallowed/internal URL: {url}")


class RateLimiter:
    """Simple in-memory token-bucket rate limiter keyed by an identity string."""

    def __init__(self, per_minute: int):
        self.capacity = float(per_minute)
        self.refill_per_sec = per_minute / 60.0
        self._buckets: dict[str, tuple[float, float]] = {}  # key -> (tokens, last_ts)
        self._lock = threading.Lock()

    def allow(self, key: str, cost: float = 1.0) -> bool:
        now = time.monotonic()
        with self._lock:
            tokens, last = self._buckets.get(key, (self.capacity, now))
            tokens = min(self.capacity, tokens + (now - last) * self.refill_per_sec)
            if tokens >= cost:
                self._buckets[key] = (tokens - cost, now)
                return True
            self._buckets[key] = (tokens, now)
            return False


def validate_api_key(raw_key: str | None, settings: Settings | None = None):
    """Return the ApiKey row if valid+active, else None."""
    if not raw_key:
        return None
    from voiceforensics.service.db import ApiKey, get_sessionmaker, init_db

    init_db(settings)
    Session = get_sessionmaker(settings)
    with Session() as session:
        key = session.query(ApiKey).filter_by(key=raw_key, active=1).one_or_none()
        return key
