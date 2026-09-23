"""
Local, no-auth profile/session identity.

This project is a single-machine, offline application with no login and no
multi-tenancy -- adding real authentication would be pure overhead for what
it's used for. What personalized correction actually needs is much smaller:
a stable string that keeps one person's corrections separate from anyone
else's who might use the same install (e.g. two family members sharing a
PC), and a per-conversation string to group corrections that happened
together. Both are just opaque identifiers the client hands us; nothing
here authenticates them.

The client (src/web_chat/index.html) generates and persists a profile id in
localStorage (survives across tabs/restarts) and a session id in
sessionStorage (fresh per tab), and sends both on /api/transcribe and
/api/feedback. Any caller that omits them -- curl, tests, a future CLI
integration -- gets the defaults below, so nothing breaks.
"""

from __future__ import annotations

DEFAULT_PROFILE_ID = "local"
DEFAULT_SESSION_ID = "unknown"

# Generous but bounded -- this is a partition key stored in SQLite text
# columns and echoed back in JSON, not a security boundary.
_MAX_ID_LENGTH = 128


def _resolve(raw: str | None, default: str) -> str:
    if raw is None:
        return default
    cleaned = raw.strip()
    if not cleaned:
        return default
    return cleaned[:_MAX_ID_LENGTH]


def resolve_profile_id(raw: str | None) -> str:
    """None / empty / whitespace-only -> DEFAULT_PROFILE_ID. Otherwise the
    trimmed, length-capped value the client sent, unchanged. This is the
    single place that defines what counts as a valid profile id -- every
    other module in this package should go through here rather than
    re-deriving the same defaulting logic.
    """
    return _resolve(raw, DEFAULT_PROFILE_ID)


def resolve_session_id(raw: str | None) -> str:
    return _resolve(raw, DEFAULT_SESSION_ID)
