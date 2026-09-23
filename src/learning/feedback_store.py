"""
Immutable storage for STT correction feedback events.

A feedback event records one user correction: the original STT output and
the corrected text, kept as two separate fields, never merged or
overwritten. This module exposes no update or delete function on purpose --
and to make that a structural guarantee rather than just "nobody happened
to write one yet", the table itself carries SQLite triggers that abort any
UPDATE or DELETE against it. Derived data (correction_memory.py's
correction_rules table) can always be recomputed from these rows, but the
rows themselves are permanent history.

Nothing in this module is called from the hot transcribe path directly --
chat_web.py calls it only from the /api/feedback endpoint, when a user
actually submits a correction, and always through a try/except so a bug
here can't take down transcription.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

from . import db as learning_db
from .diffing import DiffOp, word_diff

_SCHEMA = """
CREATE TABLE IF NOT EXISTS feedback_events (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    event_uuid         TEXT NOT NULL UNIQUE,
    profile_id         TEXT NOT NULL DEFAULT 'local',
    session_id         TEXT NOT NULL DEFAULT 'unknown',
    original_text      TEXT NOT NULL,
    corrected_text     TEXT NOT NULL,
    diff_json          TEXT NOT NULL,
    context_before     TEXT,
    context_after      TEXT,
    audio_ref          TEXT,
    language           TEXT NOT NULL DEFAULT 'en',
    stt_confidence     REAL,
    stt_model_size     TEXT,
    stt_device         TEXT,
    validation_status  TEXT NOT NULL DEFAULT 'unvalidated',
    quality_meta_json  TEXT,
    created_at         TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_feedback_events_profile
    ON feedback_events(profile_id, created_at);
CREATE TRIGGER IF NOT EXISTS feedback_events_no_update
BEFORE UPDATE ON feedback_events
BEGIN SELECT RAISE(ABORT, 'feedback_events is immutable: updates are not allowed'); END;
CREATE TRIGGER IF NOT EXISTS feedback_events_no_delete
BEFORE DELETE ON feedback_events
BEGIN SELECT RAISE(ABORT, 'feedback_events is immutable: deletes are not allowed'); END;
"""


@dataclass(frozen=True)
class FeedbackEvent:
    event_uuid: str
    profile_id: str
    session_id: str
    original_text: str
    corrected_text: str
    diffs: list[DiffOp]
    context_before: str | None
    context_after: str | None
    audio_ref: str | None
    language: str
    stt_confidence: float | None
    stt_model_size: str | None
    stt_device: str | None
    validation_status: str
    quality_meta: dict = field(default_factory=dict)
    created_at: str = ""
    id: int = 0  # SQLite autoincrement rowid; a monotonic watermark the
    # training scheduler uses to track "new since last run" without
    # depending on created_at's timezone/precision.


def init_db(db_path: str) -> None:
    conn = learning_db.connect(db_path)
    try:
        conn.executescript(_SCHEMA)
        conn.commit()
    finally:
        conn.close()


def insert_feedback_event(
    db_path: str,
    *,
    profile_id: str,
    session_id: str,
    original_text: str,
    corrected_text: str,
    context_before: str | None = None,
    context_after: str | None = None,
    audio_ref: str | None = None,
    language: str = "en",
    stt_confidence: float | None = None,
    stt_model_size: str | None = None,
    stt_device: str | None = None,
    quality_meta: dict | None = None,
) -> FeedbackEvent:
    """Writes one immutable feedback row and returns it, diffs included.
    Raises ValueError if original_text/corrected_text are empty -- callers
    (chat_web.py) are expected to validate before calling, this is a
    last-resort guard against writing a useless row.
    """
    if not original_text.strip() or not corrected_text.strip():
        raise ValueError("original_text and corrected_text must both be non-empty")

    diffs = word_diff(original_text, corrected_text)
    event = FeedbackEvent(
        event_uuid=str(uuid.uuid4()),
        profile_id=profile_id,
        session_id=session_id,
        original_text=original_text,
        corrected_text=corrected_text,
        diffs=diffs,
        context_before=context_before,
        context_after=context_after,
        audio_ref=audio_ref,
        language=language,
        stt_confidence=stt_confidence,
        stt_model_size=stt_model_size,
        stt_device=stt_device,
        validation_status="unvalidated",
        quality_meta=quality_meta or {},
        created_at=datetime.now(timezone.utc).isoformat(),
    )

    init_db(db_path)
    conn = learning_db.connect(db_path)
    try:
        conn.execute(
            """
            INSERT INTO feedback_events (
                event_uuid, profile_id, session_id, original_text, corrected_text,
                diff_json, context_before, context_after, audio_ref, language,
                stt_confidence, stt_model_size, stt_device, validation_status,
                quality_meta_json, created_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                event.event_uuid,
                event.profile_id,
                event.session_id,
                event.original_text,
                event.corrected_text,
                json.dumps([asdict(d) for d in diffs]),
                event.context_before,
                event.context_after,
                event.audio_ref,
                event.language,
                event.stt_confidence,
                event.stt_model_size,
                event.stt_device,
                event.validation_status,
                json.dumps(event.quality_meta),
                event.created_at,
            ),
        )
        conn.commit()
    finally:
        conn.close()
    return event


def _row_to_event(row: sqlite3.Row) -> FeedbackEvent:
    diff_dicts = json.loads(row["diff_json"])
    return FeedbackEvent(
        event_uuid=row["event_uuid"],
        profile_id=row["profile_id"],
        session_id=row["session_id"],
        original_text=row["original_text"],
        corrected_text=row["corrected_text"],
        diffs=[DiffOp(**d) for d in diff_dicts],
        context_before=row["context_before"],
        context_after=row["context_after"],
        audio_ref=row["audio_ref"],
        language=row["language"],
        stt_confidence=row["stt_confidence"],
        stt_model_size=row["stt_model_size"],
        stt_device=row["stt_device"],
        validation_status=row["validation_status"],
        quality_meta=json.loads(row["quality_meta_json"] or "{}"),
        created_at=row["created_at"],
        id=row["id"],
    )


def query_feedback_events(
    db_path: str, *, profile_id: str | None = None, limit: int | None = None
) -> list[FeedbackEvent]:
    """Read-only. profile_id=None returns events across all profiles (used
    by rebuild_all_rules()); pass a specific profile_id to scope the query,
    which is how per-profile isolation of derived rules is guaranteed to
    match the underlying history.
    """
    init_db(db_path)
    conn = learning_db.connect(db_path)
    try:
        sql = "SELECT * FROM feedback_events"
        params: tuple = ()
        if profile_id is not None:
            sql += " WHERE profile_id = ?"
            params = (profile_id,)
        sql += " ORDER BY id ASC"
        if limit is not None:
            sql += " LIMIT ?"
            params = params + (limit,)
        rows = conn.execute(sql, params).fetchall()
    finally:
        conn.close()
    return [_row_to_event(r) for r in rows]
