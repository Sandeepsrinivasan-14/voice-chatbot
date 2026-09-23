"""
Derived, rebuildable per-profile correction rules.

A "rule" here is just a tally: this profile corrected phrase X to phrase Y,
N times, most recently at T, usually surrounded by these words. It is
*derived* from feedback_events (feedback_store.py) -- never the other way
around -- so the correction_rules table can always be safely deleted and
rebuilt from scratch without losing any history. That's not a convention
this module has to remember to honor: feedback_events is physically
un-writable outside INSERT (see feedback_store.py's triggers), so there is
no code path, buggy or not, that can lose feedback history by touching
this table.

Only "replace" diffs (see diffing.py) become rules. "insert"/"delete" diffs
(a missed or hallucinated word) are a different kind of edit that doesn't
map cleanly onto a "phrase A -> phrase B" rule; they're still preserved in
the underlying feedback event's diff_json, just not turned into an
auto-apply candidate here. Matching for rule identity is case-insensitive
(a case-only edit like "chennai" -> "Chennai" collapses to a no-op rule and
is skipped) -- a deliberate simplification for this foundation phase.

This module never decides whether to *apply* a rule -- see
correction_scoring.py / correction_engine.py for that. It only stores and
retrieves candidates.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone

from . import db as learning_db
from . import feedback_store
from .feedback_store import FeedbackEvent

_SCHEMA = """
CREATE TABLE IF NOT EXISTS correction_rules (
    id                     INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_id             TEXT NOT NULL,
    original_phrase        TEXT NOT NULL,
    corrected_phrase       TEXT NOT NULL,
    context_before_sample  TEXT,
    context_after_sample   TEXT,
    frequency              INTEGER NOT NULL DEFAULT 1,
    first_seen_at          TEXT NOT NULL,
    last_seen_at           TEXT NOT NULL,
    source_event_ids       TEXT NOT NULL,
    enabled                INTEGER NOT NULL DEFAULT 1,
    updated_at             TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_correction_rules_key
    ON correction_rules(profile_id, original_phrase, corrected_phrase);
CREATE INDEX IF NOT EXISTS idx_correction_rules_profile
    ON correction_rules(profile_id, enabled);
"""


@dataclass(frozen=True)
class CorrectionRule:
    id: int
    profile_id: str
    original_phrase: str  # normalized (lowercased) key
    corrected_phrase: str
    context_before_sample: str
    context_after_sample: str
    frequency: int
    first_seen_at: str
    last_seen_at: str
    source_event_ids: list[str]
    enabled: bool


def init_db(db_path: str) -> None:
    conn = learning_db.connect(db_path)
    try:
        conn.executescript(_SCHEMA)
        conn.commit()
    finally:
        conn.close()


def _apply_event(conn: sqlite3.Connection, event: FeedbackEvent) -> None:
    now = datetime.now(timezone.utc).isoformat()
    for d in event.diffs:
        if d.op != "replace":
            continue
        original_norm = d.original_phrase.strip().lower()
        corrected_norm = d.corrected_phrase.strip().lower()
        if not original_norm or not corrected_norm or original_norm == corrected_norm:
            continue

        existing = conn.execute(
            """SELECT id, frequency, source_event_ids, context_before_sample,
                      context_after_sample
               FROM correction_rules
               WHERE profile_id = ? AND original_phrase = ? AND corrected_phrase = ?""",
            (event.profile_id, original_norm, corrected_norm),
        ).fetchone()

        if existing is None:
            conn.execute(
                """INSERT INTO correction_rules (
                       profile_id, original_phrase, corrected_phrase,
                       context_before_sample, context_after_sample, frequency,
                       first_seen_at, last_seen_at, source_event_ids, enabled, updated_at
                   ) VALUES (?,?,?,?,?,1,?,?,?,1,?)""",
                (
                    event.profile_id,
                    original_norm,
                    corrected_norm,
                    d.context_before,
                    d.context_after,
                    now,
                    now,
                    json.dumps([event.event_uuid]),
                    now,
                ),
            )
        else:
            source_ids = json.loads(existing["source_event_ids"])
            if event.event_uuid not in source_ids:
                source_ids.append(event.event_uuid)
            conn.execute(
                """UPDATE correction_rules SET
                       frequency = frequency + 1,
                       last_seen_at = ?,
                       source_event_ids = ?,
                       context_before_sample = COALESCE(NULLIF(context_before_sample, ''), ?),
                       context_after_sample = COALESCE(NULLIF(context_after_sample, ''), ?),
                       updated_at = ?
                   WHERE id = ?""",
                (
                    now,
                    json.dumps(source_ids),
                    d.context_before,
                    d.context_after,
                    now,
                    existing["id"],
                ),
            )


def upsert_rule_from_event(db_path: str, event: FeedbackEvent) -> None:
    """Incremental update, called right after a feedback event is stored --
    keeps rules fresh without waiting for a full rebuild.
    """
    init_db(db_path)
    conn = learning_db.connect(db_path)
    try:
        _apply_event(conn, event)
        conn.commit()
    finally:
        conn.close()


def rebuild_rules_for_profile(db_path: str, profile_id: str) -> int:
    """Wipes and re-derives this profile's rules from feedback_events.
    feedback_events is only ever read here, never written -- see module
    docstring for why that makes this safe to call at any time. Returns
    the number of rules written.
    """
    init_db(db_path)
    events = feedback_store.query_feedback_events(db_path, profile_id=profile_id)
    conn = learning_db.connect(db_path)
    try:
        conn.execute("DELETE FROM correction_rules WHERE profile_id = ?", (profile_id,))
        for event in events:
            _apply_event(conn, event)
        conn.commit()
        count = conn.execute(
            "SELECT COUNT(*) AS n FROM correction_rules WHERE profile_id = ?", (profile_id,)
        ).fetchone()["n"]
    finally:
        conn.close()
    return count


def rebuild_all_rules(db_path: str) -> dict[str, int]:
    init_db(db_path)
    conn = learning_db.connect(db_path)
    try:
        profile_ids = [
            r["profile_id"]
            for r in conn.execute("SELECT DISTINCT profile_id FROM feedback_events").fetchall()
        ]
    finally:
        conn.close()
    return {pid: rebuild_rules_for_profile(db_path, pid) for pid in profile_ids}


def _row_to_rule(row: sqlite3.Row) -> CorrectionRule:
    return CorrectionRule(
        id=row["id"],
        profile_id=row["profile_id"],
        original_phrase=row["original_phrase"],
        corrected_phrase=row["corrected_phrase"],
        context_before_sample=row["context_before_sample"] or "",
        context_after_sample=row["context_after_sample"] or "",
        frequency=row["frequency"],
        first_seen_at=row["first_seen_at"],
        last_seen_at=row["last_seen_at"],
        source_event_ids=json.loads(row["source_event_ids"]),
        enabled=bool(row["enabled"]),
    )


def get_active_rules(db_path: str, profile_id: str) -> list[CorrectionRule]:
    """Rules for exactly this profile, enabled ones only. There is no
    fallback/merge across profiles -- a rule learned under one profile can
    never be retrieved for another.
    """
    init_db(db_path)
    conn = learning_db.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT * FROM correction_rules WHERE profile_id = ? AND enabled = 1",
            (profile_id,),
        ).fetchall()
    finally:
        conn.close()
    return [_row_to_rule(r) for r in rows]


def set_rule_enabled(db_path: str, rule_id: int, enabled: bool) -> None:
    """Soft toggle -- disabling a rule (e.g. because it turned out wrong)
    never deletes it or the feedback events it was derived from; it's just
    excluded from get_active_rules() until re-enabled or a rebuild
    reconstructs the table from scratch.
    """
    init_db(db_path)
    conn = learning_db.connect(db_path)
    try:
        conn.execute(
            "UPDATE correction_rules SET enabled = ?, updated_at = ? WHERE id = ?",
            (1 if enabled else 0, datetime.now(timezone.utc).isoformat(), rule_id),
        )
        conn.commit()
    finally:
        conn.close()
