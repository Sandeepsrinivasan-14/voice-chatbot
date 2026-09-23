"""
Shared SQLite connection helper for feedback_store.py and
correction_memory.py.

Every other persistence idiom in this project is a plain file (WAV audio,
append-only CSV in confidence_check.py, whole-file-rewrite CSV in
record_web.py). Those don't fit here: the correction-rules table is
*derived from* the feedback-events table via aggregation and needs to be
atomically wiped-and-rebuilt without touching its source, and profile
lookups need indexed point queries rather than a linear file scan. SQLite
(stdlib, no new dependency) is the smallest tool that actually provides
transactions and indexes.

One connection is opened and closed per call, matching the project's
existing "open, write, close" style rather than holding a long-lived pool --
this is a single local user's app, not a server under real concurrent load.
"""

from __future__ import annotations

import os
import sqlite3


def connect(db_path: str) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=10.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn
