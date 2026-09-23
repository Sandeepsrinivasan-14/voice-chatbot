"""
Continuous-learning layer for STT corrections.

Foundation phase (what this package currently does):
  - profiles.py           local, no-auth profile/session id resolution
  - diffing.py             word/phrase-level diff between an original and a
                            corrected transcript
  - feedback_store.py      immutable storage of correction feedback events
                            (SQLite; see data/learning/feedback.db)
  - correction_memory.py   derived, rebuildable per-profile correction rules
  - correction_scoring.py  pure, deterministic scoring primitives (no ML,
                            no embeddings, no LLM calls)
  - correction_engine.py   orchestration: look up candidate rules, score
                            them, decide keep / suggest / auto-apply

Everything here is gated by the CONFIG.learning_* flags (see config.py) and
is designed to fail closed: any exception raised by this package must never
break the core transcribe/chat flow in chat_web.py, which wraps every call
into this package in a try/except and falls back to the untouched STT
output. See README.md, "Continuous learning", for the enable flags and the
explicit list of what is NOT implemented yet (dataset building, fine-tuning,
model evaluation, versioning, and rollback -- those are future phases, not
started here).
"""

from __future__ import annotations
