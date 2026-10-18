"""
Builds versioned STT fine-tuning datasets from accumulated feedback events.

STT fine-tuning needs (audio, correct_transcript) pairs, not text alone --
so this only has something to work with once corrections have been
submitted with LEARNING_PERSIST_AUDIO_ENABLED=1 (see audio_cache.py).
Building a dataset never touches feedback_events (read-only, same
principle as correction_memory.py's rebuild) and never trains anything
itself -- see training.py for that, kept in a separate module specifically
so a bug or a long-running training job can never touch the live
transcribe path (src/chat_web.py never imports this module or training.py
at all).

Selection is intentionally conservative: an event only becomes a training
example if its audio file actually exists on disk and is large enough to
plausibly be real audio, its corrected text is a sane length, and it isn't
a near-duplicate of another selected example. No LLM, no manual review
step is assumed to exist yet -- validation_status defaults to
"unvalidated" and is still eligible; "rejected" is not. A human validation
workflow (setting validation_status via some future admin surface) can
tighten this later without changing this module.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

from . import feedback_store

MANIFEST_FIELDS = [
    "event_uuid",
    "split",
    "audio_path",
    "text",
    "original_text",
    "profile_id",
    "stt_confidence",
    "created_at",
]


@dataclass
class ExclusionCounts:
    no_audio_ref: int = 0
    audio_file_missing: int = 0
    audio_file_too_small: int = 0
    rejected_validation_status: int = 0
    wrong_language: int = 0
    text_too_long: int = 0
    duplicate: int = 0


@dataclass
class DatasetManifest:
    version: str
    created_at: str
    source_db_path: str
    language: str
    total_candidates: int
    selected_count: int
    train_count: int
    eval_count: int
    excluded: ExclusionCounts
    manifest_path: str
    metadata_path: str
    content_hash: str


def _next_version(output_root: str) -> str:
    os.makedirs(output_root, exist_ok=True)
    existing = [
        d
        for d in os.listdir(output_root)
        if re.fullmatch(r"v(\d+)", d) and os.path.isdir(os.path.join(output_root, d))
    ]
    numbers = [int(d[1:]) for d in existing]
    return f"v{max(numbers) + 1 if numbers else 1}"


def _is_eval_split(event_uuid: str, eval_fraction: float) -> bool:
    """Deterministic split from a hash of the event id, so re-running the
    builder on the same underlying data reproduces the same split rather
    than randomly reshuffling which examples are held out.
    """
    digest = hashlib.sha256(event_uuid.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF  # -> [0, 1)
    return bucket < eval_fraction


def _is_eligible(
    event, language: str, min_audio_bytes: int, max_text_length: int, excluded: ExclusionCounts
) -> bool:
    """The single source of truth for "is this feedback event usable as a
    training example", shared by build_dataset() (exact selection) and
    count_new_events() (a cheap threshold check the training scheduler
    uses to decide whether a run is even worth attempting). Mutates
    `excluded` in place so callers that care about the breakdown (only
    build_dataset does) get it for free.
    """
    if event.validation_status == "rejected":
        excluded.rejected_validation_status += 1
        return False
    if event.language != language:
        excluded.wrong_language += 1
        return False
    if not event.audio_ref:
        excluded.no_audio_ref += 1
        return False
    if not os.path.exists(event.audio_ref):
        excluded.audio_file_missing += 1
        return False
    if os.path.getsize(event.audio_ref) < min_audio_bytes:
        excluded.audio_file_too_small += 1
        return False
    if len(event.corrected_text) > max_text_length:
        excluded.text_too_long += 1
        return False
    return True


def count_new_events(
    db_path: str,
    *,
    language: str = "en",
    since_event_id: int = 0,
    min_audio_bytes: int = 1000,
    max_text_length: int = 500,
) -> tuple[int, int]:
    """How many feedback events with id > since_event_id are eligible for
    training right now (same eligibility rule build_dataset() uses, minus
    the final dedup pass -- dedup only matters once rows are actually being
    assembled, not for a threshold check). Returns
    (eligible_new_count, max_event_id_seen) -- the caller persists the
    second value as its new watermark once a training run has actually
    been attempted on data through that point, so the same events are
    never counted as "new" again regardless of whether they end up
    selected into that run's dataset.
    """
    events = feedback_store.query_feedback_events(db_path)
    excluded = ExclusionCounts()
    max_id = since_event_id
    eligible = 0
    for event in events:
        max_id = max(max_id, event.id)
        if event.id <= since_event_id:
            continue
        if _is_eligible(event, language, min_audio_bytes, max_text_length, excluded):
            eligible += 1
    return eligible, max_id


def build_dataset(
    db_path: str,
    output_root: str,
    *,
    language: str = "en",
    eval_fraction: float = 0.15,
    min_audio_bytes: int = 1000,
    max_text_length: int = 500,
) -> DatasetManifest:
    """Selects eligible feedback events across all profiles and writes a
    new versioned dataset directory under output_root. Raises ValueError
    if nothing is eligible (better than silently writing an empty, useless
    dataset version).
    """
    events = feedback_store.query_feedback_events(db_path)
    excluded = ExclusionCounts()
    seen_dedup_keys: set[tuple[str, str, str]] = set()
    rows: list[dict] = []

    for event in events:
        if not _is_eligible(event, language, min_audio_bytes, max_text_length, excluded):
            continue

        dedup_key = (event.profile_id, event.original_text, event.corrected_text)
        if dedup_key in seen_dedup_keys:
            excluded.duplicate += 1
            continue
        seen_dedup_keys.add(dedup_key)

        rows.append(
            {
                "event_uuid": event.event_uuid,
                "split": "eval" if _is_eval_split(event.event_uuid, eval_fraction) else "train",
                "audio_path": event.audio_ref,
                "text": event.corrected_text,
                "original_text": event.original_text,
                "profile_id": event.profile_id,
                "stt_confidence": event.stt_confidence,
                "created_at": event.created_at,
            }
        )

    if not rows:
        raise ValueError(
            "No feedback events are eligible for a training dataset yet. "
            "This needs corrections submitted with LEARNING_PERSIST_AUDIO_"
            "ENABLED=1 -- see README, 'Continuous learning'."
        )

    version = _next_version(output_root)
    version_dir = os.path.join(output_root, version)
    os.makedirs(version_dir, exist_ok=True)

    manifest_path = os.path.join(version_dir, "manifest.csv")
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    with open(manifest_path, "rb") as f:
        content_hash = hashlib.sha256(f.read()).hexdigest()

    train_count = sum(1 for r in rows if r["split"] == "train")
    eval_count = len(rows) - train_count

    manifest = DatasetManifest(
        version=version,
        created_at=datetime.now(timezone.utc).isoformat(),
        source_db_path=db_path,
        language=language,
        total_candidates=len(events),
        selected_count=len(rows),
        train_count=train_count,
        eval_count=eval_count,
        excluded=excluded,
        manifest_path=manifest_path,
        metadata_path=os.path.join(version_dir, "dataset.json"),
        content_hash=content_hash,
    )
    with open(manifest.metadata_path, "w", encoding="utf-8") as f:
        json.dump(asdict(manifest), f, indent=2)

    return manifest


def load_manifest_rows(manifest_path: str) -> list[dict]:
    with open(manifest_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def list_dataset_versions(output_root: str) -> list[str]:
    if not os.path.isdir(output_root):
        return []
    versions = [
        d
        for d in os.listdir(output_root)
        if re.fullmatch(r"v\d+", d) and os.path.isdir(os.path.join(output_root, d))
    ]
    return sorted(versions, key=lambda v: int(v[1:]))
