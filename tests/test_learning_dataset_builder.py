"""
Tests for src/learning/dataset_builder.py -- selecting eligible feedback
events into a versioned, reproducible STT training dataset. Real (tiny)
WAV files are written to tmp_path so the audio-presence checks are
exercised against actual files, not mocks.
"""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from learning import dataset_builder, feedback_store

DB_NAME = "feedback.db"


def _write_wav(path, seconds=0.5, sample_rate=16000) -> str:
    t = np.linspace(0, seconds, int(sample_rate * seconds), endpoint=False)
    audio = (0.05 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    sf.write(str(path), audio, sample_rate)
    return str(path)


def _seed_event(
    db_path, tmp_path, *, profile_id="alice", original="four", corrected="for",
    with_audio=True, language="en",
):
    audio_ref = None
    if with_audio:
        audio_ref = _write_wav(tmp_path / f"{original}_{corrected}_{profile_id}_{language}.wav")
    return feedback_store.insert_feedback_event(
        db_path, profile_id=profile_id, session_id="s1",
        original_text=f"I said {original}", corrected_text=f"I said {corrected}",
        audio_ref=audio_ref, language=language,
    )


class TestBuildDataset:
    def test_raises_when_nothing_eligible(self, tmp_path):
        db_path = str(tmp_path / DB_NAME)
        with pytest.raises(ValueError):
            dataset_builder.build_dataset(db_path, str(tmp_path / "datasets"))

    def test_events_without_audio_are_excluded(self, tmp_path):
        db_path = str(tmp_path / DB_NAME)
        _seed_event(db_path, tmp_path, with_audio=False)
        with pytest.raises(ValueError):
            dataset_builder.build_dataset(db_path, str(tmp_path / "datasets"))

    def test_selects_eligible_events_with_audio(self, tmp_path):
        db_path = str(tmp_path / DB_NAME)
        _seed_event(db_path, tmp_path)
        manifest = dataset_builder.build_dataset(db_path, str(tmp_path / "datasets"))
        assert manifest.version == "v1"
        assert manifest.selected_count == 1
        rows = dataset_builder.load_manifest_rows(manifest.manifest_path)
        assert len(rows) == 1
        assert rows[0]["text"] == "I said for"
        assert rows[0]["split"] in ("train", "eval")

    def test_wrong_language_excluded(self, tmp_path):
        db_path = str(tmp_path / DB_NAME)
        _seed_event(db_path, tmp_path, language="fr")
        with pytest.raises(ValueError):
            dataset_builder.build_dataset(db_path, str(tmp_path / "datasets"), language="en")

    def test_duplicate_correction_only_counted_once(self, tmp_path):
        db_path = str(tmp_path / DB_NAME)
        _seed_event(db_path, tmp_path, profile_id="alice")
        _seed_event(db_path, tmp_path, profile_id="alice")  # identical original/corrected/profile
        manifest = dataset_builder.build_dataset(db_path, str(tmp_path / "datasets"))
        assert manifest.selected_count == 1
        assert manifest.excluded.duplicate == 1

    def test_versions_increment_across_builds(self, tmp_path):
        db_path = str(tmp_path / DB_NAME)
        _seed_event(db_path, tmp_path, profile_id="alice")
        datasets_root = str(tmp_path / "datasets")
        m1 = dataset_builder.build_dataset(db_path, datasets_root)
        _seed_event(db_path, tmp_path, profile_id="bob")
        m2 = dataset_builder.build_dataset(db_path, datasets_root)
        assert m1.version == "v1"
        assert m2.version == "v2"
        assert dataset_builder.list_dataset_versions(datasets_root) == ["v1", "v2"]

    def test_content_hash_is_a_sha256_hex_digest(self, tmp_path):
        db_path = str(tmp_path / DB_NAME)
        _seed_event(db_path, tmp_path)
        manifest = dataset_builder.build_dataset(db_path, str(tmp_path / "datasets"))
        assert len(manifest.content_hash) == 64
        int(manifest.content_hash, 16)  # raises if it isn't valid hex

    def test_metadata_json_is_written_alongside_the_manifest(self, tmp_path):
        db_path = str(tmp_path / DB_NAME)
        _seed_event(db_path, tmp_path)
        manifest = dataset_builder.build_dataset(db_path, str(tmp_path / "datasets"))
        import json
        import os

        assert os.path.exists(manifest.metadata_path)
        with open(manifest.metadata_path, encoding="utf-8") as f:
            data = json.load(f)
        assert data["version"] == "v1"
        assert data["selected_count"] == 1
