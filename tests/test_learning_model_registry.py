"""
Tests for src/learning/model_registry.py -- versioning, promotion gating,
and rollback for trained STT models. The central guarantee under test is
that a model can never reach production without a recorded evaluation that
actually recommends it.
"""

from __future__ import annotations

import pytest

from learning import model_registry as registry


class TestRegisterCandidate:
    def test_first_candidate_is_v1_with_status_candidate(self, tmp_path):
        path = str(tmp_path / "registry.json")
        version = registry.register_candidate(
            path, base_model="openai/whisper-small", dataset_version="v1",
            artifact_path="/models/v1/ct2",
        )
        assert version == "v1"
        models = registry.list_models(path)
        assert len(models) == 1
        assert models[0]["status"] == "candidate"
        assert registry.get_production_model(path) is None

    def test_versions_increment(self, tmp_path):
        path = str(tmp_path / "registry.json")
        registry.register_candidate(path, base_model="m", dataset_version="v1", artifact_path="a")
        v2 = registry.register_candidate(path, base_model="m", dataset_version="v2", artifact_path="b")
        assert v2 == "v2"


class TestPromotionGating:
    def test_promotion_refused_without_evaluation(self, tmp_path):
        path = str(tmp_path / "registry.json")
        version = registry.register_candidate(path, base_model="m", dataset_version="v1", artifact_path="a")
        with pytest.raises(ValueError):
            registry.promote_to_production(path, version)

    def test_promotion_refused_when_evaluation_recommends_reject(self, tmp_path):
        path = str(tmp_path / "registry.json")
        version = registry.register_candidate(path, base_model="m", dataset_version="v1", artifact_path="a")
        registry.record_evaluation(
            path, version, {"recommendation": "reject", "meets_improvement_threshold": False}
        )
        with pytest.raises(ValueError):
            registry.promote_to_production(path, version)
        assert registry.list_models(path)[0]["status"] == "rejected"

    def test_promotion_succeeds_when_evaluation_recommends_promote(self, tmp_path):
        path = str(tmp_path / "registry.json")
        version = registry.register_candidate(path, base_model="m", dataset_version="v1", artifact_path="a")
        registry.record_evaluation(
            path, version, {"recommendation": "promote", "meets_improvement_threshold": True}
        )
        registry.promote_to_production(path, version)
        prod = registry.get_production_model(path)
        assert prod["version"] == version
        assert prod["status"] == "production"

    def test_override_flag_allows_promotion_without_evaluation(self, tmp_path):
        path = str(tmp_path / "registry.json")
        version = registry.register_candidate(path, base_model="m", dataset_version="v1", artifact_path="a")
        registry.promote_to_production(path, version, allow_without_evaluation=True)
        assert registry.get_production_model(path)["version"] == version


class TestRollback:
    def test_rollback_reverts_to_previous_production_version(self, tmp_path):
        path = str(tmp_path / "registry.json")
        v1 = registry.register_candidate(path, base_model="m", dataset_version="v1", artifact_path="a")
        registry.promote_to_production(path, v1, allow_without_evaluation=True)
        v2 = registry.register_candidate(path, base_model="m", dataset_version="v2", artifact_path="b")
        registry.promote_to_production(path, v2, allow_without_evaluation=True)

        target = registry.rollback(path)
        assert target == v1
        assert registry.get_production_model(path)["version"] == v1
        models_by_version = {m["version"]: m for m in registry.list_models(path)}
        assert models_by_version[v2]["status"] == "retired"

    def test_rollback_with_no_earlier_version_raises(self, tmp_path):
        path = str(tmp_path / "registry.json")
        v1 = registry.register_candidate(path, base_model="m", dataset_version="v1", artifact_path="a")
        registry.promote_to_production(path, v1, allow_without_evaluation=True)
        with pytest.raises(ValueError):
            registry.rollback(path)


class TestRejectedCandidatesAreNotDeleted:
    def test_rejected_candidate_still_appears_in_list_models(self, tmp_path):
        path = str(tmp_path / "registry.json")
        version = registry.register_candidate(path, base_model="m", dataset_version="v1", artifact_path="a")
        registry.record_evaluation(
            path, version, {"recommendation": "reject", "meets_improvement_threshold": False}
        )
        versions = [m["version"] for m in registry.list_models(path)]
        assert version in versions
