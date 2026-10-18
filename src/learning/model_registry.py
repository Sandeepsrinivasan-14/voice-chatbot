"""
JSON-backed registry of trained STT model candidates, which one (if any)
is in production, and enough metadata to answer "what dataset version and
training run produced this" and "what do we roll back to".

A plain JSON file, not another SQLite database -- this is small, changes
rarely (once per training run at most), and is meant to be human-readable
by anyone poking around data/learning/models/registry.json, which matters
more here than query performance.

Promotion is never automatic. register_candidate() only ever adds a
"candidate" entry; promote_to_production() refuses to run unless the
candidate's recorded evaluation actually recommends it (see
evaluation.compare_models), so a model can only reach production after
someone (a human, or a script a human runs and reviews) has looked at real
WER numbers. This mirrors the user's requirement that a fine-tuned model
"must only be deployed if it demonstrates a meaningful improvement without
unacceptable regressions".
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone


def _load(registry_path: str) -> dict:
    if not os.path.exists(registry_path):
        return {"production": None, "history": [], "models": {}}
    with open(registry_path, encoding="utf-8") as f:
        return json.load(f)


def _save(registry_path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(registry_path) or ".", exist_ok=True)
    tmp_path = registry_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp_path, registry_path)  # atomic on both Windows and POSIX


def _next_version(data: dict) -> str:
    numbers = [int(m.group(1)) for v in data["models"] if (m := re.fullmatch(r"v(\d+)", v))]
    return f"v{max(numbers) + 1 if numbers else 1}"


def register_candidate(
    registry_path: str,
    *,
    base_model: str,
    dataset_version: str,
    artifact_path: str,
    training_config: dict | None = None,
) -> str:
    """Records a freshly trained model as a candidate (never as
    production). Returns the version id it was assigned (e.g. "v1").
    """
    data = _load(registry_path)
    version = _next_version(data)
    data["models"][version] = {
        "version": version,
        "status": "candidate",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "base_model": base_model,
        "dataset_version": dataset_version,
        "artifact_path": artifact_path,
        "training_config": training_config or {},
        "evaluation": None,
    }
    _save(registry_path, data)
    return version


def record_evaluation(registry_path: str, version: str, comparison: dict) -> None:
    """Attaches an evaluation.compare_models() result (as a plain dict --
    dataclasses.asdict(comparison)) to a candidate. If the comparison does
    not recommend promotion, the candidate is marked "rejected" so it's
    clear at a glance without re-reading the numbers; it stays in the
    registry either way -- rejected candidates are historical record, not
    deleted.
    """
    data = _load(registry_path)
    if version not in data["models"]:
        raise KeyError(f"No such model version: {version}")
    data["models"][version]["evaluation"] = comparison
    if not comparison.get("meets_improvement_threshold"):
        data["models"][version]["status"] = "rejected"
    _save(registry_path, data)


def promote_to_production(
    registry_path: str, version: str, *, allow_without_evaluation: bool = False
) -> None:
    """Makes `version` the production model. Refuses unless its recorded
    evaluation recommends promotion (comparison.recommendation == "promote"),
    unless allow_without_evaluation=True is passed explicitly -- an escape
    hatch for e.g. deliberately reinstating a known-good older version
    during a manual rollback, never meant to be the normal path.
    """
    data = _load(registry_path)
    entry = data["models"].get(version)
    if entry is None:
        raise KeyError(f"No such model version: {version}")

    evaluation = entry.get("evaluation")
    if not allow_without_evaluation:
        if evaluation is None:
            raise ValueError(
                f"{version} has no recorded evaluation; call record_evaluation() "
                "first, or pass allow_without_evaluation=True to override."
            )
        if evaluation.get("recommendation") != "promote":
            raise ValueError(
                f"{version}'s evaluation does not recommend promotion "
                f"(recommendation={evaluation.get('recommendation')!r})."
            )

    previous = data.get("production")
    if previous and previous != version:
        data["models"][previous]["status"] = "retired"
    entry["status"] = "production"
    data["production"] = version
    data["history"].append(version)
    _save(registry_path, data)


def get_production_model(registry_path: str) -> dict | None:
    data = _load(registry_path)
    version = data.get("production")
    return data["models"].get(version) if version else None


def rollback(registry_path: str) -> str:
    """Reverts production to whichever version was production immediately
    before the current one. Raises ValueError if there's no earlier
    version to roll back to.
    """
    data = _load(registry_path)
    history = data.get("history", [])
    if len(history) < 2:
        raise ValueError("No earlier production version to roll back to.")

    current = history[-1]
    target = history[-2]
    data["models"][current]["status"] = "retired"
    data["models"][target]["status"] = "production"
    data["production"] = target
    data["history"].append(target)  # rollback is itself a recorded transition
    _save(registry_path, data)
    return target


def _version_number(version: str) -> int:
    match = re.fullmatch(r"v(\d+)", version)
    if match is None:
        raise ValueError(f"Malformed model version {version!r}; expected 'v<number>'")
    return int(match.group(1))


def list_models(registry_path: str) -> list[dict]:
    data = _load(registry_path)
    return sorted(data["models"].values(), key=lambda m: _version_number(m["version"]))
