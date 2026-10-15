"""
Automated, scheduled continuous-learning loop: periodically checks whether
enough new corrected data has accumulated, and if so runs the existing
dataset-build -> fine-tune -> evaluate -> (maybe) promote pipeline
unattended, then optionally hot-swaps the live chatbot onto the newly
promoted model.

This module orchestrates existing pieces; it does not reimplement any of
them:
  - dataset_builder.build_dataset / count_new_events -- data selection
  - training.train                                    -- LoRA/PEFT fine-tune
  - evaluation.evaluate_model_on_dataset / compare_models -- promotion gate
  - model_registry.*                                   -- versioning, promotion

Design constraints from the spec this implements:
  - "not enough new data" must be a safe no-op, not an error.
  - only one training job may run at a time, even across process restarts
    (a plain file lock with a PID + timestamp, reclaimed if stale).
  - a candidate is only ever promoted if evaluation.compare_models() says
    so -- this module adds no new promotion criteria of its own, it just
    calls the same model_registry.promote_to_production() a human would
    call manually, and refuses exactly the same way if the evaluation
    doesn't recommend it.
  - the live chatbot must not be interrupted by a swap. This module never
    touches faster-whisper or chat_web.py's model global directly; it
    calls an injected `on_promoted(version, artifact_path)` callback after
    a successful promotion, and chat_web.py is the one place that knows
    how to load a new WhisperModel and atomically swap the module-level
    reference in-flight requests already hold a copy of.

Fully inert unless CONFIG.learning_enabled AND
CONFIG.learning_auto_training_enabled are both true -- see
start_background_scheduler()'s caller in chat_web.py.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Callable

from . import dataset_builder, evaluation, model_registry

logger = logging.getLogger("learning.scheduler")


class TrainingLockBusy(RuntimeError):
    """Raised internally when another process/thread already holds the
    training lock and it isn't stale enough to reclaim."""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# --------------------------------------------------------------------
# State: small JSON file tracking the scheduler's own progress, separate
# from model_registry.json (which tracks models) and feedback.db (which
# tracks corrections). Never the source of truth for anything else --
# safe to delete, at worst the next run re-derives its watermark from 0
# and just does a bit of redundant counting.
# --------------------------------------------------------------------

_DEFAULT_STATE = {
    "status": "idle",  # "idle" | "running"
    "last_run_started_at": None,
    "last_run_finished_at": None,
    "last_successful_training_at": None,  # last time a candidate was PROMOTED
    "last_attempted_training_at": None,  # last time train() actually ran (promoted or not)
    "last_trained_event_id": 0,  # watermark: events with id <= this were already
                                   # considered for a training attempt
    "last_result": None,  # human-readable summary of the most recent cycle
    "last_error": None,
    "consecutive_failures": 0,
}


def _load_state(state_path: str) -> dict:
    if not os.path.exists(state_path):
        return dict(_DEFAULT_STATE)
    with open(state_path, encoding="utf-8") as f:
        data = json.load(f)
    merged = dict(_DEFAULT_STATE)
    merged.update(data)
    return merged


def _save_state(state_path: str, state: dict) -> None:
    os.makedirs(os.path.dirname(state_path) or ".", exist_ok=True)
    tmp_path = state_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp_path, state_path)  # atomic on Windows and POSIX


def get_status(state_path: str, registry_path: str) -> dict:
    """Read-only combined status: scheduler state + which model is
    currently production, for a status endpoint/CLI to show a human.
    """
    state = _load_state(state_path)
    production = model_registry.get_production_model(registry_path)
    return {
        "scheduler": state,
        "production_model": production,
    }


# --------------------------------------------------------------------
# Lock: prevents two training runs overlapping, including across a crash
# and restart of the whole process. Not a distributed lock (single
# machine, single app, matches the rest of this project's concurrency
# assumptions) -- just a file the *presence* of which means "a job claims
# to be running", reclaimed if it's older than a configurable staleness
# window (i.e. the process that held it almost certainly died without
# cleaning up).
# --------------------------------------------------------------------


class _TrainingLock:
    def __init__(self, lock_path: str, stale_after_s: int):
        self.lock_path = lock_path
        self.stale_after_s = stale_after_s
        self._acquired = False

    def __enter__(self) -> "_TrainingLock":
        os.makedirs(os.path.dirname(self.lock_path) or ".", exist_ok=True)
        payload = json.dumps({"pid": os.getpid(), "acquired_at": _now_iso()}).encode("utf-8")
        try:
            fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            try:
                os.write(fd, payload)
            finally:
                os.close(fd)
            self._acquired = True
            return self
        except FileExistsError:
            if self._reclaim_if_stale():
                return self.__enter__()
            raise TrainingLockBusy(
                f"Training lock {self.lock_path} is held by another run; skipping this cycle."
            )

    def _reclaim_if_stale(self) -> bool:
        try:
            mtime = os.path.getmtime(self.lock_path)
        except OSError:
            return True  # lock vanished between the EEXIST and now; safe to retry
        if (time.time() - mtime) < self.stale_after_s:
            return False
        logger.warning(
            "Reclaiming stale training lock %s (older than %ss).",
            self.lock_path, self.stale_after_s,
        )
        try:
            os.remove(self.lock_path)
        except OSError:
            pass
        return True

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._acquired:
            try:
                os.remove(self.lock_path)
            except OSError:
                pass


# --------------------------------------------------------------------
# One training cycle
# --------------------------------------------------------------------


def run_training_cycle(
    config,
    *,
    on_promoted: Callable[[str, str], None] | None = None,
) -> dict:
    """Runs exactly one check-and-maybe-train cycle. Always safe to call
    (gated internally by config flags, wrapped so it never raises out to
    the caller -- failures are recorded in the state file and logged, not
    propagated, so a bad training run can never kill the scheduler thread
    or the web server it shares a process with).

    Returns a small result dict describing what happened -- useful for a
    manual `python -m learning.scheduler --once` invocation as well as the
    background loop's own logging.
    """
    if not config.learning_enabled or not config.learning_auto_training_enabled:
        return {"ran": False, "reason": "disabled"}

    try:
        with _TrainingLock(
            config.learning_scheduler_lock_path, config.learning_training_lock_stale_seconds
        ):
            return _run_training_cycle_locked(config, on_promoted=on_promoted)
    except TrainingLockBusy as exc:
        logger.info(str(exc))
        return {"ran": False, "reason": "lock_busy"}


def _run_training_cycle_locked(config, *, on_promoted) -> dict:
    state_path = config.learning_scheduler_state_path
    state = _load_state(state_path)
    state["status"] = "running"
    state["last_run_started_at"] = _now_iso()
    _save_state(state_path, state)

    try:
        result = _do_training_cycle(config, state, on_promoted=on_promoted)
        state["last_error"] = None
        state["consecutive_failures"] = 0
        state["last_result"] = result
        return result
    except Exception as exc:  # noqa: BLE001 -- must never crash the scheduler loop
        logger.exception("Training cycle failed.")
        state["last_error"] = str(exc)
        state["consecutive_failures"] = state.get("consecutive_failures", 0) + 1
        result = {"ran": True, "reason": "error", "error": str(exc)}
        state["last_result"] = result
        return result
    finally:
        state["status"] = "idle"
        state["last_run_finished_at"] = _now_iso()
        _save_state(state_path, state)


def _do_training_cycle(config, state: dict, *, on_promoted) -> dict:
    # Imported lazily: training.py's own module-level torch/transformers
    # imports are guarded (see its docstring), but this keeps the whole
    # scheduler module importable/testable without the training extras
    # installed, same guarantee training.py already provides.
    from . import training

    since_id = state.get("last_trained_event_id", 0)
    new_count, max_id = dataset_builder.count_new_events(
        config.learning_db_path,
        language=config.learning_training_language,
        since_event_id=since_id,
    )
    if new_count < config.learning_min_new_events_for_training:
        logger.info(
            "Not enough new corrected data yet (%d new, need %d); skipping this cycle.",
            new_count, config.learning_min_new_events_for_training,
        )
        return {
            "ran": True,
            "reason": "not_enough_data",
            "new_event_count": new_count,
            "required": config.learning_min_new_events_for_training,
        }

    logger.info(
        "%d new corrected events (>= %d threshold); starting a training run.",
        new_count, config.learning_min_new_events_for_training,
    )
    state["last_attempted_training_at"] = _now_iso()
    # Recorded regardless of outcome: these specific events have now been
    # *considered*, so a training run that ends up rejected doesn't cause
    # the same events to immediately re-trigger another attempt next cycle.
    state["last_trained_event_id"] = max_id

    manifest = dataset_builder.build_dataset(
        config.learning_db_path,
        config.learning_datasets_root,
        language=config.learning_training_language,
    )

    training_config = training.TrainingConfig(base_model=config.learning_training_base_model)
    output_dir = os.path.join(config.learning_models_root, "_pending", manifest.version)
    ct2_dir = training.train(manifest.manifest_path, output_dir, config=training_config)

    version = model_registry.register_candidate(
        config.learning_registry_path,
        base_model=training_config.base_model,
        dataset_version=manifest.version,
        artifact_path=ct2_dir,
        training_config=asdict(training_config),
    )

    manifest_rows = dataset_builder.load_manifest_rows(manifest.manifest_path)
    baseline_transcribe = training._load_ct2_transcribe_fn(
        config.learning_training_baseline_model,
        device=config.learning_training_device,
        compute_type="int8",
    )
    baseline_result = evaluation.evaluate_model_on_dataset(
        baseline_transcribe, manifest_rows,
        model_label=f"production ({config.learning_training_baseline_model})",
        dataset_version=manifest.version,
    )
    candidate_transcribe = training._load_ct2_transcribe_fn(
        ct2_dir, device=config.learning_training_device, compute_type="int8",
    )
    candidate_result = evaluation.evaluate_model_on_dataset(
        candidate_transcribe, manifest_rows,
        model_label=f"candidate ({version})", dataset_version=manifest.version,
    )
    comparison = evaluation.compare_models(
        baseline_result, candidate_result,
        min_relative_improvement=config.learning_training_min_relative_improvement,
    )
    model_registry.record_evaluation(config.learning_registry_path, version, asdict(comparison))

    result = {
        "ran": True,
        "reason": "trained",
        "dataset_version": manifest.version,
        "candidate_version": version,
        "new_event_count": new_count,
        "comparison": asdict(comparison),
        "promoted": False,
    }

    if comparison.recommendation != "promote":
        logger.info(
            "Candidate %s not promoted (relative_improvement=%+.1%%, recommendation=%s).",
            version, comparison.relative_improvement, comparison.recommendation,
        )
        return result

    model_registry.promote_to_production(config.learning_registry_path, version)
    state["last_successful_training_at"] = _now_iso()
    result["promoted"] = True
    logger.info(
        "Candidate %s promoted to production (relative_improvement=%+.1%%).",
        version, comparison.relative_improvement,
    )

    if config.learning_auto_deploy_enabled and on_promoted is not None:
        try:
            on_promoted(version, ct2_dir)
        except Exception:  # noqa: BLE001 -- a bad hot-reload must not un-promote or crash
            logger.exception(
                "on_promoted callback failed after promoting %s; production model in the "
                "registry is still %s, but the live process may not have picked it up yet.",
                version, version,
            )

    return result


# --------------------------------------------------------------------
# Background loop
# --------------------------------------------------------------------


class SchedulerThread(threading.Thread):
    """A daemon thread that calls run_training_cycle() on a fixed
    interval until stop() is called. Owned by chat_web.py's process
    lifetime -- stopping the web server stops this thread too (daemon=True)
    rather than leaving an orphaned training job's *thread* around (the
    training subprocess/lock survives independently, by design, so a
    restart doesn't lose track of an in-progress run).
    """

    def __init__(self, config, *, on_promoted: Callable[[str, str], None] | None = None):
        super().__init__(name="learning-scheduler", daemon=True)
        self._config = config
        self._on_promoted = on_promoted
        self._stop_event = threading.Event()

    def run(self) -> None:
        interval_s = max(60, self._config.learning_training_interval_minutes * 60)
        logger.info("Learning scheduler started (interval=%ds).", interval_s)
        while not self._stop_event.is_set():
            try:
                run_training_cycle(self._config, on_promoted=self._on_promoted)
            except Exception:  # noqa: BLE001 -- the loop itself must never die
                logger.exception("Unexpected error in scheduler loop iteration.")
            self._stop_event.wait(interval_s)

    def stop(self) -> None:
        self._stop_event.set()


def start_background_scheduler(
    config, *, on_promoted: Callable[[str, str], None] | None = None
) -> SchedulerThread | None:
    """Starts the scheduler thread if (and only if) both
    CONFIG.learning_enabled and CONFIG.learning_auto_training_enabled are
    set; returns None (and starts nothing) otherwise, so callers can do
    `scheduler = start_background_scheduler(CONFIG)` unconditionally.
    """
    if not config.learning_enabled or not config.learning_auto_training_enabled:
        logger.info("Automated learning scheduler is disabled; not starting.")
        return None
    thread = SchedulerThread(config, on_promoted=on_promoted)
    thread.start()
    return thread


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for a manual/cron-triggered single cycle:
        python -m learning.scheduler --once
    Useful for operators who'd rather drive this from an external
    scheduler (Windows Task Scheduler, cron, systemd timer) than run the
    always-on background thread inside chat_web.py.
    """
    import argparse

    sys_path_src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    import sys

    if sys_path_src not in sys.path:
        sys.path.insert(0, sys_path_src)
    from config import Config  # noqa: E402

    parser = argparse.ArgumentParser(description="Run one automated learning training cycle.")
    parser.add_argument("--once", action="store_true", help="Run a single cycle and exit.")
    parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO)
    config = Config.from_env()
    result = run_training_cycle(config)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
