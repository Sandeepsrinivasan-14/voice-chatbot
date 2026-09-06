"""
Phase 5: before/after benchmark -- the proof that preprocessing.py
actually helps with modulation robustness, not just a plausible story.

Runs every recorded sample (data/test_samples/, from record_test_samples.py)
through faster-whisper twice: once raw, once with normalize_audio() applied
first. Same model, same beam size, same everything else -- the only
variable is whether preprocessing ran. Whatever accuracy gap shows up is
attributable to the preprocessing layer.

Usage:
    python src/benchmark.py
"""

from __future__ import annotations

import csv
import os
import re
import string
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")  # headless-safe; we only ever save the figure to disk
import matplotlib.pyplot as plt
import soundfile as sf

sys.path.insert(0, os.path.dirname(__file__))
from preprocessing import normalize_audio  # noqa: E402

MANIFEST_PATH = os.path.join("data", "test_samples", "manifest.csv")
SAMPLES_DIR = os.path.join("data", "test_samples")
RESULTS_DIR = os.path.join("data", "results")
RESULTS_CSV = os.path.join(RESULTS_DIR, "benchmark_results.csv")
CHART_PATH = os.path.join(RESULTS_DIR, "accuracy_comparison.png")

BENCHMARK_BEAM_SIZE = 5  # fixed for both runs so preprocessing is the only variable
MODULATION_ORDER = ["normal", "whispered", "shouted", "slow", "fast"]


def normalize_text_for_match(text: str) -> str:
    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", text).strip()


def word_correct(transcribed: str, expected_word: str) -> bool:
    norm = normalize_text_for_match(transcribed)
    expected = normalize_text_for_match(expected_word)
    return expected in norm.split()


def load_manifest() -> list[dict]:
    if not os.path.exists(MANIFEST_PATH):
        raise FileNotFoundError(
            f"No manifest at {MANIFEST_PATH}. Run "
            "`python src/record_test_samples.py` first (Phase 4)."
        )
    with open(MANIFEST_PATH, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def transcribe_once(model, audio, sr) -> str:
    segments, _info = model.transcribe(
        audio, language="en", beam_size=BENCHMARK_BEAM_SIZE, vad_filter=False
    )
    return " ".join(seg.text.strip() for seg in segments).strip()


def run_benchmark():
    from faster_whisper import WhisperModel

    rows = load_manifest()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    try:
        model = WhisperModel("small", device="cuda", compute_type="int8_float16")
        # Construction alone doesn't prove the GPU path works on Windows --
        # missing cuBLAS/cuDNN DLLs only surface on first real inference.
        import numpy as _np
        list(model.transcribe(_np.zeros(4000, dtype=_np.float32), language="en")[0])
        print("[benchmark] Using GPU for faster-whisper.")
    except Exception as exc:
        print(f"[benchmark] GPU unavailable ({exc}), using CPU.")
        model = WhisperModel("small", device="cpu", compute_type="int8")

    results = []  # one dict per (file, condition)
    for i, row in enumerate(rows, 1):
        path = os.path.join(SAMPLES_DIR, row["filename"])
        expected = row["expected_word"]
        modulation = row["modulation"]
        print(f"[{i}/{len(rows)}] {row['filename']} (expected='{expected}', mod={modulation})")

        audio, sr = sf.read(path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        raw_text = transcribe_once(model, audio, sr)
        raw_ok = word_correct(raw_text, expected)

        processed_audio, _report = normalize_audio(audio, sr)
        prep_text = transcribe_once(model, processed_audio, sr)
        prep_ok = word_correct(prep_text, expected)

        print(f"    raw='{raw_text}' ({'OK' if raw_ok else 'MISS'})  "
              f"preprocessed='{prep_text}' ({'OK' if prep_ok else 'MISS'})")

        results.append(
            {
                "filename": row["filename"],
                "expected_word": expected,
                "modulation": modulation,
                "raw_transcription": raw_text,
                "raw_correct": raw_ok,
                "preprocessed_transcription": prep_text,
                "preprocessed_correct": prep_ok,
            }
        )

    write_results_csv(results)
    summary = summarize(results)
    print_summary_table(summary)
    plot_chart(summary)
    return summary


def write_results_csv(results: list[dict]) -> None:
    fieldnames = list(results[0].keys()) if results else []
    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\n[benchmark] Wrote {RESULTS_CSV}")


def summarize(results: list[dict]) -> dict:
    by_mod = defaultdict(lambda: {"raw": [], "prep": []})
    for r in results:
        by_mod[r["modulation"]]["raw"].append(r["raw_correct"])
        by_mod[r["modulation"]]["prep"].append(r["preprocessed_correct"])

    def pct(values):
        return 100.0 * sum(values) / len(values) if values else 0.0

    summary = {}
    all_raw, all_prep = [], []
    for mod in MODULATION_ORDER:
        if mod not in by_mod:
            continue
        raw_vals, prep_vals = by_mod[mod]["raw"], by_mod[mod]["prep"]
        summary[mod] = {"raw_acc": pct(raw_vals), "prep_acc": pct(prep_vals)}
        all_raw.extend(raw_vals)
        all_prep.extend(prep_vals)
    summary["Overall"] = {"raw_acc": pct(all_raw), "prep_acc": pct(all_prep)}
    return summary


def print_summary_table(summary: dict) -> None:
    print("\n" + "=" * 60)
    print(f"{'Modulation':<12} {'Raw Acc %':>12} {'Preprocessed %':>16} {'Delta':>8}")
    print("-" * 60)
    for mod, vals in summary.items():
        delta = vals["prep_acc"] - vals["raw_acc"]
        print(f"{mod:<12} {vals['raw_acc']:>11.1f}% {vals['prep_acc']:>15.1f}% {delta:>+7.1f}")
    print("=" * 60)


def plot_chart(summary: dict) -> None:
    labels = [m for m in summary if m != "Overall"] + ["Overall"]
    raw_vals = [summary[m]["raw_acc"] for m in labels]
    prep_vals = [summary[m]["prep_acc"] for m in labels]

    x = range(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar([i - width / 2 for i in x], raw_vals, width, label="Raw Whisper")
    ax.bar([i + width / 2 for i in x], prep_vals, width, label="With Preprocessing")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("STT Accuracy by Modulation: Raw vs Preprocessed")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(CHART_PATH, dpi=150)
    print(f"[benchmark] Wrote {CHART_PATH}")


if __name__ == "__main__":
    run_benchmark()
