"""
Phase 4 helper: record a personal test dataset for the benchmark.

This script needs an actual human at a microphone -- there's no way to
automate "say the word 'stop' as a whisper" from code. Run it, follow
the prompts, and it will save one .wav per (word, modulation) pair plus
a manifest CSV that benchmark.py consumes.

Usage:
    python src/record_test_samples.py
    python src/record_test_samples.py --words yes,no,stop,help,start --seconds 3
"""

from __future__ import annotations

import argparse
import csv
import os
import time

import numpy as np
import sounddevice as sd
import soundfile as sf

SAMPLE_RATE = 16000
DEFAULT_WORDS = ["yes", "no", "stop", "help", "start"]
MODULATIONS = ["normal", "whispered", "shouted", "slow", "fast"]
MODULATION_PROMPTS = {
    "normal": "Say it normally, like regular conversation.",
    "whispered": "Whisper it as quietly as you comfortably can.",
    "shouted": "Say it loudly, as if shouting across a room.",
    "slow": "Say it very slowly, stretching the word out.",
    "fast": "Say it very fast, rushed.",
}

OUT_DIR = os.path.join("data", "test_samples")
MANIFEST_PATH = os.path.join(OUT_DIR, "manifest.csv")


def record_clip(seconds: float) -> np.ndarray:
    print("  Recording", end="", flush=True)
    for _ in range(3):
        time.sleep(0.3)
        print(".", end="", flush=True)
    print(" GO!")
    audio = sd.rec(
        int(seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="float32"
    )
    sd.wait()
    print("  done.")
    return audio.flatten()


def confirm_or_retake(audio: np.ndarray, seconds: float) -> np.ndarray:
    while True:
        choice = input("  [k]eep / [r]etake / [p]lay back? [k] ").strip().lower()
        if choice in ("", "k"):
            return audio
        if choice == "p":
            sd.play(audio, SAMPLE_RATE)
            sd.wait()
            continue
        if choice == "r":
            return record_clip(seconds)


def main():
    parser = argparse.ArgumentParser(description="Record modulation-robustness test samples.")
    parser.add_argument("--words", default=",".join(DEFAULT_WORDS))
    parser.add_argument("--seconds", type=float, default=3.0)
    args = parser.parse_args()
    words = [w.strip() for w in args.words.split(",") if w.strip()]

    os.makedirs(OUT_DIR, exist_ok=True)
    manifest_rows = []

    print(f"Recording {len(words)} words x {len(MODULATIONS)} modulations = "
          f"{len(words) * len(MODULATIONS)} clips.\n")

    for word in words:
        for mod in MODULATIONS:
            print(f"\n=== Word: '{word}'  |  Modulation: {mod} ===")
            print(f"  {MODULATION_PROMPTS[mod]}")
            input("  Press Enter when ready...")
            audio = record_clip(args.seconds)
            audio = confirm_or_retake(audio, args.seconds)

            filename = f"{word}_{mod}.wav"
            filepath = os.path.join(OUT_DIR, filename)
            sf.write(filepath, audio, SAMPLE_RATE)
            manifest_rows.append(
                {"filename": filename, "expected_word": word, "modulation": mod}
            )
            print(f"  Saved {filepath}")

    with open(MANIFEST_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "expected_word", "modulation"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"\nDone. {len(manifest_rows)} clips saved to {OUT_DIR}/")
    print(f"Manifest written to {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
