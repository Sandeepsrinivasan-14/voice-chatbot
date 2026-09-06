"""
Modulation-robustness layer, part 1: audio preprocessing.

Whisper (and every other STT model trained mostly on read/conversational
speech) sees its accuracy degrade when the *acoustic* shape of the input
drifts from what it was trained on -- even though the *words* haven't
changed. Three axes matter most for a command-word chatbot:

  1. Loudness  -- whispering pushes the signal near the noise floor;
                  shouting clips/saturates it. Either way the model's
                  internal feature extractor sees an amplitude envelope
                  it wasn't tuned for.
  2. Pitch     -- a fundamental frequency (f0) far outside typical human
                  speech range (roughly 80-400 Hz) shifts formant spacing
                  in ways that confuse phoneme classification.
  3. Speed     -- very fast speech compresses phonemes together (fewer
                  frames per phoneme for the model to work with); very
                  slow speech stretches them out and can trigger
                  over-segmentation.

None of this makes Whisper *unable* to transcribe modulated speech -- it
makes the input closer, acoustically, to "normal" speech, which is the
regime the model was actually trained on. That's the whole strategy here:
we don't fix the model, we shrink the gap it has to generalize across.

UPDATE, post-benchmark (see README section 5 / data/results): on this
project's own recorded voice samples, actually *applying* the pitch and
speed corrections made accuracy dramatically WORSE (48% vs. 96% raw),
not better. Root cause, isolated with a series of controlled diagnostic
re-runs (volume-only vs. +pitch vs. +speed):

  - Speed "correction" was the dominant damage. Onset-density-based
    speaking-rate estimation needs several syllables across time to mean
    anything -- a single isolated command word gives ~1 onset, so the
    "rate" is dominated by how much silence padding surrounds the word
    in the recording, not how fast it was actually spoken. `start_fast`
    (genuinely spoken fast) was misclassified as extremely SLOW
    (ratio=0.22) and time-stretched in the wrong direction into an empty
    transcription. Every modulation, including `normal`, got flagged
    "slow" and mangled -- the signal was just noise for this input shape.
  - Pitch correction was a smaller but real second offender: on a short
    plosive-heavy word (`start`, fast), pyin locked onto a consonant
    burst as if it were the vocal fundamental (766 Hz -- clearly not a
    real pitch for this voice) and "corrected" a word that had no actual
    pitch problem.
  - Volume/RMS normalization alone matched raw Whisper's accuracy
    exactly (96%, same single failure) -- neutral to good, never harmful.

Conclusion: for short, isolated command words specifically, pitch-shift
and time-stretch as implemented here are not trustworthy enough to apply
blindly, so ENABLE_PITCH_CORRECTION and ENABLE_SPEED_CORRECTION below
default to False -- detection/logging still runs (useful signal, e.g.
for confidence_check.py diagnostics), the audio just isn't modified
based on it. This is exactly the kind of thing the benchmark exists to
catch: a plausible-sounding preprocessing idea that a real accuracy
measurement proved was actively hurting the metric it was meant to help.
Longer, continuous multi-word utterances would give onset-density a much
larger sample to estimate from and might genuinely benefit from these
corrections -- that's untested here and would need its own benchmark
before being turned back on.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import librosa
import numpy as np

logger = logging.getLogger("voice_chatbot.preprocessing")

# --- Tunable reference ranges -----------------------------------------
# These describe "normal conversational speech" and are the targets we
# normalize toward. They were chosen from typical adult speech statistics,
# not from any one voice -- re-tune PITCH_* if a much higher/lower-pitched
# speaker's normal register is being flagged as an outlier.
TARGET_RMS = 0.1                 # target RMS amplitude after normalization
PITCH_MIN_HZ = 80.0               # below this -> flagged as pitch outlier
PITCH_MAX_HZ = 400.0              # above this -> flagged as pitch outlier
PITCH_REFERENCE_HZ = 165.0        # rough neutral midpoint we correct toward
NORMAL_SPEECH_RATE = 3.5          # syllables/sec-ish proxy, see estimate below
SPEED_RATIO_LOW = 0.75            # below this fraction of normal -> "slow"
SPEED_RATIO_HIGH = 1.35           # above this fraction of normal -> "fast"

# Both default to False -- see the module docstring's "UPDATE, post-benchmark"
# section. Detection/logging always runs regardless of these flags; only the
# actual audio-modifying step (pitch_shift / time_stretch) is gated. Flip to
# True only after re-validating with your own data/results/benchmark_results.csv.
ENABLE_PITCH_CORRECTION = False
ENABLE_SPEED_CORRECTION = False


@dataclass
class PreprocessingReport:
    """Metadata about what was detected/corrected, for logging + debugging."""

    original_rms: float = 0.0
    normalized_rms: float = 0.0
    detected_pitch_hz: float | None = None
    pitch_flagged: bool = False
    pitch_shift_semitones: float = 0.0
    estimated_speed_ratio: float = 1.0
    speed_flagged: bool = False
    speed_label: str = "normal"
    notes: list[str] = field(default_factory=list)


def _normalize_volume(audio: np.ndarray) -> tuple[np.ndarray, float, float]:
    """RMS-normalize amplitude.

    RMS (not peak) normalization is used because peak normalization only
    looks at the single loudest sample -- a whispered clip can have one
    incidental loud pop and still be RMS-quiet overall, while a shouted
    clip stays loud throughout. RMS reflects the perceived/average
    loudness that actually matters to the acoustic model.
    """
    original_rms = float(np.sqrt(np.mean(audio**2)) + 1e-9)
    if original_rms < 1e-6:
        # near silence -- don't divide by ~0 and blow up noise floor
        return audio, original_rms, original_rms
    gain = TARGET_RMS / original_rms
    normalized = audio * gain
    # guard against clipping introduced by a large gain on a whispered clip
    peak = np.max(np.abs(normalized))
    if peak > 0.99:
        normalized = normalized * (0.99 / peak)
    normalized_rms = float(np.sqrt(np.mean(normalized**2)) + 1e-9)
    return normalized, original_rms, normalized_rms


def _detect_and_correct_pitch(
    audio: np.ndarray, sr: int, report: PreprocessingReport
) -> np.ndarray:
    """Detect median f0 and pitch-shift extreme outliers toward a neutral
    reference. Mild deviations are left alone -- normal voice variation
    (an excited or a flat tone) shouldn't be flattened out, only genuine
    outliers (e.g. a deliberately high-pitched or artificially deepened
    voice) that risk confusing the acoustic model.
    """
    # The whole pitch step (detection AND correction) is one try/except.
    # librosa's pyin/pitch_shift lean on numba-JIT'd DSP kernels under the
    # hood; on a memory-constrained machine running whisper + an LLM
    # simultaneously, that JIT compile has been observed to fail with a
    # transient MemoryError. Preprocessing is an accuracy *enhancement* --
    # it must never be the reason a conversation turn crashes, so any
    # failure here just skips the pitch step and passes audio through.
    try:
        f0, voiced_flag, _ = librosa.pyin(
            audio,
            fmin=librosa.note_to_hz("C2"),   # ~65 Hz
            fmax=librosa.note_to_hz("C6"),   # ~1047 Hz
            sr=sr,
        )
        voiced_f0 = f0[voiced_flag] if f0 is not None else np.array([])
        if voiced_f0.size == 0:
            report.notes.append("no voiced frames detected; pitch step skipped")
            return audio

        median_pitch = float(np.median(voiced_f0))
        report.detected_pitch_hz = median_pitch

        if median_pitch < PITCH_MIN_HZ or median_pitch > PITCH_MAX_HZ:
            report.pitch_flagged = True
            # semitone shift needed to move median pitch to the reference
            semitone_shift = 12 * np.log2(PITCH_REFERENCE_HZ / median_pitch)
            # clamp so we correct *toward* normal without overcorrecting into
            # an unnatural register -- partial correction, not a full snap
            semitone_shift = float(np.clip(semitone_shift, -6, 6))
            report.pitch_shift_semitones = semitone_shift
            if ENABLE_PITCH_CORRECTION:
                report.notes.append(
                    f"pitch outlier ({median_pitch:.0f} Hz) -> shifting "
                    f"{semitone_shift:+.1f} semitones toward reference"
                )
                audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=semitone_shift)
            else:
                report.notes.append(
                    f"pitch outlier ({median_pitch:.0f} Hz) detected but NOT "
                    "corrected (ENABLE_PITCH_CORRECTION=False, see benchmark note)"
                )
        return audio
    except Exception as exc:
        logger.warning("Pitch step failed (%s), passing audio through unshifted.", exc)
        report.notes.append(f"pitch step failed: {exc}")
        return audio


def _detect_and_correct_speed(
    audio: np.ndarray, sr: int, report: PreprocessingReport
) -> np.ndarray:
    """Estimate speaking rate via onset density (a cheap proxy for
    syllable rate that doesn't require a phoneme recognizer) and
    time-stretch outliers back toward a normal rate.

    Fast speech gets stretched out (slowed down); slow/drawn-out speech
    gets compressed. Whisper's attention window is tuned on speech that
    falls in a fairly narrow tempo band, so bringing outliers back into
    that band before transcription reduces both mis-segmentation on fast
    speech and hallucinated repeats on very slow/drawn-out speech.
    """
    duration_s = len(audio) / sr
    if duration_s < 0.2:
        report.notes.append("clip too short for speed estimation; skipped")
        return audio

    # Same rationale as the pitch step: onset/time-stretch DSP goes through
    # numba-JIT'd librosa internals that have been observed to fail
    # transiently under memory pressure. Never let that crash a turn --
    # fall back to passing audio through unmodified.
    try:
        onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
        onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
        syllable_proxy_rate = len(onsets) / duration_s if duration_s > 0 else 0.0

        ratio = (
            syllable_proxy_rate / NORMAL_SPEECH_RATE
            if syllable_proxy_rate > 0
            else 1.0
        )
        report.estimated_speed_ratio = float(ratio)

        if ratio < SPEED_RATIO_LOW:
            report.speed_flagged = True
            report.speed_label = "slow"
        elif ratio > SPEED_RATIO_HIGH:
            report.speed_flagged = True
            report.speed_label = "fast"
        else:
            report.speed_label = "normal"

        if report.speed_flagged and syllable_proxy_rate > 0:
            # time_stretch's `rate` speeds audio up when >1, slows it down
            # when <1 -- so to correct toward normal we invert our ratio and
            # blend it partway to 1.0 (a full correction can overshoot on a
            # noisy onset estimate, so only close ~60% of the gap)
            target_stretch = 1.0 + 0.6 * (ratio - 1.0)
            target_stretch = float(np.clip(target_stretch, 0.5, 2.0))
            if ENABLE_SPEED_CORRECTION:
                report.notes.append(
                    f"speech rate flagged as {report.speed_label} "
                    f"(ratio={ratio:.2f}) -> time-stretch factor {target_stretch:.2f}"
                )
                audio = librosa.effects.time_stretch(audio, rate=target_stretch)
            else:
                report.notes.append(
                    f"speech rate flagged as {report.speed_label} (ratio={ratio:.2f}) "
                    "but NOT time-stretched (ENABLE_SPEED_CORRECTION=False, see "
                    "benchmark note -- unreliable on single isolated words)"
                )

        return audio
    except Exception as exc:
        logger.warning("Speed step failed (%s), passing audio through unstretched.", exc)
        report.notes.append(f"speed step failed: {exc}")
        return audio


def normalize_audio(
    audio_array: np.ndarray, sample_rate: int
) -> tuple[np.ndarray, PreprocessingReport]:
    """Run the full modulation-robustness preprocessing chain.

    Order matters: volume first (pitch/onset detectors are more reliable
    on a signal that isn't buried near the noise floor or clipped), then
    pitch, then speed (time-stretching after pitch-shifting avoids the
    two effects compounding in unpredictable ways).

    Returns (processed_audio, report) -- the report is what
    pipeline.py logs alongside the transcription and confidence score.
    """
    audio_array = np.asarray(audio_array, dtype=np.float32)
    report = PreprocessingReport()

    normalized, orig_rms, new_rms = _normalize_volume(audio_array)
    report.original_rms = orig_rms
    report.normalized_rms = new_rms

    normalized = _detect_and_correct_pitch(normalized, sample_rate, report)
    normalized = _detect_and_correct_speed(normalized, sample_rate, report)

    logger.debug("Preprocessing report: %s", report)
    return normalized, report
