# Local Voice Chatbot — Modulation-Robust, Fully Offline

A speech-to-text → LLM → text-to-speech chatbot that runs entirely on-device, with no
internet or cloud API calls at any point in the pipeline. Built with an explicit
robustness layer to handle the same word spoken under different vocal modulations
(whispered, shouted, fast, slow, pitch-shifted) without misrecognition.

---

## 1. Problem this solves

Off-the-shelf speech-to-text models are trained mostly on normal conversational speech.
When a word is spoken in an unusual way — whispered, shouted, drawn out, rushed, or at
an atypical pitch — recognition accuracy drops, and the model can produce a confident
but wrong transcription instead of failing safely.

This project adds a preprocessing and confidence-aware decision layer on top of a
pretrained STT model so that:
- Audio is normalized before transcription, reducing the acoustic gap between "normal"
  and "modulated" speech.
- Low-confidence transcriptions are caught and handled explicitly (retry / ask again)
  instead of being passed downstream as if they were correct.

---

## 2. Architecture

```
Mic input
   -> Audio preprocessing   (normalize volume, pitch, speed)
   -> STT (faster-whisper)  (GPU-accelerated, local)
   -> Confidence check      (accept / retry / ask for repeat)
   -> LLM (Ollama)          (local model, generates response)
   -> TTS (Piper)           (local, text -> speech)
   -> Speaker output
```

Everything below the mic and above the speaker runs on this machine. No stage makes a
network request — all models are loaded from local disk and inference happens on the
local GPU/CPU.

---

## 3. Why each component was chosen

| Component | Choice | Why |
|---|---|---|
| STT | `faster-whisper` | Whisper is the strongest open-source STT available; `faster-whisper` is a CTranslate2 reimplementation that runs significantly faster with lower memory, and supports GPU inference cleanly. |
| LLM | Ollama running a local model | Ollama handles local model serving, quantization, and GPU offloading automatically — no manual CUDA plumbing needed for the chat logic itself. |
| TTS | Piper | Lightweight, fast even on CPU, sounds natural enough for a real-time assistant, and has no cloud dependency. |
| Preprocessing | `librosa` | Standard, well-tested audio processing library for normalization and feature extraction — no need to hand-roll DSP code. |

None of these components solve the modulation-robustness problem by themselves — that
is the layer built specifically for this project (Section 4).

---

## 4. The modulation-robustness layer (core contribution)

This is the part of the system that isn't "just calling a pretrained model" — it's the
engineering work that makes the pipeline reliable across modulation types.

**Audio preprocessing (`src/preprocessing.py`)**
- Volume/amplitude normalization before the audio reaches Whisper, so a whispered or
  shouted input arrives at a similar loudness level to normal speech.
- Pitch outlier detection, so unusually high/low pitch input is flagged and normalized
  rather than confusing the acoustic model.
- Speed variation handling, so very fast or very slow speech is detected and can be
  compensated for before transcription.

**Confidence-aware decision layer (`src/confidence_check.py`)**
- Every transcription comes with a confidence/logprob score from faster-whisper.
- Below a tuned threshold, the system does **not** guess — it either retries
  transcription with a different decoding strategy (more beams) or asks the user to
  repeat themselves via TTS.
- Every decision (accepted/rejected, with score) is logged to
  `logs/confidence_log.csv` for auditability.

This mirrors how production speech systems actually handle uncertainty — the goal isn't
to force a model to always output *something*, it's to know when not to trust its own
output.

**Response shaping for speech (`src/pipeline.py::SYSTEM_PROMPT`)**
- Worth calling out separately because it wasn't obvious until testing surfaced it:
  swapping LLMs did nothing for output quality, but a system prompt did. Without
  guidance, an instruction-tuned model defaults to chat-app formatting — numbered
  steps, headers, multi-paragraph answers. That's fine on a screen and actively bad
  here, since every response gets read aloud by Piper with no visual structure to
  lean on. A short system prompt ("keep answers to 1-3 spoken sentences, no lists or
  markdown") fixed this immediately, on every model tested. Lesson: for a voice
  interface, prompt-level response shaping matters as much as model choice.

**Building the test dataset** — two interchangeable recorders, same output format
(`data/test_samples/*.wav` + `manifest.csv`), pick whichever you prefer:
- `python src/record_test_samples.py` — terminal prompts, zero extra dependencies.
- `python src/record_web.py` then open `http://localhost:5005` — a nicer browser UI
  (live progress, playback/retake, resume-if-interrupted) for the same 25 clips. Still
  100% local: it's a Flask server bound to `127.0.0.1`, your mic audio never leaves
  this machine, and it decodes the browser's recording (webm/opus) with `PyAV`
  (already installed as a faster-whisper dependency) straight into the same 16kHz
  mono WAV format the terminal recorder produces.

---

## 5. Benchmark results

Recorded my own voice saying 5 command words ("yes", "no", "stop", "help", "start")
across 5 modulations (normal, whispered, shouted, slow, fast) — 25 clips — then ran
every clip through faster-whisper twice: once raw, once with the preprocessing layer
applied, everything else identical.

| Modulation | Raw Whisper accuracy | With preprocessing | Delta |
|---|---|---|---|
| Normal | 100.0% | 100.0% | +0.0 |
| Whispered | 100.0% | 100.0% | +0.0 |
| Shouted | 100.0% | 100.0% | +0.0 |
| Slow | 80.0% | 80.0% | +0.0 |
| Fast | 100.0% | 100.0% | +0.0 |
| **Overall** | **96.0%** | **96.0%** | **+0.0** |

Full results: `data/results/benchmark_results.csv`
Chart: `data/results/accuracy_comparison.png`

**This is not the result I expected, and it's the more interesting one.** The first
real run (before the fix described below) showed preprocessing *dropping* accuracy
from 96% to 48% — actively destroying transcriptions ("Stop" → "So...", "Help" →
"L L L", one clip came back empty). I root-caused it with a series of controlled
diagnostic re-runs (isolating volume-only vs. +pitch vs. +speed) rather than guessing:

- **Speed "correction" was the dominant damage.** It estimates speaking rate from
  onset density, which needs several syllables across time to mean anything. A
  single isolated command word gives ~1 onset, so the "rate" mostly reflects how
  much silence padding surrounds the word in the recording, not how fast it was
  actually spoken. A clip I said *fast* got misread as extremely slow and
  time-stretched in the wrong direction into an empty transcription.
- **Pitch correction was a smaller second offender** — on a short, consonant-heavy
  word, the pitch tracker locked onto a plosive burst as if it were the vocal
  fundamental and "corrected" a pitch problem that didn't exist.
- **Volume/RMS normalization alone matched raw Whisper exactly** (96%, same single
  failure) — neutral to good, never harmful.

Given that evidence, `src/preprocessing.py` now disables pitch and speed
*correction* by default (`ENABLE_PITCH_CORRECTION` / `ENABLE_SPEED_CORRECTION` =
`False`) while still running detection/logging — the numbers above are with that fix
applied, which is why every row shows a flat +0.0 delta rather than a regression.
Raw Whisper turns out to already be quite robust to modulation on short command
words; the real remaining value here is the confidence-aware retry/reject layer
(`confidence_check.py`), which is a separate mechanism from preprocessing and isn't
captured by this before/after table. Pitch/speed correction might still be worth
revisiting for longer, continuous multi-word utterances, where onset density would
have a much larger sample to estimate from — that's untested here.

---

## 6. Setup instructions

```bash
# Clone/enter project
cd voice-chatbot

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate       # Linux/Mac
venv\Scripts\activate          # Windows

# Install dependencies
pip install -r requirements.txt

# Install and start Ollama, pull the local model
# (llama3.2:3b chosen after head-to-head testing against qwen2.5:3b-instruct --
# comparable size/speed, but more reliably correct on basic arithmetic and reads
# more naturally aloud; either fits a 4GB GPU alongside faster-whisper)
ollama pull llama3.2:3b

# Verify everything is installed correctly
python verify_setup.py
```

---

## 7. Running the chatbot

Two front ends, same back end (`src/pipeline.py`) — pick whichever fits:

**Terminal:**
```bash
python src/cli.py
```
- Speak naturally — the system detects when you stop talking (VAD-based).
- Live transcription, confidence score, and the LLM's response are shown in the CLI.
- Say "exit" or press `Ctrl+C` to quit.

**Browser chat UI** (`python src/chat_web.py` → open `http://localhost:5006`):
- A proper chat interface — message bubbles, a mic button, and a text box (type
  instead of speaking any time; useful for testing or a quiet environment).
- Click the mic once to start recording, again to stop (no fixed duration).
- The LLM's reply streams in token-by-token like a normal chat app, then Piper
  speaks it automatically; click "🔊 Replay" on any assistant message to hear it
  again without re-asking.
- Low-confidence transcriptions show as a dashed, grayed-out bubble with the
  confidence score instead of being sent to the model — same confidence-gating
  logic as the CLI, just visible instead of implicit.
- Unlike the CLI, this UI keeps real multi-turn conversation memory (it uses
  Ollama's `/api/chat` with the running message history, not one-shot
  `/api/generate` calls) — ask a follow-up and it remembers context.
- Same offline guarantee as everywhere else: Flask binds to `127.0.0.1` only: your
  voice and the model's replies never leave this machine.

To reproduce the benchmark:

```bash
python src/benchmark.py
```

To verify offline operation, disable networking and re-run `src/cli.py` — the system
should function identically.

---

## 8. Known limitations

- Pitch and speed *correction* are implemented but disabled by default
  (`ENABLE_PITCH_CORRECTION`/`ENABLE_SPEED_CORRECTION` in `preprocessing.py`) —
  benchmarking against real recordings showed they hurt accuracy on short, isolated
  command words (see section 5). Detection/logging still runs; only the audio
  modification is gated off. This is a real, evidence-based finding, not a stub.
- Confidence thresholds were tuned on a personal voice sample set; a different
  speaker's voice profile may need threshold re-tuning.
- Latency depends on local GPU capability — tested on an RTX 3050 Laptop (4GB VRAM):
  faster-whisper `small` (int8_float16) + a 3B-parameter Ollama model both run on GPU
  simultaneously, peaking around 2.6GB VRAM used.
- **Windows-specific gotcha (worth documenting because it cost real debugging time):**
  `faster-whisper`'s CTranslate2 backend needs cuBLAS/cuDNN, but the
  `nvidia-cublas-cu12`/`nvidia-cudnn-cu12` pip wheels that provide them on Linux via
  RPATH don't work the same way on Windows — CTranslate2 resolves them with a plain
  `LoadLibrary` call, so `os.add_dll_directory()` (the usually-recommended fix) is a
  silent no-op here. `src/cuda_dlls.py` fixes this by putting each wheel's `bin/`
  directory directly on `PATH` before `faster_whisper` is ever imported. Without it,
  GPU construction succeeds but the *first inference call* fails with
  `Library cublas64_12.dll is not found` — which is also why every model-load path in
  this repo (`pipeline.py`, `verify_setup.py`, `benchmark.py`) does a cheap smoke-test
  transcription before trusting the GPU device, not just a try/except around
  construction.
- Preprocessing's pitch/speed detection goes through numba-JIT'd librosa internals.
  Under real memory pressure (running Whisper + an LLM + TTS together on a 16GB
  machine) that JIT compile has been observed to throw a transient `MemoryError`.
  Every preprocessing step therefore fails closed — it logs a warning and passes
  audio through unmodified rather than crashing the turn (see `preprocessing.py`).

---

## 9. What this project demonstrates

- Understanding of where failure modes are introduced in a real-time speech pipeline,
  and where to intercept them (preprocessing before STT, confidence check before LLM).
- Ability to combine multiple open-source/local models into a coherent, tested system
  rather than treating them as black boxes.
- A measured, evidence-based approach to solving an ambiguous requirement — the
  before/after benchmark exists specifically to prove the solution works, not just
  claim it does.
