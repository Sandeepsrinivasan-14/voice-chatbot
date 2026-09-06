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

---

## 5. Benchmark results

Recorded my own voice saying a fixed set of command words across 5 modulations
(normal, whispered, shouted, slow, fast), then ran each sample through the pipeline
with and without the preprocessing layer.

| Modulation | Raw Whisper accuracy | With preprocessing | Improvement |
|---|---|---|---|
| Normal | _fill in_ | _fill in_ | _fill in_ |
| Whispered | _fill in_ | _fill in_ | _fill in_ |
| Shouted | _fill in_ | _fill in_ | _fill in_ |
| Slow | _fill in_ | _fill in_ | _fill in_ |
| Fast | _fill in_ | _fill in_ | _fill in_ |
| **Overall** | _fill in_ | _fill in_ | _fill in_ |

Full results: `data/results/benchmark_results.csv`
Chart: `data/results/accuracy_comparison.png`

*(Fill this table in with your actual numbers from `python src/benchmark.py` before
submitting — this table is the single most persuasive artifact in the whole project.)*

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
ollama pull <model-name>

# Verify everything is installed correctly
python verify_setup.py
```

---

## 7. Running the chatbot

```bash
python src/cli.py
```

- Speak naturally — the system detects when you stop talking.
- Live transcription, confidence score, and the LLM's response are shown in the CLI.
- Say "exit" or press `Ctrl+C` to quit.

To reproduce the benchmark:

```bash
python src/benchmark.py
```

To verify offline operation, disable networking and re-run `src/cli.py` — the system
should function identically.

---

## 8. Known limitations

- Preprocessing improves robustness but does not fully replace training on modulated
  speech data — extreme modulation edge cases may still reduce accuracy somewhat.
- Confidence thresholds were tuned on a personal voice sample set; a different
  speaker's voice profile may need threshold re-tuning.
- Latency depends on local GPU capability — tested and optimized on [your GPU model].

---

## 9. What this project demonstrates

- Understanding of where failure modes are introduced in a real-time speech pipeline,
  and where to intercept them (preprocessing before STT, confidence check before LLM).
- Ability to combine multiple open-source/local models into a coherent, tested system
  rather than treating them as black boxes.
- A measured, evidence-based approach to solving an ambiguous requirement — the
  before/after benchmark exists specifically to prove the solution works, not just
  claim it does.
