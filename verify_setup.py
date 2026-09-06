"""
Phase 1: verify every piece of the stack is installed and working
*before* wiring them together. Run after installing requirements.txt,
Ollama, and Piper:

    python verify_setup.py

Checks, in order:
  1. CUDA is visible to faster-whisper's CTranslate2 backend.
  2. faster-whisper can load a model and transcribe a short synthetic clip.
  3. Ollama is running locally and responds to a test prompt.
  4. Piper can synthesize a test sentence to a .wav file.

Each check prints PASS/FAIL independently so one broken piece doesn't
hide the status of the others.
"""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np


def check_cuda() -> bool:
    print("\n[1/4] Checking CUDA availability for faster-whisper (CTranslate2)...")
    try:
        import ctranslate2

        count = ctranslate2.get_cuda_device_count()
        if count > 0:
            print(f"  PASS -- {count} CUDA device(s) visible to CTranslate2.")
            return True
        print("  WARN -- no CUDA device visible. Whisper will run on CPU "
              "(slower, but functional).")
        return False
    except Exception as exc:
        print(f"  FAIL -- {exc}")
        return False


def check_faster_whisper() -> bool:
    print("\n[2/4] Loading faster-whisper and transcribing a synthetic test clip...")
    try:
        from faster_whisper import WhisperModel

        # 1s of low-amplitude noise -- we're only checking that the model
        # loads and runs end-to-end, not that it transcribes correctly.
        audio = (np.random.randn(16000).astype(np.float32)) * 0.01

        device = "cuda"
        try:
            model = WhisperModel("small", device="cuda", compute_type="int8_float16")
            # Construction can succeed on Windows even when cuBLAS/cuDNN
            # DLLs are missing -- the failure only shows up on the first
            # real inference call, so we force one here before trusting GPU.
            list(model.transcribe(audio, language="en")[0])
        except Exception as exc:
            print(f"  GPU path unusable ({exc}), falling back to CPU.")
            device = "cpu"
            model = WhisperModel("small", device="cpu", compute_type="int8")

        segments, info = model.transcribe(audio, language="en")
        list(segments)  # force generator to run
        print(f"  PASS -- model loaded and ran inference on {device}.")
        return True
    except Exception as exc:
        print(f"  FAIL -- {exc}")
        return False


def check_ollama() -> bool:
    print("\n[3/4] Checking Ollama server + a test generation...")
    try:
        import requests

        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        resp.raise_for_status()
        models = [m["name"] for m in resp.json().get("models", [])]
        if not models:
            print("  WARN -- Ollama is running but no models are pulled. Run:\n"
                  "         ollama pull llama3.2:3b")
            return False
        print(f"  Ollama is running. Installed models: {models}")

        model_name = models[0]
        # First generation after Ollama starts pays a one-time cost to
        # load model weights into (V)RAM -- give it real headroom rather
        # than timing out and reporting a false FAIL.
        print(f"  Cold-starting '{model_name}' (first load can take a while)...")
        gen = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model_name, "prompt": "Say OK.", "stream": False},
            timeout=180,
        )
        gen.raise_for_status()
        text = gen.json().get("response", "").strip()
        print(f"  PASS -- '{model_name}' responded: {text[:80]!r}")
        return True
    except requests.exceptions.ConnectionError:
        print("  FAIL -- could not connect to http://localhost:11434. "
              "Run `ollama serve` (or start the Ollama app) first.")
        return False
    except Exception as exc:
        print(f"  FAIL -- {exc}")
        return False


def _default_piper_executable() -> str:
    # piper-tts installs its console script next to the interpreter it was
    # pip-installed into (venv/Scripts on Windows) -- resolve relative to
    # sys.executable so this works without the venv being `activate`d.
    candidate = os.path.join(
        os.path.dirname(sys.executable), "piper.exe" if os.name == "nt" else "piper"
    )
    return candidate if os.path.exists(candidate) else "piper"


def check_piper() -> bool:
    print("\n[4/4] Checking Piper TTS synthesis...")
    piper_exe = os.environ.get("PIPER_EXECUTABLE", _default_piper_executable())
    model_path = os.environ.get(
        "PIPER_MODEL_PATH", os.path.join("models", "piper", "en_US-lessac-medium.onnx")
    )
    if not os.path.exists(model_path):
        print(f"  FAIL -- voice model not found at {model_path}. Download one from "
              "https://github.com/rhasspy/piper/releases (.onnx + .onnx.json) "
              "or set PIPER_MODEL_PATH.")
        return False
    out_path = os.path.join("logs", "_verify_piper_test.wav")
    os.makedirs("logs", exist_ok=True)
    try:
        proc = subprocess.run(
            [piper_exe, "--model", model_path, "--output_file", out_path],
            input=b"This is a test of the offline text to speech system.",
            capture_output=True,
        )
        if proc.returncode != 0:
            print(f"  FAIL -- piper exited {proc.returncode}: "
                  f"{proc.stderr.decode(errors='replace')}")
            return False
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            print(f"  PASS -- wrote {out_path} ({os.path.getsize(out_path)} bytes).")
            return True
        print("  FAIL -- piper ran but produced no output file.")
        return False
    except FileNotFoundError:
        print(f"  FAIL -- '{piper_exe}' executable not found on PATH. "
              "Install piper-tts (pip) or download the piper binary.")
        return False
    except Exception as exc:
        print(f"  FAIL -- {exc}")
        return False


def main():
    print("=" * 60)
    print("Local Voice Chatbot -- setup verification")
    print("=" * 60)
    results = {
        "CUDA": check_cuda(),
        "faster-whisper": check_faster_whisper(),
        "Ollama": check_ollama(),
        "Piper": check_piper(),
    }
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for name, ok in results.items():
        print(f"  {name:<16} {'PASS' if ok else 'FAIL/WARN'}")

    critical = ["faster-whisper", "Ollama", "Piper"]
    if all(results[c] for c in critical):
        print("\nAll critical components OK. Ready for src/cli.py.")
        sys.exit(0)
    else:
        print("\nFix the FAIL items above before running the pipeline.")
        sys.exit(1)


if __name__ == "__main__":
    main()
