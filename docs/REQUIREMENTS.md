# Requirements files

Dependencies are split into small files so you only install what you use.
`requirements.txt` is the base; every other file builds on it.

| File | Install when | Notes |
| :--- | :--- | :--- |
| [`requirements.txt`](../requirements.txt) | Always (runtime) | Lower-bound pins. STT (faster-whisper), VAD, audio I/O, Piper TTS, Flask/Waitress, python-dotenv. CPU-only is fully supported. |
| [`requirements-dev.txt`](../requirements-dev.txt) | Developing / running CI locally | Includes `-r requirements.txt`, plus pytest, ruff, mypy, type stubs, pre-commit. No GPU wheels: the test suite mocks all GPU work. |
| [`requirements-gpu.txt`](../requirements-gpu.txt) | NVIDIA GPU for Whisper | Includes `-r requirements.txt`, plus CUDA 12 cuBLAS/cuDNN wheels (~1.3 GB). On Windows `src/cuda_dlls.py` adds them to `PATH` at runtime. |
| [`requirements-indic-tts.txt`](../requirements-indic-tts.txt) | Tamil speech output | torch, transformers (<5), coqui-tts. ~1 GB. Installed automatically on first Tamil TTS use unless `TAMIL_TTS_ENABLED=0`. |
| [`requirements-training.txt`](../requirements-training.txt) | Fine-tuning STT from corrections | torch, transformers, peft. Only for `python -m learning.training`; the real-time pipeline never needs it. |
| [`requirements-frozen.txt`](../requirements-frozen.txt) | Reproducing a known-good CPU environment | Fully resolved `pip freeze` snapshot. Prefer `requirements.txt` unless you need exact versions; it can lag behind Dependabot updates. |

## Common combinations

```bash
pip install -r requirements.txt                              # run on CPU
pip install -r requirements-gpu.txt                          # run with CUDA
pip install -r requirements-dev.txt                          # contribute
pip install -r requirements.txt -r requirements-indic-tts.txt  # + Tamil TTS
```

## Tooling configuration

All tool config lives in [`pyproject.toml`](../pyproject.toml): pytest
(markers, strict-markers), ruff (rule set), mypy (scope, per-module
`ignore_missing_imports` for packages that ship no types).

## Why not one consolidated file?

The heavy extras (CUDA wheels, torch) are several GB; folding them into one
file would make a CPU install pull gigabytes it never uses. Splitting keeps the
default install small while this page keeps the whole picture in one place.
Dependabot watches all of the files at the repository root.
