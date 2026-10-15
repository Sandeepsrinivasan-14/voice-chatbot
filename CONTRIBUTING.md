# Contributing

Thanks for your interest! This is a small project; the bar is simple: keep CI
green and keep the offline/no-cloud guarantee intact.

## Ground rules

- **No cloud APIs.** Do not add a dependency or code path that sends audio,
  transcripts, or prompts to a remote service at inference time.
- **Tests must not need hardware.** The suite mocks the GPU, Ollama, Piper and
  the microphone (see `tests/conftest.py`), so it runs on any CI runner. Tests
  that genuinely need a CUDA GPU, large models, or real services must carry a
  marker (`@pytest.mark.gpu`, `slow`, or `integration`); `gpu` tests are
  skipped automatically when no CUDA device is present.
- **Never commit** model weights, logs, `.env`, recordings, or anything under
  `data/learning/`. `.gitignore` covers these; check `git status` before you
  commit.

## Development setup

Python 3.11 or 3.12.

```bash
python -m venv venv
source venv/bin/activate              # Windows: .\venv\Scripts\activate

# Linux only: sounddevice needs PortAudio at import time
sudo apt-get install -y libportaudio2

pip install -r requirements-dev.txt   # runtime + pytest, ruff, mypy, pre-commit
pre-commit install                    # optional but recommended
```

See [docs/REQUIREMENTS.md](docs/REQUIREMENTS.md) for what each
`requirements-*.txt` file is for.

## The checks CI runs

```bash
ruff check .      # lint (E, F, W, B rules; see pyproject.toml)
mypy              # type check src/ (config in pyproject.toml)
pytest -v         # full suite, no GPU/Ollama/mic required
```

Run all three before opening a pull request. `pre-commit run --all-files` runs
lint and type checking plus basic hygiene hooks.

## Pull requests

1. Branch from `master`; do not push to `master` directly.
2. Keep changes focused; add or update tests alongside behavior changes.
3. Fix lint findings properly (remove the unused import, handle the error)
   rather than adding blanket `# noqa` or rule ignores. A targeted `# noqa:
   <rule>` with a reason is fine when genuinely justified.
4. Describe *why* in the PR description, and note anything you could not test
   (e.g. GPU paths).
