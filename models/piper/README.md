# Piper voice models go here

Download a voice (an `.onnx` file + its matching `.onnx.json` config) from
the official Piper voices repo:

https://huggingface.co/rhasspy/piper-voices

A good default for this project: `en_US-lessac-medium` (natural, English,
reasonable file size). Place both files in this folder:

```
models/piper/en_US-lessac-medium.onnx
models/piper/en_US-lessac-medium.onnx.json
```

`src/pipeline.py`, `src/cli.py`, and `verify_setup.py` all default to this
path via `PIPER_MODEL_PATH` — override that environment variable if you
use a different voice or location.

## Hindi voice

A second voice, `hi_IN-pratham-medium`, is also committed here for Hindi
TTS (`config.py`'s `piper_model_path_hi` / `PIPER_MODEL_PATH_HI`):

```
models/piper/hi_IN-pratham-medium.onnx
models/piper/hi_IN-pratham-medium.onnx.json
```

## Tamil

Piper ships no Tamil voice at all. Tamil speech goes through a separate,
heavier engine instead — see `src/indic_tts.py` and
`requirements-indic-tts.txt`.
