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
