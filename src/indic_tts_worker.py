"""
Standalone worker process for Tamil TTS (indic_tts.py).

Why this exists: torch/CUDA (indic_tts.py) and ctranslate2/CUDA
(faster-whisper, loaded in the main chat_web.py process) sharing one
small 4GB GPU turned out to not just be slow under memory pressure but
to occasionally segfault the entire process -- confirmed in live testing
by a chat_web.py server that died silently (no Python traceback at all,
just the process gone) while a Tamil TTS generation was running. A
Python try/except cannot catch a native crash like that; the only real
fix is putting the crash-prone code in a SEPARATE OS process, so if it
dies, only this worker dies -- chat_web.py (and Whisper, and English/
Hindi TTS) keeps running untouched.

Protocol: reads one JSON object per line from stdin --
{"text": "...", "out_path": "..."} -- synthesizes it, and writes one JSON
line to stdout in response -- {"ok": true} or {"ok": false, "error": "..."}.
The model loads once, on the first request, and stays warm for every
request after that (same as indic_tts.py's own module-level caching) for
as long as this worker process stays alive. Exits cleanly when stdin
closes (the parent closing the pipe, e.g. because it's shutting down).

Run as: python -m indic_tts_worker  (from src/, or with src/ on sys.path)
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main() -> None:
    from indic_tts import synthesize_to_file  # heavy import; deferred until the worker starts

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        response: dict[str, object]
        try:
            request = json.loads(line)
            synthesize_to_file(request["text"], request["out_path"])
            response = {"ok": True}
        except Exception as exc:  # noqa: BLE001 -- any failure reported to the parent, not raised
            response = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        sys.stdout.write(json.dumps(response) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
