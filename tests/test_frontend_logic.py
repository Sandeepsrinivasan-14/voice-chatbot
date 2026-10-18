"""
Unit tests for the framework-free, DOM-free parts of the browser UI
(src/web_chat/js/wake.js and format.js), run under Node. The rest of the
front end needs a real browser and is covered by the asset/CSP tests in
tests/test_web_assets.py; the pure logic is worth pinning down precisely
because a regression there (a missed wake phrase, a corrupted export) is
silent.

Skipped when `node` is not installed (GitHub's runners ship with it).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess

import pytest

JS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "web_chat", "js"
)

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")


def _run_node(script: str):
    # `--input-type=module` lets the snippet import the browser ES modules as-is.
    wake = os.path.join(JS_DIR, "wake.js").replace("\\", "/")
    fmt = os.path.join(JS_DIR, "format.js").replace("\\", "/")
    prelude = f'import {{ matchWakePhrase }} from "file:///{wake}";\nimport {{ buildExport, stamp }} from "file:///{fmt}";\n'
    result = subprocess.run(
        ["node", "--input-type=module", "-e", prelude + script],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


class TestWakePhrase:
    @pytest.mark.parametrize(
        ("text", "remainder"),
        [
            ("hey chatbot", ""),
            ("Hey, Chatbot", ""),
            ("hey  chatbot what's the weather", "what's the weather"),
            ("OK chatbot, Tell me a joke.", "Tell me a joke."),
            ("uh hi chatbot are you there", "are you there"),
        ],
    )
    def test_detects_phrase_and_keeps_original_casing_in_remainder(self, text, remainder):
        out = _run_node(f"console.log(JSON.stringify(matchWakePhrase({json.dumps(text)})));")
        assert out == {"hasWake": True, "remainder": remainder}

    @pytest.mark.parametrize(
        "text",
        [
            "",
            "what is a chatbot",
            "the weather is nice today, hey chatbot",  # phrase too far into the sentence
            "hey chatbots",
        ],
    )
    def test_ignores_everything_else(self, text):
        out = _run_node(f"console.log(JSON.stringify(matchWakePhrase({json.dumps(text)})));")
        assert out["hasWake"] is False


class TestTranscriptExport:
    LOG = (
        "[{role:'user',text:'Hello there',time:new Date(2026,0,2,3,4,5),language:'en',meta:'confidence -0.2'},"
        "{role:'assistant',text:'Hi!\\nHow can I help?',time:new Date(2026,0,2,3,4,9),interrupted:true}]"
    )

    def test_json_is_structured_and_omits_empty_fields(self):
        out = _run_node(
            f"const log={self.LOG}; console.log(buildExport('json', log, new Date(Date.UTC(2026,0,2))));"
        )
        assert out["app"] == "Local Voice Chatbot"
        user, bot = out["messages"]
        assert user == {
            "role": "user",
            "text": "Hello there",
            "time": user["time"],
            "language": "en",
            "note": "confidence -0.2",
        }
        assert bot["interrupted"] is True and "language" not in bot

    def test_markdown_and_text_contain_every_message(self):
        md = _run_node(
            f"const log={self.LOG}; console.log(JSON.stringify(buildExport('md', log, new Date(2026,0,2))));"
        )
        txt = _run_node(
            f"const log={self.LOG}; console.log(JSON.stringify(buildExport('txt', log, new Date(2026,0,2))));"
        )
        assert md.startswith("# Voice chat transcript")
        assert "**You**" in md and "**Assistant**" in md
        assert "Hi!  \nHow can I help?" in md  # markdown hard line break
        assert "You: Hello there" in txt and "Assistant: Hi!\nHow can I help?" in txt

    def test_filename_stamp_is_sortable(self):
        assert (
            _run_node("console.log(JSON.stringify(stamp(new Date(2026, 0, 2, 3, 4))));")
            == "2026-01-02-0304"
        )
