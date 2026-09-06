"""
Phase 7: polished CLI front-end for the voice chatbot.

Wraps pipeline.py with:
  - a rich status display (listening / transcription / confidence / response)
  - upfront checks for the three things that fail loudly and unhelpfully
    if missing: no microphone, Ollama not running, model files absent
  - a clean way to exit (Ctrl+C or saying "exit")

Run with:
    python src/cli.py
"""

from __future__ import annotations

import os
import sys
import time

import requests
import sounddevice as sd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

sys.path.insert(0, os.path.dirname(__file__))
import pipeline  # noqa: E402
from confidence_check import Decision  # noqa: E402

console = Console()


# --------------------------------------------------------------------
# Preflight checks -- fail with a clear message, not a stack trace
# --------------------------------------------------------------------
def check_microphone() -> bool:
    try:
        devices = sd.query_devices()
        inputs = [d for d in devices if d["max_input_channels"] > 0]
        if not inputs:
            console.print(
                "[bold red]No microphone detected.[/bold red] "
                "Plug one in (or check Windows sound settings / privacy "
                "permissions for microphone access) and try again."
            )
            return False
        return True
    except Exception as exc:
        console.print(f"[bold red]Could not query audio devices:[/bold red] {exc}")
        return False


def check_ollama() -> bool:
    try:
        resp = requests.get(f"{pipeline.OLLAMA_URL}/api/tags", timeout=3)
        resp.raise_for_status()
        models = [m["name"] for m in resp.json().get("models", [])]
        if pipeline.OLLAMA_MODEL not in models and not any(
            m.startswith(pipeline.OLLAMA_MODEL.split(":")[0]) for m in models
        ):
            console.print(
                f"[bold yellow]Warning:[/bold yellow] Ollama is running, but "
                f"'{pipeline.OLLAMA_MODEL}' isn't pulled. Run:\n"
                f"    ollama pull {pipeline.OLLAMA_MODEL}"
            )
        return True
    except requests.exceptions.ConnectionError:
        console.print(
            "[bold red]Ollama isn't running.[/bold red] Start it with:\n"
            "    ollama serve\n"
            "(or launch the Ollama desktop app), then try again."
        )
        return False
    except Exception as exc:
        console.print(f"[bold red]Could not reach Ollama:[/bold red] {exc}")
        return False


def check_piper_model() -> bool:
    if not os.path.exists(pipeline.PIPER_MODEL_PATH):
        console.print(
            f"[bold red]Piper voice model not found:[/bold red] "
            f"{pipeline.PIPER_MODEL_PATH}\n"
            "Download a voice from https://github.com/rhasspy/piper/releases "
            "(the .onnx and matching .onnx.json) and set PIPER_MODEL_PATH, "
            "or place it at the path above."
        )
        return False
    return True


def run_preflight() -> bool:
    console.rule("[bold]Preflight checks[/bold]")
    ok = True
    for name, check in [
        ("Microphone", check_microphone),
        ("Ollama server", check_ollama),
        ("Piper voice model", check_piper_model),
    ]:
        passed = check()
        status = "[green]OK[/green]" if passed else "[red]FAILED[/red]"
        console.print(f"  {name}: {status}")
        ok = ok and passed
    return ok


# --------------------------------------------------------------------
# Main interactive loop
# --------------------------------------------------------------------
def run_cli():
    pipeline.setup_logging()
    console.print(Panel.fit(
        "[bold cyan]Local Voice Chatbot[/bold cyan]\n"
        "Fully offline: faster-whisper + Ollama + Piper\n"
        "Say 'exit' or press Ctrl+C to quit.",
        border_style="cyan",
    ))

    if not run_preflight():
        console.print("\n[bold red]Fix the issues above and re-run.[/bold red]")
        sys.exit(1)

    with console.status("[bold green]Loading faster-whisper model..."):
        whisper_model, device = pipeline.load_whisper_model()
    console.print(f"[green]Whisper ready on {device}.[/green]\n")

    try:
        while True:
            console.print("[dim]Listening...[/dim]")
            audio = pipeline.capture_audio()
            if audio is None:
                continue

            processed_audio, report = pipeline.normalize_audio(audio, pipeline.SAMPLE_RATE)

            t0 = time.time()
            text, conf_result = pipeline.transcribe_with_confidence(whisper_model, processed_audio)
            stt_s = time.time() - t0

            if conf_result.decision == Decision.ASK_TO_REPEAT:
                console.print("[yellow]Low confidence -- asking you to repeat.[/yellow]")
                wav = pipeline.synthesize_speech("Sorry, could you say that again?")
                pipeline.play_audio(wav)
                continue

            table = Table(show_header=False, box=None)
            table.add_row("Transcription:", f"[bold]{text}[/bold]")
            table.add_row("Confidence:", f"{conf_result.avg_logprob:.2f}")
            table.add_row("STT time:", f"{stt_s:.2f}s")
            console.print(table)

            if text.strip().lower() in pipeline.EXIT_WORDS:
                console.print("[cyan]Goodbye![/cyan]")
                break
            if not text.strip():
                continue

            console.print("[dim]Thinking...[/dim]")
            t0 = time.time()
            response_parts = []
            try:
                response = pipeline.generate_response(
                    text, on_token=lambda tok: response_parts.append(tok)
                )
            except RuntimeError as exc:
                console.print(f"[bold red]{exc}[/bold red]")
                continue
            llm_s = time.time() - t0
            console.print(Panel(response, title="Response", border_style="green"))
            console.print(f"[dim]LLM time: {llm_s:.2f}s[/dim]")

            wav = pipeline.synthesize_speech(response)
            pipeline.play_audio(wav)

    except KeyboardInterrupt:
        console.print("\n[cyan]Interrupted. Goodbye![/cyan]")


if __name__ == "__main__":
    run_cli()
