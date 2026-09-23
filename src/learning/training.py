"""
LoRA/PEFT fine-tuning of the base Whisper model on a versioned dataset
(dataset_builder.py), producing a candidate CTranslate2 model that
model_registry.py tracks and evaluation.py scores before anything is
promoted to production.

Deliberately isolated from the live transcribe path: src/chat_web.py never
imports this module, and nothing in src/pipeline.py changes because this
module exists. Run it as a script, not as part of the web app:

    python -m learning.training --dataset-version v1

Needs torch, transformers, and peft, which are intentionally NOT in
requirements.txt / requirements-gpu.txt / requirements-dev.txt -- see
requirements-training.txt. Importing this module does not require them
(only calling train() does, with a clear error) so `import learning`
inside chat_web.py stays safe on a machine that never installed the
training extras. Converting the trained checkpoint to CTranslate2 uses
ct2-transformers-converter, which ships with the `ctranslate2` package
already in requirements.txt -- no extra dependency for that step.

This script trains and reports; it never promotes a model to production
on its own (see model_registry.promote_to_production's docstring) -- pass
--promote to opt in, and even then promotion is refused unless the
recorded evaluation actually recommends it.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass, field

import soundfile as sf

from . import dataset_builder, evaluation, model_registry

_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../src
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

DEFAULT_BASE_MODEL = "openai/whisper-small"  # HF hub name; matches CONFIG.whisper_model_size="small"
# Attention projection layers -- the standard LoRA target for Whisper's
# transformer stack; adapts a small fraction of the ~240M base parameters.
DEFAULT_LORA_TARGET_MODULES = ["q_proj", "v_proj"]


class MissingTrainingDependencies(RuntimeError):
    """Raised by train() -- not at import time -- when torch/transformers/
    peft aren't installed. See requirements-training.txt.
    """


@dataclass
class TrainingConfig:
    base_model: str = DEFAULT_BASE_MODEL
    epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 1e-4
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: list(DEFAULT_LORA_TARGET_MODULES))


def _import_training_stack():
    try:
        import torch
        from peft import LoraConfig, get_peft_model
        from torch.utils.data import DataLoader
        from torch.utils.data import Dataset as TorchDataset
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
    except ImportError as exc:
        raise MissingTrainingDependencies(
            "STT fine-tuning needs torch, transformers, and peft, which "
            "aren't installed. Run: pip install -r requirements-training.txt "
            "(a multi-GB download; not needed for the live app)."
        ) from exc
    return (
        torch, DataLoader, TorchDataset,
        WhisperForConditionalGeneration, WhisperProcessor,
        LoraConfig, get_peft_model,
    )


def _make_manifest_dataset_class(torch_dataset_base, processor, sample_rate: int = 16000):
    class _ManifestDataset(torch_dataset_base):
        def __init__(self, rows: list[dict]):
            self.rows = rows

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, idx: int) -> dict:
            row = self.rows[idx]
            audio, sr = sf.read(row["audio_path"], dtype="float32")
            if sr != sample_rate:
                raise ValueError(
                    f"Expected {sample_rate}Hz audio, got {sr}Hz for {row['audio_path']}"
                )
            input_features = processor.feature_extractor(
                audio, sampling_rate=sample_rate
            ).input_features[0]
            labels = processor.tokenizer(row["text"]).input_ids
            return {"input_features": input_features, "labels": labels}

    return _ManifestDataset


def _collate(batch: list[dict], processor, torch) -> dict:
    input_features = torch.stack([torch.tensor(b["input_features"]) for b in batch])
    label_lists = [b["labels"] for b in batch]
    max_len = max(len(x) for x in label_lists)
    # -100 is the ignore-index PyTorch's cross-entropy loss skips, so
    # padding never contributes to the training signal.
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    for i, ids in enumerate(label_lists):
        labels[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
    return {"input_features": input_features, "labels": labels}


def train(dataset_manifest_path: str, output_dir: str, *, config: TrainingConfig) -> str:
    """Fine-tunes config.base_model with LoRA on the "train" split of the
    given dataset manifest, merges the adapter into the base weights, and
    converts the result to CTranslate2 (loadable by faster-whisper exactly
    like the stock model). Returns the CTranslate2 output directory.
    Raises MissingTrainingDependencies if torch/transformers/peft aren't
    installed, or ValueError if the manifest has no training-split rows.
    """
    (
        torch, DataLoader, TorchDataset,
        WhisperForConditionalGeneration, WhisperProcessor,
        LoraConfig, get_peft_model,
    ) = _import_training_stack()

    rows = [
        r for r in dataset_builder.load_manifest_rows(dataset_manifest_path)
        if r["split"] == "train"
    ]
    if not rows:
        raise ValueError(f"No training-split rows in {dataset_manifest_path}.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = WhisperProcessor.from_pretrained(
        config.base_model, language="english", task="transcribe"
    )
    model = WhisperForConditionalGeneration.from_pretrained(config.base_model)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.to(device)
    model.train()

    dataset_cls = _make_manifest_dataset_class(TorchDataset, processor)
    loader = DataLoader(
        dataset_cls(rows),
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=lambda batch: _collate(batch, processor, torch),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    for epoch in range(config.epochs):
        total_loss = 0.0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        mean_loss = total_loss / max(1, len(loader))
        print(f"[training] epoch {epoch + 1}/{config.epochs}  mean loss = {mean_loss:.4f}")

    merged = model.merge_and_unload()  # fold the LoRA adapter into the base weights
    hf_dir = os.path.join(output_dir, "hf")
    os.makedirs(hf_dir, exist_ok=True)
    merged.save_pretrained(hf_dir)
    processor.save_pretrained(hf_dir)

    ct2_dir = os.path.join(output_dir, "ct2")
    _convert_to_ctranslate2(hf_dir, ct2_dir)
    return ct2_dir


def _converter_executable() -> str:
    # Same resolution strategy as config.py's _default_piper_executable:
    # the console script lives alongside whatever Python interpreter
    # ctranslate2 was pip-installed into.
    candidate = os.path.join(
        os.path.dirname(sys.executable),
        "ct2-transformers-converter.exe" if os.name == "nt" else "ct2-transformers-converter",
    )
    return candidate if os.path.exists(candidate) else "ct2-transformers-converter"


def _convert_to_ctranslate2(hf_dir: str, ct2_dir: str) -> None:
    cmd = [
        _converter_executable(),
        "--model", hf_dir,
        "--output_dir", ct2_dir,
        "--quantization", "int8_float16",
        "--force",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"CTranslate2 conversion failed:\n{proc.stderr}")


def _load_ct2_transcribe_fn(model_path_or_size: str, *, device: str, compute_type: str):
    """Wraps a faster-whisper model (already a base dependency -- no
    training extras needed for this half of the script) as a plain
    str -> str function for evaluation.evaluate_model_on_dataset(). Used
    for both the production baseline (model_path_or_size="small", etc,
    same as CONFIG.whisper_model_size) and a freshly trained candidate
    (model_path_or_size=<ct2 output dir>).
    """
    from cuda_dlls import ensure_cuda_dlls_on_path

    ensure_cuda_dlls_on_path()
    from faster_whisper import WhisperModel

    model = WhisperModel(model_path_or_size, device=device, compute_type=compute_type)

    def _transcribe(audio_path: str) -> str:
        audio, _sr = sf.read(audio_path, dtype="float32")
        segments, _info = model.transcribe(audio, language="en", beam_size=1, vad_filter=False)
        return " ".join(seg.text.strip() for seg in segments).strip()

    return _transcribe


def run_pipeline(args: argparse.Namespace) -> None:
    manifest_path = os.path.join(args.datasets_root, args.dataset_version, "manifest.csv")
    if not os.path.exists(manifest_path):
        raise SystemExit(
            f"No dataset at {manifest_path}. Build one first:\n"
            f"  python -c \"from learning import dataset_builder as d; "
            f"d.build_dataset({args.db_path!r}, {args.datasets_root!r})\""
        )

    config = TrainingConfig(
        base_model=args.base_model, epochs=args.epochs, batch_size=args.batch_size,
        learning_rate=args.learning_rate, lora_r=args.lora_r, lora_alpha=args.lora_alpha,
    )
    output_dir = os.path.join(args.models_root, "_pending", args.dataset_version)
    print(f"[training] fine-tuning {config.base_model} on {manifest_path} ...")
    ct2_dir = train(manifest_path, output_dir, config=config)
    print(f"[training] wrote candidate CTranslate2 model to {ct2_dir}")

    version = model_registry.register_candidate(
        args.registry_path,
        base_model=config.base_model,
        dataset_version=args.dataset_version,
        artifact_path=ct2_dir,
        training_config=asdict(config),
    )
    print(f"[training] registered candidate {version}")

    print("[training] evaluating baseline (production) model on the held-out split ...")
    manifest_rows = dataset_builder.load_manifest_rows(manifest_path)
    baseline_transcribe = _load_ct2_transcribe_fn(
        args.baseline_model, device=args.device, compute_type=args.baseline_compute_type
    )
    baseline_result = evaluation.evaluate_model_on_dataset(
        baseline_transcribe, manifest_rows,
        model_label=f"production ({args.baseline_model})", dataset_version=args.dataset_version,
    )

    print("[training] evaluating candidate model on the same held-out split ...")
    candidate_transcribe = _load_ct2_transcribe_fn(
        ct2_dir, device=args.device, compute_type=args.baseline_compute_type
    )
    candidate_result = evaluation.evaluate_model_on_dataset(
        candidate_transcribe, manifest_rows,
        model_label=f"candidate ({version})", dataset_version=args.dataset_version,
    )

    comparison = evaluation.compare_models(
        baseline_result, candidate_result,
        min_relative_improvement=args.min_relative_improvement,
    )
    model_registry.record_evaluation(args.registry_path, version, asdict(comparison))

    print(json.dumps(asdict(comparison), indent=2))
    print(
        f"[training] baseline WER={comparison.baseline_wer:.3f} "
        f"candidate WER={comparison.candidate_wer:.3f} "
        f"relative improvement={comparison.relative_improvement:+.1%} "
        f"-> {comparison.recommendation}"
    )

    if comparison.recommendation != "promote":
        print(f"[training] {version} NOT promoted (see recommendation above). Exiting.")
        return

    if args.promote:
        model_registry.promote_to_production(args.registry_path, version)
        print(f"[training] {version} promoted to production.")
    else:
        print(
            f"[training] {version} is eligible for promotion but --promote was not "
            f"passed. Re-run with --promote, or call "
            f"model_registry.promote_to_production({args.registry_path!r}, {version!r}) "
            "yourself after reviewing the numbers above."
        )


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Fine-tune the STT model on accumulated corrections (foundation-phase LoRA/PEFT trainer)."
    )
    p.add_argument("--dataset-version", required=True, help='e.g. "v1", from dataset_builder.build_dataset()')
    p.add_argument("--db-path", default=os.path.join("data", "learning", "feedback.db"))
    p.add_argument("--datasets-root", default=os.path.join("data", "learning", "datasets"))
    p.add_argument("--models-root", default=os.path.join("data", "learning", "models"))
    p.add_argument("--registry-path", default=os.path.join("data", "learning", "models", "registry.json"))
    p.add_argument("--base-model", default=DEFAULT_BASE_MODEL, help="HF hub model to fine-tune")
    p.add_argument("--baseline-model", default="small", help="faster-whisper model/size to compare against")
    p.add_argument("--baseline-compute-type", default="int8")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--min-relative-improvement", type=float, default=0.05)
    p.add_argument(
        "--promote", action="store_true",
        help="Promote the candidate to production if its evaluation recommends it.",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_arg_parser().parse_args(argv)
    run_pipeline(args)


if __name__ == "__main__":
    main()
