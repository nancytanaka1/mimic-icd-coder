"""Bio_ClinicalBERT / Clinical-Longformer fine-tuning — scaffold.

Implementation deferred to feat/transformer-finetune. The function signatures
here are stable — other modules can import them without waiting.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from mimic_icd_coder.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class TransformerTrainConfig:
    """Fine-tuning hyperparameters."""

    model_name: str = "emilyalsentzer/Bio_ClinicalBERT"
    max_length: int = 512
    batch_size: int = 16
    learning_rate: float = 2e-5
    epochs: int = 3
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    fp16: bool = True
    gradient_accumulation_steps: int = 1
    seed: int = 42


def tokenize_and_chunk(
    texts: list[str], tokenizer: Any, max_length: int, stride: int = 0
) -> list[dict[str, Any]]:
    """Token-level chunking for long notes.

    Uses HuggingFace's ``return_overflowing_tokens`` so a single long note
    becomes multiple training examples, each tagged with its parent doc
    index via ``doc_idx``. Callers at inference time reduce back to one
    prediction per doc by max-pooling sigmoid probabilities across every
    chunk that shares a ``doc_idx`` (see ``DECISIONS.md`` entry on the
    chunk-and-max-pool design).

    ``stride`` is the number of tokens that overlap between adjacent
    chunks. ``stride = 0`` is the original DECISIONS.md 2026-04-20 choice
    (contiguous). ``stride > 0`` (e.g. 128 for a 512-token window) is a
    sliding-window variant that avoids losing context at chunk boundaries;
    the caller picks based on the current experiment.

    Args:
        texts: Documents to tokenize. Each string is one source note.
        tokenizer: HuggingFace-compatible tokenizer with a ``__call__``
            that accepts ``return_overflowing_tokens`` and returns
            ``overflow_to_sample_mapping`` (the standard modern API).
        max_length: Max sequence length per chunk, including BERT's
            ``[CLS]`` and ``[SEP]`` special tokens.
        stride: Overlap between adjacent chunks, in tokens. ``0`` gives
            contiguous non-overlapping chunks.

    Returns:
        A flat list of chunk dicts. Each dict has three keys:

            - ``input_ids`` (list[int]): token ids for this chunk
            - ``attention_mask`` (list[int]): 1 for real tokens, 0 for pad
            - ``doc_idx`` (int): index into ``texts`` for the source note

        Order is stable: all chunks for ``texts[0]`` come before any
        chunks for ``texts[1]``, in sliding-window order within each doc.
    """
    if not texts:
        return []

    encoded = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        stride=stride,
        return_overflowing_tokens=True,
        padding=False,
    )

    mapping = encoded["overflow_to_sample_mapping"]
    input_ids_list = encoded["input_ids"]
    attention_mask_list = encoded["attention_mask"]

    chunks: list[dict[str, Any]] = [
        {
            "input_ids": list(input_ids_list[i]),
            "attention_mask": list(attention_mask_list[i]),
            "doc_idx": int(mapping[i]),
        }
        for i in range(len(input_ids_list))
    ]

    logger.info(
        "transformer.tokenize_and_chunk",
        n_docs=len(texts),
        n_chunks=len(chunks),
        avg_chunks_per_doc=round(len(chunks) / max(1, len(texts)), 2),
        max_length=max_length,
        stride=stride,
    )
    return chunks


def fine_tune(
    train_texts: list[str],
    y_train: np.ndarray,
    val_texts: list[str],
    y_val: np.ndarray,
    labels: list[str],
    cfg: TransformerTrainConfig,
    output_dir: str | Path,
) -> Path:
    """Fine-tune a transformer for multi-label ICD prediction.

    Planned implementation:
        - HuggingFace ``Trainer`` with multi-label head (BCEWithLogitsLoss)
        - Per-epoch val macro F1 + early stopping
        - MLflow autolog for params and metrics
        - Save best checkpoint + tokenizer to ``output_dir``

    Visibility contract (shared with ``baseline.fit_baseline``):
        Call ``mimic_icd_coder.logging_utils.is_debug_enabled()`` and wire the
        result into HuggingFace ``TrainingArguments``. When True, set
        ``logging_strategy="steps"``, ``logging_steps=50``,
        ``disable_tqdm=False``, and ``report_to=["mlflow"]`` so per-step loss
        / grad-norm / LR stream to stdout and MLflow. When False, keep
        ``logging_strategy="epoch"`` so INFO-mode logs stay at per-epoch
        granularity. This matches the LR baseline's "DEBUG = inner loop
        visible, INFO = stage boundaries only" contract.

    Args:
        train_texts / val_texts: Document lists.
        y_train / y_val: Dense multi-label targets, ``float32``.
        labels: Label name list.
        cfg: Training configuration.
        output_dir: Output directory for checkpoints.

    Returns:
        Path to the saved model directory.
    """
    raise NotImplementedError("Pending on feat/transformer-finetune")


def load_fine_tuned(model_dir: str | Path) -> object:
    """Load a fine-tuned model for inference."""
    raise NotImplementedError("Pending on feat/transformer-finetune")
