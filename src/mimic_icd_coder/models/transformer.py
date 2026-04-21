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
    """Token-level chunking for long notes — contiguous 512-BPE chunks by default.

    Default is ``stride = 0`` for **contiguous** (non-overlapping) chunks per
    ``DECISIONS.md`` 2026-04-20 (Chunk-and-max-pool). A non-zero stride enables
    sliding-window overlap if a future experiment needs it.

    Planned approach:
        1. Tokenize with ``return_overflowing_tokens=True``.
        2. Track parent doc index per chunk for max-pool aggregation at inference.
        3. Keep attention masks and input_ids aligned.

    Args:
        texts: Documents to tokenize.
        tokenizer: HuggingFace tokenizer.
        max_length: Max sequence length per chunk.
        stride: Overlap between chunks (tokens). Default 0 = contiguous chunks.

    Returns:
        List of tokenizer outputs, one per chunk.
    """
    raise NotImplementedError(
        "Pending on feat/transformer-finetune — implement chunking per DECISIONS.md 2026-04-20 (contiguous 512-BPE chunks + max-pool)"
    )


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
