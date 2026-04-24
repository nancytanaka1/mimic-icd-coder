"""Bio_ClinicalBERT / Clinical-Longformer fine-tuning — scaffold.

Implementation deferred to feat/transformer-finetune. The function signatures
here are stable — other modules can import them without waiting.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from mimic_icd_coder.logging_utils import get_logger, is_debug_enabled

logger = get_logger(__name__)


@dataclass
class TransformerTrainConfig:
    """Fine-tuning hyperparameters."""

    model_name: str = "emilyalsentzer/Bio_ClinicalBERT"
    max_length: int = 512
    stride: int = 128
    batch_size: int = 16
    learning_rate: float = 2e-5
    epochs: int = 3
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    fp16: bool = True
    gradient_checkpointing: bool = True
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
    """Fine-tune a transformer for multi-label ICD-10 prediction.

    Uses HuggingFace ``Trainer`` with ``problem_type="multi_label_classification"``
    on the underlying model, which triggers ``BCEWithLogitsLoss`` automatically —
    the correct loss for multi-label. Tokenizes each source document into
    ``max_length``-token chunks via ``tokenize_and_chunk`` with ``stride`` overlap,
    and every chunk of a document is trained against the parent doc's full
    label vector. Chunk-level predictions are reduced to one prediction per
    doc at inference time via max-pool on sigmoid probabilities (see
    ``load_fine_tuned``).

    Visibility contract (matches ``baseline.fit_baseline``):
        The function inspects ``mimic_icd_coder.logging_utils.is_debug_enabled()``
        at call time and wires the result into HuggingFace ``TrainingArguments``:

            debug → ``logging_strategy="steps"``, ``logging_steps=25``,
                    ``disable_tqdm=False``
            info  → ``logging_strategy="epoch"``, ``logging_steps=500``,
                    ``disable_tqdm=True``

        MLflow reporting is always on — the distinction is granularity, not
        presence. Matches the "DEBUG = inner loop visible, INFO = stage
        boundaries only" contract established in the LR baseline.

    Precision:
        ``cfg.fp16`` is honored only if CUDA is available; on CPU it is
        automatically forced to False to avoid HuggingFace raising
        "torch.amp.autocast cannot be used with CPU" on modern torch.

    Args:
        train_texts / val_texts: Document lists (one string per note).
        y_train / y_val: Multi-hot targets shape ``(n_docs, len(labels))``.
            Cast internally to float32 for BCE.
        labels: Ordered list of ICD-10 codes corresponding to the columns
            of ``y_train`` / ``y_val``.
        cfg: Training configuration (see ``TransformerTrainConfig``).
        output_dir: Directory to save the final model + tokenizer.

    Returns:
        ``Path`` to the saved model directory — ready for ``load_fine_tuned``.
    """
    # Local imports so the module is importable on systems without torch
    # (e.g. a pipeline step that only needs tokenize_and_chunk).
    import torch
    from datasets import Dataset
    from sklearn.metrics import f1_score
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )

    n_labels = len(labels)
    if y_train.shape[1] != n_labels or y_val.shape[1] != n_labels:
        raise ValueError(
            f"y_train / y_val must have {n_labels} columns matching `labels`; "
            f"got y_train={y_train.shape}, y_val={y_val.shape}"
        )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    debug = is_debug_enabled()
    effective_fp16 = cfg.fp16 and torch.cuda.is_available()

    logger.info(
        "transformer.fine_tune.start",
        model_name=cfg.model_name,
        n_train_docs=len(train_texts),
        n_val_docs=len(val_texts),
        n_labels=n_labels,
        max_length=cfg.max_length,
        stride=cfg.stride,
        batch_size=cfg.batch_size,
        epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        effective_fp16=effective_fp16,
        debug=debug,
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=n_labels,
        problem_type="multi_label_classification",
        # The pre-trained checkpoint's classification head is almost always
        # shaped for its original task (e.g. 2-class sentiment) — we always
        # reinitialize it for our 50-label multi-label ICD task, which is
        # the whole point of fine-tuning. Without this flag, transformers 5.x
        # raises on the size mismatch instead of re-initializing.
        ignore_mismatched_sizes=True,
    )

    train_chunks = tokenize_and_chunk(
        train_texts, tokenizer, max_length=cfg.max_length, stride=cfg.stride
    )
    val_chunks = tokenize_and_chunk(
        val_texts, tokenizer, max_length=cfg.max_length, stride=cfg.stride
    )

    def _chunks_to_rows(chunks: list[dict[str, Any]], y: np.ndarray) -> list[dict[str, Any]]:
        return [
            {
                "input_ids": c["input_ids"],
                "attention_mask": c["attention_mask"],
                "labels": y[c["doc_idx"]].astype(np.float32).tolist(),
            }
            for c in chunks
        ]

    train_ds = Dataset.from_list(_chunks_to_rows(train_chunks, y_train))
    val_ds = Dataset.from_list(_chunks_to_rows(val_chunks, y_val))

    args = TrainingArguments(
        output_dir=str(output_path),
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        fp16=effective_fp16,
        gradient_checkpointing=cfg.gradient_checkpointing,
        logging_strategy="steps" if debug else "epoch",
        logging_steps=25 if debug else 500,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_micro_f1",
        greater_is_better=True,
        report_to=["mlflow"],
        disable_tqdm=not debug,
        seed=cfg.seed,
    )

    def compute_metrics(eval_pred: Any) -> dict[str, float]:
        logits, labels_arr = eval_pred
        probs = 1.0 / (1.0 + np.exp(-logits))
        preds = (probs >= 0.5).astype(int)
        return {
            "micro_f1": float(f1_score(labels_arr, preds, average="micro", zero_division=0)),
            "macro_f1": float(f1_score(labels_arr, preds, average="macro", zero_division=0)),
        }

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )

    logger.info(
        "transformer.fine_tune.train_start",
        n_train_chunks=len(train_ds),
        n_val_chunks=len(val_ds),
    )
    trainer.train()
    logger.info("transformer.fine_tune.train_done")

    trainer.save_model(str(output_path))
    tokenizer.save_pretrained(str(output_path))
    logger.info("transformer.fine_tune.saved", path=str(output_path))

    return output_path


def load_fine_tuned(model_dir: str | Path) -> object:
    """Load a fine-tuned model for inference."""
    raise NotImplementedError("Pending on feat/transformer-finetune")
