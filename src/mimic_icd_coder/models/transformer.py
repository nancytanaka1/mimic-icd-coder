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
    # Save the label list alongside the model so load_fine_tuned can recover
    # human-readable code names. HuggingFace's id2label/label2id on the model
    # config works for simple cases but round-trips through strings and is
    # easy to desync; a sibling labels.json is explicit and obvious.
    import json as _json

    (output_path / "labels.json").write_text(_json.dumps(list(labels), indent=2), encoding="utf-8")
    logger.info("transformer.fine_tune.saved", path=str(output_path), n_labels=n_labels)

    return output_path


@dataclass
class FineTunedModel:
    """Loaded fine-tuned transformer — tokenizer + model + labels + chunking config.

    Chunk-pool inference contract (per ``DECISIONS.md`` 2026-04-20
    chunk-and-max-pool): ``predict_proba`` tokenizes each note into
    ``max_length``-token chunks with ``stride`` overlap, forward-passes each
    chunk independently, applies sigmoid to the logits, and then max-pools
    across every chunk that shares a parent ``doc_idx``. Rationale for
    max-pool (vs. mean): any single chunk asserting "this code is present"
    is sufficient evidence; mean-pool dilutes the signal when only one
    chunk contains the relevant clinical mention.

    Attributes:
        tokenizer: HuggingFace tokenizer loaded from the saved model dir.
        model: HuggingFace model in eval mode.
        labels: Ordered list of ICD-10 codes — column order of the output.
        max_length: Chunk window length (must match training).
        stride: Sliding-window overlap (must match training).
    """

    tokenizer: Any
    model: Any
    labels: list[str]
    max_length: int = 512
    stride: int = 128

    def predict_proba(self, texts: list[str], batch_size: int = 8) -> np.ndarray:
        """Predict per-label probabilities with chunk max-pool aggregation.

        Args:
            texts: Documents to score. One string per note.
            batch_size: Chunks per forward pass. Independent of the training
                batch size. Lower this if inference OOMs on GPU.

        Returns:
            Array shape ``(len(texts), len(self.labels))`` of sigmoid
            probabilities in ``[0, 1]``, max-pooled across chunks per doc.
            Rows of all-zeros are only possible when ``texts`` is empty
            — any non-empty note produces at least one chunk.
        """
        import torch

        if not texts:
            return np.zeros((0, len(self.labels)), dtype=np.float32)

        chunks = tokenize_and_chunk(
            texts, self.tokenizer, max_length=self.max_length, stride=self.stride
        )

        device = next(self.model.parameters()).device
        self.model.eval()

        pad_token_id = self.tokenizer.pad_token_id or 0
        chunk_probs_batches: list[np.ndarray] = []

        with torch.no_grad():
            for start in range(0, len(chunks), batch_size):
                batch = chunks[start : start + batch_size]
                max_len = max(len(c["input_ids"]) for c in batch)
                input_ids = torch.tensor(
                    [
                        c["input_ids"] + [pad_token_id] * (max_len - len(c["input_ids"]))
                        for c in batch
                    ],
                    dtype=torch.long,
                    device=device,
                )
                attention_mask = torch.tensor(
                    [
                        c["attention_mask"] + [0] * (max_len - len(c["attention_mask"]))
                        for c in batch
                    ],
                    dtype=torch.long,
                    device=device,
                )
                logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
                probs = torch.sigmoid(logits).cpu().numpy()
                chunk_probs_batches.append(probs)

        chunk_probs = np.concatenate(chunk_probs_batches, axis=0).astype(np.float32)
        doc_indices = np.array([c["doc_idx"] for c in chunks], dtype=np.int64)

        doc_probs = np.zeros((len(texts), len(self.labels)), dtype=np.float32)
        for doc_idx in range(len(texts)):
            mask = doc_indices == doc_idx
            if mask.any():
                doc_probs[doc_idx] = chunk_probs[mask].max(axis=0)

        logger.info(
            "transformer.predict_proba",
            n_docs=len(texts),
            n_chunks=len(chunks),
            n_labels=len(self.labels),
        )
        return doc_probs


def load_fine_tuned(
    model_dir: str | Path, *, max_length: int = 512, stride: int = 128
) -> FineTunedModel:
    """Load a fine-tuned model and its tokenizer from disk for inference.

    Args:
        model_dir: Directory written by ``fine_tune`` — contains the model
            weights, the tokenizer, and a ``labels.json`` sidecar.
        max_length: Chunk window length for inference. Match the value
            used during training; the 512 default matches
            ``TransformerTrainConfig.max_length``.
        stride: Sliding-window overlap for inference. Match training.

    Returns:
        A ``FineTunedModel`` wrapper with ``predict_proba`` bound and
        ready to call.
    """
    import json as _json

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_path = Path(model_dir)
    if not model_path.is_dir():
        raise FileNotFoundError(f"Model directory not found: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_path))

    labels_path = model_path / "labels.json"
    if labels_path.is_file():
        labels = list(_json.loads(labels_path.read_text(encoding="utf-8")))
    else:
        # Older training runs that predate the labels.json sidecar fall
        # back to synthetic "LABEL_N" names sized to the classifier head.
        n_labels = int(getattr(model.config, "num_labels", 0))
        labels = [f"LABEL_{i}" for i in range(n_labels)]
        logger.warning(
            "transformer.load_fine_tuned.no_labels_json",
            path=str(labels_path),
            fallback_n_labels=n_labels,
        )

    logger.info(
        "transformer.load_fine_tuned",
        model_dir=str(model_path),
        n_labels=len(labels),
        max_length=max_length,
        stride=stride,
    )
    return FineTunedModel(
        tokenizer=tokenizer,
        model=model,
        labels=labels,
        max_length=max_length,
        stride=stride,
    )
