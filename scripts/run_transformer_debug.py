"""Debug run: fine-tune Bio_ClinicalBERT on a 1.5K-note subset.

Validates the training loop end-to-end on the T1200 with explicit MLflow
wrapping AND debug-level logging so HF Trainer prints per-step loss + tqdm.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from scipy.sparse import load_npz

from mimic_icd_coder.models.transformer import TransformerTrainConfig, fine_tune

ROOT = Path(".")
SILVER = ROOT / "data" / "silver" / "notes.parquet"
GOLD_LABELS = ROOT / "data" / "gold" / "labels.npz"
GOLD_NAMES = ROOT / "data" / "gold" / "label_names.json"
SPLITS = ROOT / "data" / "gold" / "splits.parquet"
OUT_DIR = ROOT / "data" / "models" / "bio_clinicalbert_debug"

SEED = 42
N_TRAIN_SAMPLE = 1_500
N_VAL_SAMPLE = 200


def main() -> None:
    # Enable DEBUG-level logging on the root logger so transformer.fine_tune
    # detects it via is_debug_enabled() and switches to:
    #   logging_strategy="steps", logging_steps=25, disable_tqdm=False
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    mlflow.set_tracking_uri(f"file:{Path('data/mlruns').as_posix()}")
    mlflow.set_experiment("transformer_debug_local")

    print(f"Loading Silver from {SILVER}")
    silver = pd.read_parquet(SILVER)
    print(f"  {len(silver):,} cleaned notes")

    print(f"Loading Gold labels from {GOLD_LABELS}")
    y = load_npz(GOLD_LABELS)
    labels: list[str] = json.loads(GOLD_NAMES.read_text(encoding="utf-8"))
    print(f"  shape={y.shape}, {len(labels)} labels")

    splits_df = pd.read_parquet(SPLITS)
    train_idx_full = splits_df.loc[splits_df["split"] == "train", "row_idx"].to_numpy()
    val_idx_full = splits_df.loc[splits_df["split"] == "val", "row_idx"].to_numpy()
    print(f"  full train: {len(train_idx_full):,} / full val: {len(val_idx_full):,}")

    rng = np.random.default_rng(SEED)
    train_idx = rng.choice(
        train_idx_full, size=min(N_TRAIN_SAMPLE, len(train_idx_full)), replace=False
    )
    val_idx = rng.choice(val_idx_full, size=min(N_VAL_SAMPLE, len(val_idx_full)), replace=False)
    print(f"  subset: {len(train_idx):,} train / {len(val_idx):,} val (seed={SEED})")

    train_texts = silver["text"].iloc[train_idx].tolist()
    val_texts = silver["text"].iloc[val_idx].tolist()
    y_train = np.asarray(y[train_idx].todense(), dtype=np.float32)
    y_val = np.asarray(y[val_idx].todense(), dtype=np.float32)

    cfg = TransformerTrainConfig(
        model_name="emilyalsentzer/Bio_ClinicalBERT",
        max_length=512,
        stride=128,
        batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-5,
        epochs=1,
        warmup_ratio=0.1,
        weight_decay=0.01,
        fp16=True,
        gradient_checkpointing=False,
        seed=SEED,
    )
    print(f"Config: {cfg}")
    print(f"Output: {OUT_DIR}")

    print("\nStarting fine_tune in DEBUG mode (per-step logs + tqdm visible).")

    with mlflow.start_run(run_name="bio_clinicalbert_debug_1500_v2"):
        mlflow.log_params(
            {
                "n_train_docs": len(train_texts),
                "n_val_docs": len(val_texts),
                "subset_seed": SEED,
                "gradient_checkpointing": cfg.gradient_checkpointing,
                "log_level": "DEBUG",
            }
        )
        model_dir = fine_tune(
            train_texts=train_texts,
            y_train=y_train,
            val_texts=val_texts,
            y_val=y_val,
            labels=labels,
            cfg=cfg,
            output_dir=OUT_DIR,
        )
        mlflow.log_artifact(str(model_dir), artifact_path="checkpoint")
        print(f"\nDone. Model: {model_dir}")
        print("MLflow: mlflow ui --backend-store-uri file:data/mlruns")


if __name__ == "__main__":
    main()
