"""End-to-end pipeline integration test on synthetic fixtures.

Writes synthetic MIMIC-style gzipped CSVs to a temp dir, runs every
pipeline stage, and verifies expected artifacts + baseline floor metrics.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pandas as pd
import pytest
from scipy.sparse import load_npz

from mimic_icd_coder.config import AppConfig
from mimic_icd_coder.pipeline import (
    Paths,
    run_bronze,
    run_evaluate_test,
    run_gold,
    run_silver,
    run_splits,
    run_train_baseline,
)
from tests.fixtures.synthetic_notes import make_synthetic


def _write_gz_csv(df: pd.DataFrame, path: Path) -> None:
    """Write a DataFrame as gzipped CSV (mimics the PhysioNet layout)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, mode="wt", encoding="utf-8") as fh:
        df.to_csv(fh, index=False)


@pytest.fixture()
def synthetic_workspace(tmp_path: Path) -> tuple[AppConfig, Paths]:
    """Build a temp workspace with synthetic gz CSVs and a matching config."""
    corpus = make_synthetic(n_patients=120, seed=42)
    raw = tmp_path / "raw"
    _write_gz_csv(corpus["notes"], raw / "discharge.csv.gz")
    _write_gz_csv(corpus["diagnoses_icd"], raw / "diagnoses_icd.csv.gz")
    _write_gz_csv(corpus["admissions"], raw / "admissions.csv.gz")
    _write_gz_csv(corpus["patients"], raw / "patients.csv.gz")

    cfg_dict: dict[str, object] = {
        "unity_catalog": {
            "catalog": "mimic_icd_test",
            "bronze_schema": "bronze",
            "silver_schema": "silver",
            "gold_schema": "gold",
            "models_schema": "models",
        },
        "data": {
            "notes_path": str(raw / "discharge.csv.gz"),
            "diagnoses_path": str(raw / "diagnoses_icd.csv.gz"),
            "admissions_path": str(raw / "admissions.csv.gz"),
            "patients_path": str(raw / "patients.csv.gz"),
        },
        "cohort": {
            "icd_version": 10,
            "min_note_tokens": 50,
            "top_k_labels": 8,  # fixture has 10 distinct codes
            "note_types": ["DS"],
        },
        "split": {
            "train_frac": 0.70,
            "val_frac": 0.15,
            "test_frac": 0.15,
            "seed": 42,
            "strategy": "patient_stratified",
        },
        "baseline": {
            "tfidf_ngram_range": [1, 1],
            "tfidf_min_df": 1,
            "tfidf_max_features": 5000,
            "logreg_c": 1.0,
            "logreg_class_weight": None,
        },
        "transformer": {
            "model_name": "emilyalsentzer/Bio_ClinicalBERT",
            "max_length": 512,
            "batch_size": 16,
            "learning_rate": 2e-5,
            "epochs": 3,
            "warmup_ratio": 0.1,
            "weight_decay": 0.01,
            "fp16": True,
        },
        "evaluation": {
            "threshold_strategy": "per_label_pr",
            "top_k_metrics": [1, 3, 5],
        },
        "mlflow": {
            "experiment_name": "test_pipeline",
            "registry_model_name": "mimic_icd.models.test",
        },
        "logging": {"level": "WARNING", "format": "console"},
    }
    cfg = AppConfig(**cfg_dict)  # type: ignore[arg-type]
    paths = Paths(root=tmp_path / "data")
    paths.ensure()
    return cfg, paths


def test_full_pipeline_produces_all_artifacts(synthetic_workspace: tuple[AppConfig, Paths]) -> None:
    cfg, paths = synthetic_workspace

    run_bronze(cfg, paths)
    for name in ("discharge_notes", "diagnoses_icd", "admissions", "patients"):
        assert (paths.bronze / f"{name}.parquet").is_file(), f"Bronze missing {name}"

    run_silver(cfg, paths)
    silver = pd.read_parquet(paths.silver / "notes.parquet")
    assert len(silver) > 0
    assert silver["hadm_id"].is_unique
    assert (silver["n_tokens"] >= cfg.cohort.min_note_tokens).all()

    run_gold(cfg, paths)
    y = load_npz(paths.gold / "labels.npz")
    labels = json.loads((paths.gold / "label_names.json").read_text(encoding="utf-8"))
    assert y.shape == (len(silver), cfg.cohort.top_k_labels)
    assert len(labels) == cfg.cohort.top_k_labels

    run_splits(cfg, paths)
    splits = pd.read_parquet(paths.gold / "splits.parquet")
    assert set(splits["split"].unique()) == {"train", "val", "test"}
    assert len(splits) == len(silver)


def test_pipeline_baseline_beats_random(synthetic_workspace: tuple[AppConfig, Paths]) -> None:
    cfg, paths = synthetic_workspace
    run_bronze(cfg, paths)
    run_silver(cfg, paths)
    run_gold(cfg, paths)
    run_splits(cfg, paths)

    metrics = run_train_baseline(cfg, paths)
    assert metrics["micro_f1"] > 0.3, f"Baseline too weak: {metrics['micro_f1']}"

    # Saved artifacts ready for test-split eval
    assert (paths.gold / "baseline_model.joblib").is_file()
    assert (paths.gold / "baseline_thresholds.npy").is_file()

    test_metrics = run_evaluate_test(cfg, paths)
    assert "micro_f1" in test_metrics
    # mullenbach_*_delta keys were removed per DECISIONS.md 2026-04-26
    # (different dataset / coding system / cohort vs. Mullenbach 2018);
    # asserting on test_metrics["micro_f1"] alone is the right contract now.
    assert not any(
        k.startswith("mullenbach_") for k in test_metrics
    ), "mullenbach_* keys must not be present after the 2026-04-26 reframe"
    # Test metrics on a tiny synthetic corpus may trail val; sanity bound only
    assert test_metrics["micro_f1"] > 0.1
