"""Smoke tests for the mic CLI — exercise every command end-to-end."""

from __future__ import annotations

import gzip
from pathlib import Path

import yaml
from click.testing import CliRunner

from mimic_icd_coder.cli import cli
from tests.fixtures.synthetic_notes import make_synthetic


def _write_gz_csv(df, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        df.to_csv(fh, index=False)


def _build_workspace(tmp_path: Path) -> Path:
    """Produce a tiny synthetic workspace + config and return the config path."""
    corpus = make_synthetic(n_patients=80, seed=13)
    raw = tmp_path / "raw"
    _write_gz_csv(corpus["notes"], raw / "discharge.csv.gz")
    _write_gz_csv(corpus["diagnoses_icd"], raw / "diagnoses_icd.csv.gz")
    _write_gz_csv(corpus["admissions"], raw / "admissions.csv.gz")
    _write_gz_csv(corpus["patients"], raw / "patients.csv.gz")

    cfg = {
        "unity_catalog": {
            "catalog": "test",
            "bronze_schema": "b",
            "silver_schema": "s",
            "gold_schema": "g",
            "models_schema": "m",
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
            "top_k_labels": 8,
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
        "evaluation": {"threshold_strategy": "per_label_pr", "top_k_metrics": [1, 3, 5]},
        "mlflow": {
            "experiment_name": "cli_test",
            "registry_model_name": "test.models.top8",
        },
        "logging": {"level": "WARNING", "format": "console"},
    }
    cfg_path = tmp_path / "dev.yml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return cfg_path


def test_cli_run_all_end_to_end(tmp_path: Path) -> None:
    cfg_path = _build_workspace(tmp_path)
    artifacts = tmp_path / "data"
    runner = CliRunner()
    result = runner.invoke(
        cli, ["run-all", "--config", str(cfg_path), "--artifacts", str(artifacts)]
    )
    assert result.exit_code == 0, result.output
    assert (artifacts / "bronze" / "discharge_notes.parquet").is_file()
    assert (artifacts / "silver" / "notes.parquet").is_file()
    assert (artifacts / "gold" / "labels.npz").is_file()
    assert (artifacts / "gold" / "splits.parquet").is_file()
    assert (artifacts / "gold" / "baseline_model.joblib").is_file()


def test_cli_ingest_only(tmp_path: Path) -> None:
    cfg_path = _build_workspace(tmp_path)
    artifacts = tmp_path / "data"
    runner = CliRunner()
    result = runner.invoke(
        cli, ["ingest", "--config", str(cfg_path), "--artifacts", str(artifacts)]
    )
    assert result.exit_code == 0, result.output
    assert (artifacts / "bronze" / "discharge_notes.parquet").is_file()


def test_cli_help_lists_all_commands() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    for cmd in ["ingest", "silver", "gold", "splits", "train-baseline", "evaluate-test", "run-all"]:
        assert cmd in result.output
