"""Pipeline orchestration — stage-by-stage functions with Parquet checkpoints.

Each stage reads from the previous stage's artifacts on disk, runs its
transform, and writes Parquet / npz / json outputs for the next stage
to consume. This lets you iterate on one stage without recomputing upstream.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, load_npz, save_npz

from mimic_icd_coder.config import AppConfig
from mimic_icd_coder.data.clean import build_silver_notes
from mimic_icd_coder.data.ingest import (
    read_admissions,
    read_d_icd_diagnoses,
    read_diagnoses_icd,
    read_discharge_notes,
    read_patients,
)
from mimic_icd_coder.data.labels import LabelSet, build_labels
from mimic_icd_coder.data.splits import Splits, patient_split
from mimic_icd_coder.logging_utils import get_logger

logger = get_logger(__name__)


class PipelineError(Exception):
    """Raised when a pipeline stage cannot complete."""


@dataclass(frozen=True)
class Paths:
    """Canonical on-disk layout for pipeline artifacts."""

    root: Path

    @property
    def bronze(self) -> Path:
        return self.root / "bronze"

    @property
    def silver(self) -> Path:
        return self.root / "silver"

    @property
    def gold(self) -> Path:
        return self.root / "gold"

    @property
    def mlruns(self) -> Path:
        return self.root / "mlruns"

    def ensure(self) -> None:
        """Create all stage directories."""
        for p in (self.bronze, self.silver, self.gold, self.mlruns):
            p.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Bronze


def run_bronze(cfg: AppConfig, paths: Paths) -> None:
    """Read raw gz CSVs and mirror to Parquet under ``paths.bronze``.

    Writes:
        - ``discharge_notes.parquet``
        - ``diagnoses_icd.parquet``
        - ``admissions.parquet``
        - ``patients.parquet``
    """
    paths.ensure()
    logger.info("bronze.start", out=str(paths.bronze))

    notes = read_discharge_notes(cfg.data.notes_path)
    notes.to_parquet(paths.bronze / "discharge_notes.parquet", index=False)
    logger.info("bronze.wrote", table="discharge_notes", rows=len(notes))

    dx = read_diagnoses_icd(cfg.data.diagnoses_path)
    dx.to_parquet(paths.bronze / "diagnoses_icd.parquet", index=False)
    logger.info("bronze.wrote", table="diagnoses_icd", rows=len(dx))

    adm = read_admissions(cfg.data.admissions_path)
    adm.to_parquet(paths.bronze / "admissions.parquet", index=False)
    logger.info("bronze.wrote", table="admissions", rows=len(adm))

    pat = read_patients(cfg.data.patients_path)
    pat.to_parquet(paths.bronze / "patients.parquet", index=False)
    logger.info("bronze.wrote", table="patients", rows=len(pat))

    # ICD code dictionary — optional, only if configured. Feeds the top-K
    # code table (data card / EDA report) and human-readable serving output.
    if cfg.data.d_icd_path is not None:
        d_icd = read_d_icd_diagnoses(cfg.data.d_icd_path)
        d_icd.to_parquet(paths.bronze / "d_icd_diagnoses.parquet", index=False)
        logger.info("bronze.wrote", table="d_icd_diagnoses", rows=len(d_icd))
    else:
        logger.info("bronze.skipped", table="d_icd_diagnoses", reason="no d_icd_path in config")


# ---------------------------------------------------------------------------
# Silver


def run_silver(cfg: AppConfig, paths: Paths) -> None:
    """Read Bronze notes, produce cleaned Silver notes Parquet.

    Writes:
        - ``silver/notes.parquet``
    """
    paths.ensure()
    bronze_notes = paths.bronze / "discharge_notes.parquet"
    if not bronze_notes.is_file():
        raise PipelineError(f"Bronze notes missing — run 'mic ingest' first ({bronze_notes})")

    logger.info("silver.start")
    notes = pd.read_parquet(bronze_notes)
    silver = build_silver_notes(
        notes,
        note_types=cfg.cohort.note_types,
        min_tokens=cfg.cohort.min_note_tokens,
    )
    out = paths.silver / "notes.parquet"
    silver.to_parquet(out, index=False)
    logger.info("silver.wrote", path=str(out), rows=len(silver))


# ---------------------------------------------------------------------------
# Gold — labels


def run_gold(cfg: AppConfig, paths: Paths) -> None:
    """Build top-K ICD-10 label matrix aligned with Silver notes.

    Writes:
        - ``gold/labels.npz`` — scipy sparse CSR, shape (n_admissions, k)
        - ``gold/label_names.json`` — list of ICD-10 codes in column order
        - ``gold/hadm_ids.parquet`` — hadm_id per row, in y matrix order
    """
    paths.ensure()
    silver_path = paths.silver / "notes.parquet"
    bronze_dx = paths.bronze / "diagnoses_icd.parquet"
    if not silver_path.is_file():
        raise PipelineError(f"Silver notes missing — run 'mic silver' first ({silver_path})")
    if not bronze_dx.is_file():
        raise PipelineError(f"Bronze diagnoses missing — run 'mic ingest' first ({bronze_dx})")

    logger.info("gold.start", k=cfg.cohort.top_k_labels)
    silver = pd.read_parquet(silver_path)
    dx = pd.read_parquet(bronze_dx)
    label_set = build_labels(silver, dx, k=cfg.cohort.top_k_labels)

    save_npz(paths.gold / "labels.npz", label_set.y)
    (paths.gold / "label_names.json").write_text(
        json.dumps(label_set.labels, indent=2), encoding="utf-8"
    )
    pd.DataFrame({"hadm_id": label_set.hadm_ids}).to_parquet(
        paths.gold / "hadm_ids.parquet", index=False
    )
    logger.info(
        "gold.wrote",
        labels_shape=label_set.y.shape,
        density=float(label_set.y.sum() / (label_set.y.shape[0] * label_set.y.shape[1])),
    )


def load_gold(paths: Paths) -> LabelSet:
    """Read Gold artifacts back into a ``LabelSet``."""
    y = csr_matrix(load_npz(paths.gold / "labels.npz"))
    labels = json.loads((paths.gold / "label_names.json").read_text(encoding="utf-8"))
    hadm_ids = pd.read_parquet(paths.gold / "hadm_ids.parquet")["hadm_id"].to_numpy()
    return LabelSet(hadm_ids=hadm_ids, labels=labels, y=y)


# ---------------------------------------------------------------------------
# Splits


def run_splits(cfg: AppConfig, paths: Paths) -> None:
    """Compute and persist patient-level train/val/test index manifest.

    Writes:
        - ``gold/splits.parquet`` — columns ``[row_idx, split]``
    """
    paths.ensure()
    silver_path = paths.silver / "notes.parquet"
    if not silver_path.is_file():
        raise PipelineError("Silver notes missing — run 'mic silver' first")

    silver = pd.read_parquet(silver_path)
    sp = patient_split(
        silver,
        train_frac=cfg.split.train_frac,
        val_frac=cfg.split.val_frac,
        test_frac=cfg.split.test_frac,
        seed=cfg.split.seed,
    )

    rows: list[dict[str, object]] = []
    for idx in sp.train_idx:
        rows.append({"row_idx": int(idx), "split": "train"})
    for idx in sp.val_idx:
        rows.append({"row_idx": int(idx), "split": "val"})
    for idx in sp.test_idx:
        rows.append({"row_idx": int(idx), "split": "test"})
    pd.DataFrame(rows).to_parquet(paths.gold / "splits.parquet", index=False)
    logger.info(
        "splits.wrote",
        train=len(sp.train_idx),
        val=len(sp.val_idx),
        test=len(sp.test_idx),
    )


def load_splits(paths: Paths) -> Splits:
    """Read the splits manifest into a ``Splits`` object."""
    df = pd.read_parquet(paths.gold / "splits.parquet")
    train_idx = df.loc[df["split"] == "train", "row_idx"].to_numpy()
    val_idx = df.loc[df["split"] == "val", "row_idx"].to_numpy()
    test_idx = df.loc[df["split"] == "test", "row_idx"].to_numpy()
    return Splits(train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)


# ---------------------------------------------------------------------------
# Baseline training


def run_train_baseline(cfg: AppConfig, paths: Paths) -> dict[str, float]:
    """End-to-end baseline training on persisted Silver/Gold artifacts.

    Returns:
        Validation metrics as a flat dict (suitable for MLflow).
    """
    from mimic_icd_coder.evaluate import evaluate_multilabel
    from mimic_icd_coder.models.baseline import fit_baseline, log_to_mlflow
    from mimic_icd_coder.thresholds import tune_thresholds

    silver = pd.read_parquet(paths.silver / "notes.parquet")
    label_set = load_gold(paths)
    sp = load_splits(paths)

    # Sanity: silver rows must align with label matrix rows.
    if len(silver) != label_set.y.shape[0]:
        raise PipelineError(
            f"Silver rows {len(silver)} != label matrix rows {label_set.y.shape[0]}. "
            "Rebuild Gold to re-align."
        )

    texts = silver["text"].tolist()
    train_texts = [texts[i] for i in sp.train_idx]
    val_texts = [texts[i] for i in sp.val_idx]
    y = label_set.y

    # MLflow — local file store under paths.mlruns
    try:
        import mlflow
    except ImportError:
        mlflow = None  # type: ignore[assignment]

    params = {
        "tfidf_ngram_range": tuple(cfg.baseline.tfidf_ngram_range),
        "tfidf_min_df": cfg.baseline.tfidf_min_df,
        "tfidf_max_features": cfg.baseline.tfidf_max_features,
        "logreg_c": cfg.baseline.logreg_c,
        "logreg_class_weight": cfg.baseline.logreg_class_weight,
        "top_k_labels": cfg.cohort.top_k_labels,
        "n_train": len(train_texts),
        "n_val": len(val_texts),
    }

    model_ctx: object = None
    if mlflow is not None:
        mlflow.set_tracking_uri(f"file:{paths.mlruns.as_posix()}")
        mlflow.set_experiment(cfg.mlflow.experiment_name)
        model_ctx = mlflow.start_run(run_name="baseline_tfidf_lr")

    try:
        model = fit_baseline(
            train_texts,
            y[sp.train_idx],
            label_set.labels,
            tfidf_ngram_range=tuple(cfg.baseline.tfidf_ngram_range),  # type: ignore[arg-type]
            tfidf_min_df=cfg.baseline.tfidf_min_df,
            tfidf_max_features=cfg.baseline.tfidf_max_features,
            logreg_c=cfg.baseline.logreg_c,
            logreg_class_weight=cfg.baseline.logreg_class_weight,
            random_state=cfg.split.seed,
        )
        val_prob = model.predict_proba(val_texts)
        thr = tune_thresholds(y[sp.val_idx], val_prob)
        np.save(paths.gold / "baseline_thresholds.npy", thr)

        result = evaluate_multilabel(
            y[sp.val_idx], val_prob, thr, label_set.labels, top_k_list=cfg.evaluation.top_k_metrics
        )
        metrics = result.to_dict()
        logger.info("baseline.val_metrics", **metrics)

        model_path = paths.gold / "baseline_model.joblib"
        model.save(model_path)
        logger.info("baseline.saved", path=str(model_path))

        if mlflow is not None:
            log_to_mlflow(model, params, metrics)

        return metrics
    finally:
        if mlflow is not None and model_ctx is not None:
            mlflow.end_run()


# ---------------------------------------------------------------------------
# Evaluation on test split


def run_evaluate_test(cfg: AppConfig, paths: Paths) -> dict[str, float]:
    """Load saved baseline and evaluate on held-out test split.

    Returns:
        Test metrics as a flat dict.
    """
    from mimic_icd_coder.evaluate import compare_to_mullenbach, evaluate_multilabel
    from mimic_icd_coder.models.baseline import BaselineModel

    silver = pd.read_parquet(paths.silver / "notes.parquet")
    label_set = load_gold(paths)
    sp = load_splits(paths)

    model = BaselineModel.load(paths.gold / "baseline_model.joblib")
    thr = np.load(paths.gold / "baseline_thresholds.npy")

    test_texts = [silver["text"].iloc[i] for i in sp.test_idx]
    test_prob = model.predict_proba(test_texts)
    result = evaluate_multilabel(
        label_set.y[sp.test_idx],
        test_prob,
        thr,
        label_set.labels,
        top_k_list=cfg.evaluation.top_k_metrics,
    )
    metrics = result.to_dict()
    deltas = compare_to_mullenbach(result)
    logger.info("test.metrics", **metrics)
    logger.info("test.vs_mullenbach", **deltas)
    return {**metrics, **{f"mullenbach_{k}": v for k, v in deltas.items()}}
