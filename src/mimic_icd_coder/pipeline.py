"""Pipeline orchestration — one function per stage, connected by files on disk.

The pipeline runs in four stages, in order: Bronze → Silver → Gold → splits.
After those, two more stages train and evaluate the model. Each stage reads
the previous stage's output files from disk, does its job, and saves its own
output files. This is called a "medallion" layout.

Why it's organized this way:
    - You can re-run one stage without re-running everything upstream. If you
      change the cleaning rules in Silver, you don't need to re-download or
      re-read the raw CSVs — Bronze is still valid on disk.
    - A reviewer can open any stage's output folder and see exactly what the
      next stage will consume. No hidden state, no in-memory magic.
    - Files are the contract between stages. If a file is missing, the stage
      fails with a clear error telling you which earlier stage to run first.
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
    """Where each stage reads from and writes to.

    Given a ``root`` directory (usually ``./data``), this class gives you the
    four sub-folders the pipeline uses — one per stage — so stage code never
    hard-codes paths. ``ensure()`` creates any folders that don't exist yet.
    """

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
        """Create any missing stage folders. Safe to call more than once."""
        for p in (self.bronze, self.silver, self.gold, self.mlruns):
            p.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Bronze


def run_bronze(cfg: AppConfig, paths: Paths) -> None:
    """Stage 1 — copy the raw MIMIC gzipped CSVs into fast Parquet files.

    This is the "landing zone" stage. We don't change the data here — we just
    read the slow-to-parse CSVs once and save them as Parquet, which is much
    faster to re-read in every later stage. Think of it as a cache step.

    Writes these files under ``paths.bronze``:
        - ``discharge_notes.parquet``  (one row per discharge summary)
        - ``diagnoses_icd.parquet``    (one row per assigned ICD code)
        - ``admissions.parquet``       (one row per hospital visit)
        - ``patients.parquet``         (one row per patient)
        - ``d_icd_diagnoses.parquet``  (optional — ICD code → description table)
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
    """Stage 2 — clean the discharge notes and drop ones that are too short.

    Takes the raw notes from Bronze and does three things:
        1. Keeps only the note types we want (usually ``"DS"`` = discharge summary).
        2. Removes duplicate admissions — only one note per hospital visit.
        3. Drops notes shorter than ``cohort.min_note_tokens`` words (default 100),
           because very short notes don't have enough text for a model to learn from.

    Writes one file:
        - ``silver/notes.parquet``  (cleaned, de-duplicated, length-filtered notes)
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
    """Stage 3 — attach the top-50 ICD-10 labels to each note.

    For each note in Silver, look up which ICD-10 codes were assigned to that
    hospital visit. We keep only the K most frequent codes in the cohort
    (default K=50) — the "long tail" is out of scope for this project.

    One important step: we also drop notes whose admissions don't have any
    ICD-10 codes (they're ICD-9-only admissions from before 2015). Those can't
    be used for ICD-10 training, so Silver is overwritten with the smaller,
    cohort-matched set of notes. After this stage, Silver and Gold have the
    same number of rows, in the same order.

    Writes three files under ``paths.gold``:
        - ``labels.npz``        — the label matrix. Sparse CSR shape (n_notes, K).
                                  Each cell is 1 if that note has that code, else 0.
        - ``label_names.json``  — the 50 ICD-10 codes in column order.
        - ``hadm_ids.parquet``  — the hospital admission ID for each row.
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

    # Restrict Silver to the modelable cohort (admissions with at least one
    # ICD-10 diagnosis) per DECISIONS.md 2026-04-20 (ICD-10-only cohort).
    # ICD-9-only admissions are dropped here rather than at Silver so that
    # Silver remains a pure notes-cleaning stage independent of diagnoses.
    # Overwrite silver/notes.parquet so the downstream alignment invariant
    # len(silver) == labels.shape[0] holds in run_train_baseline.
    dx_cohort_hadm = set(
        dx.loc[dx["icd_version"] == cfg.cohort.icd_version, "hadm_id"].dropna().astype(int).tolist()
    )
    before_rows = len(silver)
    silver = (
        silver.loc[silver["hadm_id"].astype(int).isin(dx_cohort_hadm)].copy().reset_index(drop=True)
    )
    logger.info(
        "gold.cohort_restricted",
        silver_full=before_rows,
        silver_cohort=len(silver),
        dropped=before_rows - len(silver),
        icd_version=cfg.cohort.icd_version,
    )
    silver.to_parquet(silver_path, index=False)
    logger.info("gold.silver_rewritten", path=str(silver_path), rows=len(silver))

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
    """Read the three Gold files back into one ``LabelSet`` object.

    This is the inverse of ``run_gold``'s writes — used by the training and
    evaluation stages so they don't each need to know the file layout.
    """
    y = csr_matrix(load_npz(paths.gold / "labels.npz"))
    labels = json.loads((paths.gold / "label_names.json").read_text(encoding="utf-8"))
    hadm_ids = pd.read_parquet(paths.gold / "hadm_ids.parquet")["hadm_id"].to_numpy()
    return LabelSet(hadm_ids=hadm_ids, labels=labels, y=y)


# ---------------------------------------------------------------------------
# Splits


def run_splits(cfg: AppConfig, paths: Paths) -> None:
    """Stage 4 — decide which notes go into train, validation, and test.

    We split by **patient**, not by admission. Why: the same patient can have
    multiple admissions, and patients have personal writing styles and
    recurring conditions. If admissions from the same patient ended up in
    both the training set and the test set, the model would "memorize" that
    patient and look better than it really is. Splitting by patient prevents
    this kind of data leakage.

    Default split is 80% / 10% / 10% (train / val / test), with a fixed
    random seed so the same patients land in the same split every time.

    Writes one file:
        - ``gold/splits.parquet``  — two columns: ``row_idx`` (which row in the
                                     label matrix) and ``split`` (train/val/test).
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
    """Read the splits file back into a ``Splits`` object with three index arrays."""
    df = pd.read_parquet(paths.gold / "splits.parquet")
    train_idx = df.loc[df["split"] == "train", "row_idx"].to_numpy()
    val_idx = df.loc[df["split"] == "val", "row_idx"].to_numpy()
    test_idx = df.loc[df["split"] == "test", "row_idx"].to_numpy()
    return Splits(train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)


# ---------------------------------------------------------------------------
# Baseline training


def run_train_baseline(cfg: AppConfig, paths: Paths) -> dict[str, float]:
    """Stage 5 — train the TF-IDF + Logistic Regression baseline and save it.

    Reads the cleaned notes (Silver), the label matrix (Gold), and the split
    assignments. Trains a word-feature model on the train split, then figures
    out the best decision threshold for each of the 50 labels on the val split,
    and scores the val split to get our first metrics.

    Saves three things to disk:
        - ``gold/baseline_model.joblib``     — the trained model itself.
        - ``gold/baseline_thresholds.npy``   — one decision threshold per label.
        - ``mlruns/<run-id>/``               — MLflow run with params + metrics.

    Returns:
        A flat dictionary of validation metrics (``micro_f1``, ``macro_f1``,
        ``p_at_5``, ``p_at_8``, etc.) — the same shape we log to MLflow.
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

    # MLflow is imported lazily (inside the function, not at module top) for two
    # reasons: (1) `mic --help` should be fast — MLflow's import takes 1-2s, and
    # most CLI commands don't need it; (2) it lets the training stage degrade
    # gracefully if MLflow isn't installed (edge case, but keeps imports honest).
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
    """Stage 6 — score the saved baseline on the held-out test split.

    Loads the model and thresholds saved by stage 5, runs them on the test
    notes, and computes the headline metrics. The test split has never been
    seen during training or threshold tuning, so these numbers are the real
    scoreboard.

    Returns:
        A flat dictionary of test metrics. Also includes ``mullenbach_*_delta``
        keys — how far our numbers are from the published CAML baseline.
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
