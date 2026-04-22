"""Smoke tests — run end-to-end on synthetic fixtures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mimic_icd_coder import __version__
from mimic_icd_coder.data.clean import build_silver_notes, clean_text, token_length
from mimic_icd_coder.data.labels import LabelError, build_labels, filter_icd10, top_k_codes
from mimic_icd_coder.data.splits import SplitError, patient_split
from mimic_icd_coder.evaluate import (
    MULLENBACH_CAML_TOP50,
    evaluate_multilabel,
    precision_at_k,
)
from mimic_icd_coder.models.baseline import fit_baseline
from mimic_icd_coder.thresholds import tune_thresholds
from tests.fixtures.synthetic_notes import make_synthetic


def test_version() -> None:
    assert __version__ == "0.1.0"


def test_read_admissions_contains_demographic_columns(tmp_path: Path) -> None:
    """Regression: admissions ingest must surface demographic columns.

    MIMIC-IV v3.1 admissions header (verified against the raw CSV):
      subject_id, hadm_id, admittime, dischtime, deathtime, admission_type,
      admit_provider_id, admission_location, discharge_location, insurance,
      language, marital_status, race, edregtime, edouttime, hospital_expire_flag
    """
    import gzip

    raw = tmp_path / "admissions.csv.gz"
    header = (
        "subject_id,hadm_id,admittime,dischtime,deathtime,admission_type,"
        "admit_provider_id,admission_location,discharge_location,insurance,"
        "language,marital_status,race,edregtime,edouttime,hospital_expire_flag"
    )
    row = (
        "1,100,2150-01-01 00:00:00,2150-01-05 00:00:00,,URGENT,"
        "P1,ED,HOME,Medicare,English,MARRIED,WHITE,,,0"
    )
    with gzip.open(raw, "wt", encoding="utf-8") as fh:
        fh.write(header + "\n" + row + "\n")

    from mimic_icd_coder.data.ingest import read_admissions

    df = read_admissions(raw)
    required = {"race", "insurance", "language", "discharge_location"}
    missing = required - set(df.columns)
    assert not missing, f"read_admissions missing columns: {missing}"
    assert df.iloc[0]["race"] == "WHITE"
    assert df.iloc[0]["insurance"] == "Medicare"
    assert df.iloc[0]["language"] == "English"
    assert df.iloc[0]["discharge_location"] == "HOME"


def test_clean_text_redacts_and_normalizes() -> None:
    raw = "Patient is    Mr. ___   admitted on ___   with    CHF."
    out = clean_text(raw)
    assert "[REDACTED]" in out
    assert "    " not in out
    assert "CHF" in out


def test_clean_text_handles_non_string() -> None:
    assert clean_text(None) == ""  # type: ignore[arg-type]
    assert clean_text(42) == ""  # type: ignore[arg-type]


def test_token_length() -> None:
    assert token_length("hello world test") == 3
    assert token_length("") == 0
    assert token_length(None) == 0  # type: ignore[arg-type]


def test_build_silver_filters_and_dedups() -> None:
    corpus = make_synthetic(n_patients=20, seed=1)
    silver = build_silver_notes(corpus["notes"], min_tokens=50)
    assert set(silver.columns) == {"note_id", "subject_id", "hadm_id", "text", "n_tokens"}
    assert silver["hadm_id"].is_unique, "Silver must be deduped by hadm_id"
    assert (silver["n_tokens"] >= 50).all()


def test_filter_icd10_rejects_missing_column() -> None:
    df = pd.DataFrame({"hadm_id": [1], "icd_code": ["I50.9"]})
    with pytest.raises(LabelError, match="icd_version"):
        filter_icd10(df)


def test_top_k_codes_raises_when_too_few() -> None:
    df = pd.DataFrame(
        {"hadm_id": [1, 2, 3], "icd_code": ["A", "A", "B"], "icd_version": [10, 10, 10]}
    )
    d10 = filter_icd10(df)
    with pytest.raises(LabelError, match="distinct"):
        top_k_codes(d10, k=50)


def test_build_labels_end_to_end() -> None:
    corpus = make_synthetic(n_patients=40, seed=2)
    silver = build_silver_notes(corpus["notes"], min_tokens=50)
    # Fixture uses 10 distinct ICD codes; k=8 keeps coverage realistic for a
    # multi-label task without asserting 100% top-k saturation.
    label_set = build_labels(silver, corpus["diagnoses_icd"], k=8)
    assert label_set.y.shape == (len(silver), 8)
    assert len(label_set.labels) == 8
    # Most admissions should hit at least one top-k label with the synthetic DGP.
    coverage = float((label_set.y.sum(axis=1) > 0).mean())
    assert coverage > 0.8, f"Top-k coverage {coverage:.2f} lower than expected"


def test_build_labels_excludes_codes_absent_from_cohort() -> None:
    """Regression test for the Z20.822-style cohort-support bug.

    A code may be frequent in the full ICD-10 diagnoses pool but absent
    from the trainable cohort (e.g., COVID-era admissions without notes
    in MIMIC-IV-Note v2.2). Top-K must be computed from cohort-restricted
    diagnoses so such codes never enter the label space as constant-zero
    columns.
    """
    # Silver: two admissions (100, 101) belonging to patients 1 and 2.
    silver = pd.DataFrame(
        {
            "subject_id": [1, 2],
            "hadm_id": [100, 101],
            "text": ["sample note one longer text", "sample note two longer text"],
            "n_tokens": [5, 5],
        }
    )
    # Diagnoses: cohort admissions (100, 101) have code A.
    # Phantom admission 200 (not in the cohort) has many positives of code Z —
    # which in the FULL pool outranks A and would become the top-1 label
    # despite zero cohort support.
    diagnoses = pd.DataFrame(
        {
            "hadm_id": [100, 101, 200, 200, 200, 200, 200],
            "icd_code": ["A", "A", "Z", "Z", "Z", "Z", "Z"],
            "icd_version": [10, 10, 10, 10, 10, 10, 10],
        }
    )

    # Sanity: in the full pool, Z is top-1 (5 vs 2 for A).
    from mimic_icd_coder.data.labels import filter_icd10
    from mimic_icd_coder.data.labels import top_k_codes as topk_fn

    d10_full = filter_icd10(diagnoses)
    full_pool_top1 = topk_fn(d10_full, k=1)
    assert full_pool_top1 == ["Z"], f"Sanity check failed: full pool top-1 = {full_pool_top1}"

    # The cohort-aware build must pick A, not Z.
    label_set = build_labels(silver, diagnoses, k=1)
    assert label_set.labels == ["A"], (
        f"Cohort-aware top-1 should be ['A'] (only code present in cohort), "
        f"got {label_set.labels} — top-K is being computed from the full pool, not cohort"
    )
    # Both cohort admissions have code A → full positive coverage.
    assert int(label_set.y.sum()) == 2


def test_build_labels_cohort_restriction_logs_drop() -> None:
    """The cohort restriction should measurably drop diagnosis rows when
    the raw pool contains non-cohort admissions."""
    silver = pd.DataFrame(
        {
            "subject_id": [1],
            "hadm_id": [100],
            "text": ["a" * 500],
            "n_tokens": [1],
        }
    )
    # 1 cohort row, 9 non-cohort rows
    diagnoses = pd.DataFrame(
        {
            "hadm_id": [100] + [999] * 9,
            "icd_code": ["A"] * 10,
            "icd_version": [10] * 10,
        }
    )
    label_set = build_labels(silver, diagnoses, k=1)
    assert label_set.labels == ["A"]
    # Only the single cohort row should contribute to the matrix
    assert int(label_set.y.sum()) == 1


def test_patient_split_disjoint() -> None:
    corpus = make_synthetic(n_patients=50, seed=3)
    silver = build_silver_notes(corpus["notes"], min_tokens=50)
    sp = patient_split(silver, seed=7)

    def sids(idx: np.ndarray) -> set[int]:
        return set(silver.iloc[idx]["subject_id"].tolist())

    assert sids(sp.train_idx).isdisjoint(sids(sp.val_idx))
    assert sids(sp.train_idx).isdisjoint(sids(sp.test_idx))
    assert sids(sp.val_idx).isdisjoint(sids(sp.test_idx))


def test_patient_split_fraction_validation() -> None:
    df = pd.DataFrame({"subject_id": [1, 2, 3]})
    with pytest.raises(SplitError, match="sum"):
        patient_split(df, train_frac=0.9, val_frac=0.2, test_frac=0.1)


def test_precision_at_k() -> None:
    # Two docs, 4 labels each. Doc 1 has labels at indices 0,1. Doc 2 at 2,3.
    y = np.array([[1, 1, 0, 0], [0, 0, 1, 1]])
    prob = np.array([[0.9, 0.8, 0.1, 0.2], [0.1, 0.2, 0.9, 0.8]])
    # Top-2 predictions are all hits.
    assert precision_at_k(y, prob, k=2) == pytest.approx(1.0)
    # Top-4 = 2/4 = 0.5
    assert precision_at_k(y, prob, k=4) == pytest.approx(0.5)


def test_precision_at_k_rejects_out_of_range() -> None:
    y = np.array([[1, 0]])
    prob = np.array([[0.6, 0.4]])
    with pytest.raises(ValueError):
        precision_at_k(y, prob, k=5)


def test_mullenbach_constants_present() -> None:
    for key in ("micro_f1", "macro_f1", "p_at_5", "p_at_8"):
        assert key in MULLENBACH_CAML_TOP50


def test_baseline_end_to_end_beats_random() -> None:
    """The baseline on synthetic data should beat chance and produce a non-trivial micro F1."""
    corpus = make_synthetic(n_patients=80, seed=4)
    silver = build_silver_notes(corpus["notes"], min_tokens=50)
    label_set = build_labels(silver, corpus["diagnoses_icd"], k=5)
    sp = patient_split(silver, seed=7)

    texts = silver["text"].tolist()
    train_texts = [texts[i] for i in sp.train_idx]
    val_texts = [texts[i] for i in sp.val_idx]

    y = label_set.y
    # Small vocab + min_df=1 so the tiny synthetic corpus still has features.
    model = fit_baseline(
        train_texts,
        y[sp.train_idx],
        label_set.labels,
        tfidf_ngram_range=(1, 1),
        tfidf_min_df=1,
        tfidf_max_features=5000,
        logreg_c=1.0,
        logreg_class_weight=None,
    )
    val_prob = model.predict_proba(val_texts)
    thr = tune_thresholds(y[sp.val_idx], val_prob)
    # k list must not exceed label count; fixture uses 5 labels here.
    result = evaluate_multilabel(
        y[sp.val_idx], val_prob, thr, label_set.labels, top_k_list=[1, 3, 5]
    )

    # Synthetic signal → baseline should clear 0.3 micro F1 on this tiny set.
    assert result.micro_f1 > 0.3, f"Baseline micro F1 too low: {result.micro_f1}"


def test_threshold_tuning_returns_valid_array() -> None:
    y = np.array([[1, 0], [0, 1], [1, 1], [0, 0], [1, 0], [0, 1]])
    prob = np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.6], [0.1, 0.3], [0.8, 0.2], [0.3, 0.9]])
    thr = tune_thresholds(y, prob, min_support=1)
    assert thr.shape == (2,)
    assert (thr > 0).all() and (thr < 1).all()
