"""Unit tests for eda.py analysis functions on synthetic fixtures."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mimic_icd_coder.eda import (
    admissions_per_patient,
    admissions_per_patient_stats,
    bert_truncation_impact,
    codes_per_admission,
    codes_per_admission_stats,
    cohort_coverage,
    compute_lengths,
    date_range,
    deid_marker_stats,
    format_icd10_code,
    icd_frequency,
    icd_version_distribution,
    join_coverage,
    label_cooccurrence,
    length_of_stay,
    length_percentiles,
    mortality_rate,
    note_duplication_summary,
    note_type_distribution,
    null_rate_by_column,
    patient_demographics,
    summarize_volumetrics,
    top_k_coverage,
    version_reconciliation,
)
from tests.fixtures.synthetic_notes import make_synthetic


@pytest.fixture()
def corpus() -> dict[str, pd.DataFrame]:
    return make_synthetic(n_patients=60, seed=11)


# -- Volumetrics -------------------------------------------------------------


def test_summarize_volumetrics(corpus: dict[str, pd.DataFrame]) -> None:
    out = summarize_volumetrics(corpus)
    assert {"table", "rows", "columns", "memory_mb"} <= set(out.columns)
    assert len(out) == 4
    assert (out["rows"] > 0).all()


def test_null_rate_by_column() -> None:
    df = pd.DataFrame({"a": [1, 2, np.nan, 4], "b": [1, 2, 3, 4]})
    out = null_rate_by_column(df, "test")
    assert out.loc[out["column"] == "a", "null_rate"].iloc[0] == pytest.approx(0.25)
    assert out.loc[out["column"] == "b", "null_rate"].iloc[0] == 0.0


def test_date_range_valid() -> None:
    df = pd.DataFrame({"d": ["2020-01-01", "2020-06-15", "2020-12-31"]})
    r = date_range(df, "d")
    assert r["span_days"] == 365
    assert r["n_non_null"] == 3


def test_date_range_all_null() -> None:
    df = pd.DataFrame({"d": [None, None]})
    r = date_range(df, "d")
    assert r["n_non_null"] == 0
    assert r["span_days"] == 0


# -- Note types --------------------------------------------------------------


def test_note_type_distribution(corpus: dict[str, pd.DataFrame]) -> None:
    out = note_type_distribution(corpus["notes"])
    assert "DS" in out["note_type"].tolist()
    assert out["pct"].sum() == pytest.approx(100.0, abs=0.1)


# -- Length analysis ---------------------------------------------------------


def test_compute_lengths_adds_columns(corpus: dict[str, pd.DataFrame]) -> None:
    out = compute_lengths(corpus["notes"])
    assert "n_tokens" in out.columns
    assert "n_chars" in out.columns
    assert (out["n_tokens"] > 0).all()


def test_length_percentiles(corpus: dict[str, pd.DataFrame]) -> None:
    lens = compute_lengths(corpus["notes"])
    pct = length_percentiles(lens)
    assert {"pct", "n_tokens", "n_chars"} == set(pct.columns)
    # min/max/mean/std + 8 default percentiles must all be present
    labels = pct["pct"].tolist()
    assert {"min", "max", "mean", "std"} <= set(labels)
    assert pct.loc[pct["pct"] == "std", "n_tokens"].iloc[0] > 0


def test_bert_truncation_impact() -> None:
    df = pd.DataFrame({"n_tokens": [100, 400, 800, 1200, 2000]})
    res = bert_truncation_impact(df, max_tokens=512)
    assert res["n_total"] == 5
    assert res["n_exceeds"] == 3
    assert res["pct_exceeds"] == 60.0


# -- De-id markers -----------------------------------------------------------


def test_deid_marker_stats(corpus: dict[str, pd.DataFrame]) -> None:
    # Synthetic fixture has few de-id markers, so inject some.
    notes = corpus["notes"].copy()
    notes.loc[notes.index[:5], "text"] = notes.loc[notes.index[:5], "text"] + " ___ Dr. ___ "
    res = deid_marker_stats(notes, sample_size=None)
    assert res["total_markers"] >= 10
    assert res["mean_markers_per_note"] > 0


# -- Duplicates --------------------------------------------------------------


def test_note_duplication_summary(corpus: dict[str, pd.DataFrame]) -> None:
    res = note_duplication_summary(corpus["notes"])
    assert res["n_notes"] > 0
    assert res["duplicate_note_id"] == 0  # synthetic generator makes unique IDs


# -- ICD version distribution ------------------------------------------------


def test_icd_version_distribution_without_admissions(corpus: dict[str, pd.DataFrame]) -> None:
    out = icd_version_distribution(corpus["diagnoses_icd"])
    assert (out["icd_version"] == 10).any()
    assert (out["n_codes"] > 0).all()


def test_icd_version_distribution_with_admissions(corpus: dict[str, pd.DataFrame]) -> None:
    out = icd_version_distribution(corpus["diagnoses_icd"], corpus["admissions"])
    assert "year" in out.columns


# -- ICD frequency and top-K coverage ---------------------------------------


def test_format_icd10_code_adds_period_for_long_codes() -> None:
    assert format_icd10_code("E785") == "E78.5"
    assert format_icd10_code("I4891") == "I48.91"
    assert format_icd10_code("Z20822") == "Z20.822"


def test_format_icd10_code_leaves_short_codes_unchanged() -> None:
    # 3-character category codes have no sub-category; no period inserted.
    assert format_icd10_code("I10") == "I10"
    assert format_icd10_code("F32") == "F32"
    assert format_icd10_code("") == ""


def test_icd_frequency_returns_sorted(corpus: dict[str, pd.DataFrame]) -> None:
    freq = icd_frequency(corpus["diagnoses_icd"], version=10)
    assert freq["rank"].iloc[0] == 1
    assert freq["n_codes"].is_monotonic_decreasing


def test_top_k_coverage_monotonic(corpus: dict[str, pd.DataFrame]) -> None:
    cov = top_k_coverage(corpus["diagnoses_icd"], k_list=(2, 5, 10))
    assert cov["pct_admissions_covered"].is_monotonic_increasing
    assert cov["pct_admissions_covered"].iloc[-1] <= 100.0


# -- Codes per admission -----------------------------------------------------


def test_codes_per_admission(corpus: dict[str, pd.DataFrame]) -> None:
    cpa = codes_per_admission(corpus["diagnoses_icd"], version=10)
    assert cpa["n_codes"].min() >= 1
    stats = codes_per_admission_stats(cpa)
    assert stats["n_admissions"] == len(cpa)
    assert stats["min"] >= 1


# -- Co-occurrence -----------------------------------------------------------


def test_label_cooccurrence_shape(corpus: dict[str, pd.DataFrame]) -> None:
    cooc = label_cooccurrence(corpus["diagnoses_icd"], top_k=5, version=10)
    assert cooc.shape == (5, 5)
    # Diagonal should equal per-code admission count (≥ any off-diagonal in that row)
    for i, code in enumerate(cooc.index):
        assert cooc.iloc[i, i] >= cooc.iloc[i].drop(code).max()


def test_label_cooccurrence_symmetric(corpus: dict[str, pd.DataFrame]) -> None:
    cooc = label_cooccurrence(corpus["diagnoses_icd"], top_k=4, version=10)
    arr = cooc.to_numpy()
    assert np.allclose(arr, arr.T), "Co-occurrence matrix must be symmetric"


# -- Patient and admission-level --------------------------------------------


def test_patient_demographics(corpus: dict[str, pd.DataFrame]) -> None:
    demo = patient_demographics(corpus["patients"])
    assert {"dimension", "bucket", "n", "pct"} == set(demo.columns)
    assert (demo["dimension"] == "gender").any()
    assert (demo["dimension"] == "age_bucket").any()


def test_admissions_per_patient_stats(corpus: dict[str, pd.DataFrame]) -> None:
    per = admissions_per_patient(corpus["admissions"])
    stats = admissions_per_patient_stats(per)
    assert stats["n_patients"] > 0
    assert stats["min"] >= 1


def test_length_of_stay(corpus: dict[str, pd.DataFrame]) -> None:
    los = length_of_stay(corpus["admissions"])
    # Synthetic has 4-day LOS
    assert los.mean() == pytest.approx(4.0, abs=0.5)


def test_mortality_rate_synthetic(corpus: dict[str, pd.DataFrame]) -> None:
    res = mortality_rate(corpus["admissions"])
    assert res["n_admissions"] > 0
    assert 0.0 <= res["rate"] <= 1.0


# -- Join coverage -----------------------------------------------------------


def test_join_coverage_total_all_three(corpus: dict[str, pd.DataFrame]) -> None:
    out = join_coverage(corpus["notes"], corpus["admissions"], corpus["diagnoses_icd"])
    all_three = out.loc[out["slice"] == "all_three", "count"].iloc[0]
    total_notes = out.loc[out["slice"] == "total_notes", "count"].iloc[0]
    # In synthetic, every note has both an admission and ICD-10 codes
    assert all_three == total_notes


# -- Version reconciliation -------------------------------------------------


def test_version_reconciliation_no_drop(corpus: dict[str, pd.DataFrame]) -> None:
    res = version_reconciliation(corpus["notes"], corpus["admissions"])
    assert res["n_notes_missing_from_hosp"] == 0
    assert res["pct_notes_missing_from_hosp"] == 0.0


def test_version_reconciliation_partial_drop() -> None:
    notes = pd.DataFrame({"hadm_id": [1, 2, 3, 4]})
    hosp = pd.DataFrame({"hadm_id": [1, 2]})
    res = version_reconciliation(notes, hosp)
    assert res["n_notes_missing_from_hosp"] == 2
    assert res["pct_notes_missing_from_hosp"] == 50.0


def test_cohort_coverage_decomposition() -> None:
    """drift and cohort-filter losses are separated, and sum to n_notes."""
    # 5 notes: hadm_ids 1..5
    # admissions has 1,2,3 — so hadm 4,5 are drift losses
    # cohort has 1 only — so hadm 2,3 are filter losses (present in admissions, excluded)
    notes = pd.DataFrame({"hadm_id": [1, 2, 3, 4, 5]})
    admissions = pd.DataFrame({"hadm_id": [1, 2, 3]})
    cohort = pd.DataFrame({"hadm_id": [1]})

    res = cohort_coverage(notes, admissions, cohort)

    assert res["n_notes"] == 5
    assert res["n_retained"] == 1
    assert res["n_drift_loss"] == 2
    assert res["n_cohort_filter_loss"] == 2
    # Invariant
    assert res["n_retained"] + res["n_drift_loss"] + res["n_cohort_filter_loss"] == 5
    # Percentages sum to ~100
    total_pct = res["pct_retained"] + res["pct_drift_loss"] + res["pct_cohort_filter_loss"]
    assert total_pct == pytest.approx(100.0, abs=0.01)


def test_cohort_coverage_all_retained(corpus: dict[str, pd.DataFrame]) -> None:
    """On synthetic data with no drift and a non-restrictive cohort, everything is retained."""
    # Use admissions itself as the cohort — nothing is filtered out.
    res = cohort_coverage(corpus["notes"], corpus["admissions"], corpus["admissions"])
    assert res["n_drift_loss"] == 0
    assert res["n_cohort_filter_loss"] == 0
    assert res["n_retained"] == res["n_notes"]
