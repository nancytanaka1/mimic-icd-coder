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
from mimic_icd_coder.logging_utils import configure_logging, is_debug_enabled
from mimic_icd_coder.models.baseline import fit_baseline
from mimic_icd_coder.models.transformer import TransformerTrainConfig, fine_tune, tokenize_and_chunk
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
    # Mullenbach 2018 Table 5 (MIMIC-III top-50) reports Micro F1, Macro F1,
    # and P@5 for CAML. P@8 is NOT reported for the top-50 setting — only for
    # full-codes tables — so it is deliberately omitted from the constant to
    # avoid citing a wrong-setting baseline. See evaluate.py for rationale.
    for key in ("micro_f1", "macro_f1", "p_at_5"):
        assert key in MULLENBACH_CAML_TOP50
    assert "p_at_8" not in MULLENBACH_CAML_TOP50, (
        "p_at_8 must stay out of MULLENBACH_CAML_TOP50 — Mullenbach Table 5 "
        "does not report it for the top-50 setting"
    )


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


def test_fit_baseline_verbose_is_gated_by_debug_log_level() -> None:
    """Contract: sklearn verbose knobs fire only when root logger is at DEBUG.

    INFO mode should stay quiet (no per-iter liblinear spam, no per-label
    joblib spam). DEBUG mode flips both verbose knobs on. This protects the
    "--log-level DEBUG" UX — a reviewer reading INFO-mode logs shouldn't see
    a wall of convergence output.
    """
    configure_logging(level="INFO")
    assert is_debug_enabled() is False
    # Fit with INFO — LogisticRegression.verbose should resolve to 0.
    corpus = make_synthetic(n_patients=15, seed=11)
    silver = build_silver_notes(corpus["notes"], min_tokens=50)
    label_set = build_labels(silver, corpus["diagnoses_icd"], k=3)
    model = fit_baseline(
        silver["text"].tolist(),
        label_set.y,
        label_set.labels,
        tfidf_ngram_range=(1, 1),
        tfidf_min_df=1,
        tfidf_max_features=500,
        logreg_c=1.0,
        logreg_class_weight=None,
    )
    # OvR wraps one LR per label; they all share the same verbose setting.
    assert all(est.verbose == 0 for est in model.classifier.estimators_)

    # Flip to DEBUG — verbose must now be on.
    configure_logging(level="DEBUG")
    assert is_debug_enabled() is True
    model2 = fit_baseline(
        silver["text"].tolist(),
        label_set.y,
        label_set.labels,
        tfidf_ngram_range=(1, 1),
        tfidf_min_df=1,
        tfidf_max_features=500,
        logreg_c=1.0,
        logreg_class_weight=None,
    )
    assert all(est.verbose == 1 for est in model2.classifier.estimators_)

    # Reset so downstream tests aren't stuck in DEBUG.
    configure_logging(level="INFO")


def test_tokenize_and_chunk_preserves_doc_idx_and_shapes() -> None:
    """A long note should split into multiple chunks; each chunk keeps its
    parent ``doc_idx`` and chunk lengths stay within ``max_length``.

    Uses ``distilbert-base-uncased`` as a small, universally-cached
    tokenizer so the test runs quickly in CI without downloading the
    ~440 MB Bio_ClinicalBERT model weights (the production tokenizer is
    API-compatible with this one — identical chunking semantics).
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    # Doc 0: long enough that a 64-token window needs multiple chunks.
    # Doc 1: short, single-chunk.
    long_text = "patient presents with acute onset chest pain and dyspnea " * 20
    short_text = "brief stable patient"

    chunks = tokenize_and_chunk([long_text, short_text], tokenizer, max_length=64, stride=16)

    # Multiple chunks from doc 0, exactly one from doc 1.
    doc0_chunks = [c for c in chunks if c["doc_idx"] == 0]
    doc1_chunks = [c for c in chunks if c["doc_idx"] == 1]
    assert len(doc0_chunks) >= 2, f"long doc should split; got {len(doc0_chunks)}"
    assert len(doc1_chunks) == 1, f"short doc should stay single; got {len(doc1_chunks)}"

    # Every chunk respects max_length and has aligned masks.
    for c in chunks:
        assert len(c["input_ids"]) <= 64
        assert len(c["input_ids"]) == len(c["attention_mask"])
        assert all(tok in (0, 1) for tok in c["attention_mask"])

    # Ordering invariant: all doc 0 chunks come before doc 1's.
    doc_indices = [c["doc_idx"] for c in chunks]
    assert doc_indices == sorted(doc_indices), "doc order must be stable across chunks"


def test_tokenize_and_chunk_handles_empty_input() -> None:
    """An empty list is a valid input and should produce zero chunks."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    assert tokenize_and_chunk([], tokenizer, max_length=64, stride=16) == []


def test_tokenize_and_chunk_stride_increases_chunk_count() -> None:
    """Sliding-window (stride > 0) must produce at least as many chunks as
    contiguous (stride = 0) on the same input. This is the correctness
    check for the stride parameter — positional overlap is hard to verify
    with a repeating-vocab test, but the chunk-count monotonicity is
    deterministic and matches the sliding-window contract.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    long_text = "patient presents with acute onset chest pain and dyspnea " * 20

    contiguous = tokenize_and_chunk([long_text], tokenizer, max_length=32, stride=0)
    sliding = tokenize_and_chunk([long_text], tokenizer, max_length=32, stride=8)

    assert len(contiguous) >= 2, "long doc with small window must chunk multiple times"
    assert len(sliding) >= len(contiguous), (
        f"stride=8 must produce at least as many chunks as stride=0; "
        f"got sliding={len(sliding)}, contiguous={len(contiguous)}"
    )


def test_fine_tune_on_tiny_synthetic_runs_end_to_end(tmp_path: Path) -> None:
    """Smoke test for the transformer fine_tune path.

    Uses ``hf-internal-testing/tiny-random-bert`` (~1 MB, random weights,
    ~500 params) so the test runs on CPU in under 90 seconds with a
    negligible first-run download. The goal is API correctness — forward
    pass, backward pass, checkpoint save, MLflow logging hook — NOT model
    quality; we never assert on loss or F1 values because the fixture is
    too small to train meaningfully.
    """
    import os

    import mlflow

    # Defensive reset: prior tests may have left an active MLflow run or
    # cached an experiment ID whose on-disk directory has been cleaned up
    # by pytest's tmp_path teardown. Without this, HuggingFace Trainer's
    # MLflow integration raises "Could not find experiment with ID ..."
    # when the cached ID no longer has a backing directory.
    if mlflow.active_run() is not None:
        mlflow.end_run()
    mlflow.set_tracking_uri(f"file:{(tmp_path / 'mlruns').as_posix()}")
    mlflow.set_experiment("fine-tune-smoke-test")
    os.environ["MLFLOW_DISABLE_TELEMETRY"] = "1"

    corpus = make_synthetic(n_patients=12, seed=11)
    silver = build_silver_notes(corpus["notes"], min_tokens=50)
    label_set = build_labels(silver, corpus["diagnoses_icd"], k=3)

    texts = silver["text"].tolist()
    y = np.asarray(label_set.y.todense()).astype(np.float32)

    n = len(texts)
    split = max(2, int(n * 0.75))
    train_texts, val_texts = texts[:split], texts[split:]
    y_train, y_val = y[:split], y[split:]

    cfg = TransformerTrainConfig(
        model_name="hf-internal-testing/tiny-random-bert",
        max_length=64,
        stride=16,
        batch_size=2,
        learning_rate=1e-3,
        epochs=1,
        warmup_ratio=0.0,
        weight_decay=0.0,
        fp16=False,
        gradient_checkpointing=False,
        gradient_accumulation_steps=1,
        seed=42,
    )

    output_dir = tmp_path / "fine_tune_output"

    result_path = fine_tune(
        train_texts=train_texts,
        y_train=y_train,
        val_texts=val_texts,
        y_val=y_val,
        labels=label_set.labels,
        cfg=cfg,
        output_dir=output_dir,
    )

    # Contract: fine_tune returns a Path to a saved-model directory.
    assert isinstance(result_path, Path)
    assert result_path.exists()
    # Model saved (HF writes config.json and weights).
    assert (result_path / "config.json").exists(), "config.json missing from saved model"
    # Tokenizer saved (HF writes tokenizer_config.json or at minimum vocab files).
    tokenizer_marker = any(
        (result_path / name).exists() for name in ("tokenizer_config.json", "vocab.txt")
    )
    assert tokenizer_marker, "tokenizer not saved alongside the model"


def test_threshold_tuning_returns_valid_array() -> None:
    y = np.array([[1, 0], [0, 1], [1, 1], [0, 0], [1, 0], [0, 1]])
    prob = np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.6], [0.1, 0.3], [0.8, 0.2], [0.3, 0.9]])
    thr = tune_thresholds(y, prob, min_support=1)
    assert thr.shape == (2,)
    assert (thr > 0).all() and (thr < 1).all()
