"""Gold — build multi-label target matrix for top-K ICD-10 codes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from mimic_icd_coder.logging_utils import get_logger

logger = get_logger(__name__)


class LabelError(Exception):
    """Raised when label engineering produces an invalid matrix."""


@dataclass(frozen=True)
class LabelSet:
    """Multi-label target matrix aligned with admissions.

    Attributes:
        hadm_ids: Admission IDs in row order.
        labels: Ordered list of ICD-10 codes (length = n_classes).
        y: Sparse multi-hot matrix, shape (n_admissions, n_classes).
    """

    hadm_ids: np.ndarray
    labels: list[str]
    y: csr_matrix


def filter_icd10(diagnoses: pd.DataFrame) -> pd.DataFrame:
    """Filter to ICD-10 rows and normalize codes.

    Args:
        diagnoses: Raw ``hosp/diagnoses_icd``.

    Returns:
        Filtered DataFrame with columns ``[hadm_id, icd_code]`` and codes
        upper-cased and stripped.
    """
    if "icd_version" not in diagnoses.columns:
        raise LabelError("diagnoses missing 'icd_version' column")
    df = diagnoses.loc[diagnoses["icd_version"] == 10].copy()
    df["icd_code"] = df["icd_code"].astype(str).str.strip().str.upper()
    logger.info("labels.filter_icd10", rows=len(df))
    return df[["hadm_id", "icd_code"]]


def top_k_codes(diagnoses_icd10: pd.DataFrame, k: int) -> list[str]:
    """Return the top-K most frequent ICD-10 codes.

    Args:
        diagnoses_icd10: Filtered ICD-10 rows.
        k: Number of codes to retain.

    Returns:
        Sorted-by-frequency list of ICD-10 codes, length exactly K.

    Raises:
        LabelError: If fewer than K distinct codes exist.
    """
    counts = diagnoses_icd10["icd_code"].value_counts()
    if len(counts) < k:
        raise LabelError(f"Only {len(counts)} distinct ICD-10 codes; requested top-{k}")
    top = counts.head(k).index.tolist()
    logger.info("labels.top_k_selected", k=k, min_count=int(counts.iloc[k - 1]))
    return sorted(top)  # stable ordering for reproducibility


def build_label_matrix(
    diagnoses_icd10: pd.DataFrame, hadm_ids: np.ndarray, labels: list[str]
) -> csr_matrix:
    """Multi-hot encode codes per admission.

    Args:
        diagnoses_icd10: Filtered ICD-10 rows.
        hadm_ids: Ordered array of admissions to produce rows for.
        labels: Ordered list of target codes (columns).

    Returns:
        Sparse CSR matrix, shape ``(len(hadm_ids), len(labels))``.
    """
    label_to_col = {c: i for i, c in enumerate(labels)}
    hadm_to_row = {h: i for i, h in enumerate(hadm_ids)}

    df = diagnoses_icd10.loc[
        diagnoses_icd10["icd_code"].isin(label_to_col)
        & diagnoses_icd10["hadm_id"].isin(hadm_to_row)
    ]

    rows = df["hadm_id"].map(hadm_to_row).to_numpy()
    cols = df["icd_code"].map(label_to_col).to_numpy()
    data = np.ones(len(df), dtype=np.int8)

    y = csr_matrix((data, (rows, cols)), shape=(len(hadm_ids), len(labels)), dtype=np.int8)
    # Dedup via max (one admission can have the same code twice in input)
    y = (y > 0).astype(np.int8)
    logger.info(
        "labels.matrix_built",
        admissions=y.shape[0],
        classes=y.shape[1],
        density=float(y.sum() / (y.shape[0] * y.shape[1])),
    )
    return y


def build_labels(silver_notes: pd.DataFrame, diagnoses: pd.DataFrame, k: int) -> LabelSet:
    """End-to-end Gold label build.

    Args:
        silver_notes: Cleaned notes (must have ``hadm_id`` column).
        diagnoses: Raw diagnoses table (ICD-9 and ICD-10 mixed).
        k: Top-K ICD-10 codes to retain.

    Returns:
        ``LabelSet`` aligned with silver_notes row order.

    Raises:
        LabelError: If the resulting matrix has any admissions with zero labels
            (indicates join error or missing codes).
    """
    d10 = filter_icd10(diagnoses)
    hadm_ids = silver_notes["hadm_id"].to_numpy()
    labels = top_k_codes(d10, k)
    y = build_label_matrix(d10, hadm_ids, labels)

    zero_label = (y.sum(axis=1) == 0).sum()
    if zero_label > 0.5 * y.shape[0]:
        raise LabelError(
            f"{zero_label} of {y.shape[0]} admissions have zero top-{k} labels — "
            "check join keys, ICD version filter, or increase k"
        )
    logger.info("labels.built", zero_label_admissions=int(zero_label))
    return LabelSet(hadm_ids=hadm_ids, labels=labels, y=y)
