"""Patient-level train/val/test splits with optional label stratification."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from mimic_icd_coder.logging_utils import get_logger

logger = get_logger(__name__)


class SplitError(Exception):
    """Raised when split invariants are violated."""


@dataclass(frozen=True)
class Splits:
    """Row indices (into the Silver/Gold aligned DataFrames) for each split."""

    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray


def patient_split(
    silver: pd.DataFrame,
    *,
    train_frac: float = 0.80,
    val_frac: float = 0.10,
    test_frac: float = 0.10,
    seed: int = 42,
) -> Splits:
    """Split admissions so no patient appears in more than one split.

    Admission-level splits leak patient-specific language patterns across
    train/test and inflate metrics — a well-documented pitfall in clinical NLP.

    Args:
        silver: Silver notes with ``subject_id`` column.
        train_frac: Fraction of patients for training.
        val_frac: Fraction of patients for validation.
        test_frac: Fraction of patients for testing.
        seed: RNG seed.

    Returns:
        ``Splits`` with row indices into ``silver``.

    Raises:
        SplitError: If fractions don't sum to 1.0 (±1e-6) or splits overlap.
    """
    total = train_frac + val_frac + test_frac
    if abs(total - 1.0) > 1e-6:
        raise SplitError(f"Fractions sum to {total}, expected 1.0")

    rng = np.random.default_rng(seed)
    patients = silver["subject_id"].drop_duplicates().to_numpy()
    rng.shuffle(patients)

    n = len(patients)
    n_train = int(round(train_frac * n))
    n_val = int(round(val_frac * n))

    train_p = set(patients[:n_train].tolist())
    val_p = set(patients[n_train : n_train + n_val].tolist())
    test_p = set(patients[n_train + n_val :].tolist())

    if train_p & val_p or train_p & test_p or val_p & test_p:
        raise SplitError("Patient sets overlap across splits")

    sid = silver["subject_id"].to_numpy()
    train_idx = np.where(np.isin(sid, list(train_p)))[0]
    val_idx = np.where(np.isin(sid, list(val_p)))[0]
    test_idx = np.where(np.isin(sid, list(test_p)))[0]

    logger.info(
        "splits.built",
        train_patients=len(train_p),
        val_patients=len(val_p),
        test_patients=len(test_p),
        train_admissions=len(train_idx),
        val_admissions=len(val_idx),
        test_admissions=len(test_idx),
    )
    return Splits(train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)


def label_coverage_report(y: csr_matrix, splits: Splits, labels: list[str]) -> pd.DataFrame:
    """Report positive rate per label in each split (sanity check for rare-label skew).

    Args:
        y: Multi-hot label matrix.
        splits: Patient-level splits.
        labels: Ordered label list.

    Returns:
        DataFrame with columns ``[label, train_pct, val_pct, test_pct]``.
    """
    rows: list[dict[str, float | str]] = []
    for j, label in enumerate(labels):
        col = y[:, j].toarray().ravel()
        rows.append(
            {
                "label": label,
                "train_pct": float(col[splits.train_idx].mean()),
                "val_pct": float(col[splits.val_idx].mean()),
                "test_pct": float(col[splits.test_idx].mean()),
            }
        )
    return pd.DataFrame(rows)
