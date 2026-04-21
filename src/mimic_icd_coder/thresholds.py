"""Per-label threshold tuning via PR curve optimization on the validation set."""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix, issparse
from sklearn.metrics import precision_recall_curve

from mimic_icd_coder.logging_utils import get_logger

logger = get_logger(__name__)


def tune_thresholds(
    y_true: csr_matrix | np.ndarray,
    y_prob: np.ndarray,
    *,
    metric: str = "f1",
    min_support: int = 5,
    default_threshold: float = 0.5,
) -> np.ndarray:
    """Find per-label thresholds that maximize a binary metric on val.

    Args:
        y_true: Validation targets, shape ``(n, L)``.
        y_prob: Validation probabilities, shape ``(n, L)``.
        metric: Either ``"f1"`` or ``"youden"``.
        min_support: Minimum positives required to tune; fallback otherwise.
        default_threshold: Fallback threshold for labels with insufficient support.

    Returns:
        Array of thresholds, shape ``(L,)``.
    """
    if metric not in {"f1", "youden"}:
        raise ValueError(f"Unknown metric: {metric!r}")

    y = np.asarray(y_true.todense()) if issparse(y_true) else np.asarray(y_true)
    n_labels = y.shape[1]
    thresholds = np.full(n_labels, default_threshold, dtype=np.float32)
    tuned = 0

    for j in range(n_labels):
        support = int(y[:, j].sum())
        if support < min_support:
            continue

        precisions, recalls, thr = precision_recall_curve(y[:, j], y_prob[:, j])
        if metric == "f1":
            # f1 vector is of length n_thresholds (thr is length n_thresholds)
            # precisions/recalls have length n_thresholds+1 (the final point has threshold=+inf)
            p = precisions[:-1]
            r = recalls[:-1]
            denom = np.where((p + r) > 0, (p + r), 1.0)
            f1 = np.where((p + r) > 0, 2 * p * r / denom, 0.0)
            best_idx = int(np.argmax(f1))
        else:  # youden
            j_stat = recalls[:-1] - (1 - precisions[:-1])
            best_idx = int(np.argmax(j_stat))

        if 0 <= best_idx < len(thr):
            thresholds[j] = float(thr[best_idx])
            tuned += 1

    logger.info(
        "thresholds.tuned",
        metric=metric,
        n_labels=n_labels,
        n_tuned=tuned,
        n_default=n_labels - tuned,
        min_support=min_support,
    )
    return thresholds
