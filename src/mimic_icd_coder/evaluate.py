"""Multi-label evaluation — micro/macro F1, P@k, benchmark vs. Mullenbach 2018."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import csr_matrix, issparse
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from mimic_icd_coder.logging_utils import get_logger

logger = get_logger(__name__)

# Published CAML baseline from Mullenbach et al. (2018),
# "Explainable Prediction of Medical Codes from Clinical Text" (NAACL).
# Table 5, "Results on MIMIC-III, 50 labels", CAML row (p. 1107).
# https://arxiv.org/abs/1802.05695
#
# Table 5 does NOT report P@8 for the 50-label setting — only P@5 is tabled
# for top-50. The 0.709 / 0.523 P@8 values that appear in the paper are
# from Table 4 (MIMIC-III full codes) and Table 6 (MIMIC-II full codes)
# respectively, neither of which is an apples-to-apples baseline for a
# MIMIC-IV top-50 comparison. We deliberately omit p_at_8 here rather
# than cite a wrong-setting number.
#
# Kept as sanity benchmark for MIMIC-IV top-50 — not a promise.
MULLENBACH_CAML_TOP50 = {
    "micro_f1": 0.614,  # Table 5, CAML Micro-F1
    "macro_f1": 0.532,  # Table 5, CAML Macro-F1
    "p_at_5": 0.609,  # Table 5, CAML P@5
}


@dataclass
class EvalResult:
    """Evaluation summary."""

    micro_f1: float
    macro_f1: float
    micro_auc: float | None = None
    macro_auc: float | None = None
    micro_auprc: float | None = None
    macro_auprc: float | None = None
    precision_at_k: dict[int, float] = field(default_factory=dict)
    per_label: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, float]:
        """Flatten for MLflow logging."""
        out: dict[str, float] = {
            "micro_f1": self.micro_f1,
            "macro_f1": self.macro_f1,
        }
        if self.micro_auc is not None:
            out["micro_auc"] = self.micro_auc
        if self.macro_auc is not None:
            out["macro_auc"] = self.macro_auc
        if self.micro_auprc is not None:
            out["micro_auprc"] = self.micro_auprc
        if self.macro_auprc is not None:
            out["macro_auprc"] = self.macro_auprc
        for k, v in self.precision_at_k.items():
            out[f"p_at_{k}"] = v
        return out


def _to_dense(y: csr_matrix | np.ndarray) -> np.ndarray:
    if issparse(y):
        return np.asarray(y.todense())
    return np.asarray(y)


def precision_at_k(y_true: np.ndarray, y_prob: np.ndarray, k: int) -> float:
    """Precision at k — fraction of top-k predicted labels that are positive.

    Args:
        y_true: Binary target, shape ``(n, L)``.
        y_prob: Predicted probabilities, shape ``(n, L)``.
        k: Cutoff.

    Returns:
        Mean precision@k across rows.
    """
    if y_true.shape != y_prob.shape:
        raise ValueError(f"Shape mismatch: {y_true.shape} vs {y_prob.shape}")
    if k < 1 or k > y_true.shape[1]:
        raise ValueError(f"k={k} out of range [1, {y_true.shape[1]}]")

    topk = np.argsort(-y_prob, axis=1)[:, :k]
    hits = np.take_along_axis(y_true, topk, axis=1).sum(axis=1)
    return float((hits / k).mean())


def evaluate_multilabel(
    y_true: csr_matrix | np.ndarray,
    y_prob: np.ndarray,
    thresholds: np.ndarray,
    labels: list[str],
    top_k_list: list[int] | None = None,
) -> EvalResult:
    """Compute multi-label metrics using per-label thresholds.

    Args:
        y_true: Binary targets.
        y_prob: Predicted probabilities.
        thresholds: Per-label thresholds, shape ``(n_labels,)``.
        labels: Label names.
        top_k_list: List of k values for precision@k (e.g. [5, 8, 15]).

    Returns:
        ``EvalResult`` with micro/macro F1, AUC, AUPRC, P@k, and per-label metrics.
    """
    if top_k_list is None:
        top_k_list = [5, 8, 15]

    y_true_d = _to_dense(y_true)
    if y_true_d.shape != y_prob.shape:
        raise ValueError(f"Shape mismatch: {y_true_d.shape} vs {y_prob.shape}")
    if thresholds.shape != (y_prob.shape[1],):
        raise ValueError(f"thresholds shape {thresholds.shape} != (n_labels={y_prob.shape[1]},)")

    y_pred = (y_prob >= thresholds[None, :]).astype(np.int8)

    micro = f1_score(y_true_d, y_pred, average="micro", zero_division=0)
    macro = f1_score(y_true_d, y_pred, average="macro", zero_division=0)

    # AUC / AUPRC are defined only when both classes are present per-label.
    try:
        micro_auc = roc_auc_score(y_true_d, y_prob, average="micro")
        macro_auc = roc_auc_score(y_true_d, y_prob, average="macro")
        micro_auprc = average_precision_score(y_true_d, y_prob, average="micro")
        macro_auprc = average_precision_score(y_true_d, y_prob, average="macro")
    except ValueError:
        micro_auc = macro_auc = micro_auprc = macro_auprc = None

    p_at_k = {k: precision_at_k(y_true_d, y_prob, k) for k in top_k_list}

    precisions, recalls, f1s, supports = precision_recall_fscore_support(
        y_true_d, y_pred, average=None, zero_division=0
    )
    per_label = {
        labels[i]: {
            "precision": float(precisions[i]),
            "recall": float(recalls[i]),
            "f1": float(f1s[i]),
            "support": int(supports[i]),
            "threshold": float(thresholds[i]),
        }
        for i in range(len(labels))
    }

    result = EvalResult(
        micro_f1=float(micro),
        macro_f1=float(macro),
        micro_auc=None if micro_auc is None else float(micro_auc),
        macro_auc=None if macro_auc is None else float(macro_auc),
        micro_auprc=None if micro_auprc is None else float(micro_auprc),
        macro_auprc=None if macro_auprc is None else float(macro_auprc),
        precision_at_k=p_at_k,
        per_label=per_label,
    )
    logger.info(
        "evaluate.summary",
        micro_f1=result.micro_f1,
        macro_f1=result.macro_f1,
        p_at_5=p_at_k.get(5),
        p_at_8=p_at_k.get(8),
    )
    return result


def compare_to_mullenbach(result: EvalResult) -> dict[str, float]:
    """Return absolute deltas from the Mullenbach 2018 CAML top-50 baseline.

    Positive = we beat the baseline.
    """
    deltas = {
        "micro_f1_delta": result.micro_f1 - MULLENBACH_CAML_TOP50["micro_f1"],
        "macro_f1_delta": result.macro_f1 - MULLENBACH_CAML_TOP50["macro_f1"],
    }
    if 5 in result.precision_at_k and "p_at_5" in MULLENBACH_CAML_TOP50:
        deltas["p_at_5_delta"] = result.precision_at_k[5] - MULLENBACH_CAML_TOP50["p_at_5"]
    # Mullenbach Table 5 does not report P@8 for the 50-label setting — no
    # apples-to-apples baseline exists, so we deliberately skip the delta.
    if 8 in result.precision_at_k and "p_at_8" in MULLENBACH_CAML_TOP50:
        deltas["p_at_8_delta"] = result.precision_at_k[8] - MULLENBACH_CAML_TOP50["p_at_8"]
    return deltas
