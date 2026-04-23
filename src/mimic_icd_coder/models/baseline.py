"""TF-IDF + multi-label Logistic Regression baseline.

Real, runnable code. Use ``fit_baseline`` to train on Silver notes + Gold
label matrix; logs to MLflow if an active run is set.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import joblib
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline

from mimic_icd_coder.logging_utils import get_logger, is_debug_enabled

logger = get_logger(__name__)


@dataclass(frozen=True)
class BaselineModel:
    """Trained baseline — vectorizer + multi-label classifier + label list."""

    vectorizer: TfidfVectorizer
    classifier: OneVsRestClassifier
    labels: list[str]

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """Predict per-label probabilities.

        Args:
            texts: Input documents.

        Returns:
            Array of shape ``(len(texts), n_labels)``.
        """
        x = self.vectorizer.transform(texts)
        return cast("np.ndarray", self.classifier.predict_proba(x))

    def save(self, path: str | Path) -> None:
        """Persist the model to disk via joblib.

        Args:
            path: Output file (typically ``.joblib``).
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"vectorizer": self.vectorizer, "classifier": self.classifier, "labels": self.labels},
            p,
        )
        logger.info("baseline.saved", path=str(p))

    @classmethod
    def load(cls, path: str | Path) -> BaselineModel:
        """Load a model previously saved with ``save``."""
        data = joblib.load(Path(path))
        return cls(
            vectorizer=data["vectorizer"],
            classifier=data["classifier"],
            labels=data["labels"],
        )


def fit_baseline(
    train_texts: list[str],
    y_train: csr_matrix,
    labels: list[str],
    *,
    tfidf_ngram_range: tuple[int, int] = (1, 2),
    tfidf_min_df: int = 5,
    tfidf_max_features: int = 200_000,
    logreg_c: float = 1.0,
    logreg_class_weight: str | None = "balanced",
    random_state: int = 42,
    n_jobs: int = 1,
) -> BaselineModel:
    """Fit TF-IDF + OneVsRest LogisticRegression.

    Args:
        train_texts: Training documents.
        y_train: Sparse multi-label targets, shape ``(n_docs, n_labels)``.
        labels: Ordered label list (column names for y).
        tfidf_ngram_range: N-gram range for TF-IDF.
        tfidf_min_df: Minimum document frequency.
        tfidf_max_features: Vocabulary cap.
        logreg_c: L2 inverse regularization strength.
        logreg_class_weight: ``"balanced"`` or ``None``.
        random_state: RNG seed.
        n_jobs: Parallelism for OvR.

    Returns:
        Trained ``BaselineModel``.
    """
    if y_train.shape[1] != len(labels):
        raise ValueError(
            f"y_train has {y_train.shape[1]} columns but {len(labels)} labels provided"
        )
    if y_train.shape[0] != len(train_texts):
        raise ValueError(
            f"y_train has {y_train.shape[0]} rows but {len(train_texts)} texts provided"
        )

    logger.info(
        "baseline.fit.start",
        n_train=len(train_texts),
        n_labels=len(labels),
        ngram_range=tfidf_ngram_range,
        min_df=tfidf_min_df,
        max_features=tfidf_max_features,
    )

    vectorizer = TfidfVectorizer(
        ngram_range=tfidf_ngram_range,
        min_df=tfidf_min_df,
        max_features=tfidf_max_features,
        sublinear_tf=True,
        lowercase=True,
        strip_accents="unicode",
    )
    logger.debug("baseline.tfidf.fit_transform_start", n_docs=len(train_texts))
    x_train = vectorizer.fit_transform(train_texts)
    logger.info(
        "baseline.tfidf.done",
        vocab_size=len(vectorizer.vocabulary_),
        x_shape=x_train.shape,
        x_nnz=int(x_train.nnz),
    )

    # Debug mode turns on framework-native verbose output — liblinear's
    # per-iteration convergence ("iter 1 act ... nnz ... cg ...") and joblib's
    # per-label progress ("Done N out of 50, elapsed Xs, remaining Ys").
    # INFO mode stays quiet: logs only the stage-boundary events above.
    debug = is_debug_enabled()
    lr_verbose = 1 if debug else 0
    ovr_verbose = 10 if debug else 0

    base = LogisticRegression(
        C=logreg_c,
        class_weight=logreg_class_weight,
        max_iter=1000,
        solver="liblinear",
        random_state=random_state,
        verbose=lr_verbose,
    )
    # n_jobs=1 by default: joblib's multi-process backend memmaps sparse
    # x_train / y_train as read-only, and liblinear's C writeback-cast then
    # fails with "WRITEBACKIFCOPY base is read-only". Serial fit sidesteps
    # the whole issue at a ~2-3x runtime cost. Override only with a solver
    # that tolerates memmap input (e.g. saga).
    clf = OneVsRestClassifier(base, n_jobs=n_jobs, verbose=ovr_verbose)
    logger.info("baseline.classifier.fit_start", n_labels=len(labels), n_train=x_train.shape[0])
    clf.fit(x_train, y_train)

    logger.info("baseline.fit.done", vocab_size=len(vectorizer.vocabulary_))
    return BaselineModel(vectorizer=vectorizer, classifier=clf, labels=labels)


def log_to_mlflow(model: BaselineModel, params: dict[str, Any], metrics: dict[str, float]) -> None:
    """Log baseline model + params + metrics to the active MLflow run.

    No-op if MLflow is not installed or no run is active.

    Args:
        model: Trained baseline.
        params: Hyperparameters to log.
        metrics: Evaluation metrics.
    """
    try:
        import mlflow  # noqa: WPS433 (runtime-only import)
        import mlflow.sklearn  # noqa: WPS433
    except ImportError:
        logger.warning("mlflow.unavailable_skipping_log")
        return

    if mlflow.active_run() is None:
        logger.warning("mlflow.no_active_run_skipping")
        return

    mlflow.log_params(params)
    mlflow.log_metrics(metrics)
    # Persist both the full pipeline as sklearn flavor + an artifact copy.
    pipeline = Pipeline(steps=[("tfidf", model.vectorizer), ("clf", model.classifier)])
    mlflow.sklearn.log_model(
        sk_model=pipeline,
        artifact_path="baseline_model",
        registered_model_name=None,
    )
    mlflow.log_dict({"labels": model.labels}, "labels.json")
    logger.info("mlflow.logged")
