"""Databricks job entry point — baseline training.

Reads Silver + Gold from Unity Catalog, fits TF-IDF + LR, logs to MLflow,
registers the model in Unity Catalog Model Registry.

Run via ``databricks bundle run train_baseline --target dev``.
"""

from __future__ import annotations

import argparse

# On Databricks, these imports assume the package is installed via
# ``pip install .`` or distributed as a wheel artifact.
from mimic_icd_coder.logging_utils import configure_logging, get_logger


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--silver-schema", default="silver")
    parser.add_argument("--gold-schema", default="gold")
    parser.add_argument("--models-schema", default="models")
    parser.add_argument("--registered-model", default="discharge_top50_baseline")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    configure_logging(level="INFO", fmt="json")
    log = get_logger("jobs.train_baseline")
    log.info("train_baseline.start", **vars(args))

    # Implementation outline for feat/databricks-migration:
    #   1. Read Silver notes from {catalog}.{silver_schema}.discharge_notes
    #   2. Read Gold labels from {catalog}.{gold_schema}.labels_top50
    #   3. Join on hadm_id, split manifest → train/val/test arrays
    #   4. mlflow.start_run; fit_baseline(...); log params + metrics
    #   5. mlflow.register_model to {catalog}.{models_schema}.{registered_model}
    #   6. Stage alias: set Staging on the new version
    raise NotImplementedError(
        "Spark reads + MLflow registry calls pending on feat/databricks-migration. "
        "See src/mimic_icd_coder/models/baseline.py and pipeline.py::run_train_baseline "
        "for the local reference implementation."
    )


if __name__ == "__main__":
    main()
