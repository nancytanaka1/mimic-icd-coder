"""Typed YAML configuration with environment-variable substitution."""

from __future__ import annotations

import os
import re
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

_ENV_PATTERN = re.compile(r"\$\{(\w+)\}")


class UnityCatalogConfig(BaseModel):
    catalog: str
    bronze_schema: str
    silver_schema: str
    gold_schema: str
    models_schema: str


class DataConfig(BaseModel):
    notes_path: str
    diagnoses_path: str
    admissions_path: str
    patients_path: str
    d_icd_path: str | None = None


class CohortConfig(BaseModel):
    icd_version: int = Field(ge=9, le=10)
    min_note_tokens: int = Field(ge=0)
    top_k_labels: int = Field(ge=1, le=1000)
    note_types: list[str]


class SplitConfig(BaseModel):
    train_frac: float = Field(gt=0, lt=1)
    val_frac: float = Field(gt=0, lt=1)
    test_frac: float = Field(gt=0, lt=1)
    seed: int
    strategy: str


class BaselineConfig(BaseModel):
    tfidf_ngram_range: tuple[int, int]
    tfidf_min_df: int
    tfidf_max_features: int
    logreg_c: float
    logreg_class_weight: str | None


class TransformerConfig(BaseModel):
    model_name: str
    max_length: int
    batch_size: int
    learning_rate: float
    epochs: int
    warmup_ratio: float
    weight_decay: float
    fp16: bool


class EvaluationConfig(BaseModel):
    threshold_strategy: str
    top_k_metrics: list[int]


class MLflowConfig(BaseModel):
    experiment_name: str
    registry_model_name: str


class AppConfig(BaseModel):
    unity_catalog: UnityCatalogConfig
    data: DataConfig
    cohort: CohortConfig
    split: SplitConfig
    baseline: BaselineConfig
    transformer: TransformerConfig
    evaluation: EvaluationConfig
    mlflow: MLflowConfig
    logging: dict[str, str]


def _substitute_env(raw: str) -> str:
    """Replace ``${VAR}`` with env-var values.

    Raises:
        KeyError: if a referenced env var is missing.
    """

    def _replace(match: re.Match[str]) -> str:
        var = match.group(1)
        value = os.environ.get(var)
        if value is None:
            raise KeyError(f"Environment variable {var!r} is not set")
        return value

    return _ENV_PATTERN.sub(_replace, raw)


def load_config(path: str | Path) -> AppConfig:
    """Load and validate a YAML config.

    Args:
        path: Config file path.

    Returns:
        Parsed ``AppConfig``.

    Raises:
        FileNotFoundError: If the file doesn't exist.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Config not found: {p}")
    raw = p.read_text(encoding="utf-8")
    data = yaml.safe_load(_substitute_env(raw))
    return AppConfig(**data)
