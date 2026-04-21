"""Command-line entry points — one per pipeline stage.

Each command reads a YAML config, sets up structured logging, and runs
a single stage. Stages are checkpointed to Parquet / npz so you can
iterate on any one step without recomputing upstream.

Commands:
    mic ingest          — read raw gz CSVs → Bronze Parquet
    mic silver          — Bronze → Silver (cleaned notes)
    mic gold            — Silver + diagnoses → Gold (label matrix)
    mic splits          — Silver → patient-level split manifest
    mic train-baseline  — Silver + Gold → trained TF-IDF+LR, val metrics
    mic evaluate-test   — held-out test metrics
    mic run-all         — run every stage in order
"""

from __future__ import annotations

from pathlib import Path

import click

from mimic_icd_coder.config import AppConfig, load_config
from mimic_icd_coder.logging_utils import configure_logging, get_logger
from mimic_icd_coder.pipeline import (
    Paths,
    run_bronze,
    run_evaluate_test,
    run_gold,
    run_silver,
    run_splits,
    run_train_baseline,
)


def _bootstrap(config_path: Path, artifacts_dir: str) -> tuple[AppConfig, Paths]:
    cfg = load_config(config_path)
    configure_logging(
        level=cfg.logging.get("level", "INFO"),
        fmt=cfg.logging.get("format", "console"),
    )
    paths = Paths(root=Path(artifacts_dir))
    paths.ensure()
    return cfg, paths


CONFIG_OPTION = click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to YAML config (see configs/dev.example.yml).",
)

ARTIFACTS_OPTION = click.option(
    "--artifacts",
    "artifacts_dir",
    type=click.Path(file_okay=False, path_type=str),
    default="./data",
    show_default=True,
    help="Directory for stage outputs (bronze/silver/gold/mlruns).",
)


@click.group()
def cli() -> None:
    """mimic-icd-coder local pipeline."""


@cli.command("ingest")
@CONFIG_OPTION
@ARTIFACTS_OPTION
def cmd_ingest(config_path: Path, artifacts_dir: str) -> None:
    """Read raw gz CSVs → Bronze Parquet."""
    cfg, paths = _bootstrap(config_path, artifacts_dir)
    get_logger("cli").info("stage.ingest")
    run_bronze(cfg, paths)


@cli.command("silver")
@CONFIG_OPTION
@ARTIFACTS_OPTION
def cmd_silver(config_path: Path, artifacts_dir: str) -> None:
    """Bronze → Silver (cleaned, deduped, token-filtered notes)."""
    cfg, paths = _bootstrap(config_path, artifacts_dir)
    get_logger("cli").info("stage.silver")
    run_silver(cfg, paths)


@cli.command("gold")
@CONFIG_OPTION
@ARTIFACTS_OPTION
def cmd_gold(config_path: Path, artifacts_dir: str) -> None:
    """Silver + diagnoses → Gold (top-K ICD-10 multi-hot label matrix)."""
    cfg, paths = _bootstrap(config_path, artifacts_dir)
    get_logger("cli").info("stage.gold")
    run_gold(cfg, paths)


@cli.command("splits")
@CONFIG_OPTION
@ARTIFACTS_OPTION
def cmd_splits(config_path: Path, artifacts_dir: str) -> None:
    """Silver → patient-level train/val/test split manifest."""
    cfg, paths = _bootstrap(config_path, artifacts_dir)
    get_logger("cli").info("stage.splits")
    run_splits(cfg, paths)


@cli.command("train-baseline")
@CONFIG_OPTION
@ARTIFACTS_OPTION
def cmd_train_baseline(config_path: Path, artifacts_dir: str) -> None:
    """Silver + Gold → trained TF-IDF + LR, val metrics logged to MLflow."""
    cfg, paths = _bootstrap(config_path, artifacts_dir)
    log = get_logger("cli")
    log.info("stage.train_baseline")
    metrics = run_train_baseline(cfg, paths)
    log.info("cli.val_metrics", **metrics)


@cli.command("evaluate-test")
@CONFIG_OPTION
@ARTIFACTS_OPTION
def cmd_evaluate_test(config_path: Path, artifacts_dir: str) -> None:
    """Evaluate saved baseline on the held-out test split."""
    cfg, paths = _bootstrap(config_path, artifacts_dir)
    log = get_logger("cli")
    log.info("stage.evaluate_test")
    metrics = run_evaluate_test(cfg, paths)
    log.info("cli.test_metrics", **metrics)


@cli.command("run-all")
@CONFIG_OPTION
@ARTIFACTS_OPTION
def cmd_run_all(config_path: Path, artifacts_dir: str) -> None:
    """Run every pipeline stage end-to-end."""
    cfg, paths = _bootstrap(config_path, artifacts_dir)
    log = get_logger("cli")
    log.info("stage.run_all.start")
    run_bronze(cfg, paths)
    run_silver(cfg, paths)
    run_gold(cfg, paths)
    run_splits(cfg, paths)
    metrics = run_train_baseline(cfg, paths)
    log.info("cli.val_metrics", **metrics)
    test_metrics = run_evaluate_test(cfg, paths)
    log.info("cli.test_metrics", **test_metrics)


if __name__ == "__main__":
    cli()
