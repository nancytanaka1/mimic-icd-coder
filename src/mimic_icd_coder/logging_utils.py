"""Structured logging via structlog."""

from __future__ import annotations

import logging
import sys

import structlog


def configure_logging(level: str = "INFO", fmt: str = "console") -> None:
    """Configure structlog with JSON or console output.

    Args:
        level: Log level name.
        fmt: ``"json"`` for production, ``"console"`` for local dev.
    """
    resolved_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=resolved_level,
    )
    # basicConfig is a no-op on second invocation (the root logger already
    # has handlers). Force the level directly so --log-level overrides take
    # effect when the CLI is invoked multiple times in the same process
    # (e.g. from tests, or `mic run-all` calling multiple stages).
    logging.getLogger().setLevel(resolved_level)

    renderer: structlog.types.Processor = (
        structlog.processors.JSONRenderer()
        if fmt == "json"
        else structlog.dev.ConsoleRenderer(colors=False)
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a bound structlog logger."""
    return structlog.get_logger(name)


def is_debug_enabled() -> bool:
    """Return True if the root logger is at DEBUG level.

    Training modules call this to decide whether to turn on framework-native
    verbose knobs — sklearn's ``verbose=`` parameters, HuggingFace Trainer's
    per-step logging, etc. Keeps per-iteration spam out of INFO-mode logs
    while giving DEBUG-mode runs full visibility into each inner loop.
    """
    return logging.getLogger().isEnabledFor(logging.DEBUG)
