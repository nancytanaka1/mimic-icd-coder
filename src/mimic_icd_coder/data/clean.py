"""Silver — clean discharge notes and dedup per admission.

TODO (date un-shifting, tracked in reports/eda_report.md §6):
    MIMIC-IV de-identifies dates with per-patient shifts of 100+ years,
    making raw `admittime`, `dischtime`, `charttime`, `deathtime`,
    `patients.dod` uninterpretable as real calendar years. Before any
    model consumer uses these fields (temporal split, drift monitoring,
    fairness-by-era, data card year coverage), add a transform that
    computes `real_year_approx` per admission:

        real_year_approx = admittime.year
                         - patients.anchor_year
                         + midpoint(patients.anchor_year_group)

    where `anchor_year_group` is a 3-year bucket like "2014 - 2016"
    (midpoint 2015). Precision ±1.5 years. Persist as new columns
    alongside the raw shifted values — do not overwrite. Not required
    for text-only ICD coding baseline; defer until first temporal use.
"""

from __future__ import annotations

import re

import pandas as pd

from mimic_icd_coder.logging_utils import get_logger

logger = get_logger(__name__)

# MIMIC notes use ``___`` to mark de-identified spans (names, dates).
DEID_PATTERN = re.compile(r"_{2,}")
WHITESPACE_PATTERN = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Normalize whitespace and collapse de-id markers.

    Args:
        text: Raw note text.

    Returns:
        Cleaned text. Returns empty string for ``None`` or non-string input.
    """
    if not isinstance(text, str):
        return ""
    t = DEID_PATTERN.sub(" [REDACTED] ", text)
    t = WHITESPACE_PATTERN.sub(" ", t).strip()
    return t


def token_length(text: str) -> int:
    """Whitespace-split token count (fast approximation)."""
    if not isinstance(text, str):
        return 0
    return len(text.split())


def build_silver_notes(
    notes: pd.DataFrame,
    *,
    note_types: list[str] | None = None,
    min_tokens: int = 100,
) -> pd.DataFrame:
    """Dedup, filter, and clean discharge notes.

    Rules:
        1. Keep rows where ``note_type`` is in ``note_types`` (default: ``["DS"]``).
        2. Dedup by ``hadm_id`` — if multiple notes exist, keep the one with
           the highest ``note_seq`` (final/amended discharge summary).
        3. Clean text via ``clean_text``.
        4. Drop notes shorter than ``min_tokens`` (whitespace tokens).

    Args:
        notes: Raw notes from ``read_discharge_notes``.
        note_types: Acceptable ``note_type`` values.
        min_tokens: Minimum whitespace token count to retain.

    Returns:
        Silver DataFrame with columns ``[note_id, subject_id, hadm_id, text, n_tokens]``.
    """
    if note_types is None:
        note_types = ["DS"]

    before = len(notes)
    df = notes.loc[notes["note_type"].isin(note_types)].copy()
    logger.info(
        "silver.filter_note_type", note_types=note_types, kept=len(df), dropped=before - len(df)
    )

    # Dedup: keep last note_seq per hadm_id
    df = df.sort_values(["hadm_id", "note_seq"]).drop_duplicates(subset=["hadm_id"], keep="last")
    logger.info("silver.dedup_hadm", remaining=len(df))

    df["text"] = df["text"].map(clean_text)
    df["n_tokens"] = df["text"].map(token_length)

    before = len(df)
    df = df.loc[df["n_tokens"] >= min_tokens].copy()
    logger.info(
        "silver.token_filter",
        min_tokens=min_tokens,
        kept=len(df),
        dropped=before - len(df),
    )

    return df[["note_id", "subject_id", "hadm_id", "text", "n_tokens"]].reset_index(drop=True)
