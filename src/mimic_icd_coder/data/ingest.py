"""Bronze ingestion — raw gzipped CSVs → pandas DataFrames.

On Databricks, swap these to ``spark.read.csv(..., header=True)`` writing
to Delta. This module keeps a local code path for development and CI.

Reads via ``pyarrow.csv.read_csv`` directly (not pandas' ``engine="pyarrow"``
shim) so we can pass ``newlines_in_values=True`` — MIMIC discharge summaries
have quoted text fields containing embedded newlines, which pyarrow's
default parser rejects. The native pyarrow reader is also multi-threaded
and 2–5× faster than pandas' C engine on large files.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv

from mimic_icd_coder.logging_utils import get_logger

logger = get_logger(__name__)


class IngestError(Exception):
    """Raised when raw data ingestion fails."""


def _read_gz_csv(path: str | Path, columns: list[str] | None = None) -> pd.DataFrame:
    """Read a gzipped CSV with optional column whitelist.

    Uses the native PyArrow CSV reader with ``newlines_in_values=True`` so
    quoted multi-line fields (e.g. MIMIC discharge note text) parse correctly.
    Gzip compression is auto-detected from the ``.gz`` extension.

    Args:
        path: Local path to a ``.csv.gz`` file.
        columns: If provided, read only these columns.

    Returns:
        DataFrame with columns lowercased.

    Raises:
        IngestError: If the file is missing or unreadable.
    """
    p = Path(path)
    if not p.is_file():
        raise IngestError(f"Missing raw file: {p}")

    read_opts = pacsv.ReadOptions(block_size=16 * 1024 * 1024)
    parse_opts = pacsv.ParseOptions(newlines_in_values=True)
    convert_opts = (
        pacsv.ConvertOptions(include_columns=columns) if columns else pacsv.ConvertOptions()
    )

    try:
        table = pacsv.read_csv(
            str(p),
            read_options=read_opts,
            parse_options=parse_opts,
            convert_options=convert_opts,
        )
    except (OSError, pa.ArrowInvalid) as exc:
        raise IngestError(f"Failed to read {p}") from exc

    df = table.to_pandas()
    df.columns = [c.lower() for c in df.columns]
    logger.info("ingest.read", path=str(p), rows=len(df), cols=len(df.columns))
    return df


def read_discharge_notes(path: str | Path) -> pd.DataFrame:
    """Read MIMIC-IV-Note discharge summaries.

    Expected columns: ``note_id, subject_id, hadm_id, note_type, note_seq,
    charttime, storetime, text``.

    Args:
        path: Path to ``discharge.csv.gz``.

    Returns:
        DataFrame of discharge notes.
    """
    return _read_gz_csv(
        path,
        columns=["note_id", "subject_id", "hadm_id", "note_type", "note_seq", "charttime", "text"],
    )


def read_diagnoses_icd(path: str | Path) -> pd.DataFrame:
    """Read MIMIC-IV Hosp ``diagnoses_icd.csv.gz``.

    Expected columns: ``subject_id, hadm_id, seq_num, icd_code, icd_version``.
    """
    return _read_gz_csv(
        path, columns=["subject_id", "hadm_id", "seq_num", "icd_code", "icd_version"]
    )


def read_admissions(path: str | Path) -> pd.DataFrame:
    """Read MIMIC-IV Hosp ``admissions.csv.gz``."""
    return _read_gz_csv(
        path,
        columns=[
            "subject_id",
            "hadm_id",
            "admittime",
            "dischtime",
            "deathtime",
            "hospital_expire_flag",
            "admission_type",
        ],
    )


def read_patients(path: str | Path) -> pd.DataFrame:
    """Read MIMIC-IV Hosp ``patients.csv.gz``."""
    return _read_gz_csv(
        path,
        columns=["subject_id", "gender", "anchor_age", "anchor_year", "anchor_year_group", "dod"],
    )


def read_d_icd_diagnoses(path: str | Path) -> pd.DataFrame:
    """Read MIMIC-IV Hosp ``d_icd_diagnoses.csv.gz`` — the ICD code dictionary.

    Expected columns: ``icd_code, icd_version, long_title``. Used downstream
    to attach human-readable descriptions to predicted codes and for the
    top-50 code table in the EDA / data card.
    """
    return _read_gz_csv(path, columns=["icd_code", "icd_version", "long_title"])
