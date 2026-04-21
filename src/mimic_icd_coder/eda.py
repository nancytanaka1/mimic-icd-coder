"""Exploratory Data Analysis — analysis functions for MIMIC-IV cohort.

Each function takes raw Bronze DataFrames and returns analysis-ready
DataFrames or dicts. Visualization helpers take a matplotlib ``Axes`` and
return the ``Axes`` so the notebook can compose multi-panel figures.

Design principles:
    - Functions are pure: no global state, no print side effects.
    - Returns are DataFrames (for the report), not terminal output.
    - Type-hinted, docstring'd, testable on synthetic fixtures.
    - Callers control figure sizing, saving, and layout.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from mimic_icd_coder.data.clean import token_length
from mimic_icd_coder.logging_utils import get_logger

logger = get_logger(__name__)

_DEID_MARKER = re.compile(r"_{2,}")


# ---------------------------------------------------------------------------
# 1. Volumetrics
# ---------------------------------------------------------------------------


def summarize_volumetrics(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Row count, column count, and memory footprint for each table.

    Args:
        tables: Mapping from table name to DataFrame.

    Returns:
        DataFrame with columns ``[table, rows, columns, memory_mb]``.
    """
    rows: list[dict[str, Any]] = []
    for name, df in tables.items():
        rows.append(
            {
                "table": name,
                "rows": len(df),
                "columns": df.shape[1],
                "memory_mb": round(df.memory_usage(deep=True).sum() / 1e6, 2),
            }
        )
    return pd.DataFrame(rows)


def null_rate_by_column(df: pd.DataFrame, table_name: str) -> pd.DataFrame:
    """Per-column null rate as DataFrame.

    Args:
        df: Table to analyze.
        table_name: Human-readable label for the ``table`` column.

    Returns:
        DataFrame ``[table, column, nulls, null_rate]`` sorted by null_rate desc.
    """
    data = [
        {
            "table": table_name,
            "column": col,
            "nulls": int(df[col].isna().sum()),
            "null_rate": float(df[col].isna().mean()),
        }
        for col in df.columns
    ]
    return pd.DataFrame(data).sort_values("null_rate", ascending=False).reset_index(drop=True)


def date_range(df: pd.DataFrame, date_column: str) -> dict[str, Any]:
    """Min, max, and span for a datetime-like column.

    Args:
        df: Table with the date column.
        date_column: Column name (will be coerced via ``pd.to_datetime``).

    Returns:
        Dict with ``min``, ``max``, ``span_days``, ``n_non_null``.
    """
    s = pd.to_datetime(df[date_column], errors="coerce")
    non_null = s.dropna()
    if non_null.empty:
        return {"min": None, "max": None, "span_days": 0, "n_non_null": 0}
    span = (non_null.max() - non_null.min()).days
    return {
        "min": non_null.min().isoformat(),
        "max": non_null.max().isoformat(),
        "span_days": int(span),
        "n_non_null": int(len(non_null)),
    }


# ---------------------------------------------------------------------------
# 2. Note types
# ---------------------------------------------------------------------------


def note_type_distribution(notes: pd.DataFrame) -> pd.DataFrame:
    """Count and percentage for each ``note_type``.

    Returns:
        DataFrame ``[note_type, count, pct]`` sorted by count desc.
    """
    counts = notes["note_type"].value_counts(dropna=False)
    total = max(counts.sum(), 1)
    df = counts.reset_index()
    df.columns = ["note_type", "count"]
    df["pct"] = (df["count"] / total * 100).round(3)
    return df


# ---------------------------------------------------------------------------
# 3. Token / char length
# ---------------------------------------------------------------------------


def compute_lengths(notes: pd.DataFrame) -> pd.DataFrame:
    """Return notes with ``n_tokens`` and ``n_chars`` appended.

    Args:
        notes: Must contain a ``text`` column.

    Returns:
        Copy of ``notes`` with two new columns. Original is not mutated.
    """
    out = notes.copy()
    out["n_chars"] = out["text"].fillna("").str.len().astype(int)
    out["n_tokens"] = out["text"].map(token_length)
    return out


def length_percentiles(
    notes_with_lengths: pd.DataFrame,
    percentiles: tuple[float, ...] = (0.01, 0.05, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99),
) -> pd.DataFrame:
    """Percentile summary of token and char lengths.

    Args:
        notes_with_lengths: Output of ``compute_lengths``.
        percentiles: Quantiles to compute (in [0, 1]).

    Returns:
        DataFrame with one row per percentile and columns for tokens & chars,
        plus a ``min`` and ``max`` row.
    """
    rows: list[dict[str, Any]] = []
    for q in percentiles:
        rows.append(
            {
                "pct": f"p{int(q * 100):02d}",
                "n_tokens": float(notes_with_lengths["n_tokens"].quantile(q)),
                "n_chars": float(notes_with_lengths["n_chars"].quantile(q)),
            }
        )
    rows.insert(
        0,
        {
            "pct": "min",
            "n_tokens": float(notes_with_lengths["n_tokens"].min()),
            "n_chars": float(notes_with_lengths["n_chars"].min()),
        },
    )
    rows.append(
        {
            "pct": "max",
            "n_tokens": float(notes_with_lengths["n_tokens"].max()),
            "n_chars": float(notes_with_lengths["n_chars"].max()),
        }
    )
    rows.append(
        {
            "pct": "mean",
            "n_tokens": float(notes_with_lengths["n_tokens"].mean()),
            "n_chars": float(notes_with_lengths["n_chars"].mean()),
        }
    )
    rows.append(
        {
            "pct": "std",
            "n_tokens": float(notes_with_lengths["n_tokens"].std()),
            "n_chars": float(notes_with_lengths["n_chars"].std()),
        }
    )
    return pd.DataFrame(rows)


def bert_truncation_impact(
    notes_with_lengths: pd.DataFrame, max_tokens: int = 512
) -> dict[str, float]:
    """What fraction of notes exceed a given token cap?

    Args:
        notes_with_lengths: Output of ``compute_lengths``.
        max_tokens: Cap (e.g. 512 for Bio_ClinicalBERT, 4096 for Longformer).

    Returns:
        Dict with ``n_total``, ``n_exceeds``, ``pct_exceeds``,
        ``median_tokens_lost_for_exceeders``.
    """
    n_total = len(notes_with_lengths)
    exceeds = notes_with_lengths["n_tokens"] > max_tokens
    n_exceeds = int(exceeds.sum())
    over = notes_with_lengths.loc[exceeds, "n_tokens"] - max_tokens
    median_lost = float(over.median()) if n_exceeds > 0 else 0.0
    return {
        "max_tokens": max_tokens,
        "n_total": n_total,
        "n_exceeds": n_exceeds,
        "pct_exceeds": round(n_exceeds / max(n_total, 1) * 100, 2),
        "median_tokens_lost_for_exceeders": median_lost,
    }


# ---------------------------------------------------------------------------
# 4. De-identification marker density
# ---------------------------------------------------------------------------


def deid_marker_stats(notes: pd.DataFrame, sample_size: int | None = 5000) -> dict[str, float]:
    """Density of ``___`` de-id markers per note (sample-based if large).

    Args:
        notes: Must contain ``text`` column.
        sample_size: If set and len(notes) > sample_size, sample for speed.

    Returns:
        Dict with marker counts and ratio metrics.
    """
    if sample_size is not None and len(notes) > sample_size:
        sample = notes.sample(n=sample_size, random_state=0)
    else:
        sample = notes

    text = sample["text"].fillna("")
    marker_counts = text.map(lambda t: len(_DEID_MARKER.findall(t)))
    char_counts = text.str.len()
    token_counts = text.map(token_length)

    total_markers = int(marker_counts.sum())
    total_chars = int(char_counts.sum())
    total_tokens = int(token_counts.sum())

    return {
        "sample_size": len(sample),
        "total_markers": total_markers,
        "mean_markers_per_note": float(marker_counts.mean()),
        "median_markers_per_note": float(marker_counts.median()),
        "markers_per_1000_tokens": round(total_markers / max(total_tokens / 1000, 1), 3),
        "markers_per_10000_chars": round(total_markers / max(total_chars / 10000, 1), 3),
    }


# ---------------------------------------------------------------------------
# 5. Duplicate analysis
# ---------------------------------------------------------------------------


def note_duplication_summary(notes: pd.DataFrame) -> dict[str, int | float]:
    """Dedup surface area beyond ``hadm_id``.

    Args:
        notes: Must contain ``note_id``, ``hadm_id``.

    Returns:
        Dict with counts of duplicate ``note_id``, admissions with >1 note,
        and max note_seq.
    """
    return {
        "n_notes": len(notes),
        "unique_note_id": int(notes["note_id"].nunique()),
        "duplicate_note_id": int(len(notes) - notes["note_id"].nunique()),
        "unique_hadm_id": int(notes["hadm_id"].nunique()),
        "admissions_with_multiple_notes": int((notes.groupby("hadm_id").size() > 1).sum()),
        "max_notes_per_hadm": int(notes.groupby("hadm_id").size().max()),
        "max_note_seq": int(notes["note_seq"].max()) if "note_seq" in notes.columns else -1,
    }


# ---------------------------------------------------------------------------
# 6. ICD version split
# ---------------------------------------------------------------------------


def icd_version_distribution(
    diagnoses: pd.DataFrame, admissions: pd.DataFrame | None = None
) -> pd.DataFrame:
    """ICD-9 vs ICD-10 counts, optionally broken out by admission year.

    Args:
        diagnoses: Must contain ``hadm_id``, ``icd_version``.
        admissions: Optional — if provided, join to get admission year.

    Returns:
        If ``admissions`` is None: DataFrame ``[icd_version, n_codes, n_admissions]``.
        Else: DataFrame ``[year, icd_version, n_codes]``.
    """
    if admissions is None:
        by_version = diagnoses.groupby("icd_version").agg(
            n_codes=("icd_code", "size"), n_admissions=("hadm_id", "nunique")
        )
        return by_version.reset_index()

    adm = admissions[["hadm_id", "admittime"]].copy()
    adm["admittime"] = pd.to_datetime(adm["admittime"], errors="coerce")
    adm["year"] = adm["admittime"].dt.year
    merged = diagnoses.merge(adm[["hadm_id", "year"]], on="hadm_id", how="left")
    out = (
        merged.groupby(["year", "icd_version"])
        .size()
        .reset_index(name="n_codes")
        .sort_values(["year", "icd_version"])
        .reset_index(drop=True)
    )
    return out


# ---------------------------------------------------------------------------
# 7. ICD frequency and top-K coverage
# ---------------------------------------------------------------------------


def format_icd10_code(code: str) -> str:
    """Insert the standard ICD-10-CM period into a raw MIMIC code.

    MIMIC-IV stores ICD-10 codes without a separator (e.g. ``E785``).
    Published and clinical notation uses a period after the three-character
    category (e.g. ``E78.5``). This helper converts stored format to
    display format for reports, data cards, and serving output. Codes of
    length three or less (e.g. ``I10``) are returned unchanged.

    Args:
        code: Raw ICD-10 code as stored in MIMIC (no period).

    Returns:
        Code with period inserted after position three, or unchanged for
        short codes.
    """
    return f"{code[:3]}.{code[3:]}" if len(code) > 3 else code


def icd_frequency(diagnoses: pd.DataFrame, version: int = 10) -> pd.DataFrame:
    """Frequency of each ICD code at a given version.

    Returns:
        DataFrame ``[icd_code, n_codes, n_admissions, rank]`` sorted desc by count.
    """
    sub = diagnoses.loc[diagnoses["icd_version"] == version].copy()
    sub["icd_code"] = sub["icd_code"].astype(str).str.strip().str.upper()
    agg = (
        sub.groupby("icd_code")
        .agg(n_codes=("icd_code", "size"), n_admissions=("hadm_id", "nunique"))
        .reset_index()
        .sort_values("n_codes", ascending=False)
        .reset_index(drop=True)
    )
    agg["rank"] = np.arange(1, len(agg) + 1)
    return agg


def top_k_coverage(
    diagnoses: pd.DataFrame,
    k_list: tuple[int, ...] = (10, 25, 50, 100, 250, 500, 1000),
    version: int = 10,
) -> pd.DataFrame:
    """Fraction of admissions with at least one code in the top-K.

    This is THE key chart for choosing ``top_k_labels``.

    Returns:
        DataFrame ``[k, n_admissions_covered, pct_admissions_covered]``.
    """
    sub = diagnoses.loc[diagnoses["icd_version"] == version].copy()
    sub["icd_code"] = sub["icd_code"].astype(str).str.strip().str.upper()
    code_counts = sub["icd_code"].value_counts()
    total_admissions = sub["hadm_id"].nunique()

    rows: list[dict[str, Any]] = []
    for k in k_list:
        top_codes = set(code_counts.head(k).index)
        covered_hadm = sub.loc[sub["icd_code"].isin(top_codes), "hadm_id"].nunique()
        rows.append(
            {
                "k": k,
                "n_admissions_covered": int(covered_hadm),
                "pct_admissions_covered": round(covered_hadm / max(total_admissions, 1) * 100, 2),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 8. Codes per admission
# ---------------------------------------------------------------------------


def codes_per_admission(diagnoses: pd.DataFrame, version: int = 10) -> pd.DataFrame:
    """How many distinct codes does each admission get?

    Returns:
        DataFrame ``[hadm_id, n_codes]`` — one row per admission at this version.
    """
    sub = diagnoses.loc[diagnoses["icd_version"] == version]
    counts = sub.groupby("hadm_id")["icd_code"].nunique().reset_index(name="n_codes")
    return counts


def codes_per_admission_stats(per_admission: pd.DataFrame) -> dict[str, float]:
    """Summary stats on codes-per-admission distribution."""
    s = per_admission["n_codes"]
    return {
        "min": int(s.min()),
        "p25": float(s.quantile(0.25)),
        "median": float(s.median()),
        "mean": float(s.mean()),
        "p75": float(s.quantile(0.75)),
        "p95": float(s.quantile(0.95)),
        "max": int(s.max()),
        "n_admissions": len(s),
    }


# ---------------------------------------------------------------------------
# 9. Label co-occurrence (top-K × top-K)
# ---------------------------------------------------------------------------


def label_cooccurrence(diagnoses: pd.DataFrame, top_k: int = 20, version: int = 10) -> pd.DataFrame:
    """Pairwise co-occurrence count among the top-K ICD codes.

    Args:
        diagnoses: Diagnoses table.
        top_k: How many top codes to include (matrix will be ``top_k × top_k``).
        version: ICD version.

    Returns:
        DataFrame of shape (top_k, top_k) with code labels as index & columns.
        Diagonal is the marginal count; off-diagonal is co-occurrence count.
    """
    sub = diagnoses.loc[diagnoses["icd_version"] == version].copy()
    sub["icd_code"] = sub["icd_code"].astype(str).str.strip().str.upper()
    code_counts = sub["icd_code"].value_counts()
    top = list(code_counts.head(top_k).index)
    top_set = set(top)

    # Build per-admission code sets restricted to top-K
    sub_top = sub.loc[sub["icd_code"].isin(top_set)]
    per_adm = sub_top.groupby("hadm_id")["icd_code"].apply(set)

    idx = {c: i for i, c in enumerate(top)}
    mat = np.zeros((top_k, top_k), dtype=np.int64)
    for codes in per_adm:
        code_list = list(codes)
        for i, a in enumerate(code_list):
            ai = idx[a]
            mat[ai, ai] += 1
            for b in code_list[i + 1 :]:
                bi = idx[b]
                mat[ai, bi] += 1
                mat[bi, ai] += 1
    return pd.DataFrame(mat, index=top, columns=top)


# ---------------------------------------------------------------------------
# 10. Patient & admission-level
# ---------------------------------------------------------------------------


def patient_demographics(patients: pd.DataFrame) -> pd.DataFrame:
    """Gender and age-bucket distribution.

    Args:
        patients: Must contain ``gender`` and ``anchor_age``.

    Returns:
        DataFrame ``[dimension, bucket, n, pct]`` with rows for gender and age buckets.
    """
    rows: list[dict[str, Any]] = []
    if "gender" in patients.columns:
        g = patients["gender"].value_counts(dropna=False)
        total = max(g.sum(), 1)
        for k, v in g.items():
            rows.append(
                {
                    "dimension": "gender",
                    "bucket": str(k),
                    "n": int(v),
                    "pct": round(v / total * 100, 2),
                }
            )

    if "anchor_age" in patients.columns:
        bins = [0, 18, 30, 45, 55, 65, 75, 85, 200]
        labels = ["<18", "18-29", "30-44", "45-54", "55-64", "65-74", "75-84", "85+"]
        age = pd.cut(patients["anchor_age"], bins=bins, labels=labels, right=False)
        a = age.value_counts(dropna=False).reindex(labels)
        total = max(int(a.sum()), 1)
        for k, v in a.items():
            v_int = int(v) if not pd.isna(v) else 0
            rows.append(
                {
                    "dimension": "age_bucket",
                    "bucket": str(k),
                    "n": v_int,
                    "pct": round(v_int / total * 100, 2),
                }
            )
    return pd.DataFrame(rows)


def admissions_per_patient(admissions: pd.DataFrame) -> pd.DataFrame:
    """Distribution of admissions per patient.

    Returns:
        DataFrame ``[subject_id, n_admissions]`` — one row per patient.
    """
    return admissions.groupby("subject_id").size().reset_index(name="n_admissions")


def admissions_per_patient_stats(per_patient: pd.DataFrame) -> dict[str, float]:
    s = per_patient["n_admissions"]
    return {
        "n_patients": int(len(s)),
        "min": int(s.min()),
        "median": float(s.median()),
        "mean": float(s.mean()),
        "p95": float(s.quantile(0.95)),
        "max": int(s.max()),
        "patients_with_single_admission": int((s == 1).sum()),
        "patients_with_5plus_admissions": int((s >= 5).sum()),
    }


def length_of_stay(admissions: pd.DataFrame) -> pd.Series:
    """Length of stay in days (dischtime - admittime).

    Returns:
        Series of LOS in days, aligned with admissions rows (may contain NaN).
    """
    admit = pd.to_datetime(admissions["admittime"], errors="coerce")
    disch = pd.to_datetime(admissions["dischtime"], errors="coerce")
    return (disch - admit).dt.total_seconds() / 86400.0


def mortality_rate(admissions: pd.DataFrame) -> dict[str, Any]:
    """In-hospital mortality rate from ``hospital_expire_flag``.

    Returns:
        Dict with ``n_admissions``, ``n_deaths``, ``rate``.
    """
    if "hospital_expire_flag" not in admissions.columns:
        return {"n_admissions": len(admissions), "n_deaths": None, "rate": None}
    flag = pd.to_numeric(admissions["hospital_expire_flag"], errors="coerce").fillna(0)
    n = len(admissions)
    deaths = int((flag == 1).sum())
    return {"n_admissions": n, "n_deaths": deaths, "rate": round(deaths / max(n, 1), 4)}


# ---------------------------------------------------------------------------
# 11. Join coverage (notes ∩ admissions ∩ diagnoses)
# ---------------------------------------------------------------------------


def join_coverage(
    notes: pd.DataFrame, admissions: pd.DataFrame, diagnoses: pd.DataFrame, version: int = 10
) -> pd.DataFrame:
    """Venn-style coverage of ``hadm_id`` across the three tables.

    Args:
        notes, admissions, diagnoses: Bronze tables.
        version: Filter diagnoses to this ICD version before counting.

    Returns:
        DataFrame with one row per slice (A∪B∪C regions) and counts.
    """
    note_h = set(notes["hadm_id"].dropna().astype(int))
    adm_h = set(admissions["hadm_id"].dropna().astype(int))
    dx = diagnoses.loc[diagnoses["icd_version"] == version]
    dx_h = set(dx["hadm_id"].dropna().astype(int))

    rows = [
        {"slice": "notes_only", "count": len(note_h - adm_h - dx_h)},
        {"slice": "adm_only", "count": len(adm_h - note_h - dx_h)},
        {"slice": "dx_only", "count": len(dx_h - note_h - adm_h)},
        {"slice": "notes_and_adm", "count": len((note_h & adm_h) - dx_h)},
        {"slice": "notes_and_dx", "count": len((note_h & dx_h) - adm_h)},
        {"slice": "adm_and_dx", "count": len((adm_h & dx_h) - note_h)},
        {"slice": "all_three", "count": len(note_h & adm_h & dx_h)},
        {"slice": "total_notes", "count": len(note_h)},
        {"slice": "total_adm", "count": len(adm_h)},
        {"slice": f"total_dx_v{version}", "count": len(dx_h)},
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 12. Version reconciliation (v2.2 notes vs. v3.1 Hosp)
# ---------------------------------------------------------------------------


def version_reconciliation(notes: pd.DataFrame, admissions: pd.DataFrame) -> dict[str, float | int]:
    """Quantify **pure version drift** between v2.2 Notes and v3.1 Hosp admissions.

    Measures only the admissions that exist in MIMIC-IV-Note v2.2 but have been
    removed from / never existed in MIMIC-IV v3.1 Hosp. Does NOT measure
    cohort-filter effects (e.g. loss from filtering to ICD-10 only) — for that,
    use :func:`cohort_coverage`, which separates drift from filter loss.

    Args:
        notes: Notes DataFrame with ``hadm_id``.
        admissions: **Full, unfiltered** admissions table with ``hadm_id``.
            Pre-filtering this table will conflate version drift with the
            filter's effect and produce a misleading drift rate.

    Returns:
        Dict with ``n_notes``, ``n_admissions``, ``n_intersection``,
        ``n_notes_missing_from_hosp``, ``pct_notes_missing_from_hosp``.
    """
    note_h = set(notes["hadm_id"].dropna().astype(int))
    adm_h = set(admissions["hadm_id"].dropna().astype(int))
    intersection = note_h & adm_h
    missing = note_h - adm_h
    return {
        "n_notes": len(note_h),
        "n_admissions": len(adm_h),
        "n_intersection": len(intersection),
        "n_notes_missing_from_hosp": len(missing),
        "pct_notes_missing_from_hosp": round(len(missing) / max(len(note_h), 1) * 100, 3),
    }


def cohort_coverage(
    notes: pd.DataFrame,
    admissions: pd.DataFrame,
    cohort: pd.DataFrame,
) -> dict[str, float | int]:
    """Decompose note-level coverage loss into **drift** vs **cohort filter**.

    When joining v2.2 notes to a cohort-filtered slice of v3.1 Hosp (e.g.
    "admissions with ≥1 ICD-10 diagnosis"), two losses are conflated:

    1. **Drift loss** — notes whose admission is absent from v3.1 Hosp at all
       (true version mismatch).
    2. **Cohort-filter loss** — notes whose admission IS in v3.1 Hosp but has
       been excluded by the cohort filter (e.g. ICD-9 admission).

    This function returns each loss separately so callers can distinguish
    data-quality issues from deliberate cohort scoping.

    Invariant: ``n_retained + n_drift_loss + n_cohort_filter_loss == n_notes``.

    Args:
        notes: Notes DataFrame with ``hadm_id``.
        admissions: **Full, unfiltered** admissions table (drift baseline).
        cohort: Any DataFrame with a ``hadm_id`` column representing the
            post-filter cohort (filtered admissions, filtered diagnoses, etc.).

    Returns:
        Dict with ``n_notes``, ``n_admissions_full``, ``n_admissions_cohort``,
        ``n_retained``, ``n_drift_loss``, ``n_cohort_filter_loss``, and the
        matching ``pct_*`` fields (rounded to 3 decimals).
    """
    note_h = set(notes["hadm_id"].dropna().astype(int))
    adm_h = set(admissions["hadm_id"].dropna().astype(int))
    cohort_h = set(cohort["hadm_id"].dropna().astype(int))

    n_notes = len(note_h)
    retained = note_h & cohort_h
    drift_loss = note_h - adm_h
    cohort_filter_loss = (note_h & adm_h) - cohort_h

    def _pct(n: int) -> float:
        return round(n / max(n_notes, 1) * 100, 3)

    return {
        "n_notes": n_notes,
        "n_admissions_full": len(adm_h),
        "n_admissions_cohort": len(cohort_h),
        "n_retained": len(retained),
        "n_drift_loss": len(drift_loss),
        "n_cohort_filter_loss": len(cohort_filter_loss),
        "pct_retained": _pct(len(retained)),
        "pct_drift_loss": _pct(len(drift_loss)),
        "pct_cohort_filter_loss": _pct(len(cohort_filter_loss)),
    }


# ---------------------------------------------------------------------------
# Visualization helpers (matplotlib Axes in/out; callers manage figure lifecycle)
# ---------------------------------------------------------------------------


def plot_length_distribution(
    notes_with_lengths: pd.DataFrame, ax: Any, column: str = "n_tokens", log_y: bool = True
) -> Any:
    """Histogram of note lengths with percentile markers.

    Args:
        notes_with_lengths: Output of ``compute_lengths``.
        ax: matplotlib Axes to draw on.
        column: ``n_tokens`` or ``n_chars``.
        log_y: Log-scale the y-axis.

    Returns:
        The same ``ax`` for chaining.
    """
    s = notes_with_lengths[column]
    ax.hist(s, bins=80, alpha=0.8, color="#4682B4")
    for q, color in zip((0.5, 0.95, 0.99), ("#2ca02c", "#ff7f0e", "#d62728"), strict=False):
        v = float(s.quantile(q))
        ax.axvline(v, linestyle="--", color=color, label=f"p{int(q * 100)} = {v:.0f}")
    if column == "n_tokens":
        ax.axvline(512, linestyle=":", color="#111", label="BERT cap (512)")
        ax.axvline(4096, linestyle=":", color="#555", label="Longformer cap (4096)")
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel(column)
    ax.set_ylabel("count")
    ax.set_title(f"Distribution of {column}")
    ax.legend()
    return ax


def plot_icd_frequency_curve(icd_freq: pd.DataFrame, ax: Any, log_y: bool = True) -> Any:
    """Rank-frequency plot for ICD codes (Zipf-style).

    Args:
        icd_freq: Output of ``icd_frequency``.
        ax: matplotlib Axes.
        log_y: Log y.

    Returns:
        ``ax``.
    """
    ax.plot(icd_freq["rank"], icd_freq["n_codes"], linewidth=1.2, color="#4682B4")
    ax.set_xlabel("Rank")
    ax.set_ylabel("# codes assigned")
    if log_y:
        ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_title("ICD-10 code frequency (rank-order, log-log)")
    return ax


def plot_top_k_coverage(coverage: pd.DataFrame, ax: Any) -> Any:
    """Coverage curve: % admissions covered as K grows.

    Returns:
        ``ax``.
    """
    ax.plot(coverage["k"], coverage["pct_admissions_covered"], marker="o", color="#4682B4")
    ax.set_xlabel("Top-K codes retained")
    ax.set_ylabel("% admissions with ≥ 1 top-K code")
    ax.set_xscale("log")
    ax.set_ylim(0, 105)
    ax.set_title("Cohort coverage vs. label-space size")
    ax.grid(alpha=0.3)
    return ax


def plot_codes_per_admission(per_admission: pd.DataFrame, ax: Any, max_bin: int = 40) -> Any:
    """Histogram of codes per admission, capped to ``max_bin`` on the x-axis."""
    capped = per_admission["n_codes"].clip(upper=max_bin)
    ax.hist(capped, bins=np.arange(0, max_bin + 2) - 0.5, color="#4682B4", alpha=0.8)
    ax.set_xlabel(f"Codes per admission (capped at {max_bin})")
    ax.set_ylabel("Admissions")
    ax.set_title("Distribution: codes per admission")
    return ax


def plot_cooccurrence_heatmap(cooc: pd.DataFrame, ax: Any, log_color: bool = True) -> Any:
    """Heatmap of top-K label co-occurrence.

    Args:
        cooc: Output of ``label_cooccurrence``.
        ax: matplotlib Axes.
        log_color: Apply log1p to the matrix before plotting.

    Returns:
        ``ax``.
    """
    data = np.log1p(cooc.to_numpy()) if log_color else cooc.to_numpy()
    im = ax.imshow(data, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(cooc.columns)))
    ax.set_yticks(range(len(cooc.index)))
    ax.set_xticklabels(cooc.columns, rotation=90, fontsize=8)
    ax.set_yticklabels(cooc.index, fontsize=8)
    ax.set_title("Top-K label co-occurrence" + (" (log1p)" if log_color else ""))
    # ask caller to add colorbar if needed
    ax.figure.colorbar(im, ax=ax, shrink=0.8)
    return ax
