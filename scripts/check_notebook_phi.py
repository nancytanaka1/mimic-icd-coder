"""Enforce MIMIC-IV DUA compliance on committed notebooks.

Scans notebook **source** (code, markdown, raw cells) AND code-cell **outputs**
for indicators of PHI leakage, raw clinical text, MIMIC redaction markers,
hardcoded raw-data paths, or dangerous pandas display settings. Exits
non-zero on any finding.

Compliance axes enforced (PhysioNet Credentialed Health Data License v1.5.0):

  1. **Clinical text** — discharge-summary section headers and clinical prose.
  2. **PHI markers** — MRN, SSN, DOB, phone numbers, "X year-old" phrases.
  3. **MIMIC redaction leaks** — underscore runs, `[**bracketed**]` redactions.
  4. **Raw-data paths** — hardcoded MIMIC filesystem paths in source.
  5. **Dangerous display settings** — pandas options that render full text columns.
  6. **Long clinical-looking prose** — heuristic for bleed-through of note bodies.

Usage:
    python scripts/check_notebook_phi.py
    python scripts/check_notebook_phi.py --scope . --verbose

Exit codes:
    0 — all scanned notebooks clean
    1 — at least one violation found
    2 — scanner invocation error (bad path, malformed notebook, etc.)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class Severity(Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"


@dataclass(frozen=True)
class Rule:
    """A single compliance rule."""

    category: str
    severity: Severity
    pattern: re.Pattern[str]
    scans: tuple[str, ...]  # subset of {"source", "output"}
    applies_to: tuple[str, ...]  # subset of {"code", "markdown", "raw"}
    description: str


# ---------------------------------------------------------------------------
# Rule set — ordered by category, each with severity and scan scope.
# ---------------------------------------------------------------------------

# Clinical-text section headers (discharge summaries, H&Ps).
_CLINICAL_HEADERS = [
    r"Admission\s+Date\s*:",
    r"Discharge\s+Date\s*:",
    r"Chief\s+Complaint\s*:",
    r"History\s+of\s+Present\s+Illness",
    r"Past\s+Medical\s+History",
    r"Hospital\s+Course\s*:",
    r"Discharge\s+Diagnosis",
    r"Discharge\s+Medications",
    r"Physical\s+Exam\s*:",
    r"Allergies?\s*:",
    r"Service\s*:\s*MEDICINE",
    r"Service\s*:\s*SURGERY",
    r"Review\s+of\s+Systems",
    r"Major\s+Surgical\s+or\s+Invasive\s+Procedure",
    r"Brief\s+Hospital\s+Course",
    r"Pertinent\s+Results\s*:",
    r"Facility\s*:",
]

# Direct PHI markers.
_PHI_MARKERS = [
    r"\bMRN\s*[:#]",
    r"\bSSN\s*[:#]?",
    r"\bDOB\s*[:#]?",
    r"Patient\s+is\s+a\s+\d+",
    r"\b\d+\s*(?:yo|y\.?o\.?|year[- ]old)\b",
    r"\(\d{3}\)\s*\d{3}-\d{4}",  # phone
    r"\b\d{3}-\d{3}-\d{4}\b",  # phone, alternate
]

# MIMIC-specific de-identification artifacts leaking into outputs.
_MIMIC_REDACTION = [
    r"_{4,}",  # four or more underscores — likely a de-id span
    r"\[\*\*[^\]]{1,120}\*\*\]",  # MIMIC-III-style bracketed redaction
    r"\[\*\*Hospital\s+\d+\*\*\]",
    r"\[\*\*Known\s+(?:first|last)name[^\]]*\*\*\]",
]

# Hardcoded raw-data paths — source files (or markdown) that reveal
# MIMIC raw-file locations on the author's filesystem.
_RAW_DATA_PATHS = [
    r"[/\\]data[/\\]physionet",
    r"physionet[/\\]mimic",
    r"mimic-iv-note[/\\]note[/\\]",
    r"mimic-iv[/\\]hosp[/\\]",
    r"[A-Za-z]:[/\\]data[/\\]physionet",
    r"physionet\.org[/\\]files[/\\]mimic",
]

# Pandas display settings that could render the full `text` column.
_DANGEROUS_DISPLAY = [
    r"display\.max_colwidth[\"']?\s*,\s*(?:None|-?\s*1\b)",
    r"set_option\(\s*[\"']display\.max_colwidth[\"']\s*,\s*(?:None|-?\s*1\b)",
    r"set_option\(\s*[\"']display\.max_rows[\"']\s*,\s*None",  # + max_colwidth is scary, but this is lone-useful
]


def _compile(patterns: list[str]) -> list[re.Pattern[str]]:
    return [re.compile(p, re.IGNORECASE) for p in patterns]


RULES: tuple[Rule, ...] = (
    *(
        Rule(
            category="clinical_text",
            severity=Severity.CRITICAL,
            pattern=p,
            scans=("source", "output"),
            applies_to=("code", "markdown", "raw"),
            description="Discharge-summary section header — indicates raw clinical text.",
        )
        for p in _compile(_CLINICAL_HEADERS)
    ),
    *(
        Rule(
            category="phi_marker",
            severity=Severity.CRITICAL,
            pattern=p,
            scans=("source", "output"),
            applies_to=("code", "markdown", "raw"),
            description="Direct PHI marker (MRN / SSN / DOB / age / phone).",
        )
        for p in _compile(_PHI_MARKERS)
    ),
    *(
        Rule(
            category="mimic_redaction_leak",
            severity=Severity.HIGH,
            pattern=p,
            scans=("output",),
            applies_to=("code",),
            description="MIMIC de-identification marker in output — raw text bleeding through Silver cleaning.",
        )
        for p in _compile(_MIMIC_REDACTION)
    ),
    *(
        Rule(
            category="raw_data_path",
            severity=Severity.HIGH,
            pattern=p,
            scans=("source",),
            applies_to=("code", "markdown", "raw"),
            description="Hardcoded raw MIMIC data path — use gitignored config instead.",
        )
        for p in _compile(_RAW_DATA_PATHS)
    ),
    *(
        Rule(
            category="dangerous_display_setting",
            severity=Severity.MEDIUM,
            pattern=p,
            scans=("source",),
            applies_to=("code",),
            description="Pandas display option may render full `text` column — high leak risk if paired with a notes.head() style output.",
        )
        for p in _compile(_DANGEROUS_DISPLAY)
    ),
)


# ---------------------------------------------------------------------------
# Long-prose heuristic: catch raw clinical narrative that doesn't match a
# specific pattern but looks like a note body bleeding into outputs.
# ---------------------------------------------------------------------------

# Clinical vocabulary used for prose detection. Biased toward verbs and
# narrative constructions that appear in note *bodies*, not toward nouns
# that double as table/column names (admissions, diagnoses, etc.).
_CLINICAL_VOCAB = re.compile(
    r"\b(?:admitted|prescribed|presented|reported|complained|denied|"
    r"tolerated|discharged\s+home|transferred|followed[- ]up|"
    r"status\s+post|s\s*/\s*p\b|chief\s+complaint|brief\s+hospital\s+course|"
    r"past\s+medical\s+history|history\s+of\s+present|review\s+of\s+systems|"
    r"physical\s+exam|pertinent\s+results|social\s+history|family\s+history|"
    r"medications?\s+on\s+admission|medications?\s+on\s+discharge)\b",
    re.IGNORECASE,
)

MIN_PROSE_CHARS = 1500
MIN_CLINICAL_VOCAB_HITS = 5


def _looks_like_clinical_prose(text: str) -> tuple[bool, int, str]:
    """Heuristic: does this output look like a raw clinical note body?

    Conservative — biased to miss rather than false-alarm on legitimate
    aggregate outputs (DataFrame reprs, ICD description tables, schema
    summaries). Returns ``(is_prose, vocab_hits, reason)``.
    """
    if len(text) < MIN_PROSE_CHARS:
        return False, 0, "below length floor"

    hits = len(_CLINICAL_VOCAB.findall(text))
    if hits < MIN_CLINICAL_VOCAB_HITS:
        return False, hits, "below vocab floor"

    # Reject tabular output: pandas DataFrame reprs and ICD description
    # tables have very short average line lengths and many runs of
    # consecutive spaces used for column alignment.
    nonblank_lines = [ln for ln in text.split("\n") if ln.strip()]
    if nonblank_lines:
        avg_words_per_line = sum(len(ln.split()) for ln in nonblank_lines) / len(nonblank_lines)
        if avg_words_per_line < 12:
            return False, hits, f"tabular output (avg {avg_words_per_line:.1f} words/line)"

    # Reject alignment-padded output (tables use many "   " runs).
    triple_space_density = len(re.findall(r"   +", text)) / max(len(text), 1) * 1000
    if triple_space_density > 2:
        return (
            False,
            hits,
            f"column-alignment padding (density {triple_space_density:.2f}/1000 chars)",
        )

    # Require sentence structure — clinical prose ends sentences and starts
    # new ones. Tables and schema dumps don't.
    sentence_breaks = len(re.findall(r"[.!?]\s+[A-Z]", text))
    if sentence_breaks < 4:
        return False, hits, f"insufficient sentence structure ({sentence_breaks} breaks)"

    return True, hits, "matches clinical-prose heuristic"


# ---------------------------------------------------------------------------
# Redaction for reporting — never echo matched PHI to stderr. Show bounded
# context around each match with the matched content replaced by [REDACTED].
# ---------------------------------------------------------------------------

_CONTEXT_WINDOW = 40  # chars before and after the match


def _redacted_context(text: str, match: re.Match[str]) -> str:
    """Return ±40 chars of context with the match itself replaced by `[REDACTED]`."""
    start, end = match.span()
    before = text[max(0, start - _CONTEXT_WINDOW) : start]
    after = text[end : end + _CONTEXT_WINDOW]
    snippet = f"...{before}[REDACTED]{after}...".replace("\n", " ")
    return re.sub(r"\s+", " ", snippet).strip()


# ---------------------------------------------------------------------------
# Cell-content extraction
# ---------------------------------------------------------------------------


def _cell_source(cell: dict) -> str:
    src = cell.get("source", "")
    return "".join(src) if isinstance(src, list) else str(src)


def _output_text(output: dict) -> str:
    parts: list[str] = []
    data = output.get("data", {})
    for mime in ("text/plain", "text/html"):
        v = data.get(mime, "")
        parts.append("".join(v) if isinstance(v, list) else str(v))
    stream = output.get("text", "")
    parts.append("".join(stream) if isinstance(stream, list) else str(stream))
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Finding record
# ---------------------------------------------------------------------------


@dataclass
class Finding:
    notebook: Path
    cell_index: int
    location: str  # e.g., "source" or "output[0]"
    cell_type: str
    category: str
    severity: Severity
    pattern: str
    context: str

    def format(self) -> str:
        return (
            f"  [{self.severity.value}] cell[{self.cell_index}] "
            f"({self.cell_type}, {self.location}) — {self.category}\n"
            f"    pattern: /{self.pattern}/\n"
            f"    context: {self.context}"
        )


# ---------------------------------------------------------------------------
# Scanning
# ---------------------------------------------------------------------------


def _apply_rules(
    rules: tuple[Rule, ...],
    scope: str,
    cell_type: str,
    text: str,
) -> list[tuple[Rule, re.Match[str]]]:
    matches: list[tuple[Rule, re.Match[str]]] = []
    for rule in rules:
        if scope not in rule.scans:
            continue
        if cell_type not in rule.applies_to:
            continue
        m = rule.pattern.search(text)
        if m:
            matches.append((rule, m))
    return matches


def scan_notebook(path: Path) -> list[Finding]:
    try:
        nb = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"cannot parse notebook {path}: {exc}", file=sys.stderr)
        sys.exit(2)

    findings: list[Finding] = []

    for cell_idx, cell in enumerate(nb.get("cells", [])):
        cell_type = cell.get("cell_type", "")
        src = _cell_source(cell)

        # Source-level scan (all cell types).
        for rule, match in _apply_rules(RULES, "source", cell_type, src):
            findings.append(
                Finding(
                    notebook=path,
                    cell_index=cell_idx,
                    location="source",
                    cell_type=cell_type,
                    category=rule.category,
                    severity=rule.severity,
                    pattern=rule.pattern.pattern,
                    context=_redacted_context(src, match),
                )
            )

        # Output-level scan (code cells only).
        if cell_type != "code":
            continue

        for out_idx, output in enumerate(cell.get("outputs", [])):
            out_text = _output_text(output)
            for rule, match in _apply_rules(RULES, "output", cell_type, out_text):
                findings.append(
                    Finding(
                        notebook=path,
                        cell_index=cell_idx,
                        location=f"output[{out_idx}]",
                        cell_type=cell_type,
                        category=rule.category,
                        severity=rule.severity,
                        pattern=rule.pattern.pattern,
                        context=_redacted_context(out_text, match),
                    )
                )

            # Long-prose heuristic (narrative bleed-through check)
            is_prose, vocab_hits, reason = _looks_like_clinical_prose(out_text)
            if is_prose:
                findings.append(
                    Finding(
                        notebook=path,
                        cell_index=cell_idx,
                        location=f"output[{out_idx}]",
                        cell_type=cell_type,
                        category="long_clinical_prose",
                        severity=Severity.HIGH,
                        pattern=f"len={len(out_text)}, narrative_vocab={vocab_hits}, {reason}",
                        context=(
                            f"{len(out_text)}-char output with {vocab_hits} narrative clinical "
                            "terms and clear sentence structure — likely a note body."
                        ),
                    )
                )

    return findings


def _find_notebooks(scope: Path) -> list[Path]:
    if scope.is_file():
        return [scope] if scope.suffix == ".ipynb" else []
    # Skip common virtualenv / cache directories
    skip = {".venv", "venv", ".ipynb_checkpoints", "node_modules", "__pycache__"}
    found: list[Path] = []
    for p in scope.rglob("*.ipynb"):
        if any(part in skip for part in p.parts):
            continue
        found.append(p)
    return sorted(found)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "paths",
        type=Path,
        nargs="*",
        help=(
            "Notebook files to scan (positional). Overrides --scope. Pre-commit "
            "passes matched files this way."
        ),
    )
    parser.add_argument(
        "--scope",
        type=Path,
        default=Path("."),
        help="Directory or .ipynb file to scan (default: .) when no positional paths given.",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print OK lines for clean notebooks."
    )
    args = parser.parse_args()

    # Positional paths (pre-commit's idiom) take precedence; --scope is the
    # fallback for direct invocation without args.
    if args.paths:
        missing = [p for p in args.paths if not p.exists()]
        if missing:
            print(f"paths not found: {missing}", file=sys.stderr)
            return 2
        notebooks = sorted(p for p in args.paths if p.suffix == ".ipynb")
    else:
        if not args.scope.exists():
            print(f"scope not found: {args.scope}", file=sys.stderr)
            return 2
        notebooks = _find_notebooks(args.scope)
    if not notebooks:
        print(f"no notebooks found under {args.scope}; nothing to scan.")
        return 0

    all_findings: list[Finding] = []
    for nb in notebooks:
        findings = scan_notebook(nb)
        if findings:
            print(f"FAIL: {nb} — {len(findings)} violation(s):", file=sys.stderr)
            for f in sorted(findings, key=lambda x: (x.severity.value, x.cell_index)):
                print(f.format(), file=sys.stderr)
            print("", file=sys.stderr)
            all_findings.extend(findings)
        elif args.verbose:
            print(f"OK:   {nb}")

    if all_findings:
        by_sev: dict[str, int] = {}
        for f in all_findings:
            by_sev[f.severity.value] = by_sev.get(f.severity.value, 0) + 1
        counts = ", ".join(f"{v} {k}" for k, v in sorted(by_sev.items()))

        print(
            f"\n{len(all_findings)} total violation(s): {counts}\n\n"
            "Remediation steps (pick the one that fits):\n"
            "  - Clear the offending output in Jupyter:\n"
            "      Cell → All Output → Clear  (then save)\n"
            "  - Strip outputs via nbstripout:\n"
            "      nbstripout <notebook>\n"
            "  - For source-level hits (raw data paths, dangerous display settings),\n"
            "    edit the cell directly and move the value into a gitignored config\n"
            "    (see configs/dev.example.yml).\n"
            "  - Pre-commit will re-run this check on every commit.\n",
            file=sys.stderr,
        )
        return 1

    if not args.verbose:
        # Default quiet-success output — one line so CI logs stay clean
        print(f"OK: {len(notebooks)} notebook(s) scanned, no violations.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
