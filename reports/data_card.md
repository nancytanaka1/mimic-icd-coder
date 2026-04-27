# Data Card — MIMIC-IV ICD Coding Dataset

## Identity

- **Name:** MIMIC-IV top-50 ICD-10 auto-coding dataset
- **Version:** 0.1.0
- **Sources:** MIMIC-IV-Note v2.2 (notes, Jan 2023) + MIMIC-IV v3.1 Hosp (admissions, diagnoses, patients; Oct 2024)
- **License:** PhysioNet Credentialed Health Data License v1.5.0. Not redistributable.
- **Steward:** MIT Laboratory for Computational Physiology
- **Card format:** Pushkarna et al. 2022

## Version Mismatch

Notes come from v2.2 because it is the only public release. Structured tables come from v3.1 because it fixes defects in `diagnoses_icd`. The `hadm_id` join key is stable across versions. Only 61 of 331,793 notes (0.018%) are orphaned by the mismatch. All joins use inner join on `hadm_id`.

## Composition

Cohort construction is a four-stage filter. Final modelable cohort: **122,288 admissions**.

| Stage | N after | Loss |
|---|---|---|
| Raw discharge notes (v2.2) | 331,793 | — |
| Inner join with v3.1 admissions | 331,732 | 61 (version drift) |
| Restrict to ICD-10 admissions | 122,288 | 209,444 (ICD-9 admissions, by design) |
| Minimum 100 whitespace tokens | 122,283 | 5 (notes shorter than 100 whitespace tokens) |

Top-50 ICD-10 codes cover 94.12% of cohort admissions. Long tail: 19,440 distinct codes.

## Data Dictionary

### Silver (`catalog.silver.discharge_notes`)

| Column | Type | Description |
|---|---|---|
| `note_id` | string | Primary key |
| `subject_id` | bigint | Patient ID |
| `hadm_id` | bigint | Admission ID; join key to Hosp |
| `text` | string | Cleaned discharge summary |
| `n_tokens` | int | Whitespace token count |

### Gold (`catalog.gold.labels_top50`)

Sparse multi-hot matrix over top-50 ICD-10 codes, one row per `hadm_id`. Stored as a Delta table with one row per `hadm_id` × `icd_code` and a `present` flag.

## Collection

Beth Israel Deaconess Medical Center EHR capture, 2008–2019. De-identified by MIT-LCP per HIPAA Safe Harbor (text redaction + per-patient date shift; `anchor_age` capped at 89).

## Preprocessing

- Filter notes to `note_type = DS`.
- Replace `___` redaction markers with `[REDACTED]`.
- Normalize whitespace.
- Drop notes below 100 whitespace tokens.
- Deduplicate by `hadm_id` (keep highest `note_seq`) — no-op in v2.2.

## Restrictions

- PhysioNet credential required for raw data access.
- No redistribution of raw notes, derived text, or labels.
- No transmission of clinical text to third-party LLM APIs.
- Repository code is Apache 2.0; data is not.

## Known Limitations

- **Single institution.** All data from BIDMC. No external validation.
- **Billing codes are not ground truth.** ICD-10 codes are coder judgment for billing, with documented comorbidity underreporting.
- **De-identification artifacts.** Date-shifted timelines; age capped at 89; names, dates, and some numbers redacted. Temporal analysis requires real-year recovery (deferred to Silver).
- **Class imbalance.** The rarest top-50 code may fall below 1% prevalence.
- **Cohort ceiling.** The modelable cohort is 48% of all ICD-10 admissions in v3.1 Hosp — limited by MIMIC-IV-Note v2.2 sampling, not by project filters. A future note release could nearly double cohort N without code changes.
- **ICD-10 sub-code fragmentation.** Common conditions (CHF, CKD, COPD, T2DM) spread across multiple sub-codes. Top-50 on MIMIC-IV is not equivalent to top-50 on MIMIC-III despite the same K.
- **Demographic coverage.** Race, insurance, language, and discharge location are ingested into Bronze `admissions` (per `feat/ingest-demographic`). These columns have not yet flowed through to the Silver/Gold cohort join used by the modeling stage, so race-stratified fairness metrics are not yet computed — that slicing pass is deferred to a future branch. Ethnicity is not separately represented: MIMIC-IV's `race` field conflates race and ethnicity, and a disaggregated ethnicity column would require further work. Race is held strictly for future evaluation-time cohort stratification per the model card's deferred analyses; it is **not** used as a prediction feature.
