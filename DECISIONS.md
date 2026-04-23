# DECISIONS.md

Record every non-obvious architectural or methodological choice.
Format: `[Decision]: [What was chosen] — [Why] — [Alternatives considered]`

---

## 2026-04-20 — Target top-50 ICD-10 codes
- **What:** Predict the 50 most frequent ICD-10 codes in the MIMIC-IV Hosp diagnoses table.
- **Why:** Published benchmark (Mullenbach et al. 2018) used top-50 on MIMIC-III. Direct comparability; keeps the label space tractable for BERT fine-tuning.
- **Alternatives:** All ~18K ICD-10 codes (intractable), top-10 (too easy), disease-category roll-up (less useful clinically).
- **Update 2026-04-22:** Cohort-aware top-50 verified after `mic gold`. `Z20.822` (COVID exposure, post-hoc, insufficient cohort support in MIMIC-IV-Note v2.2 window) dropped; `N18.3`, `J18.9`, `Y92.239`, `Z23` added in its place. Final cohort 122,283 admissions; `data/gold/label_names.json` confirmed.

## 2026-04-20 — Patient-level train/val/test split
- **What:** Partition by `subject_id`, not by `hadm_id`. 80/10/10 stratified by label presence.
- **Why:** Admission-level splits leak patient-specific language patterns (writing style, comorbidity clusters) across train/test and inflate metrics.
- **Alternatives:** Temporal split by admit date. Rejected for this study — patient-level is the methodologically standard choice for clinical NLP benchmarks; temporal split could be added as a secondary evaluation.

## 2026-04-20 — Use MIMIC-IV v3.1 labels joined to MIMIC-IV-Note v2.2 notes
- **What:** Labels from `hosp/diagnoses_icd.csv.gz` (v3.1, Oct 2024); notes from `note/discharge.csv.gz` (v2.2, Jan 2023).
- **Why:** v3.1 is the only version with the corrected `diagnoses_icd` table. MIMIC-IV-Note has only one public release (v2.2). `hadm_id` join key is stable across versions.
- **Alternatives:** Downgrade labels to v2.2 (loses corrections), upgrade notes (no newer version exists).
- **Risk:** Document clearly in `reports/data_card.md`; a small number of admissions in notes v2.2 may have been removed from v3.1 Hosp; handle as inner join.

## 2026-04-20 — ICD-10-only cohort
- **What:** Filter to admissions where all assigned codes are ICD-10 (`icd_version = 10`).
- **Why:** MIMIC-IV spans ICD-9 (pre-2015) and ICD-10 (post-2015). Mixing breaks the label space. ICD-10 is the operationally relevant system today.
- **Alternatives:** Include ICD-9 with code mapping (GEM crosswalk). Adds complexity and mapping loss; reject for now.

## 2026-04-20 — Bio_ClinicalBERT as primary model family; Clinical-Longformer as fallback
- **What:** Primary transformer is Bio_ClinicalBERT. Clinical-Longformer is the fallback, triggered only if Bio_ClinicalBERT misses the Micro F1 target by more than 3 points. This entry scopes the **model family choice** only. The context-handling mechanism (chunk-and-max-pool vs single-pass) is defined in the Chunk-and-max-pool entry below.
- **Why:** Bio_ClinicalBERT is the dominant clinical-pretrained encoder with established benchmarks in clinical NLP (Alsentzer et al. 2019). It is faster per epoch than Longformer and easier to debug. Longformer's 4K context is valuable only if Bio_ClinicalBERT with chunking leaves substantial signal on the floor — an empirical question to answer with data, not a priori.
- **Alternatives:** (1) Start with Longformer — rejected, slower and premature without evidence that Bio_ClinicalBERT underperforms. (2) General-domain BERT or RoBERTa — rejected, domain mismatch on clinical text. (3) Other clinical encoders (PubMedBERT, BioLinkBERT, Clinical-T5) — comparable choices; Bio_ClinicalBERT picked for direct lineage to Alsentzer 2019 and prior MIMIC precedent.

## 2026-04-20 — Chunk-and-max-pool over full notes; Longformer deferred pending empirical result
- **What:** Tokenize each note into contiguous 512-BPE-token chunks, run Bio_ClinicalBERT over all chunks, and max-pool logits across chunks per label. Escalate to Clinical-Longformer only if chunked-BERT misses the Micro F1 target (≥ 0.70) by more than 3 points.
- **Why:** EDA on MIMIC-IV-Note v2.2 discharge summaries (N = 331,793) —
  truncation-impact analysis in whitespace tokens:
  | cap  | % truncated | median tokens lost per truncated note |
  |------|------------:|--------------------------------------:|
  |  512 |      98.74% |                                   998 |
  | 1024 |      81.78% |                                   621 |
  | 2048 |      21.11% |                                   372 |
  | 4096 |       0.51% |                                   455 |
  Exceedance rate collapses between 1K→2K (the "knee"); gains from 2K→4K are marginal. Applying ~1.3× BPE inflation for clinical text: BERT's single-window 512-BPE cap ≈ 400 whitespace tokens (near-universal truncation); Longformer's 4K-BPE cap ≈ 3,150 whitespace tokens (~5–8% still truncate). Discharge summaries front-load Assessment/Plan — chunked BERT over 6 × 512-BPE chunks covers ~3,072 BPE per note, slightly more than Longformer's single-pass 4K cap, at 3–5× lower training cost per epoch.
- **Alternatives:** (1) Single-window BERT (first 512 tokens) — rejected, throws away the majority of signal. (2) Start directly with Clinical-Longformer — rejected, slower and no evidence the back half of notes carries decisive signal. (3) Hierarchical transformer (chunk encoder + cross-chunk attention) — rejected for scope; revisit if both chunked-BERT and Longformer miss targets.
- **Evidence:** `reports/eda_report.md` §3 (token length). Even Longformer (4096 BPE) truncates the top ~1% of notes — no transformer at current context lengths sees the full tail.

## 2026-04-21 — Defer date un-shifting to Silver; EDA chart flagged as diagnostic-only
- **What:** Do not un-shift MIMIC-IV de-identified dates in the EDA layer. Compute `real_year_approx` per admission/patient in Silver using `anchor_year` + midpoint(`anchor_year_group`) **only when a consumer needs real years** (temporal split, drift monitoring, fairness-by-era, data card year coverage). Persist as new columns alongside raw shifted values; never overwrite.
- **Why:** Separation of concerns — EDA is diagnostic, not a production transform. Date un-shifting is a Silver-level contract so all downstream consumers get consistent semantics. The ICD-coding-from-text baseline doesn't use dates, so implementing the transform preemptively would be premature.
- **Alternatives:** (1) Un-shift in EDA so the §6 transition chart shows the Oct-2015 ICD-9→ICD-10 cutoff cleanly — rejected, pushes a production transform into a diagnostic notebook. (2) Leave dates shifted permanently — rejected, prevents drift monitoring and temporal fairness analysis needed for the model card.
- **Evidence / breadcrumbs:** `reports/eda_report.md` §6 action item + TODO in `src/mimic_icd_coder/data/clean.py` module docstring.


## [YYYY-MM-DD] — [Next decision]
- **What:**
- **Why:**
- **Alternatives:**
