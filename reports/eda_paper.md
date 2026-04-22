# Exploratory Data Analysis of MIMIC-IV-Note v2.2: Cohort Construction for Multi-Label ICD-10 Auto-Coding

**Author:** Nancy Tanaka
**Affiliation:** Northwestern University, MS in Data Science
**Date:** 2026-04-21
**Dataset:** MIMIC-IV-Note v2.2 (Jan 2023) + MIMIC-IV v3.1 Hosp (Oct 2024)

---

## Abstract

<!-- 150-250 words. Single paragraph. Covers:
     - Research problem: multi-label ICD-10 auto-coding from discharge summaries
     - Dataset: MIMIC-IV-Note v2.2 + MIMIC-IV v3.1 Hosp; version-mismatch rationale
     - Approach: cohort construction via notes ∩ admissions ∩ ICD-10 diagnoses
     - Key quantitative findings:
         - N = 122,288 modelable admissions (2.6× Mullenbach MIMIC-III)
         - Top-50 ICD-10 coverage = 91.04%
         - 44.83% of patients have >1 admission (patient-level splits required)
         - Note length: median 1,501 tokens, p90 2,447 (chunking required for BERT)
         - Redaction density 3.66% of tokens
     - Implications: chunked Bio_ClinicalBERT justified over single-window or Longformer
     - One-sentence "so what" for management audience -->

---

## 1. Introduction

<!-- Scope: why ICD-10 auto-coding matters, what MIMIC-IV offers, what this EDA decides.
     Draw from README §1 (Product summary) and the underlying clinical/business motivation.

     Paragraphs:
     1. Clinical context — ICD coding is a manual, bottlenecked step in revenue cycle;
        coder-assist AI reduces turnaround and improves consistency.
     2. Benchmark context — Mullenbach et al. (2018) set the reference bar on
        MIMIC-III top-50 at Micro F1 ≈ 0.539, Macro F1 ≈ 0.088; the field has
        advanced since (clinical BERT, long-context transformers).
     3. This work's scope — MIMIC-IV-Note v2.2 discharge summaries joined to
        MIMIC-IV v3.1 Hosp diagnoses; top-50 ICD-10 codes; patient-level splits.
     4. Research question — can a chunked Bio_ClinicalBERT pipeline exceed the
        CAML benchmark on the more recent MIMIC-IV cohort?
     5. Contribution of this paper — cohort construction and EDA that informs
        the downstream modeling decisions (chunking strategy, label space size,
        split methodology, fairness considerations). -->

---

## 2. Literature Review

<!-- Four sub-topics, each a paragraph or two with the cited work tied back to
     a specific decision in this project. Keep academic register; use
     (Author, Year) inline citation format, APA. -->

### 2.1 Automated Medical Coding

<!-- Paragraph 1: The CAML benchmark (Mullenbach et al., 2018). Attention-based
     CNN over MIMIC-III top-50 ICD-9. Metric conventions. Limitations noted
     (short-window attention, ICD-9 only). Why this is still the comparison
     anchor for medical coding papers.

     Paragraph 2: Subsequent work — label-attention transformers,
     retrieval-augmented coders, hierarchical ICD classifiers. Position
     this work as a return to the basics with updated tooling (BERT +
     chunking) on a more recent cohort. -->

### 2.2 Clinical Language Models

<!-- Paragraph 1: Foundation — Devlin et al. (2019) BERT introduced the
     pretrain-finetune paradigm. Limits: 512-token context.

     Paragraph 2: Clinical adaptations — Alsentzer et al. (2019) Bio_ClinicalBERT
     trained on MIMIC-III notes; superior to general BERT on clinical NLP.
     This is the primary model choice for this project.

     Paragraph 3: Long-context alternatives — Beltagy et al. (2020) Longformer
     extends context to 4,096 tokens via sliding-window attention. Trade-off:
     3-5× slower training than BERT. Used here as the deferred fallback if
     chunked BERT underperforms. -->

### 2.3 The MIMIC-IV Dataset

<!-- Paragraph 1: MIMIC-IV Hosp v3.1 (Johnson et al., 2023a) — structured EHR
     data from Beth Israel Deaconess Medical Center, 2008–2019 (shifted).
     Contains ICD-9 and ICD-10 diagnoses; v3.1 introduced corrections to
     diagnoses_icd table.

     Paragraph 2: MIMIC-IV-Note v2.2 (Johnson et al., 2023b) — free-text
     clinical notes released separately. Only public release as of April 2026;
     motivates the version-mismatch that this work resolves. De-identification
     methodology (Safe Harbor + date shifting).

     Paragraph 3: Comparison to MIMIC-III — newer cohort, ICD-10 inclusive,
     post-2015 admissions (including COVID-era codes like Z20.822). Larger
     baseline training set than CAML's MIMIC-III derivation. -->

### 2.4 Responsible Medical AI Documentation

<!-- Paragraph: Mitchell et al. (2019) Model Cards framework;
     Pushkarna et al. (2022) Data Cards framework. Position these as the
     deliverables that will accompany the trained model, grounded in this
     EDA's fairness and coverage findings. -->

---

## 3. Methods

### 3.1 Data Source

<!-- Describe in prose:
     - MIMIC-IV-Note v2.2: 331,793 discharge summaries, one per hadm_id,
       fields [note_id, subject_id, hadm_id, note_type, note_seq, charttime,
       text]. Single note_type value ("DS") in this release.
     - MIMIC-IV Hosp v3.1: admissions (546,028), patients (364,627),
       diagnoses_icd (6,364,488 rows; ICD-9 and ICD-10 mixed).
     - Version-mismatch justification — notes v2.2 is the only public
       release; Hosp v3.1 incorporates diagnoses_icd corrections.
       Join key hadm_id is stable across versions.
     - De-identification: per-patient date shift of 100+ years; anchor_age
       capped at 89.
     - DUA compliance posture (single sentence).

     Source evidence: `reports/eda_report.md` §1a, §1b, §12; DECISIONS.md
     2026-04-20 version-mismatch entry. -->

### 3.2 Data Preparation and Analysis Strategies

<!-- Paragraphs covering the cohort-construction pipeline:

     Paragraph 1 — Schema validation (from §1b). All timestamp columns
     auto-parsed by pyarrow CSV reader except patients.dod (89.5% null
     defeated type inference; deferred to Silver cast). No numeric columns
     coerced to object. hospital_expire_flag clean int64.

     Paragraph 2 — Cleaning rules. Discharge-only filter (note_type='DS').
     De-identification markers collapsed (see §3.3 for density findings).
     Deduplication by hadm_id (no-op on this release — PhysioNet
     pre-deduplicated). Minimum token filter at 100 whitespace tokens.

     Paragraph 3 — Cohort filtering. ICD-10-only admissions. Inner join
     notes ∩ admissions ∩ ICD-10 diagnoses. Final modelable N = 122,288.
     Loss decomposition: 0.018% version drift + 63.14% cohort-filter
     effect (ICD-9 admissions). Reference Figure 6.1.

     Paragraph 4 — Train/val/test split methodology. Patient-stratified
     80/10/10 by subject_id. Leakage-risk quantification (44.83% of
     patients multi-admitted) justifies patient-level over admission-level.
     Reference Figure 10.1.

     Source evidence: §1b, §4, §5, §6, §10, §11; `DECISIONS.md`
     entries 2026-04-20 × 3 and 2026-04-21. -->

### 3.3 Data Visualization

<!-- Narrative describing each visualization and its key observation.
     Figures numbered by the section of the EDA they support (see Appendix).

     Paragraph 1 — Note length distribution (Figure 3.1). Right-skewed,
     median 1,501 tokens, p90 2,447, p99 3,691. The "knee" of the
     exceedance curve sits at 1K-2K tokens. Motivates chunked-BERT
     decision (§3.4.2).

     Paragraph 2 — ICD version transition artifact (Figure 6.1).
     Shifted-year x-axis misrepresents the real Oct-2015 ICD-9→ICD-10
     transition because per-patient date shifts smear the cutoff across
     the full cohort window. Retained as diagnostic evidence only.
     Action item deferred to Silver (DECISIONS.md 2026-04-21).

     Paragraph 3 — Rank-frequency and coverage curves (Figure 7.1).
     Top-50 covers 91.04% of admissions; marginal coverage collapses
     past K=50. Classic long-tail distribution over 19,440 distinct
     codes.

     Paragraph 4 — Label co-occurrence heatmap (Figure 9.1). Confirms
     three canonical comorbidity clusters (metabolic, behavioral
     health, acute complications) and a negative-control result:
     the near-zero Z87.891 × F17.210 cell validates coding discipline
     (past vs. current smoker are mutually exclusive under ICD-10-CM).

     Paragraph 5 — Admissions-per-patient distribution (Figure 10.1).
     Heavy right skew; median 1, mean 2.44, max 238. 55% of patients
     admitted once; 11% admitted 5+ times. Justifies patient-level splits. -->

### 3.4 Algorithms and Modeling Methods

#### 3.4.1 Multi-Label Classification Approach

<!-- Paragraph: problem formulation as multi-label BCE over 50 binary
     labels. Baseline: TF-IDF n-gram features (1,2-gram, min_df=5,
     max_features=200K) + one-vs-rest LogisticRegression with
     class_weight='balanced'. Justify choice against the alternative
     of multi-class (which would force exactly one label per admission
     — inconsistent with §8 finding that median codes/admission = 12). -->

#### 3.4.2 Transformer Fine-Tuning Strategy

<!-- Paragraph 1: Primary — Bio_ClinicalBERT (Alsentzer et al., 2019)
     with chunk-and-max-pool decoding. Rationale from EDA:
     single-window BERT truncates 98.74% of notes with ~1,000 tokens
     lost each; chunking over 6 × 512-BPE chunks covers ~3,072 BPE
     per note, which slightly exceeds Longformer's single-pass 4K-BPE
     cap at 3-5× lower training cost.

     Paragraph 2: Escalation trigger — if chunked-BERT misses the
     Micro F1 target of 0.70 by more than 3 points, re-evaluate
     against Clinical-Longformer (Beltagy et al., 2020).

     Reference DECISIONS.md 2026-04-20 "Chunk-and-max-pool" entry. -->

#### 3.4.3 Threshold Tuning

<!-- Paragraph: per-label threshold selection on validation split,
     maximizing per-label F1. Rationale: imbalanced label distribution
     (see §8 — top-50 positive rates range ~1%-20%); a single global
     threshold under-performs on rare labels. -->

#### 3.4.4 Patient-Level Train/Val/Test Splits

<!-- Paragraph: 80/10/10 by subject_id, seed=42. Fraction of patients
     with >1 admission = 44.83%; under admission-level splits, the
     majority of multi-admission patients would contaminate test/val
     (for k=5 admissions, 66% leakage probability under random
     admission-level split). Admission-level evaluation would inflate
     reported metrics by an amount proportional to within-patient
     language-style consistency. -->

### 3.5 EDA Results

<!-- Synthesis section. Integrate the findings from §3.1-3.4 into a
     coherent summary of what the cohort supports:

     Paragraph 1 — Cohort viability. N = 122,288, 2.6× Mullenbach CAML's
     MIMIC-III training set; passes the 80,000-admission floor by 53%.
     Top-50 coverage at 91.04% exceeds the 80% target and is 11
     percentage points above MIMIC-III top-50 coverage (~80% per
     Mullenbach 2018). The cohort is both larger and more concentrated
     than the MIMIC-III benchmark — a favorable setting for the baseline.

     Paragraph 2 — Constraint findings. Note length distribution forces
     chunking (98.74% single-window truncation). Patient-level splits
     are mandatory (44.83% multi-admit leakage risk). The v3.1 Hosp
     upgrade adds data relative to v2.2 but limits the cohort to
     ICD-10-only admissions, removing 63.14% of the note inventory by
     design rather than by data quality.

     Paragraph 3 — Unexpected findings worth recording. COVID-19
     exposure code Z20.822 is in top-10, confirming the MIMIC-IV cohort
     includes 2020+ admissions. The expected chronic cardiometabolic
     cluster (E11.9 T2DM, I50.9 CHF, J44.9 COPD, N18.6 ESRD) is absent
     from top-10 due to ICD-10 sub-code fragmentation, but is recovered
     at top-50. The Z87.891 × F17.210 near-zero co-occurrence validates
     coding discipline. -->

### 3.6 Model Evaluation

<!-- Paragraph 1 — Primary metrics:
     - Micro F1 — global TP/FP/FN aggregation, penalizes rare-label errors
       less than Macro F1.
     - Macro F1 — unweighted mean of per-label F1; Mullenbach CAML scored
       0.088 on this due to weak rare-label performance.
     - P@5, P@8, P@15 — ranked-prediction precision, aligns with coder-assist UX.

     Paragraph 2 — Targets and floors:
     Target Micro F1 ≥ 0.70, Macro F1 ≥ 0.55, P@8 ≥ 0.65. Floor
     (below which something is broken): Micro F1 ≥ 0.55.

     Paragraph 3 — Mullenbach delta reporting. Report Micro F1 delta and
     Macro F1 delta vs. Mullenbach's MIMIC-III numbers, with the
     fragmentation caveat from §3.5 acknowledged.

     Paragraph 4 — Subgroup reporting deferred to model card — frequent-
     vs.-single-admission patients, gender-conditional F1, age-bucket
     F1. Temporal subgroup analysis deferred pending Silver-stage date
     un-shifting (DECISIONS.md 2026-04-21). -->

---

## 4. Conclusion

### 4.1 Exposition

<!-- 1-2 paragraphs on the big picture. What this EDA established about
     MIMIC-IV ICD coding feasibility. Why the cohort clears the gate
     for a chunked-BERT baseline. What the findings say about the
     evolution of the problem since Mullenbach 2018 (newer cohort,
     ICD-10 inclusive, more granular label space, larger N, includes
     pandemic-era codes). -->

### 4.2 Problem Description

<!-- 1 paragraph articulating the specific modeling challenge the data
     reveals: heavily right-skewed notes requiring chunking, fragmented
     label space requiring care in top-K selection, high patient-level
     leakage risk requiring subject-id splits, and a cohort sampled
     from MIMIC-IV-Note v2.2 which itself samples 48% of all v3.1 ICD-10
     admissions. -->

### 4.3 Management Recommendations

<!-- Numbered list. Each item: what to do, why, by when. Draw from the
     deferred-items sections at the bottom of eda_report.md and the
     DECISIONS.md action items. Target 5-8 recommendations. -->

1. **Proceed to baseline training.** Cohort N = 122,288 and top-50 coverage
   of 91.04% both clear the pre-registered gate. Expected TF-IDF+LR baseline
   at Micro F1 ≥ 0.55.

2. **Plan for chunked Bio_ClinicalBERT, not single-window.** 98.74% of
   notes exceed the 512-token BERT cap; any single-window architecture
   would discard the majority of signal.

3. **Defer Clinical-Longformer evaluation** until chunked-BERT results
   are available. Escalation trigger: Micro F1 miss by >3 points
   below the 0.70 target.

4. **Schedule date un-shifting for the Silver stage** before any consumer
   uses real-year fields (drift monitoring, temporal splits, fairness-by-era,
   data card year coverage).

5. **Extend the Bronze admissions schema** to ingest `race` and related
   fields before model-card development. Current extract excludes these
   by design decision (deferred), but the model card requires subgroup
   fairness evaluation.

6. **Document the cohort-ceiling finding** in the data card: the
   modelable population is 48% of MIMIC-IV v3.1 ICD-10 admissions,
   limited by MIMIC-IV-Note v2.2's sampling rather than this project's
   filters. Cohort N would roughly double under a future v3.x note release.

7. **Cap fairness reporting on the 85+ age bucket** due to the
   `anchor_age = 89` re-identification cap. Report this limitation
   explicitly in the model card.

### 4.4 Closing

<!-- Short paragraph — tie the analysis back to the larger goal
     (portfolio demonstration of end-to-end clinical NLP pipeline
     with proper EDA discipline and methodological transparency).
     One sentence on what the next deliverable will be (baseline
     results + BERT fine-tune on Azure Databricks). -->

---

## References

<!-- APA 7th edition format. Alphabetical by first author. -->

Alsentzer, E., Murphy, J. R., Boag, W., Weng, W.-H., Jin, D., Naumann, T.,
  & McDermott, M. (2019). Publicly available clinical BERT embeddings.
  *Proceedings of the 2nd Clinical Natural Language Processing Workshop*,
  72–78. https://doi.org/10.18653/v1/W19-1909

Beltagy, I., Peters, M. E., & Cohan, A. (2020). Longformer: The
  long-document transformer. *arXiv preprint arXiv:2004.05150*.
  https://arxiv.org/abs/2004.05150

Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT:
  Pre-training of deep bidirectional transformers for language
  understanding. *Proceedings of the 2019 Conference of the North
  American Chapter of the Association for Computational Linguistics*,
  4171–4186. https://doi.org/10.18653/v1/N19-1423

Johnson, A. E. W., Bulgarelli, L., Shen, L., Gayles, A., Shammout, A.,
  Horng, S., Pollard, T. J., Hao, S., Moody, B., Gow, B., Lehman,
  L. H., Celi, L. A., & Mark, R. G. (2023). MIMIC-IV, a freely
  accessible electronic health record dataset. *Scientific Data*,
  *10*(1), 1. https://doi.org/10.1038/s41597-022-01899-x

Johnson, A. E. W., Pollard, T. J., Horng, S., Celi, L. A., & Mark,
  R. G. (2023). MIMIC-IV-Note: Deidentified free-text clinical notes
  (version 2.2) [Data set]. *PhysioNet*.
  https://doi.org/10.13026/1n74-ne17

Mitchell, M., Wu, S., Zaldivar, A., Barnes, P., Vasserman, L.,
  Hutchinson, B., Spitzer, E., Raji, I. D., & Gebru, T. (2019).
  Model cards for model reporting. *Proceedings of the Conference
  on Fairness, Accountability, and Transparency (FAT\**), 220–229.
  https://doi.org/10.1145/3287560.3287596

Mullenbach, J., Wiegreffe, S., Duke, J., Sun, J., & Eisenstein, J.
  (2018). Explainable prediction of medical codes from clinical text.
  *Proceedings of the 2018 Conference of the North American Chapter
  of the Association for Computational Linguistics*, 1101–1111.
  https://doi.org/10.18653/v1/N18-1100

Pushkarna, M., Zaldivar, A., & Kjartansson, O. (2022). Data cards:
  Purposeful and transparent dataset documentation for responsible AI.
  *Proceedings of the 2022 ACM Conference on Fairness, Accountability,
  and Transparency (FAccT)*, 1776–1826.
  https://doi.org/10.1145/3531146.3533231

<!-- Additional references to add if desired:
     - Lee et al. (2020) BioBERT — predecessor to Bio_ClinicalBERT
     - U.S. Department of Health & Human Services HIPAA Safe Harbor method
     - Goldberger et al. (2000) PhysioNet foundation paper
     - Huang et al. (2020) ClinicalBERT for hospital readmission
     - Vu et al. (2020) Label attention for automated medical coding -->

---

## Appendix

### Figures

Figures are numbered `Figure <section>.<index>` where `<section>` corresponds
to the EDA section that produced or referenced the figure, as organized in
the original cohort-construction notebook.

**Section 1 — Volumetrics**

- *Figure 1.1:* Row counts per source table with expected vs. observed comparison.
  <!-- Optional: add as a bar chart if useful; currently a table in eda_report §1a. -->
- *Figure 1.2:* Per-column null rate summary, four source tables.
  <!-- Optional: current report has this as prose in §1b. -->

**Section 2 — Note Types**

- (No figure — single-category distribution reported in text.)

**Section 3 — Token and Character Length**

- *Figure 3.1:* Distribution of note lengths (whitespace tokens and characters)
  with percentile markers at p50, p95, p99, and the BERT (512) and Longformer (4096)
  context-length references.
  <!-- File: `reports/figures/length_distribution.png` -->

**Section 4 — De-identification Markers**

- *Figure 4.1:* Optional — histogram of `___` marker counts per note.
  <!-- Currently stats-only in the report; add only if narrative references it. -->

**Section 5 — Duplicate Analysis**

- (No figure — tabular results reported in text.)

**Section 6 — ICD Version Split**

- *Figure 6.1:* Stacked-area distribution of ICD-9 vs. ICD-10 code assignments
  by shifted admit year. Retained as diagnostic only; real-year transition is
  not visible in the shifted timeline (see §3.3 paragraph 2).
  <!-- File: `reports/figures/icd_version_by_year.png` -->

**Section 7 — ICD-10 Code Frequency and Top-K Coverage**

- *Figure 7.1:* Left panel — rank-frequency curve of ICD-10 codes, log-log.
  Right panel — fraction of admissions covered by the top-K most-frequent codes
  as K grows.
  <!-- File: `reports/figures/icd_frequency_and_coverage.png` -->

**Section 8 — Codes per Admission**

- *Figure 8.1:* Distribution of distinct ICD-10 codes assigned per admission.
  <!-- File: `reports/figures/codes_per_admission.png` -->

**Section 9 — Label Co-occurrence**

- *Figure 9.1:* Top-20 × Top-20 ICD-10 label co-occurrence heatmap, log-scaled.
  Clinically coherent clusters (metabolic, behavioral health, acute) are
  visible; the near-zero Z87.891 × F17.210 cell confirms coding-discipline
  mutual-exclusivity.
  <!-- File: `reports/figures/label_cooccurrence.png` -->

**Section 10 — Patient Demographics and Admission Patterns**

- *Figure 10.1:* Distribution of admissions per patient, capped at 20 for
  readability. Heavy right-skew consistent with chronic-care cohort tail.
  <!-- File: `reports/figures/admissions_per_patient.png` -->
- *Figure 10.2:* Optional — gender and age-bucket distributions of the
  admitted-patient cohort (distinct from the registry-level distribution
  reported in §10).
  <!-- Currently table-only; generate if narrative wants visual support. -->

**Section 11 — Join Coverage (notes ∩ admissions ∩ diagnoses)**

- *Figure 11.1:* Optional — Venn or set-bar diagram showing the three-way
  intersection. Current report uses a tabular decomposition.

**Section 12 — Version Reconciliation**

- (No figure — version-drift result is a single scalar, 0.018%; cohort-filter
  decomposition is a table.)

### Supplementary Tables

<!-- Place any large auxiliary tables here that would break reading flow in
     Methods. Candidates:
     - Full top-50 ICD-10 code list with clinical descriptions
     - Schema column reference for all four Bronze tables
     - Full cohort-coverage decomposition (§11 slice table) -->
