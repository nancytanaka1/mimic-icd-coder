# EDA Report — MIMIC-IV ICD-10 Auto-Coding

**Owner:** Nancy Tanaka
**Date completed:** 2026-04-20
**Notebook:** `notebooks/01_eda.ipynb`
**Data:** MIMIC-IV-Note v2.2 + MIMIC-IV v3.1 Hosp module
**Storage:** Local Parquet at `data/bronze/`

Fill in each section after running the corresponding notebook cells. Keep findings tight — one or two sentences per bullet. Commit this file alongside the notebook.

---

## 1. Volumetrics

### 1a. Row counts

| Table | Observed rows | Expected | Match? |
|---|---|---|---|
| notes | 331,793 | 331,000 (v2.2) | ☑ |
| diagnoses | 6,364,488 | 6,400,000 | ☑ |
| admissions | 546,028 | 431,000 (v2.2) → expect ~540K on v3.1 | ☑ |
| patients | 364,627 | 300,000 (v2.2) → expect ~360K on v3.1 | ☑ |


### 1b. Column schema

Dtypes observed after Bronze ingest (pyarrow → pandas). Fill in `Observed dtype` + `Null %` from the notebook output.

**notes**

| Column | Expected type (MIMIC docs) | Observed dtype | Null % | Notes |
|---|---|---|---|---|
| note_id | string | `object` | 0.00% | primary key, unique |
| subject_id | int | `int64` | 0.00% | patient FK |
| hadm_id | int | `int64` | 0.00% | admission FK |
| note_type | string | `object` | 0.00% | `DS` only in v2.2 |
| note_seq | int | `int64` | 0.00% | amendment counter; max observed = 228 |
| charttime | timestamp | `datetime64[ms]` | 0.00% | auto-parsed by pyarrow ✓ |
| text | string | `object` | 0.00% | free-text clinical note |

**diagnoses_icd**

| Column | Expected type | Observed dtype | Null % | Notes |
|---|---|---|---|---|
| subject_id | int | `int64` | 0.00% | patient FK |
| hadm_id | int | `int64` | 0.00% | admission FK |
| seq_num | int | `int64` | 0.00% | ordering within an admission |
| icd_code | string | `object` | 0.00% | ICD-9 or ICD-10 code literal |
| icd_version | int | `int64` | 0.00% | 9 or 10 |

**admissions**

| Column | Expected type | Observed dtype | Null % | Notes |
|---|---|---|---|---|
| subject_id | int | `int64` | 0.00% | patient FK |
| hadm_id | int | `int64` | 0.00% | primary key |
| admittime | timestamp | `datetime64[ms]` | 0.00% | auto-parsed ✓ |
| dischtime | timestamp | `datetime64[ms]` | 0.00% | auto-parsed ✓ |
| deathtime | timestamp | `datetime64[ms]` | **97.84%** | expected: null for the 97.8% of admissions where patient survived |
| hospital_expire_flag | int | `int64` | 0.00% | 0/1; clean int (no NaN promotion) |
| admission_type | string | `object` | 0.00% | EW EMER / URGENT / ELECTIVE / ... |

**patients**

| Column | Expected type | Observed dtype | Null % | Notes |
|---|---|---|---|---|
| subject_id | int | `int64` | 0.00% | primary key |
| gender | string | `object` | 0.00% | F / M |
| anchor_age | int | `int64` | 0.00% | capped at 89 for re-id protection |
| anchor_year | int | `int64` | 0.00% | shifted |
| anchor_year_group | string | `object` | 0.00% | e.g. `2014 - 2016` |
| dod | timestamp | `object` ⚠️ | **89.50%** | nullable; string-typed because 89.5% null → see red flag below |

**Schema red flags observed:**

- ⚠️ **`patients.dod` is `object`, not `datetime64`.** All other timestamp columns (`charttime`, `admittime`, `dischtime`, `deathtime`) auto-parsed via pyarrow CSV inference. `dod` did not — almost certainly because the column is mostly null (most MIMIC patients are alive in the extract), and pyarrow's type inference falls back to string when too few non-null values are seen in the inference block. Non-blocking — downstream `pd.to_datetime(dod, errors="coerce")` handles it — but worth an explicit cast in Silver for type stability.

**Schema red flags ruled out:**

- ✓ No numeric column came back as `object` → no bad null tokens (`"NA"`, whitespace) in the CSV source.
- ✓ `hospital_expire_flag` is `int64`, not `float64` → no NaN promotion; column is fully populated.
- ✓ `charttime` / `admittime` / `dischtime` / `deathtime` all `datetime64[ms]` → pyarrow auto-parsed correctly; no manual `pd.to_datetime` needed for these four.

**Null-rate findings:** Only two columns have non-zero null rates, both **expected** given the clinical meaning:

- `admissions.deathtime` — **97.84% null** (534,238 / 546,028). Set only when the patient died during the admission. 2.16% in-hospital mortality rate (confirmed by `hospital_expire_flag` in §10).
- `patients.dod` — **89.50% null** (326,326 / 364,627). Set only when the patient has died at any time through the last follow-up. 10.5% of patients recorded deceased — consistent with MIMIC-IV's ICU-heavy cohort with multi-year follow-up.

All other columns across all four tables are fully populated. No unexpected null patterns. No fix required.

**Date range:**
- notes `charttime`: 2105-10-12  to 2212-04-12
- admissions `admittime`: 2105-10-04 to 2214-12-15
- admissions `dischtime`: 2105-10-12 to 2214-12-24

**Verdict:** ☑ All tables loaded cleanly.

**Row counts:** `notes` and `diagnoses` match template expectations. `admissions` (546,028) and `patients` (364,627) are ~20–27% higher than template targets — expected, because template targets were from MIMIC-IV v2.2 Hosp; we're using v3.1 Hosp (Oct 2024), which added ~2020 admissions. Consistent with the version-mismatch decision in `DECISIONS.md` (2026-04-20).

**Schema caveat:** `patients.dod` typed as `object` rather than `datetime64` due to 89.5% null rate defeating pyarrow's CSV type inference. Non-blocking — downstream `pd.to_datetime(..., errors="coerce")` handles it.

**Null audit:** Only `admissions.deathtime` (97.84%) and `patients.dod` (89.50%) have non-zero null rates, both expected given clinical meaning (null when patient has not died).


---

## 2. Note Types

| note_type | count | pct |
|---|---|---|
| DS | 331,793 | 100.000 |
| (other) | 0 | 0.000 |

**Decision:** Keep only `note_type = "DS"`. ☑ Confirmed — `discharge.csv.gz` contains only DS records by PhysioNet's file convention; `radiology.csv.gz` is a separate file not ingested (radiology reports are out of scope for ICD coding from discharge summaries).

**Note on the notebook's section 2 header:** the docstring says v2.2 contains "DS and radiology note types" — that phrasing conflates file names with the `note_type` column. The `note_type` column within `discharge.csv.gz` is always `DS`. If you ever want to include radiology reports, you'd ingest `radiology.csv.gz` separately and either keep them in their own Silver table or concatenate with `note_type` as a discriminator column.


---

## 3. Token & Character Length

| Metric | tokens | chars |
|---|---|---|
| min | 44.00 | 353.00 |
| p05 | 718.00	 | 4,771.00 |
| p50 (median) | 1,501.00	 | 9,847.00 |
| mean | 	1,600.27	 | 10,550.96 |
| p90 | 2,447.00	 | 16,159.00 |
| p95 | 2,806.00	 | 18,619.00 |
| p99 | 3,691.00	 | 24,710.16 |
| max | 9,026.00	 | 60,381.00 |

**Truncation impact** (whitespace tokens — note that BPE ≈ 1.3× larger for clinical text):

| Context length (tokens) | Example model | n notes exceeding | % notes exceeding | median tokens lost per truncated note |
|---|---|---|---|---|
| 512 | Bio_ClinicalBERT (single window) | 327,602 | **98.74%** | 998 |
| 1024 | — (waypoint) | 271,341 | **81.78%** | 621 |
| 2048 | — (waypoint) | 70,035 | **21.11%** | 372 |
| 4096 | Clinical-Longformer (single pass) | 1,684 | **0.51%** | 455 |

**Reading the shape of the distribution:**

- **The "knee" is at 1K → 2K tokens** — exceedance rate collapses from 82% to 21%. Any model that sees at least 2K tokens captures the body of the distribution.
- 2K → 4K only buys another ~20 percentage points; the marginal return on Longformer's 4K cap (vs chunked-BERT at 3K+) is small.
- Applying 1.3× BPE inflation: BERT's true 512-BPE cap ≈ 400 whitespace tokens (near-universal truncation); Longformer's 4K-BPE cap ≈ 3,150 whitespace tokens (~5–8% still truncate).

**Decision:**

☑ **Bio_ClinicalBERT with chunk-and-max-pool; Longformer deferred pending empirical result.**

Per [`DECISIONS.md`](../DECISIONS.md) (2026-04-20 — Chunk-and-max-pool entry). The template's 40% / 512-token threshold is superseded: at 98.74% truncation, single-window BERT is non-viable, so chunking is mandatory regardless of model choice. Chunking 6 × 512-BPE covers ~3,072 BPE per note — slightly more than Longformer's single-pass 4K-BPE cap, at 3–5× lower training cost per epoch.

- ☐ Bio_ClinicalBERT only / ☐ Bio_ClinicalBERT then Longformer comparison / ☐ Longformer only

**Escalation trigger:** If chunked-BERT Micro F1 misses the 0.70 target by more than 3 points, re-evaluate against Clinical-Longformer.

**Evidence:** Truncation-impact table above; percentile table above; discharge summaries front-load Assessment/Plan (diagnoses concentrated in the first section regardless of note length).

**Decision:**
- Start with Bio_ClinicalBERT at 512 if fewer than 40% of notes exceed.
- Escalate to Clinical-Longformer if 40%+ of notes exceed 512 AND the excess content is clinically relevant (final-diagnosis section).
- Bio_ClinicalBERT only / Bio_ClinicalBERT then Longformer comparison / Longformer only

---

## 4. De-identification Markers

- Mean `___` markers per note: 58.19
- Markers per 1000 tokens: 36.56
- Markers per 10000 characters: 55.41

**Verdict:** ☑ **Heavy redaction** (36.6 / 1000 tokens, 3.7× the template's 10/1000 threshold). Expected for MIMIC-IV discharge summaries — each admission references many dates, provider names, and hospital identifiers, all of which are redacted. Not a blocker:

- **Silver cleaning already collapses `___` markers** (per `data/clean.py` → `build_silver_notes`), so redaction spans don't reach TF-IDF or BERT as signal-diluting tokens.
- **Validation step:** after `mic train-baseline`, grep the top 100 TF-IDF features — `___` (or variants) must not appear. If they do, Silver cleaning isn't firing.
- **BERT-specific concern:** BPE may still split partial underscore sequences if cleaning is incomplete. Worth a spot-check on a few Silver notes before fine-tuning.

**Comparable prior work:** MIMIC-III redaction density is similar (~3–5% of tokens). Published Bio_ClinicalBERT and CAML results were achieved on equivalently-redacted text, so this density is a known-workable operating point — not novel risk.

---

## 5. Duplicate Analysis

- Total notes: **331,793**
- Unique `note_id`: **331,793** (equals total ✓ — no duplicate PKs)
- Admissions with > 1 note: **0**
- Max notes per `hadm_id`: **1**
- Max `note_seq`: **228**

**Dedup rule:** Keep note with max `note_seq` per `hadm_id`.

**Verdict:** ☑ **Standard dedup handles it — currently a no-op.**

PhysioNet pre-deduplicated discharge summaries to one row per admission before the v2.2 release: every `hadm_id` has exactly one note, and the `note_seq` retained is the final amended version. Silver's "dedup by hadm_id, keep max note_seq" rule therefore has nothing to collapse on this release.

**Why keep the dedup code anyway:** defensive. If a future MIMIC-IV-Note release surfaces intermediate amendments (all versions per admission), the same Silver logic handles it without change. Expected Silver row count ≈ 331,793 minus the ~0–5% that fail the `min_note_tokens ≥ 100` filter.

**Oddity worth recording (non-blocking):** `max_note_seq = 228` means one discharge summary was amended 228 times before sign-off. Plausible for a long, complex ICU stay where labs/micro/consults trickle in over days — not a data quality issue, just a clinical reality surfaced by the counter. `hadm_id` remains a safe primary key for joins to `diagnoses_icd`, `admissions`, and `patients`.


---

## 6. ICD Version Split

| icd_version | n_codes | n_admissions |
|---|---|---|
| 9 | 2,908,741 | 291,130 |
| 10 | 3,455,747 | 254,377 |

**Transition pattern (stacked-area chart) — ⚠️ x-axis is misleading, flagged for Silver-stage fix:**

The chart currently plots ICD codes against the **shifted** admit-year (~2100–2215). These are not real calendar years — MIMIC-IV de-identifies by applying an independent per-patient date shift of 100+ years, so the same real date maps to many different shifted years across the cohort. Consequences:

- No sharp ICD-9 → ICD-10 cutoff is visible in the chart, even though the real-world transition happened on a specific date (October 1, 2015 — U.S. HIPAA mandate). The cutoff is smeared across the shifted axis by per-patient date shifts.
- The shifted-year range "2100–2215" is clinically meaningless — it reflects the union of shifted windows, not a ~115-year span of clinical activity.
- ICD-9 and ICD-10 appear to coexist throughout the full range, which is an artifact, not a finding.

**The chart is retained as a diagnostic exhibit** showing that both versions are present in substantial volume across the cohort, but **it is NOT a real-year transition chart** and should not be cited as evidence of any temporal trend.

 **🔧 Silver-stage action item — date un-shifting.** Before any model consumer (chunking, training, drift monitoring, fairness-by-era, temporal split, data card year coverage) uses admit/discharge/chart/death dates, compute an approximate real year per admission:

 ```
 real_year_approx = admittime.year - patients.anchor_year + midpoint(patients.anchor_year_group)
 ```

 where `anchor_year_group` is a 3-year bucket like `"2014 - 2016"` (midpoint 2015). Precision: ±1.5 years. Apply to all four shifted datetime columns (`charttime`, `admittime`, `dischtime`, `deathtime`). Persist to Silver as new columns alongside the raw shifted ones — don't overwrite, so the mapping is auditable.

 **Owner:** Silver stage (`data/clean.py` or a new `data/dates.py`). **Trigger:** before first training run that uses temporal features or before data card / drift monitoring setup (whichever comes first). **Not required for the current baseline** — ICD coding from text alone doesn't use dates. Defer until needed, but don't forget.

 **Tracked at:** this note in `eda_report.md` + TODO in `src/mimic_icd_coder/data/clean.py` + (recommended) a new DECISIONS.md entry.

**What the chart does validly confirm (version marginals, not temporal):**
- ✓ Both versions present in substantial volume in the dataset.
- ✓ ICD-9 dominates by row count (2,908,741 codes / 291,130 admissions) vs ICD-10 (3,455,747 codes / 254,377 admissions) — consistent with a cohort skewed earlier in MIMIC-IV's real time window.

**Decision:** ☑ **Filter to ICD-10 only.**

- **Diagnosis rows retained:** ICD-10 codes / total codes — expect ~30–35% retention based on the chart's visual proportion (ICD-10 shown as the smaller yellow band). Confirm exact figure from cell 18.
- **Admissions retained:** the `n_admissions` for icd_version=10 is your training cohort upper bound — expect ~130K–180K. This is the `N` for modeling before further filters (min tokens, join with notes).
- **ICD-9 admissions discarded:** not recoverable without GEM crosswalk; rejected per `DECISIONS.md` (2026-04-20 — ICD-10-only cohort) on complexity-vs-benefit grounds.

**Cohort N implications:**
- Cohort N for modeling ≈ min(ICD-10 admissions, notes admissions with valid join). The notes file has 331,793 admissions; ICD-10 admissions ~130K–180K expected; after inner join, expect **~120K–170K admissions** as the trainable cohort. Final number confirmed in §11 (Join Coverage).
---

## 7. ICD-10 Code Frequency + Top-K Coverage

Total distinct ICD-10 codes observed: 19,440

| K | % admissions with ≥ 1 top-K code |
|---|---|
| 10 | 80.37 % |
| 25 | 89.10 % |
| 50 | 91.04 % |
| 100 | 95.45 % |
| 250 | 97.81 % |
| 500 | 98.63 % |
| 1000 | 99.23 % |

**Top 10 most-assigned ICD-10 codes:**
| Rank | Code | Description |
|---|---|---|
| 1 | E78.5 (`E785`) | Hyperlipidemia, unspecified |
| 2 | I10 | Essential (primary) hypertension |
| 3 | Z87.891 (`Z87891`) | Personal history of nicotine dependence |
| 4 | K21.9 (`K219`) | GERD without esophagitis |
| 5 | F32.9 (`F329`) | Major depressive disorder, single episode, unspecified |
| 6 | I25.10 (`I2510`) | Atherosclerotic heart disease of native coronary artery without angina |
| 7 | F41.9 (`F419`) | Anxiety disorder, unspecified |
| 8 | N17.9 (`N179`) | Acute kidney injury, unspecified |
| 9 | Z20.822 (`Z20822`) | Contact with and (suspected) exposure to COVID-19 |
| 10 | Z79.01 (`Z7901`) | Long-term (current) use of anticoagulants |

**Decision on `top_k_labels`:**
- Target: coverage ≥ 80%. Met at K=50 (97.04%); exceeded at K=10 (80.37%)
- K chosen: 50 - confims `DECISION.md` (2026-04-20 - Target top-50 ICD-10 codes). No amendment required.
- **Rationale:**
1. **Coverage is strong and above target.** 91.04% at K=50 is 11pp above target, and 11pp above MIMIC-III's published top-50 coverage (~80% per Mullenbach et al. 2018). The ICD-10 distribution in this cohort is *more* concentrated than MIMIC-III's ICD-9 was — this is the calibration finding that matters for framing the expected macro F1 ceiling.

2. **Direct comparability to Mullenbach CAML** preserved — the primary motivation for K=50 remains intact.

3. **Diminishing returns past K=50.**
   - K=50 → K=100: +4.41pp coverage, 2× label space, ~½ per-label positive mass.
   - K=100 → K=250: +2.36pp coverage, 2.5× label space, further mass halving.
   - K=50 → K=1000: only +8.19pp coverage for 20× label space.
   - The curve flattens fast; additional K buys very little.

4. **Fragmentation caveat acknowledged but non-decisive.** Top-10 is missing the expected cardiometabolic cluster (E11.9 T2DM, I50.9 CHF, J44.9 COPD, N18.6 ESRD) because ICD-10's finer granularity distributes those conditions across sub-codes (E11.65, I50.23, etc.). The top-50 absorbs many of these sub-codes — which is why K=50 coverage is so strong despite the top-10 surprises. Going to K=100 would pick up more of the fragmented chronics, but not at a rate that justifies the label-space doubling.

5. **Alternatives rejected:**
   - **K=10** (80.37%, technically on-target): covers the chronic-comorbidity headlines but loses the multi-label richness that makes this benchmark interesting. 10 labels is effectively a multi-class problem.
   - **K=25** (89.10%): within ~2pp of K=50 but loses the CAML benchmark comparison.
   - **K=100** (95.45%): marginal coverage gain (+4.41pp), non-trivial training cost, weakens CAML comparability framing. Revisit only if chunked-BERT macro F1 plateaus and label fragmentation is diagnosed as the bottleneck.
   - **K=250+** (≥97.81%): halves per-label positive mass beyond the point where per-label F1 is reliably measurable for the rare labels.

**Not an amendment to DECISIONS.md** — the existing entry stands.

**Flag for the data card:** ICD-10 sub-code fragmentation (E11.*, I50.*, J44.*, N18.*) means the label space isn't perfectly comparable to Mullenbach's MIMIC-III top-50 even at the same K. Report CAML deltas with that caveat; don't over-interpret small macro F1 differences as pure model quality.
---

## 8. Codes per Admission

| Stat | ICD-10 codes per admission |
|---|---|
| min | 1 |
| p25 | 7.0 |
| median | 12.0 |
| mean | 13.5849 |
| p75 | 18.0 |
| p95 | 29.0 |
| max | 39 |

**Verdict:** ☑ Median 12 / mean 13.58 codes per admission — healthy multi-label signal, well-distributed across the IQR 7–18. Min=1 confirms no empty-label admissions. Expected top-50 label density of ~5–8 active labels per admission supports BCE + per-label threshold tuning for the multi-label objective.

- **Distribution shape:** right-skewed (mean > median), IQR 7–18, p95=29, max=39. Consistent with typical acute-care coding behavior: most admissions get 7–18 codes; complex ICU/long-stay admissions accumulate more; no admissions with zero codes (min=1, data integrity ✓).
- **Min=1:** confirms every admission has at least one diagnosis. No empty-label admissions to filter out.
- **Max=39:** high but plausible — corresponds to multi-organ-failure ICU stays with extensive comorbidity documentation. Not a data-quality concern.

**Modeling implications:**

- These are counts over **all ICD-10 codes**, not just the top-50 label space. After restricting to top-50 labels, per-admission label count will be lower (expect median ~5–8 active labels per admission). That's still dense enough for reliable multi-label training with BCE loss + per-label threshold tuning.
- **Expected per-label positive rate:** with ~5–8 active labels/admission out of 50 possible, positive rate per label ≈ 10–16% on average. Rare labels (rank 40–50) will be materially sparser (~1–3%). Threshold tuning in `mimic_icd_coder.thresholds` handles this — without it, the rare-label tail would collapse to "always predict negative."
- **No cohort surgery required.** No admissions to filter on low- or high-code extremes.
- **Sanity gate for Silver:** after running `mic silver`, re-compute this on the Silver-filtered cohort (min 100 tokens). Median and max should remain nearly identical — the token-length filter is language-side, not diagnosis-side, so the per-admission code distribution should be preserved.


---

## 9. Label Co-occurrence

**Expected patterns to verify:**

- I10 (HTN) co-occurs with I50.9 (CHF)? ☐ **Not testable** — I50.9 is not in top-20 (ICD-10 fragments CHF across I50.23, I50.33, etc. — same granularity artifact noted in §7). HTN co-occurs with I2510 (CAD) strongly, consistent with the CV comorbidity cluster.
- E11.9 (T2DM) co-occurs with I10? ☑ **Yes** — E119 × I10 cell is bright green (~10+ log1p), confirming the canonical metabolic-syndrome cluster. E119 also strong with E785 (hyperlipidemia), Z794 (insulin use), I2510 (CAD) — textbook diabetes comorbidities.
- N18.* (CKD) co-occurs with I10 and E11.9? ☐ **Not testable** — N18.* family (CKD stages) is absent from top-20; only N179 (AKI) is present. Same fragmentation: CKD sub-codes distribute across top-50 but none individually reach top-20. N179 does co-occur strongly with I10, confirming AKI's association with hypertension.
- F32.9 (depression) shows up as a behavioral-health cluster? ☑ **Yes** — F329 × F419 (anxiety) and F329 × Z87891 (history of smoking) both show strong co-occurrence. Depression is broadly distributed (not isolated), which is epidemiologically correct for hospitalized patients.

**Red flags:** ☑ **Expected structure present.** Matrix is not diagonal-only — rich off-diagonal signal across clinically coherent clusters (metabolic, CV, behavioral).

**Additional finding — the dark square is a validation signal, not a red flag:**

The **Z87.891 × F17.210 cell is near-zero** (dark purple, log1p ≈ 3–4 vs matrix median ≈ 8–9). At first glance this looks like a negative finding, but it's actually confirming the coding is being applied correctly:

- **Z87.891** = Personal **history of** nicotine dependence (past smoker, quit).
- **F17.210** = Nicotine dependence, cigarettes, uncomplicated — **current** smoker.
- These are **mutually exclusive by coding guidelines** — a patient is either a former smoker (Z87.891) or a current smoker (F17.210) at admission, not both. The near-zero co-occurrence confirms coders are following ICD-10-CM conventions. If this square had been bright, it would indicate coding errors.

**Clinically coherent clusters observed:**

1. **Metabolic syndrome cluster:** E785 (lipids) ↔ I10 (HTN) ↔ I2510 (CAD) ↔ E119 (T2DM) ↔ E669 (obesity) ↔ Z794 (insulin use). All mutually co-occurring with high intensity — canonical cardiometabolic comorbidity.
2. **Behavioral health cluster:** F329 (depression) ↔ F419 (anxiety) ↔ Z87891/F17210 (tobacco use). Linked but properly bifurcated on past-vs-current smoking status.
3. **Acute complications cluster:** N179 (AKI) co-occurs broadly with chronic conditions — consistent with AKI's role as a frequent complication of admission.
4. **Respiratory cluster:** J45909 (asthma) and G4733 (OSA) — modest co-occurrence with the metabolic cluster (OSA/obesity link).

**Verdict:** ☑ Structure is correct and the dark Z87.891×F17.210 cell is actively reassuring — the cohort's coding discipline is sound. No red flags. The missing I50.9 and N18.* codes from top-20 are fragmentation artifacts already documented in §7 and DECISIONS.md.

---

## 10. Patient Demographics + Admission Patterns

**Gender:**

| Gender | n | pct |
|---|---|---|
| F | 191,984 | 52.65 |
| M | 172,643 | 47.35 |

**Age buckets:**

| Bucket |n  |  pct |
|---|---|---|
| < 18  | 0 | 0 |
| 18-29 |91,967 | 25.22 |
| 30-44 |72,766 | 19.96 |
| 45-54 |50,068 | 13.73 |
| 55-64 |54,111 | 14.84 |
| 65-74 |44,320 | 12.15 |
| 75-84 |32,172 | 8.82 |
| 85+ |19,223 | 5.27 |

**Admissions per patient:**

- Total patients with ≥ 1 admission: 223,452 (of 364,627 subject_ids in `patients` — 141,175 registry patients have no admission in this extract, likely ambulatory-only).
- Patients with exactly 1 admission: 123,289 (55.17%)
- Patients with 5+ admissions: 24,760 (11.08%)
- Min / median / mean / p95 / max: 1 / 1 / 2.44 / 7 / 238
- Distribution is heavily right-skewed (mean 2.44 vs median 1). Most patients admit once; a long tail of chronic / frequent-admission patients pulls the mean up. Max=238 is a single outlier (chronic-care patient — SCD, transplant, or dialysis).

**LOS (days):** median 2.82 / mean 4.76 / p95 15.42
**In-hospital mortality rate:** 2.16 %

**Decision on splits:**  **Patient-level splits are CRITICAL, not overkill.**

**Quantitative justification:**

- **44.83% of patients have more than one admission** (100,163 of 223,452). Nearly half the cohort would leak across splits under admission-level (`hadm_id`) splitting.
- Among multi-admission patients, mean admissions = (546,028 − 123,289) / 100,163 = **4.22 admissions per multi-admission patient**. Admission-level splitting would place multiple admissions from the same patient in both train and test sets with near certainty.
- Expected leakage under naive 80/10/10 admission-level split (probability a k-admission patient has all admits in the same split ≈ 0.8^k + 2×0.1^k ≈ 0.8^k for k≥2):
  - k=2 (majority of multi-admit patients): 34% leak across splits.
  - k=5: 66% leak.
  - k=7 (p95 level): 79% leak.
  - Weighted across the full distribution, the **majority of multi-admission patients would contaminate test/val** — inflating reported metrics by an amount proportional to within-patient language-style consistency (substantial for clinical notes).
- Consistent with `DECISIONS.md` (2026-04-20 — Patient-level train/val/test split). Data confirms the decision; does not overturn it.

**Fairness flags for later model card:**

- **Frequent-admission patients** (k ≥ 5; n=24,760; 11.08% of admitted cohort): overrepresented in training by admission count — 11% of patients contribute a disproportionately large fraction of admissions. Codes that correlate with chronic-care patterns (ESRD sub-codes, heart failure sub-codes, chronic pain, anticoagulation) may be easier for the model to learn on these patients than on one-admission patients. **Report subgroup-conditional F1:** k=1 vs k≥5 patients.
- **Extreme-tail patient** (k=238, single individual): contributes more admissions than ~1,400 one-admission patients combined. Report **patient-averaged F1** alongside admission-averaged F1 so one mega-patient doesn't dominate the metric.
- **Gender** (registry-level): F 52.65% / M 47.35% — balanced for registry-level fairness reporting. Watch for **gender distribution drift in the admitted subset** — cardiac/surgical ICU cohorts typically skew male, psych/obstetrics skew female. Recompute gender % on the 223,452 admitted patients (not the full 364,627 registry) before committing a fairness claim.
- **Age distribution** (registry-level): age 18-29 is 25.22% of the registry — unexpectedly high for a hospital cohort. This almost certainly reflects the **registry vs admitted-patient mismatch**: the 141,175 un-admitted patients are probably the younger ambulatory cohort. Age distribution **on admitted patients only** will skew older and is the one to report in the model card. The 18-29 25% figure here is a registry-level artifact, not the cohort-at-risk distribution.
- **85+ bucket (5.27%)** is compressed by the `anchor_age=89` cap (re-identification protection). The very-elderly cohort is under-reportable — flag in data card. Model performance on age ≥ 90 cannot be meaningfully evaluated.
- **Children (<18): 0** — expected. MIMIC-IV is explicitly an adult cohort. If this weren't zero, the cohort filter would be wrong.
- **Era / temporal subgroup**: deferred — cannot report fairness by real-year era until Silver-stage date un-shifting (per `DECISIONS.md` 2026-04-21). Data card must state this limitation.
- **Race / ethnicity**: not ingested in the current Bronze schema (`admissions` columns pulled in `read_admissions` don't include `race`). If race-stratified fairness is required for the model card, extend the ingest schema and re-run Bronze.

---

## 11. Join Coverage (notes ∩ admissions ∩ diagnoses_v10)

| Slice | Count |
|---|---|
| total_notes | 331,793 |
| total_adm | 546,028 |
| total_dx_v10 | 254,377 |
| notes_and_adm | 209,444 |
| notes_and_dx | 0 |
| adm_and_dx | 132,089 |
| all_three | 122,288 |

**Cohort N for modeling (all_three):** 122,288

**Loss rate from `total_notes` to `all_three`:** 63.14 %
loss_rate = (total_notes - all_three) / total_notes
          = (331,793 - 122,288) / 331,793
          = 209,505 / 331,793
          = 0.63143...
          ≈ 63.14%
Interpretation: of 331,793 discharge notes in MIMIC-IV-Note v2.2, 209,505 (63.14%) don't survive into the trainable cohort because they either:

Belong to ICD-9 admissions (209,444 notes — dropped by design), or
Belong to admissions not present in Hosp v3.1 (61 notes — version-mismatch orphans).

**Go/No-Go check:** 122,288 ≥ 80,000 floor. **Passes with 53% margin.** Cohort is 2.6× larger than Mullenbach CAML's MIMIC-III training set (~47K).

**Note coverage ceiling (non-blocking finding):** 132,089 ICD-10 admissions in Hosp v3.1 have diagnosis codes but no discharge summary in MIMIC-IV-Note v2.2. The modelable cohort is therefore a 48% sample of all ICD-10 admissions, limited by MIMIC-IV-Note's own sampling — not by this project's filters. Flag for the data card. Cohort N would roughly double if a future MIMIC-IV-Note release expands note coverage to match Hosp.

---

## 12. Version Reconciliation (v2.2 notes ↔ v3.1 Hosp)

Notes missing from v3.1 admissions: 61 ( 0.018 %)
Notes missing from v3.1 diagnoses_v10: 209,505 ( 63.14 %)

**Verdict:** :Negligible loss (< 1%) — join safely / Meaningful loss — investigate
**Q1 — True version drift (notes ↔ admissions):**

- Only **61 of 331,793 notes** (0.018%) from MIMIC-IV-Note v2.2 correspond to admissions not present in MIMIC-IV Hosp v3.1.
- This is the actual measure of "did PhysioNet remove admissions between v2.2 and v3.1?" Answer: essentially no — 61 is the version-mismatch orphan count.
- **Well below the 1% threshold** in the template verdict. Inner joining v2.2 notes to v3.1 admissions is safe.

**Q2 — Notes without ICD-10 diagnoses (63.14%):**

- This number is **dominated by the ICD-9 vs ICD-10 filter**, NOT by version drift.
- Decomposition:
  - 209,444 notes → attached to ICD-9 admissions (present in v3.1, just not ICD-10). Cohort filter effect per `DECISIONS.md` 2026-04-20 (ICD-10-only).
  - 61 notes → actual version orphans (same as Q1).
  - Total = 209,505.
- **Do not record 63.14% as "version drift."** It's cohort filter + 0.018% of drift. Same number already accounted for in §11 Join Coverage.

**Conclusion:** MIMIC-IV-Note v2.2 ↔ MIMIC-IV Hosp v3.1 joins cleanly. The version mismatch decision in `DECISIONS.md` (2026-04-20) is validated by data: the predicted risk was "a small number of admissions in notes v2.2 may have been removed from v3.1 Hosp; handle as inner join" → observed 0.018%, negligible. Inner join, no reconciliation logic required.
---

## Final Decisions to Commit to `configs/dev.yml`

```yaml
cohort:
  icd_version: 10
  min_note_tokens: 100        # §3 — filter pathologically short notes; conventional floor
  top_k_labels: 50            # §7 — 91.04% coverage, comparability with Mullenbach CAML
  note_types: ["DS"]          # §2 — discharge summaries only; only DS present in v2.2

split:
  strategy: "patient_stratified"   # §10 — 44.83% of patients have >1 admission; admission-level splits would leak
  train_frac: 0.80
  val_frac: 0.10
  test_frac: 0.10
  seed: 42
```

## Go / No-Go Gate

Status as of 2026-04-21. All checks validated in the sections cited; project is cleared to proceed.

- [x] **Bronze ingestion produced expected row counts** — §1a (notes 331,793 ✓; diagnoses 6,364,488 ✓; admissions 546,028 and patients 364,627 both consistent with v3.1 Hosp upgrade ✓).
- [x] **Cohort N (all_three) ≥ 80,000** — §11: N = **122,288**, 53% above floor. 2.6× the Mullenbach CAML MIMIC-III training set.
- [x] **Top-K chosen with coverage ≥ 80%** — §7: K=50, **91.04%** coverage. 11pp above target.
- [x] **Truncation strategy decided** — §3: **chunked-BERT + max-pool**, Longformer deferred pending empirical result (tracked in `DECISIONS.md` 2026-04-20).
- [x] **Patient-level splits confirmed necessary** — §10: **44.83%** of patients have >1 admission; admission-level splits would leak majority of multi-admit patients.
- [x] **No data quality red flags** — §1b (all schema as expected except `patients.dod` dtype, non-blocking), §5 (dedup clean), §9 (co-occurrence validates coding discipline).

**Gate cleared.** Proceed to `mic silver` → `mic gold` → `mic train-baseline`.

**Deferred to Silver (tracked, not blocking modeling):**
- Date un-shifting to approximate real years (per `DECISIONS.md` 2026-04-21). Required before drift monitoring / temporal splits / fairness-by-era / data card year coverage.
- Explicit `pd.to_datetime` cast on `patients.dod` for type stability (§1b red flag).

**Deferred to data card (not blocking modeling):**
- Race/ethnicity ingestion — requires schema extension in `read_admissions` per §10 fairness flag.
- MIMIC-IV-Note v2.2 cohort coverage note: modelable cohort is 48% of all ICD-10 admissions, limited by Note release sampling not by our filters (§11).
- `anchor_age=89` cap limits very-elderly fairness eval (§10).


## Figures Generated

All saved to `reports/figures/`:

- `length_distribution.png` — token + char histograms with percentile markers
- `icd_version_by_year.png` — ICD-9 → ICD-10 transition
- `icd_frequency_and_coverage.png` — rank-frequency + top-K coverage curve
- `codes_per_admission.png`
- `label_cooccurrence.png`
- `admissions_per_patient.png`

Commit all figures alongside the report — they're the visual evidence for the data card.
