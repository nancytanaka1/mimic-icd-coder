# Model Card — mimic-icd-coder

Template: Mitchell et al. 2019.

*Fill performance numbers after baseline and transformer training runs complete.*

## Model Details

- **Model name:** mimic-icd-coder / discharge_top50
- **Version:** 0.1.0
- **Base model:** Bio_ClinicalBERT (`emilyalsentzer/Bio_ClinicalBERT`)
- **Task:** Multi-label classification of top-50 ICD-10 codes from discharge summaries
- **Architecture:** BERT encoder + linear → sigmoid multi-label head. Long notes are processed via **chunk-and-max-pool**: each note is split into contiguous 512-token chunks, each chunk runs through the encoder, and per-label logits are max-pooled across chunks.
- **Training framework:** HuggingFace Transformers 4.40+, PyTorch 2.2+, fp16.
- **Compute:** Azure Databricks, single-node `Standard_NC6s_v3` (1× V100).
- **Owner:** Nancy Tanaka
- **License:** Apache 2.0 for code. Weights are trained on PhysioNet-credentialed data and are **not redistributable**.

## Intended Use

- **Primary use:** Production-grade benchmark for clinical NLP ICD-10 auto-coding on MIMIC-IV, deployed as an Azure Databricks Model Serving endpoint for reproducible inference. Demonstrates the full MLOps lifecycle — feature engineering, model validation against a published benchmark (Mullenbach et al. 2018 CAML), Unity Catalog Model Registry, drift monitoring.
- **Out of scope:** Clinical decision support, real-world coding workflows, any production health-system deployment. The MIMIC-IV license restricts use to research and benchmarking on credentialed PhysioNet data only.
- **Users:** Data scientists and ML engineers evaluating production clinical NLP patterns; research teams benchmarking against this implementation on MIMIC-IV top-50 ICD-10.

## Training Data

Cohort definition, composition, and limitations are documented in [`reports/data_card.md`](data_card.md). Model-specific details:

- Modelable cohort: **122,288 admissions**.
- Splits: patient-level 80/10/10 by `subject_id`, seed = 42.
- Labels: top-50 ICD-10 codes, covering 91.04% of cohort admissions.

## Evaluation Data

- **Test split:** Held-out 10% of patients, no overlap with train or val.
- **Metrics:** Micro F1, Macro F1, P@5, P@8, per-label F1.
- **Reference baseline:** Mullenbach et al. 2018 CAML on MIMIC-III top-50 ICD-9 — Table 5 of the paper. Micro F1 = 0.614, Macro F1 = 0.532, P@5 = 0.609. P@8 is not reported for the top-50 setting.

Full evaluation methodology and comparison caveats are in [`reports/eval_report.qmd`](eval_report.qmd).

## Performance

| Metric | Target | Floor | Mullenbach 2018 CAML (MIMIC-III top-50) | Target Δ vs. CAML | Result |
|---|---|---|---|---|---|
| Micro F1 | ≥ 0.70 | 0.55 | 0.614 | +0.086 | **0.617** (baseline) / TBD (transformer) |
| Macro F1 | ≥ 0.55 | 0.40 | 0.532 | +0.018 | **0.584** (baseline) / TBD (transformer) |
| P@5 | ≥ 0.70 | — | 0.609 | +0.091 | 0.526 (baseline) / TBD (transformer) |
| P@8 | ≥ 0.65 | — | n/a (Mullenbach Table 5 reports P@5 only) | — | 0.433 (baseline) / TBD (transformer) |

Baseline = TF-IDF (1–2 gram, min_df=5, max_features=200k) + One-vs-Rest Logistic Regression (`class_weight="balanced"`, liblinear solver, max_iter=1000) with per-label F1-optimal thresholds. Evaluated on held-out patient-level test split, n=12,091 admissions, seed=42. MLflow run `4e577699a67a4027bc27628e9b237ac5`. Baseline beats Mullenbach CAML on both F1 metrics; P@k is intentionally de-emphasized for the baseline and is the transformer branch's primary gate — see [`DECISIONS.md`](../DECISIONS.md) 2026-04-23.

CAML values from Mullenbach et al. 2018 Table 5 (MIMIC-III, 50 labels). See [`src/mimic_icd_coder/evaluate.py::MULLENBACH_CAML_TOP50`](../src/mimic_icd_coder/evaluate.py) for the citation.

## Fairness Analysis

Macro F1 reported separately for:

- **Gender** (F vs. M)
- **Age bucket** (18-29, 30-44, 45-54, 55-64, 65-74, 75-84, 85+)
- **Admission frequency** (single-admit vs. 5+ admits)

Any subgroup disparity of 5 percentage points or more is flagged for explicit discussion.

**Deferred analyses:**

- **Race and ethnicity** — not in the current Bronze schema; requires schema extension before inclusion.
- **Temporal era** — requires Silver-stage date un-shifting before admissions can be grouped by real calendar year. See [`DECISIONS.md`](../DECISIONS.md) entry dated 2026-04-21.

## Ethical Considerations

- **Single institution.** All training data from Beth Israel Deaconess Medical Center. Performance on other populations is untested.
- **Billing codes are not clinical truth.** ICD-10 assignments are coder decisions made for billing, with known comorbidity underreporting. The model learns this noise.
- **Age cap.** `anchor_age` is capped at 89 in MIMIC; evaluation on patients aged 89 and older cannot be fully disaggregated.
- **No clinical use.** Outputs are predictions, not diagnoses. Do not deploy in any setting where the output influences a care decision.

## Caveats and Limitations

- **Cohort ceiling.** Modelable population is 48% of all ICD-10 admissions in v3.1 Hosp, limited by MIMIC-IV-Note v2.2 note sampling.
- **Note length handled by chunking.** Single-window BERT would truncate 98.74% of notes; the chunk-and-max-pool architecture recovers this content. Chunked coverage reaches ~3,072 BPE tokens per note over six chunks.
- **Longformer fallback.** If chunked Bio_ClinicalBERT misses the Micro F1 target by more than 3 points, re-evaluate against Clinical-Longformer (4K tokens). Trigger-driven, not speculative.
- **ICD-10 sub-code fragmentation.** Conditions such as CHF, CKD, COPD, and T2DM distribute across multiple sub-codes in ICD-10 but were single top-10 codes in ICD-9. Top-50 label space is not equivalent to Mullenbach's MIMIC-III top-50.
- **Long-tail codes excluded.** Rare codes outside the top 50 are out of scope. Production use would require a hierarchical or retrieval-based extension.
- **External generalization untested.** Notes from other institutions may use different formatting, abbreviations, and documentation conventions.
