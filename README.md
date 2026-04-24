# mimic-icd-coder

**Multi-label ICD-10 auto-coder for hospital discharge summaries — a production MLOps benchmark study.**

End-to-end clinical NLP pipeline deployed on Azure Databricks (Delta Lake + Unity Catalog + MLflow + Model Serving), benchmarked against Mullenbach et al. 2018 (CAML) on MIMIC-IV top-50 ICD-10. Reproducible on a single workstation or in the cloud without code branches; every methodological choice is pre-registered in [`DECISIONS.md`](DECISIONS.md) and defended in [`reports/EDA_Report.docx`](reports/EDA_Report.docx).

**Headline result** (baseline complete; transformer pending):

| Metric | Baseline (TF-IDF + LR) | Chunked Bio_ClinicalBERT | Mullenbach 2018 CAML (MIMIC-III top-50) |
|---|---|---|---|
| Micro F1 | **0.617** (+0.003 vs. CAML) | TBD | 0.614 |
| Macro F1 | **0.584** (+0.052 vs. CAML) | TBD | 0.532 |
| P@5 | 0.526 (−0.083 vs. CAML) | TBD | 0.609 |
| P@8 | 0.433 | TBD | n/a (not reported for top-50) |

Source: Mullenbach et al. 2018 Table 5. Baseline evaluated on held-out patient-level test split (n=12,091 admissions, seed=42). Val→test drift <0.01 on every metric, confirming val-tuned thresholds generalize. Leakage prevention is architectural, not inferred from the drift: train/val/test are disjoint by `subject_id`, verified in [`tests/test_smoke.py::test_patient_split_disjoint`](tests/test_smoke.py). Baseline uses `class_weight="balanced"` + per-label F1-optimal thresholds, which trades ranking calibration (P@k) for per-label F1 — a deliberate baseline choice. P@k reclaim is the transformer branch's objective. See [`DECISIONS.md`](DECISIONS.md) 2026-04-23.

**Reproducibility.** `mic run-all --config configs/dev.nancy.yml` on fresh MIMIC-IV v3.1 Hosp + MIMIC-IV-Note v2.2 raw CSVs reproduces every headline metric to **15+ decimal places** (verified 2026-04-24, MLflow run `6e809d5dfd3b46dbafae84ddba710bd7`). Reproducibility comes from a fixed patient-split seed (42), a deterministic liblinear solver, and file-on-disk stage boundaries that make each step independently re-runnable. A reviewer with PhysioNet credentials cloning this repo will get identical numbers end-to-end.

For the full data card, model card, EDA paper, and evaluation methodology, see [`reports/`](reports/).
For AI-assistance disclosure, see [`ACKNOWLEDGMENTS.md`](ACKNOWLEDGMENTS.md).

---

## 1. Study & deployment summary

| Attribute | Value |
|---|---|
| **Domain** | Clinical NLP — automated medical coding |
| **Input** | Free-text discharge summaries (MIMIC-IV-Note v2.2) |
| **Output** | Per-code probability and thresholded binary labels over top-50 ICD-10 codes |
| **Training cohort** | **122,288 admissions** (MIMIC-IV v3.1 ICD-10 cohort ∩ v2.2 notes) |
| **Top-50 coverage** | 91.04% of cohort admissions |
| **License — code** | Apache-2.0 |
| **License — data** | PhysioNet Credentialed Health Data License v1.5.0 (not redistributed) |

### Metric targets (top-50 ICD-10, patient-level test split)

| Metric | Target | Floor | Mullenbach 2018 CAML (MIMIC-III top-50) | Target Δ vs. CAML |
|---|---|---|---|---|
| Micro F1 | ≥ 0.70 | 0.55 | 0.614 | +0.086 |
| Macro F1 | ≥ 0.55 | 0.40 | 0.532 | +0.018 |
| P@5 | ≥ 0.70 | — | 0.609 | +0.091 |
| P@8 | ≥ 0.65 | — | n/a (Mullenbach Table 5 reports P@5 only) | — |

Targets are for chunked Bio_ClinicalBERT. The TF-IDF+LR baseline is expected to clear Micro F1 ≥ 0.55. Below that, something upstream is broken — cohort filter, split leakage, or label misalignment.

CAML baseline values are from Mullenbach et al. 2018 Table 5 (MIMIC-III, 50 labels). See [`src/mimic_icd_coder/evaluate.py::MULLENBACH_CAML_TOP50`](src/mimic_icd_coder/evaluate.py) for the citation.

---

## 2. System architecture

### 2.1 Logical topology

```
                         Raw MIMIC-IV (PhysioNet)
                         discharge.csv.gz
                         diagnoses_icd.csv.gz
                         admissions.csv.gz
                         patients.csv.gz
                         d_icd_diagnoses.csv.gz
                                 │
                                 ▼
     ┌────────────── Bronze ─────────────┐     Raw mirror (Parquet / Delta)
     │  gz CSV → columnar; no transforms │
     └────────────────┬──────────────────┘
                      ▼
     ┌────────────── Silver ─────────────┐     Cleaned notes, one per hadm_id
     │  de-id collapse, dedup, min 100tk │
     └────────────────┬──────────────────┘
                      ▼
     ┌────────────── Gold ───────────────┐     Model-ready artifacts
     │  top-50 ICD-10 multi-hot matrix   │     labels.npz, label_names.json,
     │  patient-level split manifest     │     splits.parquet
     └────────────────┬──────────────────┘
                      ▼
     ┌────────── Training / Eval ────────┐     MLflow tracking + Model Registry
     │  • TF-IDF + LogReg (baseline)     │     per-label threshold tuning
     │  • Chunked Bio_ClinicalBERT       │
     │  • Clinical-Longformer (fallback) │
     └────────────────┬──────────────────┘
                      ▼
     ┌────────── Serving + Monitoring ───┐     Databricks Model Serving (GPU)
     │  FastAPI-compatible scoring API   │     Evidently drift checks
     └───────────────────────────────────┘
```

### 2.2 Deployment surfaces

The same pipeline runs in two environments with no code branching. Only config paths change.

| Surface | Storage | Compute | Orchestration | Tracking | Use |
|---|---|---|---|---|---|
| **Local workstation** | Parquet on `E:\` | CPU (16 threads) | `mic` CLI | File-backed MLflow | Cohort construction, EDA, baseline iteration, tests |
| **Azure Databricks** | ADLS Gen2 + Delta | CPU + GPU job clusters (NC6s_v3 for transformer) | Databricks Asset Bundles | Managed MLflow + Unity Catalog Model Registry | Transformer fine-tune, Model Serving, drift monitoring |

---

## 3. Data contracts

Full cohort composition and preprocessing logic live in [`reports/data_card.md`](reports/data_card.md). Quick reference:

### 3.1 Inputs

| Source | Version | Key fields | Notes |
|---|---|---|---|
| `mimic-iv-note/note/discharge.csv.gz` | v2.2 (Jan 2023) | `note_id, subject_id, hadm_id, note_type, note_seq, charttime, text` | 331,793 rows; `note_type = 'DS'` is the only value |
| `mimic-iv/hosp/diagnoses_icd.csv.gz` | v3.1 (Oct 2024) | `subject_id, hadm_id, seq_num, icd_code, icd_version` | 6,364,488 rows; `icd_version ∈ {9, 10}` |
| `mimic-iv/hosp/admissions.csv.gz` | v3.1 | `subject_id, hadm_id, admittime, dischtime, ...` | 546,028 rows |
| `mimic-iv/hosp/patients.csv.gz` | v3.1 | `subject_id, gender, anchor_age, ...` | 364,627 rows |
| `mimic-iv/hosp/d_icd_diagnoses.csv.gz` | v3.1 | `icd_code, icd_version, long_title` | ICD dictionary for human-readable descriptions |

The v2.2/v3.1 mismatch is deliberate. `hadm_id` is stable across versions; only 61 of 331,793 notes (0.018%) are orphaned. Full rationale in [`DECISIONS.md`](DECISIONS.md) (2026-04-20).

### 3.2 Stage outputs

| Stage | Artifact | Shape | Contract |
|---|---|---|---|
| Bronze | `bronze/{discharge_notes,diagnoses_icd,admissions,patients,d_icd_diagnoses}.parquet` | source schema | Lossless columnar mirror |
| Silver | `silver/notes.parquet` | `hadm_id, subject_id, text, n_tokens` | One row per admission; `n_tokens ≥ 100` |
| Gold | `gold/labels.npz` | CSR `(n_admissions, 50)` | Rows aligned 1:1 to `silver/notes.parquet` |
| Gold | `gold/label_names.json` | list[str] length 50 | ICD-10 codes in column order |
| Gold | `gold/hadm_ids.parquet` | `hadm_id` | Row-to-hadm_id lookup |
| Gold | `gold/splits.parquet` | `row_idx, split` | Patient-level 80/10/10; no `subject_id` spans splits |
| Gold | `gold/baseline_model.joblib` | vectorizer + 50 LR heads | Output of `fit_baseline` |
| Gold | `gold/baseline_thresholds.npy` | `float64[50]` | Per-label thresholds tuned on val |

**Alignment invariant** (`pipeline.py`): `len(silver) == labels.shape[0]`. Violation means Gold must be rebuilt.

### 3.3 Cohort rules

Defined in `configs/*.yml` under `cohort:`.

| Rule | Default | Rationale |
|---|---|---|
| `icd_version` | `10` | ICD-10 is operationally current; mixing fragments the label space |
| `note_types` | `['DS']` | Discharge summaries only; v2.2 contains only DS |
| `min_note_tokens` | `100` | Drops near-empty notes that hurt baseline precision |
| `top_k_labels` | `50` | Direct comparability to Mullenbach et al. 2018 |

---

## 4. Pipeline stages

Each stage is an idempotent function in [`src/mimic_icd_coder/pipeline.py`](src/mimic_icd_coder/pipeline.py) with a Parquet or npz checkpoint. Downstream stages read from disk, so any step can be re-run without recomputing upstream.

| Stage | Entry point | Reads | Writes | Runtime (laptop) |
|---|---|---|---|---|
| Bronze | `mic ingest` | 5 gz CSVs | 5 Parquet mirrors | 5–10 min |
| Silver | `mic silver` | `bronze/discharge_notes.parquet` | `silver/notes.parquet` | 2–3 min |
| Gold | `mic gold` | Silver + `bronze/diagnoses_icd.parquet` | `labels.npz`, `label_names.json`, `hadm_ids.parquet` | ~30 s |
| Splits | `mic splits` | Silver | `splits.parquet` | < 10 s |
| Baseline train | `mic train-baseline` | Silver + Gold + Splits | `baseline_model.joblib`, `baseline_thresholds.npy`, MLflow run | 15–25 min (CPU, 16 threads) |
| Test eval | `mic evaluate-test` | Silver + Gold + Splits + saved model | Test metrics | ~1 min |
| Run-all | `mic run-all` | raw gz | everything | 25–40 min end-to-end |

Checkpoint layout, rooted at `Paths.root` (default `./data`):

```
data/
  bronze/   discharge_notes.parquet  diagnoses_icd.parquet  admissions.parquet
            patients.parquet         d_icd_diagnoses.parquet
  silver/   notes.parquet
  gold/     labels.npz  label_names.json  hadm_ids.parquet  splits.parquet
            baseline_model.joblib  baseline_thresholds.npy
  mlruns/   <MLflow experiment tree>
```

---

## 5. Models

Full details, architecture rationale, and ethics in [`reports/model_card.md`](reports/model_card.md).

### 5.1 Baseline — TF-IDF + one-vs-rest LogisticRegression

[`src/mimic_icd_coder/models/baseline.py`](src/mimic_icd_coder/models/baseline.py)

| Parameter | Default | Config key |
|---|---|---|
| n-gram range | (1, 2) | `baseline.tfidf_ngram_range` |
| min doc freq | 5 | `baseline.tfidf_min_df` |
| max features | 200,000 | `baseline.tfidf_max_features` |
| LR C | 1.0 | `baseline.logreg_c` |
| class_weight | `balanced` | `baseline.logreg_class_weight` |

Per-label decision thresholds are tuned on the validation split by maximizing per-label F1 ([`src/mimic_icd_coder/thresholds.py`](src/mimic_icd_coder/thresholds.py)).

### 5.2 Transformer — Chunked Bio_ClinicalBERT (primary)

[`src/mimic_icd_coder/models/transformer.py`](src/mimic_icd_coder/models/transformer.py), [`jobs/train_transformer.py`](jobs/train_transformer.py)

Each note is split into contiguous 512-token chunks. Each chunk runs through the BERT encoder. Per-label logits are max-pooled across chunks. This recovers the signal that a single 512-token window would lose — 98.74% of notes exceed 512 whitespace tokens.

| Parameter | Default | Config key |
|---|---|---|
| model | `emilyalsentzer/Bio_ClinicalBERT` | `transformer.model_name` |
| max sequence length per chunk | 512 | `transformer.max_length` |
| batch size | 16 | `transformer.batch_size` |
| learning rate | 2e-5 | `transformer.learning_rate` |
| epochs | 3 | `transformer.epochs` |
| warmup ratio | 0.1 | `transformer.warmup_ratio` |
| weight decay | 0.01 | `transformer.weight_decay` |
| fp16 | true | `transformer.fp16` |

Early stop on validation macro F1.

### 5.3 Fallback — Clinical-Longformer

Triggered only if chunked Bio_ClinicalBERT misses the Micro F1 target by more than 3 points. 4K-token context; ~3–5× slower training. Rationale in [`DECISIONS.md`](DECISIONS.md) (2026-04-20).

---

## 6. Evaluation

Full methodology and Mullenbach comparison caveats in [`reports/eval_report.qmd`](reports/eval_report.qmd).

### Test-split results

Held-out patient-level test split, n=12,091 admissions, 6,567 patients. Seed 42. MLflow run `4e577699a67a4027bc27628e9b237ac5`.

| Metric | Value | Target | Mullenbach CAML (MIMIC-III-50) |
|---|---:|---:|---|
| Micro F1 | 0.6174 | ≥ 0.70 | 0.614 (+0.003) |
| Macro F1 | 0.5843 | ≥ 0.55 | 0.532 (+0.052) |
| P@5 | 0.5259 | ≥ 0.70 | 0.609 (−0.083) |
| P@8 | 0.4326 | ≥ 0.65 | n/a (not reported for top-50) |
| P@15 | 0.2935 | — | — |
| Micro AUC | 0.9284 | — | — |
| Macro AUC | 0.9097 | — | — |
| Micro AUPRC | 0.6263 | — | — |
| Macro AUPRC | 0.5739 | — | — |

Model: `sklearn.OneVsRestClassifier(LogisticRegression(class_weight="balanced"))` over TF-IDF (1–2 grams, `min_df=5`, 200 K vocab cap). Per-label thresholds tuned on val via F1 maximization. Val→test drift ≤ 0.005 on every metric, confirming val-tuned thresholds generalize. Train/val/test are disjoint by `subject_id` (verified in [`tests/test_smoke.py::test_patient_split_disjoint`](tests/test_smoke.py)) — this prevents the patient-writing-style leakage an admission-level split would allow.

### Metrics used

| Metric | Use |
|---|---|
| Micro F1 | Primary operational metric — stable under class imbalance |
| Macro F1 | Rare-label performance across all 50 codes, equally weighted |
| P@5 / P@8 / P@15 | Ranked-prediction precision for coder-assist workflow |
| Per-label F1 | Error analysis on worst-performing labels |

Mullenbach CAML deltas are computed in `compare_to_mullenbach` and logged as MLflow metrics.

---

## 7. Interfaces

### 7.1 Local CLI

Entry points registered in [`pyproject.toml`](pyproject.toml), implemented in [`src/mimic_icd_coder/cli.py`](src/mimic_icd_coder/cli.py).

```bash
mic ingest          --config configs/dev.yml
mic silver          --config configs/dev.yml
mic gold            --config configs/dev.yml
mic splits          --config configs/dev.yml
mic train-baseline  --config configs/dev.yml
mic evaluate-test   --config configs/dev.yml
mic run-all         --config configs/dev.yml
```

`configs/dev.yml` is gitignored; copy [`configs/dev.example.yml`](configs/dev.example.yml) and fill in your MIMIC paths. `--artifacts <dir>` overrides the default `./data` checkpoint root.

### 7.2 Databricks Asset Bundle

[`databricks.yml`](databricks.yml). Two targets:

| Target | Catalog | Run-as | Compute |
|---|---|---|---|
| `dev` | `mimic_icd_dev` | workspace user | `Standard_DS4_v2` × 2 (Bronze), `Standard_DS5_v2` × 2 (baseline) |
| `prod` | `mimic_icd` | service principal `mimic-icd-sp` | same + `Standard_NC6s_v3` single-node (1× V100) for transformer |

```bash
databricks bundle validate --target dev
databricks bundle deploy   --target dev
databricks bundle run ingest_bronze     --target dev
databricks bundle run train_baseline    --target dev
databricks bundle run train_transformer --target prod
```

### 7.3 Model Serving API

Databricks Model Serving endpoint, GPU-backed.

```json
POST /serving-endpoints/mimic-icd-discharge/invocations
{
  "dataframe_records": [
    {"text": "<discharge summary text>"}
  ]
}

Response:
{
  "predictions": [
    {
      "codes":        ["I10", "I50.9", "N18.6", "E11.9", ...],
      "scores":       [0.94, 0.87, 0.72, 0.68, ...],
      "thresholded":  ["I10", "I50.9"]
    }
  ]
}
```

---

## 8. Configuration

Template: [`configs/dev.example.yml`](configs/dev.example.yml). User overrides go in `configs/dev.yml` or `configs/dev.<username>.yml`, both gitignored. Schema validated by Pydantic `AppConfig` in [`src/mimic_icd_coder/config.py`](src/mimic_icd_coder/config.py).

| Section | Purpose |
|---|---|
| `unity_catalog` | Catalog + schema names for Bronze / Silver / Gold / Models |
| `data` | Input paths (local gz or ADLS `abfss://`), including `d_icd_path` |
| `cohort` | Cohort filters (see §3.3) |
| `split` | Train/val/test fractions, seed, strategy |
| `baseline` | TF-IDF + LR hyperparameters |
| `transformer` | Bio_ClinicalBERT hyperparameters |
| `evaluation` | Threshold strategy, top-k metric list |
| `mlflow` | Experiment name, registry model name |
| `logging` | Level + format (console or JSON) |

---

## 9. Observability

| Channel | Backing | Captured |
|---|---|---|
| Structured logs | `structlog` — console locally, JSON on Databricks | Stage start/end, row counts, label density, metric values |
| MLflow runs | Local file store (`data/mlruns`) or Databricks-managed | Params, metrics, model artifact, signature, thresholds, label list |
| Model Registry | Unity Catalog (`mimic_icd.models.discharge_top50`) | Staging / Production aliases; train-data fingerprint and git SHA tags |
| Drift monitoring | Evidently scheduled job (prod only) | Input distribution, prediction, and label drift |

---

## 10. Security & compliance

Full details in [`reports/data_card.md`](reports/data_card.md). Headlines:

- MIMIC-IV data is credentialed under the PhysioNet Health Data License v1.5.0.
- `.gitignore` blocks CSV, Parquet, gz, npz, joblib, and user-specific configs. No raw data enters this repository.
- Training runs in the user's own Azure tenant (single-tenant Databricks workspace, private ADLS Gen2).
- Clinical text must not be sent to third-party LLM APIs. Only open-weights models hosted inside the workspace are used.
- CI runs only on synthetic fixtures in [`tests/fixtures/synthetic_notes.py`](tests/fixtures/synthetic_notes.py).
- Notebook outputs are PHI-scanned by [`scripts/check_notebook_phi.py`](scripts/check_notebook_phi.py) in CI and pre-commit.
- Service-principal credentials are stored in Databricks secret scopes.

---

## 11. Quality gates

| Gate | Tool | Enforced in |
|---|---|---|
| Lint | `ruff check` | Pre-commit + CI |
| Format | `black --check` (line length 100) | Pre-commit + CI |
| Types | `mypy src` (strict) | CI |
| Unit + integration tests | `pytest` (28 tests on synthetic fixtures, ~5 s) | CI + local |
| Notebook output hygiene | `nbstripout` + PHI scanner | Pre-commit + CI |
| Data-safety guards | Large-file check, private-key detection | Pre-commit |
| Bundle validity | `databricks bundle validate --target dev` | Pre-deploy |
| Metric floor | Baseline Micro F1 ≥ 0.55 on dev split | Manual review gate after `mic train-baseline` |

Pre-commit config: [`.pre-commit-config.yaml`](.pre-commit-config.yaml). CI workflow: [`.github/workflows/ci.yml`](.github/workflows/ci.yml).

---

## 12. Quick start

### 12.1 Local (no credentialed data required)

```bash
git clone git@github.com:nancytanaka1/mimic-icd-coder.git
cd mimic-icd-coder

python -m venv .venv
# Windows: .\.venv\Scripts\activate    POSIX: source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install

pytest -q                     # 28 tests, synthetic fixtures
```

### 12.2 Local end-to-end on real MIMIC (requires PhysioNet credentials)

See [`LOCAL_SETUP.md`](LOCAL_SETUP.md) for the workstation walkthrough (memory profile, expected row counts, runtime envelopes, GPU prerequisites).

```bash
cp configs/dev.example.yml configs/dev.yml        # then edit data paths
mic run-all --config configs/dev.yml
mlflow ui --backend-store-uri file:./data/mlruns --port 5000
```

### 12.3 Databricks

```bash
pip install databricks-cli
databricks configure --token
databricks bundle validate --target dev
databricks bundle deploy   --target dev
databricks bundle run ingest_bronze --target dev
```

---

## 13. Repository layout

```
mimic-icd-coder/
├── src/mimic_icd_coder/
│   ├── cli.py                CLI entry points (mic ...)
│   ├── config.py             Pydantic AppConfig
│   ├── pipeline.py           Stage orchestration + Paths
│   ├── logging_utils.py      structlog configuration
│   ├── eda.py                EDA analysis helpers (used by notebook)
│   ├── evaluate.py           Metrics + Mullenbach comparison
│   ├── thresholds.py         Per-label threshold tuner
│   ├── data/
│   │   ├── ingest.py         gz CSV → DataFrame readers (pyarrow CSV engine)
│   │   ├── clean.py          Silver transforms
│   │   ├── labels.py         Top-K multi-hot label builder
│   │   └── splits.py         Patient-level splitter
│   └── models/
│       ├── baseline.py       TF-IDF + LogReg + MLflow logger
│       └── transformer.py    Chunked Bio_ClinicalBERT fine-tune wrapper
├── jobs/                     Databricks-entry-point scripts
│   ├── bronze.py
│   └── train_baseline.py
├── notebooks/
│   └── 01_eda.ipynb          Cohort + label distribution EDA
├── scripts/
│   └── check_notebook_phi.py PHI scanner (CI + pre-commit)
├── configs/
│   ├── dev.example.yml       Template (checked in)
│   └── dev.*.yml             User-specific configs (gitignored)
├── tests/
│   ├── test_smoke.py         Import + config smoke
│   ├── test_eda.py           EDA helpers
│   ├── test_pipeline_integration.py  End-to-end on synthetic fixtures
│   └── fixtures/synthetic_notes.py   Synthetic MIMIC-shaped generator
├── reports/
│   ├── data_card.md          Dataset provenance, composition, DUA
│   ├── model_card.md         Model architecture, intended use, ethics
│   ├── eval_report.qmd       Evaluation methodology + metrics
│   ├── eda_report.md         Internal EDA audit
│   └── EDA_Report.docx       Academic paper (IMRaD, Literature Review, citations)
├── .github/workflows/ci.yml  Lint, type, test, PHI scan
├── .pre-commit-config.yaml   Pre-commit hooks
├── databricks.yml            Asset Bundle definition
├── pyproject.toml            Build, tools, console scripts
├── DECISIONS.md              Architectural decision log
├── LOCAL_SETUP.md            Workstation setup playbook
├── ACKNOWLEDGMENTS.md        AI-assistance disclosure
├── LICENSE                   Apache 2.0 (code)
└── README.md                 (this file)
```

---

## 14. Implementation status

| Component | Status |
|---|---|
| Scaffold, CI, pre-commit, Asset Bundle | Ready |
| EDA notebook + paper + data card + model card + eval report | Complete |
| Bronze ingestion (5 tables including ICD dictionary) | Implemented and run on real data |
| Silver (clean + min-token filter) | Implemented; not yet run end-to-end |
| Gold (top-50 label matrix + patient splits) | Implemented; not yet run end-to-end |
| TF-IDF + LR baseline | Implemented; not yet run |
| Per-label threshold tuning | Implemented |
| Evaluation (Micro/Macro F1, P@k, Mullenbach deltas) | Implemented |
| MLflow + Unity Catalog Model Registry | Local MLflow wired; Registry write on Databricks only |
| Chunked Bio_ClinicalBERT fine-tune | Scaffolded — [`jobs/train_transformer.py`](jobs/train_transformer.py) |
| Clinical-Longformer fallback | Not started — trigger-driven |
| Model Serving endpoint | Not started |
| Evidently drift monitoring | Not started |

---

## 15. References

- Mullenbach, Wiegreffe, Duke, Sun, Eisenstein (2018). *Explainable Prediction of Medical Codes from Clinical Text.* NAACL. https://arxiv.org/abs/1802.05695
- Alsentzer, Murphy, Boag, Weng, Jin, Naumann, McDermott (2019). *Publicly Available Clinical BERT Embeddings.* ClinicalNLP Workshop. https://arxiv.org/abs/1904.03323
- Beltagy, Peters, Cohan (2020). *Longformer: The Long-Document Transformer.* https://arxiv.org/abs/2004.05150
- Devlin, Chang, Lee, Toutanova (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.* NAACL.
- Johnson et al. (2023). *MIMIC-IV-Note: Deidentified free-text clinical notes.* PhysioNet.
- Mitchell et al. (2019). *Model Cards for Model Reporting.* FAccT.
- Pushkarna, Zaldivar, Kjartansson (2022). *Data Cards: Purposeful and Transparent Dataset Documentation.* FAccT.

---

## 16. License

Code licensed under **Apache-2.0** ([LICENSE](LICENSE)). MIMIC data is licensed separately under the **PhysioNet Credentialed Health Data License v1.5.0** and is not redistributed via this repository.
