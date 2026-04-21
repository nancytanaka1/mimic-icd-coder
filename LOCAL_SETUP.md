# Local Setup — Dell Precision 5560

Step-by-step for Nancy's exact laptop: 32 GB RAM, i7-11800H (16 threads), NVIDIA T1200 (4 GB), 3.7 TB free on E:.

## 0. Prerequisites

- Python 3.11 (Windows installer from python.org, *not* the Microsoft Store version)
- Git for Windows
- PowerShell 7+ recommended
- ~10 GB free disk for venv + Parquet intermediates (you have 3.7 TB on E:, no concern)

## 1. First-time setup (10 min)

```powershell
# Clone or copy the scaffold
cd E:\NU
# (scaffold is already at E:\NU\mimic-icd-coder)
cd mimic-icd-coder

# Create venv on E: so it doesn't eat C: space
python -m venv .venv
.\.venv\Scripts\activate

# Core install
python -m pip install --upgrade pip
pip install -e ".[dev]"

# Verify on synthetic fixtures — should be 15 passed
pytest -q

# Set your dev config (from the provided template)
copy configs\dev.nancy.yml configs\dev.yml
# dev.yml is gitignored; your real credentials / paths stay off GitHub
```

Expected: `15 passed` in about 5 seconds.

## 2. Run the pipeline (`feat/local-pipeline`)

All commands from `E:\NU\mimic-icd-coder` with venv active. Each stage writes to `E:\NU\mimic-icd-coder\data\<stage>\`.

### 2a. Bronze — raw gz → Parquet mirror

```powershell
mic ingest --config configs\dev.yml
```

**What it does:** Reads your 4 gzipped CSVs from `E:\data\physionet\...` and writes Parquet to `data\bronze\`.

**Expected:**
- `discharge_notes.parquet` — ~331,000 rows, ~2 GB
- `diagnoses_icd.parquet` — ~6,400,000 rows, ~80 MB
- `admissions.parquet` — ~431,000 rows, ~30 MB
- `patients.parquet` — ~300,000 rows, ~10 MB
- Runtime: **~5–10 minutes** (most of it is reading 1.1 GB of gzipped text)

**Memory peak:** ~8–10 GB during the notes read. Close Chrome if tight.

### 2b. Silver — clean + dedup

```powershell
mic silver --config configs\dev.yml
```

**What it does:** Loads Bronze notes, collapses `___` de-id markers, normalizes whitespace, dedups to one note per `hadm_id` (keeping the latest `note_seq`), drops notes under 100 tokens.

**Expected:**
- `data\silver\notes.parquet` — ~331,000 rows (DS is the only note type in v2.2)
- Runtime: **~2–3 minutes**

### 2c. Gold — top-50 ICD-10 label matrix

```powershell
mic gold --config configs\dev.yml
```

**What it does:** Filters diagnoses to ICD-10 only, picks top-50 most-frequent codes, builds a sparse multi-hot matrix aligned with Silver row order.

**Expected:**
- `data\gold\labels.npz` — scipy sparse CSR, shape ~(122288, 50)
- `data\gold\label_names.json` — ICD-10 code list
- `data\gold\hadm_ids.parquet` — row-to-hadm_id lookup
- Runtime: **~30 seconds**

Inspect the labels once:
```powershell
python -c "import json; print(json.load(open('data/gold/label_names.json')))"
```

### 2d. Splits — patient-level manifest

```powershell
mic splits --config configs\dev.yml
```

**What it does:** Shuffles unique `subject_id`, assigns 80/10/10 to train/val/test. No patient appears in more than one split.

**Expected:**
- `data\gold\splits.parquet` — one row per admission
- Runtime: **< 10 seconds**

### 2e. Baseline — TF-IDF + LogisticRegression

```powershell
mic train-baseline --config configs\dev.yml
```

**What it does:** Fits TF-IDF on training notes, trains 50 one-vs-rest logistic regressions, tunes per-label thresholds on val, evaluates on val, logs to local MLflow.

**Expected:**
- Runtime: **~15–25 minutes** on your 16-thread CPU
- Outputs:
  - `data\gold\baseline_model.joblib` — pickled vectorizer + classifier
  - `data\gold\baseline_thresholds.npy` — per-label thresholds
  - `data\mlruns\*` — MLflow experiment files
- Logged val metrics: `micro_f1`, `macro_f1`, `p_at_5`, `p_at_8`, `p_at_15`

**Floor (if you miss this, something upstream is broken):**
- Micro F1 ≥ 0.55
- Macro F1 ≥ 0.35

### 2f. Test split evaluation

```powershell
mic evaluate-test --config configs\dev.yml
```

**Expected:** Same metrics as val plus `mullenbach_*_delta` comparisons.

### 2g. View MLflow UI

```powershell
mlflow ui --backend-store-uri file:E:/NU/mimic-icd-coder/data/mlruns --host 127.0.0.1 --port 5000
```

Open http://127.0.0.1:5000 in a browser. You should see one run with your baseline metrics, hyperparameters, model artifact, and the labels list.

## 3. One-shot option

If you want to run everything end-to-end in one command (useful for a clean rerun):

```powershell
mic run-all --config configs\dev.yml
```

Runtime: **~25–40 minutes total** on your laptop.

## 4. Memory troubleshooting

If Bronze ingestion OOMs:
- Close Chrome, VS Code, Outlook
- Check Task Manager → Memory — ensure ~20 GB free before starting
- Last resort: add `--chunksize 50000` to `ingest_mod.read_discharge_notes` and aggregate (future enhancement; your 32 GB should not need this)

## 5. GPU sanity check (prep for `feat/transformer-finetune` — not needed for `feat/local-pipeline` or `feat/baseline-model`)

```powershell
# Install PyTorch with CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Verify
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
# Expected: CUDA: True NVIDIA T1200 Laptop GPU
```

If `False`, install the CUDA 12.1 runtime from NVIDIA (not the full toolkit — just the runtime libraries).

## 6. Git hygiene before pushing

```powershell
# Confirm data\ is gitignored
git status
# Should show untracked: .venv/, data/, configs/dev.yml — NONE should be under the arrow for staged files.

# First commit
git add .
git commit -m "feat: initial scaffold with local staged pipeline, 15 smoke tests passing"

# Private repo on GitHub
gh repo create mimic-icd-coder --private --source . --push
```

## 7. What happens when

| Branch | Scope |
|---|---|
| `feat/local-pipeline` | Sections 1, 2a–2d (Bronze → Silver → Gold → Splits on workstation). Validate row counts and label distribution. |
| `feat/baseline-model` | Section 2e–2g (Baseline training + MLflow). First resume-eligible metric. |
| `feat/databricks-migration` | Port Silver/Gold Parquet to ADLS Gen2. Spin up Databricks GPU for BERT fine-tune. |
| `feat/transformer-finetune` | Bio_ClinicalBERT fine-tune + Clinical-Longformer fallback. Error analysis. |
| `feat/model-serving-drift` | Deploy Model Serving endpoint. Evidently drift config. |
| `feat/reports-and-cards` | Model card, data card, Quarto eval report. |

## 8. Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: mimic_icd_coder` | venv not active. Run `.\.venv\Scripts\activate`. |
| `FileNotFoundError: ...discharge.csv.gz` | Path typo in `configs\dev.yml`. Double-check `E:/data/physionet/...`. Forward slashes work in Python on Windows. |
| MemoryError in Bronze | Close other apps. 32 GB should be enough; if not, reduce `discharge.csv.gz` to 200K rows for a test pass. |
| `pytest` fails with coverage error | Run `pytest -q -p no:cacheprovider -o addopts=""` to bypass coverage if something's off with pytest-cov install. |
| `mic: command not found` | Re-run `pip install -e ".[dev]"` to register the console script. Or use `python -m mimic_icd_coder.cli <command>`. |

## Ready signal

You know `feat/local-pipeline` is done when:

- [ ] `mic run-all` completes without error
- [ ] `data\silver\notes.parquet` has ~122,288 rows after ICD-10 filter intersection
- [ ] `data\gold\label_names.json` lists 50 ICD-10 codes, including common ones like `I10`, `I50.9`, `N18.6`, `E11.9`
- [ ] MLflow UI shows a run with `val_micro_f1 >= 0.55` and `val_macro_f1 >= 0.35`
- [ ] `git status` shows no data files staged

If all five check, `feat/local-pipeline` and `feat/baseline-model` are complete on your laptop for zero Databricks cost. With the MLflow numbers in, move to `feat/databricks-migration`.
