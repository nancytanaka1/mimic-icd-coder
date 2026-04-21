"""Generate synthetic MIMIC-IV-style tables for CI and local dev.

Never commit real MIMIC data. These synthetic tables mimic the column
schema and include plausible clinical phrasing built from public templates
so ``TfidfVectorizer`` has words to work with.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Mapping from ICD-10 code → clinical phrase fragment. Used to ensure the
# baseline has learnable signal in the synthetic fixtures.
_ICD_PHRASES: dict[str, list[str]] = {
    "I50.9": [
        "congestive heart failure unspecified",
        "decompensated heart failure",
        "bnp elevated volume overload",
    ],
    "N18.6": [
        "end stage renal disease on hemodialysis",
        "esrd requiring dialysis",
        "chronic kidney disease stage 5",
    ],
    "E11.9": [
        "type 2 diabetes mellitus without complications",
        "poorly controlled diabetes hyperglycemia",
        "niddm with hba1c elevation",
    ],
    "J44.9": [
        "chronic obstructive pulmonary disease exacerbation",
        "copd flare requiring steroids",
        "bronchodilator therapy",
    ],
    "I10": [
        "essential hypertension",
        "elevated blood pressure on home medications",
        "hypertensive urgency",
    ],
    "I48.91": [
        "atrial fibrillation unspecified",
        "new onset afib with rvr",
        "anticoagulation started warfarin",
    ],
    "Z79.01": [
        "long term use of anticoagulants",
        "chronic warfarin therapy",
        "apixaban daily",
    ],
    "F32.9": [
        "major depressive disorder",
        "depressed mood sertraline started",
        "psychiatry consult for depression",
    ],
    "N17.9": ["acute kidney injury", "rising creatinine aki", "prerenal azotemia"],
    "J18.9": ["pneumonia unspecified", "community acquired pneumonia", "lobar infiltrate"],
}

_FRAME = "Patient is a {age} year old {sex} with history of {hist}. {chief} Hospital course notable for {course}. Discharge meds include {meds}. Follow up in {days} days."


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def make_synthetic(
    n_patients: int = 60, n_admissions_per_patient: tuple[int, int] = (1, 3), seed: int = 0
) -> dict[str, pd.DataFrame]:
    """Build a small synthetic corpus for tests and local dev.

    Args:
        n_patients: Number of distinct ``subject_id`` values.
        n_admissions_per_patient: Inclusive low/high bounds on admissions per patient.
        seed: RNG seed.

    Returns:
        Dict with keys ``notes``, ``diagnoses_icd``, ``admissions``, ``patients``.
    """
    rng = _rng(seed)
    labels = list(_ICD_PHRASES.keys())

    patients: list[dict[str, object]] = []
    admissions: list[dict[str, object]] = []
    diagnoses: list[dict[str, object]] = []
    notes: list[dict[str, object]] = []

    hadm_counter = 1_000_000
    for sid in range(1, n_patients + 1):
        n_admits = int(rng.integers(n_admissions_per_patient[0], n_admissions_per_patient[1] + 1))
        patients.append(
            {
                "subject_id": sid,
                "gender": rng.choice(["F", "M"]),
                "anchor_age": int(rng.integers(40, 90)),
                "anchor_year": 2150,
                "anchor_year_group": "2014 - 2016",
                "dod": "",
            }
        )
        for _ in range(n_admits):
            hadm = hadm_counter
            hadm_counter += 1

            k = int(rng.integers(2, 6))  # 2–5 labels per admission
            admit_labels = list(rng.choice(labels, size=k, replace=False))

            phrases: list[str] = []
            for code in admit_labels:
                phrases.append(rng.choice(_ICD_PHRASES[code]))
                diagnoses.append(
                    {
                        "subject_id": sid,
                        "hadm_id": hadm,
                        "seq_num": len(phrases),
                        "icd_code": code,
                        "icd_version": 10,
                    }
                )

            text = _FRAME.format(
                age=int(rng.integers(40, 90)),
                sex=rng.choice(["female", "male"]),
                hist=", ".join(phrases[: max(1, len(phrases) // 2)]),
                chief=rng.choice(
                    [
                        "Chief complaint shortness of breath.",
                        "Chief complaint chest pain.",
                        "Chief complaint altered mental status.",
                    ]
                ),
                course=", ".join(phrases),
                meds=rng.choice(
                    [
                        "furosemide, lisinopril, metoprolol",
                        "insulin, metformin",
                        "warfarin, atorvastatin",
                    ]
                ),
                days=int(rng.choice([7, 14, 30])),
            )
            # Pad text so it passes the 100-token filter.
            text = text + " " + " ".join(phrases * 8)

            notes.append(
                {
                    "note_id": f"{hadm}-DS-1",
                    "subject_id": sid,
                    "hadm_id": hadm,
                    "note_type": "DS",
                    "note_seq": 1,
                    "charttime": "2150-01-01 00:00:00",
                    "storetime": "2150-01-01 01:00:00",
                    "text": text,
                }
            )
            admissions.append(
                {
                    "subject_id": sid,
                    "hadm_id": hadm,
                    "admittime": "2150-01-01 00:00:00",
                    "dischtime": "2150-01-05 00:00:00",
                    "deathtime": "",
                    "hospital_expire_flag": 0,
                    "admission_type": "URGENT",
                }
            )

    return {
        "notes": pd.DataFrame(notes),
        "diagnoses_icd": pd.DataFrame(diagnoses),
        "admissions": pd.DataFrame(admissions),
        "patients": pd.DataFrame(patients),
    }
