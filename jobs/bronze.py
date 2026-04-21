"""Databricks job — Bronze ingestion from ADLS Gen2 raw gz CSVs to Delta.

Stub. Implementation plan (mirrors `src/mimic_icd_coder/pipeline.py::run_bronze`):

    - spark.read.csv(notes_path,                        header=True) → {catalog}.bronze.notes_raw
    - spark.read.csv(hosp_path + "diagnoses_icd.csv.gz",    header=True) → {catalog}.bronze.diagnoses_raw
    - spark.read.csv(hosp_path + "admissions.csv.gz",       header=True) → {catalog}.bronze.admissions_raw
    - spark.read.csv(hosp_path + "patients.csv.gz",         header=True) → {catalog}.bronze.patients_raw
    - spark.read.csv(hosp_path + "d_icd_diagnoses.csv.gz",  header=True) → {catalog}.bronze.d_icd_diagnoses_raw
    - Stamp _ingested_at per Databricks medallion best practice
    - Set Delta table properties: delta.autoOptimize.optimizeWrite = true

The five tables mirror the local Bronze stage. `d_icd_diagnoses` is the ICD
code dictionary (`icd_code, icd_version, long_title`); joined downstream for
human-readable code descriptions in the data card and serving output.
"""

from __future__ import annotations

import argparse

from mimic_icd_coder.logging_utils import configure_logging, get_logger


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--notes-path", required=True)
    parser.add_argument("--hosp-path", required=True)
    parser.add_argument("--bronze-schema", default="bronze")
    args = parser.parse_args()

    configure_logging(level="INFO", fmt="json")
    log = get_logger("jobs.bronze")
    log.info("bronze.start", **vars(args))

    raise NotImplementedError(
        "Spark ingestion to Delta Bronze pending on feat/databricks-migration. "
        "See src/mimic_icd_coder/pipeline.py::run_bronze for the local reference."
    )


if __name__ == "__main__":
    main()
