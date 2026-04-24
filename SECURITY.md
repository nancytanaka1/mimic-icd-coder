# Security and Responsible-Use Policy

This repository is a scientific-research artifact: a replication of Mullenbach et al. 2018 (CAML) on MIMIC-IV top-50 ICD-10 auto-coding, plus an MLOps methodology demonstration for credentialed clinical NLP. It contains code, aggregate research results, and synthetic examples only. **No MIMIC-IV data and no trained model weights are distributed through this repository.**

## Reporting a PHI or DUA concern

This repository is designed to exclude any patient data or identifying information. If you believe you have found any content in this repository that could be patient data, that is derived from unaltered MIMIC-IV note text, or that could potentially re-identify an individual:

**Do two things in parallel.**

1. **Report to PhysioNet immediately** at <phi-report@physionet.org>. This is the canonical PhysioNet Credentialed Health Data License v1.5.0 reporting contact. Include the file path, line number(s), and why the content concerns you. Do not reproduce the suspected content in the email.

2. **Open a private security advisory on this repository** via GitHub's "Security → Report a vulnerability" workflow. Do not open a public issue — that would further expose the suspected content. If the repository's security-advisory feature is unavailable, contact the maintainer through GitHub (`nancytanaka1`).

The maintainer will acknowledge within 72 hours and remove the content within 24 hours of confirming the concern. PhysioNet's determination governs.

## Reporting a software-security vulnerability

For code-level vulnerabilities (dependency CVEs, path traversal, injection, secret leakage in code, etc.) that are **not** related to MIMIC data:

- Use GitHub's private security advisory feature on this repository.
- Please do not open a public issue or pull request describing the vulnerability until it has been addressed.

This is a portfolio / research project, not a commercial product. No bounty program exists; responsible disclosure is appreciated.

## Responsible-use requirements for users reproducing this work

This repository does not grant any access to MIMIC-IV data. To reproduce the training pipeline from raw data, you must independently:

1. Complete [PhysioNet credentialing](https://physionet.org/login/) (create a credentialed account on physionet.org).
2. Complete a [CITI "Data or Specimens Only Research" training course](https://about.citiprogram.org/course/data-or-specimens-only-research/) — or equivalent human-subjects training — and upload the completion certificate to your PhysioNet profile.
3. Sign the [PhysioNet Credentialed Health Data License v1.5.0](https://physionet.org/content/mimiciv/view-license/3.1/) for MIMIC-IV v3.1 Hosp and, separately, for MIMIC-IV-Note v2.2.
4. Download MIMIC-IV locally on a device you control with secured storage. Do not copy the data to any shared drive, cloud service, or third-party system not covered by your DUA.
5. Never pass MIMIC-IV note text or PHI to third-party hosted LLM APIs (OpenAI, Anthropic, Google, etc.). Only open-weights models hosted on infrastructure you control are DUA-compatible.

Users who reproduce this pipeline agree to abide by PhysioNet's DUA at all times, independent of this repository.

## What this repository does not contain

- No MIMIC-IV source CSVs, Parquet extracts, or Delta tables.
- No individual patient records, discharge note text, admission IDs, or subject IDs.
- No trained model weights, serialized tokenizers, or MLflow model artifacts.
- No credentialing certificates, DUA paperwork, or personal MIMIC access tokens.
- No real-time clinical decision-support functionality; this is a research benchmark, not a clinical tool.

## What this repository does contain

- Source code for the full pipeline (Bronze → Silver → Gold → baseline training → evaluation).
- Aggregate cohort statistics and per-label performance metrics equivalent to those published in peer-reviewed MIMIC research.
- Two PNG figures derived from aggregate metrics (reliability curve, confusion-pair heatmap).
- Three synthetic discharge-note excerpts authored by the repository owner from domain knowledge — zero real MIMIC text.
- ICD-10 code descriptions sourced from the public ICD-10-CM dictionary (not from MIMIC).
- Decision logs, EDA narrative, data card, model card, and evaluation report — all text-only, all aggregate.

## Maintainer

Nancy Tanaka — via GitHub (`nancytanaka1`).
