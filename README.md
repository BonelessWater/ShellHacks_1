# ShellHacks — Fraud Detection Suite

This repository contains a multi-agent fraud detection suite built around
Google Generative AI, BigQuery data access, and local agent components. It
was developed to exercise data access and large-language-model driven
coordination of specialist agents that analyze invoices and other documents.

This README documents the repository layout, runtime and developer
instructions, architecture overview, and deployment considerations.

## Table of contents
- Repository layout
- Architecture overview
- How to run locally (development)
- Tests and CI
- BigQuery & data access
- Compatibility shims and migration notes
- Security and secrets
- Next steps and integration plan (see `nextstep.txt`)

## Repository layout

- `backend/` — core backend modules: agents, pipeline orchestration, data
  models and archived implementations (some modules live under
  `backend/archive/` and are re-exported by lightweight shims for
  backward compatibility).
- `data_pipeline/` — data access utilities, feature engineering, monitoring,
  and integration helpers for ML and streaming.
- `api/` and `frontend/` — illustrative API and frontend wiring used in
  demos or locally hosted integrations.
- `tests/` — unit and integration tests. The test suite is run with
  Poetry and is configured to allow read-only BigQuery access by default.
- `tensorflow.py` — a minimal pure-Python shim included to avoid pulling
  the real `tensorflow` package in developer environments and CI where
  it's unnecessary. Replace with the real package as needed.
- `bigquery_config.py` — small convenience wrapper for BigQuery clients and
  dataset qualification.

## Architecture overview

The system uses a coordination LLM (core) that analyzes an invoice and
selects a set of specialist agents. Each specialist agent performs a
domain-specific check (amount validation, vendor checks, tax calculations,
format inspection, etc.). The results are synthesized into a final
fraud-detection result.

Data flow (high level):

1. Ingest invoice data (JSON or structured records).
2. Core coordinator calls the LLM to decide which agents to summon.
3. Agents run locally or call external services (BigQuery reads, 3rd-party
   checks) and return structured `AgentResponse` objects.
4. Results are compiled and optionally written to BigQuery or emitted to
   other pipelines.

## How to run locally (development)

Prerequisites
- Python 3.11
- Poetry (recommended for managing the venv and dev deps)
- A Google service account with BigQuery read access for integration tests
  (set `GOOGLE_APPLICATION_CREDENTIALS` to its JSON path or rely on ADC).

Install and run

1. Install dependencies (local):

```bash
poetry install
```

2. Run tests:

```bash
poetry run pytest -q
```

Notes
- The `tensorflow.py` shim exists to keep the test environment light; if
  you need full TensorFlow, add it via `poetry add tensorflow` (may be
  heavy for local dev).
- Tests will read from real BigQuery datasets by default (configured via
  `TEST_BQ_DATASET` or `BQ_DEFAULT_DATASET`) but **will not perform writes
  by default**. There is an autouse pytest fixture `prevent_bq_writes` that
  no-ops write operations unless `ALLOW_REAL_BQ_WRITES` is set.

## Tests and CI

- Unit and integration tests live under `tests/`; integration tests that
  require GCP access are marked accordingly. To run only unit tests:

```bash
poetry run pytest tests/unit -q
```

## BigQuery & data access

- `bigquery_config.py` provides `BigQueryManager` which centralizes client
  creation and dataset qualification. Datasets may be referenced by name and
  will be qualified with the default dataset when necessary.
- For safety, test code uses an autouse fixture to prevent writes; to enable
  real writes (dangerous for CI) export `ALLOW_REAL_BQ_WRITES=1` in your
  environment first.

## Compatibility shims and migration notes

- During a large repo merge, many modules were moved under `backend/archive/`.
  To keep imports stable the repo includes small shim modules under
  `backend/` that re-export public symbols from the archived modules. See
  `backend/compat.py` for the central re-export helper.
- These shims are temporary — plan a cleanup PR to migrate call sites and
  remove the shims.

## Security & secrets

- Do NOT commit `.env`, service-account JSON files, or other secret
  material. `.gitignore` should contain `.env` and similar entries. If you
  ever exported stash content containing secrets, delete those exports and
  rotate credentials.
- The repository contains a `.env` sample in your local workspace — make
  sure it is never added to version control.

## Next steps

- See `nextstep.txt` for a concrete integration plan and milestones to turn
  this suite into a maintained product.
# ShellHacks
Set up for competition

## Testing & DB notes

A few developer notes to run the test suite safely and reproduce integration
behaviour.

- Tests are run under Poetry: `poetry run pytest -q`.
- By default tests may read from real BigQuery datasets; real writes are
  disabled during tests. The autouse pytest fixture `prevent_bq_writes`
  no-ops writes unless `ALLOW_REAL_BQ_WRITES` is explicitly set in your
  environment.
- Unqualified BigQuery table names are resolved against `TEST_BQ_DATASET`
  (or `BQ_DEFAULT_DATASET`) which defaults to `transactional_fraud`.
- The repo includes a small pure-Python `tensorflow.py` shim used to avoid
  pulling the heavy `tensorflow` dependency during test runs in CI or on
  developer machines. If you need full TF behaviour, install the real
  package with Poetry.

If you need help running tests against a specific dataset or restoring
local stash changes, ping me and I can walk through the safe steps.
# ShellHacks
Set up for competition
