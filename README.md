# ShellHacks — Fraud Detection Suite (Holistic Overview)

This repository is a multi-agent fraud detection platform that combines an
LLM-based coordinator with a set of specialist agents, a data pipeline
backed by BigQuery, and a lightweight frontend/API for demoing and local
integration. The goal is to provide a modular ADK (Agent Development Kit)
and runtime so teams can add agents, test locally, and stage integrations
into production safely.

This README gives a holistic view of the system: the coordinator, agent
types and ADK, the frontend, data flow (ingest → analyze → persist), and
developer and deployment guidance.

Table of contents
- High-level architecture (coordinator, agents, data pipeline, frontend)
- The Agent Development Kit (ADK)
- Agents (types and responsibilities)
- Frontend & API
- Data pipeline & BigQuery
- How to run locally
- Testing, CI, and safety
- Deployment & observability
- Contributing and next steps

High-level architecture
------------------------

The platform is centered around a coordinator component that uses a
Generative LLM to orchestrate specialist agents. Agents are small, focused
units that analyze one aspect of an invoice or document (amounts, vendor
signals, line items, attachments, etc.). The coordinator asks the LLM which
agents to run for a given input, aggregates their structured responses, and
produces a final assessment.

Architecture components
- Coordinator (core): LLM-driven decision maker that selects agents,
  schedules runs, and compiles results.
- Agents: modular analyzers implemented as Python classes or processes that
  return typed results (AgentResponse). They encapsulate logic or call
  external services.
- Data Pipeline: ingestion, feature engineering, and storage (BigQuery).
- API / Frontend: a lightweight HTTP API and demo frontend to submit inputs
  and visualize results.
- Worker/Runtime: optional async workers (RQ/Celery) for heavy or long
  tasks and retries.

The Agent Development Kit (ADK)
-------------------------------

The ADK is a developer-facing set of helpers and conventions to write new
agents quickly and safely. Core ADK pieces:
- BaseAgent class: defines the interface and lifecycle (validate, run,
  serialize output).
- AgentResponse dataclass: standardized output (agent_type, confidence,
  risk_score, details, structured fields) to simplify aggregation.
- Test harness: local mocks and fixtures for LLM and BigQuery to run unit
  tests without external credentials.
- Registration: small registry for available agents so the coordinator can
  discover them dynamically.

Agents: types and responsibilities
----------------------------------

Common agent categories in the repo:
- VendorAgent: validates vendor names, checks blacklists and fuzzy
  matching.
- AmountAgent / TotalsAgent: verifies totals, line-item sums, tax calcs.
- LineItemAgent: inspects line items for unusual prices, quantities, or
  suspicious descriptions.
- DocumentAgent: inspects attachments or document metadata for tampering.
- FraudRuleAgent: codified heuristics that complement LLM judgements.

Each agent should be small, testable, and produce deterministic structured
outputs where possible — the coordinator resolves conflicts and uses
confidence scores in downstream decisions.

Frontend & API
---------------

The `frontend/` folder hosts a simple demo (React) that talks to a local
API. The API exposes endpoints such as:
- POST /analyze — submit an invoice and receive a synthesized assessment.
- GET /models — list available agent names and descriptions.

For production usage, wrap the coordinator behind an authenticated API and
use async processing for heavy workloads: accept the request, enqueue a
job, and return a job id. Clients can poll for results or receive a webhook.

Data pipeline & BigQuery
-------------------------

Data sources and sinks:
- Raw ingestion (JSON, uploads) → transformation → features.
- Feature store (in repo under `data_pipeline/features`) for model inputs.
- BigQuery: canonical storage for transactional and derived tables.

Safety around BigQuery
- Tests are allowed to read from real datasets by default but writes are
  no-op in test runs (autouse fixture `prevent_bq_writes`).
- Use `TEST_BQ_DATASET` or `BQ_DEFAULT_DATASET` to qualify unqualified
  table references for consistency.
- For real writes, use a sandbox dataset and require `ALLOW_REAL_BQ_WRITES`
  explicitly in the environment.

How to run locally (developer quickstart)
-----------------------------------------

1. Clone and install with Poetry:

```bash
git clone <repo>
cd ShellHacks_1
poetry install
```

2. Set up credentials only if you plan to run integration tests:

```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"
```

3. Run unit tests (fast):

```bash
poetry run pytest tests/unit -q
```

4. Run full test suite (integration tests will attempt read-only BigQuery):

```bash
poetry run pytest -q
```

5. Start local API (example using FastAPI):

```bash
# example using FastAPI uvicorn
uvicorn api.main:app --reload
```

Testing, CI, and safety
-----------------------

- Use GitHub Actions or your CI provider; keep integration tests gated and
  run them only when secrets are available. Unit tests should run on every
  PR.
- Add pre-commit hooks to block committing `.env` or service-account
  files. Consider `detect-secrets` for CI scanning.

Deployment & observability
--------------------------

- Containerize the API and workers with a Dockerfile. Provide a Docker
  Compose for local development and a Helm chart for k8s deployment.
- Add structured logging (JSON), metrics (Prometheus), and traces
  (optional). Monitor costs for LLM calls and BigQuery queries.

Contributing & next steps
-------------------------

- See `nextstep.txt` for prioritized milestones (cleanup shims, API
  skeleton, CI gating).
- When adding a new agent:
  1. Create a small module under `backend/agents` or `backend/archive/`.
  2. Implement `BaseAgent` interface, add unit tests that mock LLM/GCP.
  3. Register the agent in the registry so the coordinator can find it.

Security reminder
- Never commit `.env` or credentials. The repo includes a README note on
  this and the stash exports containing secrets were removed.

If you want, I can now:
- scaffold a FastAPI skeleton for `/analyze` and `/models` endpoints,
- open a cleanup PR to migrate away from shims and fully document the
  module mapping,
- or draft GitHub Actions workflows for unit and gated integration tests.
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
