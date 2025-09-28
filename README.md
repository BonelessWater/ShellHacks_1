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
- Architecture overview
- How to run locally (development)
- Tests and CI
- BigQuery & data access
- Compatibility shims and migration notes
- Security and secrets
- Next steps and integration plan (see `nextstep.txt`)

## New features (summary)

These recent additions aim to make ML experimentation and auditability safer
and easier to use in developer and CI environments:

- Dataset fingerprinting: a lightweight utility `backend/ml/dataset_utils.py`
  computes stable SHA-256 fingerprints for CSVs and row iterables. The
  training helper `TransactionAnomalyTrainer.train_and_evaluate` now
  persists a `metadata.json` alongside saved models which includes the
  computed `dataset_fingerprint` and evaluation metrics for reproducibility.

- Optuna opt-in CLI & trial logging: an example CLI `examples/run_optuna_study.py`
  demonstrates a guarded Optuna run. Optuna runs are opt-in; set
  `RUN_OPTUNA=1` or pass `--confirm` to execute. Trial metadata is logged to
  a JSONL file when `trial_log_path` is provided. The helper wrapper is in
  `backend/ml/optuna_utils.py` and raises a clear error when Optuna isn't
  installed.

- Lightweight ML & Graph agents: `backend/archive/ml_agents.py` provides
  `TransactionAnomalyAgent` and `GraphFraudAgent` which accept injected
  predictors/graph-builders. These agents are test-friendly and safe to
  import in environments without TensorFlow or graph libraries.

- Separate ML analysis endpoint: `POST /analyze/ml` (in
  `backend/api/app.py`) runs ML/graph agents and returns their per-agent
  outputs. The endpoint is conservative: it will not perform training or
  external deployments and returns safe defaults if ML agents are not
  configured.

See the `backend/ml` and `examples/` folders for usage examples and tests.

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
5. BigQuery: ingest & schema

### BigQuery: ingest & schema

This repo includes a recommended BigQuery schema and sample data for invoices. I created a dataset `invoice_dataset` and a nested `invoices` table in project `vaulted-timing-473322-f9` as an example. Files added to the repo under `bq/`:

- `bq/schema_invoices.json` — BigQuery schema JSON for the invoices table
- `bq/sample_invoices.ndjson` — a small newline-delimited JSON example row

Quick commands (replace `PROJECT_ID` with your project if different):

```bash
# set gcloud project
gcloud config set project PROJECT_ID

# create dataset
bq --location=US mk --dataset --description "Invoice data for clients (ingest for training & analytics)" PROJECT_ID:invoice_dataset

# create table (DDL with nested vendor and repeated line_items)
bq query --use_legacy_sql=false 'CREATE TABLE `PROJECT_ID.invoice_dataset.invoices` (
  invoice_id STRING,
  invoice_number STRING,
  vendor_name STRING,
  vendor STRUCT<name STRING, address STRING, phone STRING, email STRING, tax_id STRING>,
  invoice_date DATE,
  due_date DATE,
  subtotal NUMERIC,
  tax_amount NUMERIC,
  total_amount NUMERIC,
  payment_terms STRING,
  purchase_order STRING,
  notes STRING,
  line_items ARRAY<STRUCT<description STRING, quantity NUMERIC, unit_price NUMERIC, total NUMERIC, sku STRING>>,
  received_date DATE,
  processed_ts TIMESTAMP,
  verification_status STRING,
  confidence_score FLOAT64,
  ingestion_time TIMESTAMP,
  invoice_hash STRING,
  source STRING
) PARTITION BY invoice_date CLUSTER BY vendor_name, invoice_number;'

# load sample NDJSON
bq load --source_format=NEWLINE_DELIMITED_JSON PROJECT_ID:invoice_dataset.invoices bq/sample_invoices.ndjson bq/schema_invoices.json
```

Sample view to flatten line items and extract training features:

```sql
CREATE OR REPLACE VIEW `PROJECT_ID.invoice_dataset.invoice_line_items_flat` AS
SELECT
  invoice_id,
  invoice_number,
  vendor_name,
  vendor.name AS vendor_full_name,
  invoice_date,
  total_amount,
  li.description AS item_description,
  CAST(li.quantity AS FLOAT64) AS item_quantity,
  CAST(li.unit_price AS NUMERIC) AS item_unit_price,
  CAST(li.total AS NUMERIC) AS item_total,
  LENGTH(vendor_name) AS vendor_len
FROM `PROJECT_ID.invoice_dataset.invoices`,
UNNEST(line_items) AS li;
```

If you'd like, I can commit a small SQL file with the view and/or run the `bq load` for the sample data in your environment.

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

### ML agents integration (TransactionAnomalyAgent & GraphFraudAgent)

This repository includes lightweight, test-friendly ML and graph agents
under `backend/archive/ml_agents.py`. They are intentionally import-safe
and accept injected predictors/graph builders so you can run them in unit
tests without TensorFlow or graph libraries.

Integration steps (safe, non-invasive)

1. Wire agents into the runtime coordinator at startup (no source edits
   required). This attaches the agents dynamically so the core
   coordinator doesn't require heavy deps:

```python
from backend.archive.ml_agents import (
    TransactionAnomalyAgent,
    GraphFraudAgent,
)
from backend.archive.agents import AgentCoordinator

coordinator = AgentCoordinator()

# Provide your predictor/graph builder or test stubs
txn_agent = TransactionAnomalyAgent(predictor=my_predictor)
graph_agent = GraphFraudAgent(graph_builder=my_graph_builder)

# Attach dynamically (safe)
coordinator.transaction_agent = txn_agent
coordinator.graph_agent = graph_agent
```

2. Add an optional API route to expose ML/graph analysis separately from
   the main `/analyze` flow. Keep this endpoint gated so it only runs in
   environments that have the required dependencies:

```python
from fastapi import APIRouter
from backend.archive.main_pipeline import get_pipeline

router = APIRouter()

@router.post("/analyze/ml")
def analyze_ml(payload: dict):
    pipeline = get_pipeline()
    # Convert payload to Invoice using the pipeline helper
    invoice = pipeline.agent_coordinator.execute_tasks.__self__.agent_coordinator if False else None
    # call agents directly (they should be attached on the coordinator)
    txn = pipeline.agent_coordinator.transaction_agent.run(invoice)
    graph = pipeline.agent_coordinator.graph_agent.run(invoice)
    return {"transaction": txn, "graph": graph}
```

Testing guidance

- Unit tests should inject simple stubs for predictors and graph builders.
  See `tests/unit/test_ml_agents.py` for examples. Stubs keep tests fast
  and deterministic.
- Keep heavy model training and Vertex AI deployment behind environment
  guards and explicit scripts. CI sets `NO_NETWORK=1` for test steps; do
  not perform network calls or deploy models during standard CI runs.

Persistence & ops

- If your agents maintain learned state, persist model pointers or state
  via the coordinator persistence helpers (file-based example):

```python
coordinator.persist_state("./agent_state.json")
```

Training examples

- A guarded trainer helper lives in `backend/ml/transaction_trainer.py`.
  It raises a friendly error when TensorFlow or cloud SDKs are missing.
  Install required packages and run training only in provisioned
  environments:

```bash
# in a provisioned environment
# poetry add tensorflow google-cloud-aiplatform
# python examples/train_transaction_model.py
```

Safety summary

- Keep ML/graph code injectable and mockable. Gate heavy deps behind
  optional imports or environment checks. Only enable model training or
  cloud deploys in controlled environments.

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

## API keys & DSPy integration (recent changes)

This repository now includes a flexible API key discovery and prioritization
mechanism to make local development and CI more resilient to differing
environment setups.

- Key discovery: The system will scan environment variables for names
  containing common keywords (case-insensitive) such as `API_KEY`,
  `GOOGLE`, `GENAI`, `GCP`, `BIGQUERY`, or `DOCUMENT`. That means keys named
  `MY_PROJECT_GENAI_KEY`, `ILAN_GOOGLE_KEY`, or `PROD_GCP_API_KEY` will be
  discovered automatically.

- Prioritization: At startup a best-effort validation pass runs and any
  keys that validate are moved to the front of the list so the runtime
  will prefer working keys first. Validation attempts the official SDK
  if available, otherwise it performs a minimal REST call to the
  `v1beta/{model}:generateContent` endpoint to check a key's usability.

- DSPy integration: Lightweight shims are provided so `backend.dspy_*`
  imports work even if the full archived implementations are in
  `backend/archive/`. There is also a `scripts/dspy_integration_test.py`
  script you can run to quickly verify that keys, imports, and basic
  DSPy initialization succeed:

```bash
poetry run python scripts/dspy_integration_test.py
```

If you plan to run integration tests on CI, consider adding a CI-only
mock mode (opt-in) so tests don't call real Generative AI endpoints in
pull-request builds. Contact the maintainer if you'd like me to scaffold
that mock mode and a GitHub Actions workflow.

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
