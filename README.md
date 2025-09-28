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
