BigQuery smoke test

This repository includes a tiny script to verify local BigQuery access using a service account JSON.

Quick steps:

1. Make sure you have a service account JSON file on your machine. For example:

   /Users/ilandanial/Downloads/vaulted-timing-473322-f9-5f312b3321cc.json

2. Export the env var (zsh):

   export GOOGLE_APPLICATION_CREDENTIALS="/Users/ilandanial/Downloads/vaulted-timing-473322-f9-5f312b3321cc.json"

3. Run the smoke test with Poetry (uses the project's interpreter):

   poetry run python scripts/smoke_bq.py

If the script prints "Smoke test passed", your local environment can initialize BigQuery and run queries using that service account.

Notes:
- Never commit service account JSONs into source control. Keep them out of the repository.
- For CI, prefer using secret managers or environment variables injected securely in your pipeline.
