"""Quick smoke test for BigQuery access using GOOGLE_APPLICATION_CREDENTIALS.

Usage:
  export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"
  poetry run python scripts/smoke_bq.py

This script runs a tiny query `SELECT 1 as ok` to confirm client initialization and ability to run a query.
"""

import os
import sys
from google.cloud import bigquery

def main():
    creds = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
    if not creds:
        print('GOOGLE_APPLICATION_CREDENTIALS not set. Please export path to service account JSON.')
        sys.exit(2)

    print('Using credentials:', creds)
    client = bigquery.Client()
    print('Client project:', client.project)
    query = 'SELECT 1 as ok'
    job = client.query(query)
    result = job.result()
    rows = list(result)
    if rows and rows[0].ok == 1:
        print('Smoke test passed: query returned ok=1')
        return 0
    else:
        print('Smoke test failed: unexpected result', rows)
        return 1

if __name__ == '__main__':
    sys.exit(main())
