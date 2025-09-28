ETL: build_invoice_training_table.py
=================================

Purpose
-------
Discover invoice-like tables across specified source datasets in BigQuery, normalize their fields to a canonical invoice schema, and consolidate them into a single training table.

Usage
-----
Run from the project's Poetry environment. The script uses the GOOGLE_APPLICATION_CREDENTIALS environment variable for authentication.

Example:

```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/key.json"
poetry run python scripts/build_invoice_training_table.py \
  --project vaulted-timing-473322-f9 \
  --source-datasets invoice_dataset,document_forgery,ieee_cis_fraud \
  --dest-dataset training \
  --dest-table invoice_training
```

Options
-------
- --dry-run: print planned SQL queries without executing

Output schema
-------------
The consolidated `training.invoice_training` table has the following canonical columns:

- invoice_id: STRING
- vendor_name: STRING
- invoice_date: STRING
- subtotal: FLOAT
- tax_amount: FLOAT
- total_amount: FLOAT
- line_items_json: STRING (JSON string of line items if present)
- raw_record: STRING (original row as JSON)
- source_table: STRING (origin table)

Notes
-----
- The script writes per-source staging tables before consolidating to handle schema differences across sources.
- Deduplication is performed by invoice_id using a source_rank (order of datasets passed on the command line) to break ties.
- If fields are missing in a source, the script leaves them NULL (or empty string for vendor_name).
