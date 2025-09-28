"""Export a small, obfuscated sample from training.invoice_training for frontend dev.

This script queries BigQuery for a small sample and writes a privacy-preserving
JSON file to `frontend/public/sample_invoices.json` for local frontend development.

Usage:
  export GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json
  poetry run python scripts/export_sample_invoices.py --project vaulted-timing-473322-f9 --limit 100

Notes:
- The sample is obfuscated: vendor_name is hashed/masked and numeric totals are
  slightly perturbed to avoid exact memorization by any downstream model.
- The script is explicitly for development; do NOT use this file for training.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import hashlib
import math
from typing import Any, Dict

from google.cloud import bigquery


def obfuscate_vendor(name: str) -> str:
    if not name:
        return ""
    # deterministic hash prefix to keep uniqueness but hide original
    h = hashlib.sha256(name.encode('utf-8')).hexdigest()
    return f"vendor_{h[:8]}"


def perturb_amount(amount: Any) -> float:
    try:
        a = float(amount)
    except Exception:
        return 0.0
    # Guard against NaN or infinite values coming from BigQuery
    if not math.isfinite(a):
        return 0.0
    # perturb by up to +/-5%
    factor = 1.0 + random.uniform(-0.05, 0.05)
    return round(a * factor, 2)


def sanitize_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(rec)
    # obfuscate vendor
    out['vendor'] = {'name': obfuscate_vendor(rec.get('vendor') or rec.get('vendor_name') or '')}
    # perturb totals
    out['total_amount'] = perturb_amount(rec.get('total_amount') or rec.get('total') or 0)
    out['subtotal'] = perturb_amount(rec.get('subtotal') or 0)
    out['tax_amount'] = perturb_amount(rec.get('tax_amount') or 0)
    # remove raw_record to avoid leaking original payloads
    out.pop('raw_record', None)
    # keep line_items as-is but if large, truncate
    if isinstance(out.get('line_items'), list) and len(out['line_items']) > 5:
        out['line_items'] = out['line_items'][:5]
    return out


def export_sample(project: str, limit: int, output_path: str):
    client = bigquery.Client(project=project)
    table = f"{project}.training.invoice_training"
    sql = f"SELECT * FROM `{table}` LIMIT {limit}"
    # If the table doesn't exist or query fails, raise a helpful error
    try:
        query_job = client.query(sql)
        # Avoid using the BigQuery Storage API (which requires
        # bigquery.readsessions.create). Fall back to the standard
        # download path which uses the regular BigQuery API.
        df = query_job.result().to_dataframe(create_bqstorage_client=False)
    except Exception as e:
        raise RuntimeError(f"Failed to query BigQuery table {table}: {e}")

    records = []
    for _, row in df.iterrows():
        rec = {}
        for col in df.columns:
            val = row[col]
            try:
                if hasattr(val, 'isoformat'):
                    rec[col] = val.isoformat()
                else:
                    rec[col] = (val.item() if hasattr(val, 'item') else val)
            except Exception:
                rec[col] = str(val)
        records.append(sanitize_record(rec))

    # Ensure output directory exists
    out_dir = os.path.dirname(output_path)
    os.makedirs(out_dir, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(records)} sample rows to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', required=True)
    parser.add_argument('--limit', type=int, default=100)
    parser.add_argument('--out', default='frontend/public/sample_invoices.json')
    args = parser.parse_args()

    export_sample(args.project, args.limit, args.out)


if __name__ == '__main__':
    main()
