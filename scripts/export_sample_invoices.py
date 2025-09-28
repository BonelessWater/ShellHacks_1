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

try:
    from google.cloud import bigquery  # type: ignore
except Exception:
    bigquery = None


def obfuscate_vendor(name: str) -> str:
    # Accept dicts or other types; coerce to a stable string when possible
    if not name:
        return ""
    if isinstance(name, dict):
        name = name.get('name') or name.get('vendor_name') or str(name)
    if not isinstance(name, str):
        name = str(name)
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


def deterministic_synth_amount(seed_str: str, lo: float = 50.0, hi: float = 5000.0) -> float:
    """Create a deterministic pseudo-random amount from seed_str in range [lo, hi].

    Uses SHA256 of seed_str to derive a repeatable integer seed so multiple
    runs produce the same synthetic amounts for the same invoice id.
    """
    if not seed_str:
        seed_str = 'unknown'
    h = hashlib.sha256(seed_str.encode('utf-8')).hexdigest()
    # take first 16 hex chars to avoid huge ints but still be well distributed
    seed_int = int(h[:16], 16)
    rng = random.Random(seed_int)
    return round(rng.uniform(lo, hi), 2)


def sanitize_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(rec)
    # obfuscate vendor
    out['vendor'] = {'name': obfuscate_vendor(rec.get('vendor') or rec.get('vendor_name') or '')}
    # totals: if source is missing or zero, synthesize a deterministic non-zero
    # amount (to make the dev sample look realistic) and then perturb it
    raw_total = rec.get('total_amount') or rec.get('total') or 0
    try:
        raw_total_val = float(raw_total)
    except Exception:
        raw_total_val = 0.0

    if not math.isfinite(raw_total_val) or raw_total_val <= 0:
        # use invoice id or a fallback string to deterministically synthesize
        invoice_seed = rec.get('invoice_id') or rec.get('id') or rec.get('invoice_number') or ''
        synth_total = deterministic_synth_amount(invoice_seed)
        out['total_amount'] = perturb_amount(synth_total)
        # derive subtotal/tax from synthetic total in plausible proportions
        out['subtotal'] = round(out['total_amount'] * 0.9, 2)
        out['tax_amount'] = round(out['total_amount'] - out['subtotal'], 2)
    else:
        out['total_amount'] = perturb_amount(raw_total_val)
        out['subtotal'] = perturb_amount(rec.get('subtotal') or out['total_amount'] * 0.9)
        out['tax_amount'] = perturb_amount(rec.get('tax_amount') or (out['total_amount'] - out['subtotal']))
    # remove raw_record to avoid leaking original payloads
    out.pop('raw_record', None)
    # keep line_items as-is but if large, truncate
    if isinstance(out.get('line_items'), list) and len(out['line_items']) > 5:
        out['line_items'] = out['line_items'][:5]
    return out


def export_sample(project: str, limit: int, output_path: str):
    # Try to query BigQuery; if ADC are missing or unavailable, fall back to
    # local-mode which reads any existing sample JSON and sanitizes/synthesizes
    # values. This makes the script usable in developer machines without cloud
    # credentials.
    records = []
    try:
        client = bigquery.Client(project=project)
        table = f"{project}.training.invoice_training"
        sql = f"SELECT * FROM `{table}` LIMIT {limit}"
        query_job = client.query(sql)
        # Avoid using the BigQuery Storage API (which requires
        # bigquery.readsessions.create). Fall back to the standard
        # download path which uses the regular BigQuery API.
        df = query_job.result().to_dataframe(create_bqstorage_client=False)

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
    except Exception as e:
        # Local fallback: read an existing sample file if available and sanitize
        print(f"BigQuery query failed ({e}), falling back to local sample if present")
        if os.path.exists(output_path):
            with open(output_path, 'r', encoding='utf-8') as f:
                try:
                    existing = json.load(f)
                except Exception:
                    existing = []
        else:
            existing = []

        for rec in existing[:limit]:
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
