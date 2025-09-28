"""Invoice mapping agent

Provides helpers to infer a mapping from arbitrary BigQuery column names to the
canonical invoice fields used by the frontend, and to apply that mapping to raw
records. This is a deterministic, rule-based agent (no external calls).
"""
from typing import Dict, List, Any
import difflib
import re

# Canonical target fields we want for frontend consumption
EXPECTED_FIELDS = [
    'invoice_id', 'invoice_number', 'vendor', 'vendor_name', 'invoice_date',
    'due_date', 'subtotal', 'tax_amount', 'total_amount', 'payment_terms',
    'purchase_order', 'line_items', 'notes', 'raw_record'
]

# Common alternative names observed in BigQuery tables
ALIASES = {
    'invoice_id': ['invoice_id', 'inv_id', 'id', 'invoiceid'],
    'invoice_number': ['invoice_number', 'invoice_no', 'inv_number'],
    'vendor': ['vendor', 'vendor_name', 'company', 'supplier', 'vendor_company'],
    'invoice_date': ['invoice_date', 'date', 'issued_date', 'invoice_dt'],
    'due_date': ['due_date', 'due', 'payment_due'],
    'subtotal': ['subtotal', 'sub_total', 'amount_before_tax', 'net_amount'],
    'tax_amount': ['tax_amount', 'tax', 'tax_total'],
    'total_amount': ['total_amount', 'total', 'amount', 'grand_total'],
    'payment_terms': ['payment_terms', 'terms'],
    'purchase_order': ['purchase_order', 'po_number', 'po'],
    'line_items': ['line_items', 'items', 'line_items_json', 'lineitems'],
    'notes': ['notes', 'description', 'memo', 'analysis_summary'],
}


def _normalize_col_name(c: str) -> str:
    if c is None:
        return ''
    c = str(c).lower()
    c = re.sub(r'[^a-z0-9_]+', '_', c)
    return c


def infer_mapping_from_columns(columns: List[str]) -> Dict[str, str]:
    """Given a list of column names, suggest a mapping to canonical fields.

    Returns a dict mapping canonical_field -> source_column (or empty if none).
    """
    norm_cols = {c: _normalize_col_name(c) for c in columns}
    mapping: Dict[str, str] = {}

    # First try alias exact match
    for target, aliases in ALIASES.items():
        found = None
        for a in aliases:
            # match normalized alias to normalized cols
            for src, nsrc in norm_cols.items():
                if nsrc == _normalize_col_name(a):
                    found = src
                    break
            if found:
                break

        # fuzzy match as fallback
        if not found:
            best = difflib.get_close_matches(target, list(norm_cols.values()), n=1)
            if best:
                # get original column name
                inv = {v: k for k, v in norm_cols.items()}
                found = inv.get(best[0])

        mapping[target] = found or ''

    return mapping


def apply_mapping(record: Dict[str, Any], mapping: Dict[str, str]) -> Dict[str, Any]:
    """Apply a mapping to a raw record and return a normalized invoice dict.

    mapping: canonical_field -> source_column
    """
    out: Dict[str, Any] = {}

    # Helper to grab nested vendor fields
    def _get(src):
        if not src:
            return None
        # direct nested dict
        if isinstance(record.get(src), dict):
            return record.get(src)
        # otherwise try simple access
        return record.get(src)

    # Map straightforward fields
    for target in EXPECTED_FIELDS:
        src = mapping.get(target) or mapping.get(target.replace('vendor_', 'vendor'))
        if src:
            val = _get(src)
            out[target] = val
        else:
            out[target] = None

    # Post-process amounts into numeric floats when possible
    for amount_field in ('subtotal', 'tax_amount', 'total_amount'):
        v = out.get(amount_field)
        try:
            if v is None:
                out[amount_field] = 0.0
            else:
                out[amount_field] = float(v)
        except Exception:
            try:
                s = str(v).replace(',', '')
                out[amount_field] = float(s)
            except Exception:
                out[amount_field] = 0.0

    # Normalize vendor into object with name key if it's a string
    vendor_raw = out.get('vendor')
    if isinstance(vendor_raw, str):
        out['vendor'] = {'name': vendor_raw}
    elif isinstance(vendor_raw, dict):
        # keep as-is
        out['vendor'] = vendor_raw
    else:
        out['vendor'] = {'name': out.get('vendor_name') or 'Unknown Vendor'}

    # Ensure invoice_id exists
    if not out.get('invoice_id') and out.get('invoice_number'):
        out['invoice_id'] = out['invoice_number']

    # Line items: if JSON string try to keep as list
    li = out.get('line_items')
    if isinstance(li, str):
        try:
            import json

            parsed = json.loads(li)
            if isinstance(parsed, list):
                out['line_items'] = parsed
            else:
                out['line_items'] = []
        except Exception:
            out['line_items'] = []
    elif isinstance(li, list):
        out['line_items'] = li
    else:
        out['line_items'] = []

    # Pass-through raw record for debugging
    out['raw_record'] = record

    return out
