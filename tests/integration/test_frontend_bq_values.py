import pytest
from fastapi.testclient import TestClient
import minimal_backend as main_app


client = TestClient(main_app.app)


def _has_real_amounts(invoices):
    """Return True if invoices list contains at least one non-zero, non-placeholder amount."""
    placeholders = {0.0, 0, "0", "0.00", 123.45}
    for inv in invoices:
        amt = inv.get("amount") or inv.get("total_amount") or inv.get("total") or 0
        try:
            a = float(amt)
        except Exception:
            continue
        if a not in placeholders and a != 0:
            return True
    return False


def test_invoices_amounts_nonzero_or_dynamic():
    """Call /api/invoices and assert amounts are not all zero/static placeholders.

    This test requires BigQuery to be available; fail loudly if it's not.
    """
    dbg = client.get("/api/debug/env")
    assert dbg.status_code == 200, "debug/env endpoint failed"
    dbgdata = dbg.json()
    bq_present = dbgdata.get("bq", {}).get("exists") or dbgdata.get("bq_manager", {}).get("exists")
    assert bq_present, "BigQuery not available — this test requires access to BigQuery"

    resp = client.get("/api/invoices?limit=10&offset=0")
    assert resp.status_code == 200
    data = resp.json()
    invoices = data.get("invoices") or []
    assert invoices, "No invoices returned from /api/invoices — expected BQ-backed rows"

    assert isinstance(invoices, list)
    assert any(isinstance(inv.get("amount"), (int, float)) or inv.get("total_amount") for inv in invoices)
    assert _has_real_amounts(invoices), "Expected at least one invoice with non-zero/non-placeholder amount"


def test_sample_dynamic_uses_bq_or_skips():
    """Call /api/invoices/sample?dynamic=true and, if BigQuery is available, assert amounts look real.

    The test queries /api/debug/env to decide whether to run strict checks.
    """
    dbg = client.get("/api/debug/env")
    assert dbg.status_code == 200, "debug/env endpoint failed"
    dbgdata = dbg.json()
    bq_present = dbgdata.get("bq", {}).get("exists") or dbgdata.get("bq_manager", {}).get("exists")
    assert bq_present, "BigQuery not available — this test requires access to BigQuery for dynamic sample"

    resp = client.get("/api/invoices/sample?dynamic=true&limit=10")
    assert resp.status_code == 200
    samples = resp.json() or []
    assert samples, "No sample invoices returned from dynamic sample — expected BQ-backed rows"

    # If BigQuery is present, assert at least one sample has a non-zero amount
    assert _has_real_amounts(samples), "Dynamic sample from BQ appears to contain only zero/static amounts"
