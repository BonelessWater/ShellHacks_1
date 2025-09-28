from backend.api.app import analyze_per_agent, apply_feedback, analyze_invoice
from backend.archive.data_models import Invoice, InvoiceItem


def test_analyze_per_agent_basic():
    payload = {
        "invoice_id": "TEST-1",
        "vendor": "ACME Corp",
        "date": "2024-01-01",
        "items": [{"description": "Item A", "quantity": 1, "unit_price": 10.0}],
        "total": 10.0,
    }

    res = analyze_per_agent(payload)
    assert isinstance(res, dict)
    assert "vendor" in res and "totals" in res and "patterns" in res


def test_feedback_apply():
    fb = {"vendor": {"approve": ["New Vendor"]}}
    res = apply_feedback(fb)
    assert res.get("ok") is True


def test_analyze_invoice_direct_call():
    payload = {
        "invoice_id": "TEST-2",
        "vendor": "ACME Corp",
        "date": "2024-01-01",
        "items": [{"description": "Item A", "quantity": 1, "unit_price": 10.0}],
        "total": 10.0,
    }
    res = analyze_invoice(payload)
    assert isinstance(res, dict)
    assert "summary" in res
