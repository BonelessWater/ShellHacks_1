import json
from fastapi.testclient import TestClient
import backend.main as main_app


client = TestClient(main_app.app)


def assert_invoice_shape(inv):
    # required top-level keys
    keys = set(inv.keys())
    expected = {"id", "vendor", "amount", "status", "confidence", "issues", "date", "description", "line_items", "_raw"}
    assert expected.issubset(keys), f"missing keys: {expected - keys}"
    # vendor should be object with name
    assert isinstance(inv["vendor"], dict)
    assert "name" in inv["vendor"]
    assert isinstance(inv["line_items"], list)


def test_get_invoices_list():
    resp = client.get("/api/invoices?limit=5&offset=0")
    assert resp.status_code == 200
    data = resp.json()
    assert "invoices" in data
    assert isinstance(data["invoices"], list)
    if data["invoices"]:
        assert_invoice_shape(data["invoices"][0])


def test_get_invoices_bq_fallback(monkeypatch):
    # Force bigquery access to raise so endpoint falls back to mock
    def fake_get_bq_manager():
        raise RuntimeError("BQ not available")

    monkeypatch.setattr("backend.api.routes.get_bq_manager", fake_get_bq_manager)

    resp = client.get("/api/invoices/bq?limit=5&offset=0")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    if data:
        assert_invoice_shape(data[0])
