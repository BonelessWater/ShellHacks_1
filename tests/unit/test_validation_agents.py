import math
from backend.archive.data_models import Invoice, InvoiceItem, create_test_invoice
from backend.archive import agents


def test_metadata_validation_agent_basic():
    agent = agents.MetadataValidationAgent()
    inv = create_test_invoice(include_issues=False)
    res = agent.run(inv)
    assert isinstance(res, dict)
    assert res.get("risk_score") >= 0.0


def test_frequency_anomaly_agent():
    agent = agents.FrequencyAnomalyAgent()
    inv = create_test_invoice(include_issues=False)
    # simulate multiple invoices in short window
    for _ in range(6):
        r = agent.run(inv)
    assert isinstance(r, dict)
    assert r.get("risk_score") >= 0.0
    assert any("High invoice frequency" in s for s in r.get("issues", []))


def test_benfords_law_agent():
    agent = agents.BenfordsLawAgent()
    # create invoice with amounts that skew toward 9 leading digit
    items = [InvoiceItem("Item", 1, 9.0), InvoiceItem("Item2", 1, 99.0), InvoiceItem("Item3", 1, 900.0)]
    inv = Invoice(invoice_id="X", vendor="ACME Corp", date="2024-01-01", items=items, total=1008.0)
    r = agent.run(inv)
    assert isinstance(r, dict)
    assert 0.0 <= r.get("risk_score") <= 1.0


def test_simple_threshold_agent():
    agent = agents.SimpleThresholdAgent(amount_threshold=100.0, quantity_threshold=10)
    items = [InvoiceItem("Item", 20, 200.0), InvoiceItem("Item2", 5, 10.0)]
    inv = Invoice(invoice_id="T1", vendor="ACME Corp", date="2024-01-01", items=items, total=4100.0)
    r = agent.run(inv)
    assert isinstance(r, dict)
    assert any("High unit price" in s for s in r.get("issues", []))
