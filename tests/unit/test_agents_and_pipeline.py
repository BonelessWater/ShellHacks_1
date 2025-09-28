import pytest
from backend.agents import (
    VendorAgent,
    TotalsAgent,
    PatternAgent,
    AgentCoordinator,
)
from backend.archive.main_pipeline import FraudDetectionPipeline
from data_models import Invoice, InvoiceItem


def make_mock_invoice():
    items = [
        InvoiceItem(description="Laptop", quantity=1, unit_price=1200.0),
        InvoiceItem(description="USB Cable", quantity=2, unit_price=10.0),
    ]
    invoice = Invoice(
        invoice_id="INV-1001",
        vendor="ACME Corp",
        date="2024-01-15",
        items=items,
        total=1220.0,
    )
    return invoice


def test_vendor_agent_basic():
    agent = VendorAgent()
    invoice = make_mock_invoice()
    result = agent.check_vendor(invoice)
    assert result.vendor_valid is True
    assert result.risk_factor in ("LOW", "MEDIUM", "HIGH")


def test_totals_agent_match():
    agent = TotalsAgent()
    invoice = make_mock_invoice()
    result = agent.check_totals(invoice)
    assert result.totals_match is True
    assert result.difference == pytest.approx(0.0)


def test_pattern_agent_basic():
    agent = PatternAgent()
    invoice = make_mock_invoice()
    result = agent.analyze_patterns(invoice)
    assert isinstance(result.anomalies_found, int)
    assert result.risk_factor in ("LOW", "MEDIUM", "HIGH")


def test_agent_coordinator_execute():
    coordinator = AgentCoordinator()
    invoice = make_mock_invoice()
    results = coordinator.execute_tasks(invoice, ["CheckVendor", "CheckTotals", "AnalyzePatterns"])
    assert "vendor" in results and "totals" in results and "patterns" in results
    assert isinstance(results["vendor"], dict)


def test_in_memory_pipeline_run():
    pipeline = FraudDetectionPipeline()
    invoice = make_mock_invoice()
    # Use the existing run_detection interface with a dict
    out = pipeline.run_detection(invoice.to_dict(), max_iterations=1)
    # run_detection may return a FraudDetectionResult object or a dict
    if hasattr(out, "to_dict"):
        out_dict = out.to_dict()
    else:
        out_dict = out

    assert isinstance(out_dict, dict)
    assert "results" in out_dict or "summary" in out_dict
