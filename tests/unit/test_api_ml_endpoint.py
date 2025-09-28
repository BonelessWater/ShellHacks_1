from backend.api import app
from backend.api.app import analyze_ml_agents
from backend.archive.main_pipeline import get_pipeline
from backend.archive.agents import AgentCoordinator
from backend.archive.data_models import DataValidator


def test_analyze_ml_agents_integration(monkeypatch):
    # Prepare a fake coordinator with mocked agents
    coord = get_pipeline().agent_coordinator

    class FakeTxAgent:
        def run(self, invoice):
            return {"risk_score": 0.42, "details": "fake_model"}

    class FakeGraphAgent:
        def run(self, invoice):
            return {"risk_score": 0.12, "details": "fake_graph"}

    coord.transaction_agent = FakeTxAgent()
    coord.graph_agent = FakeGraphAgent()

    # Build a minimal invoice payload
    payload = {
        "invoice_id": "inv-1",
        "vendor": "ACME Corp",
        "date": "2025-01-01",
        "items": [{"line_total": 100.0}],
        "total": 100.0,
    }

    res = analyze_ml_agents(payload)
    assert "transaction_anomaly" in res
    assert "graph_fraud" in res
    assert res["transaction_anomaly"]["details"] == "fake_model"
