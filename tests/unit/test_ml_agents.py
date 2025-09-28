import pytest

from backend.archive.ml_agents import TransactionAnomalyAgent, GraphFraudAgent


class DummyPredictor:
    def __call__(self, features):
        # return increasing scores for amounts
        return [min(f.get("amount", 0) / 100.0, 1.0) for f in features]


class DummyGraph:
    def __init__(self, edges):
        self.edges = edges

    def score_node(self, node_id):
        # simple score: number of outgoing edges normalized
        out = len([e for e in self.edges if e[0] == node_id])
        return min(out / 5.0, 1.0)


class DummyGraphBuilder:
    def __call__(self, edges):
        return DummyGraph(edges)


def make_invoice_simple():
    class Item:
        def __init__(self, qty, unit_price, desc="item"):
            self.quantity = qty
            self.unit_price = unit_price
            self.line_total = qty * unit_price
            self.description = desc

    class Invoice:
        def __init__(self):
            self.items = [Item(1, 10), Item(2, 250)]
            self.total = 510
            self.vendor = "ACME Corp"

    return Invoice()


def test_transaction_anomaly_agent_default():
    inv = make_invoice_simple()
    ag = TransactionAnomalyAgent()
    res = ag.run(inv)
    assert res["risk_score"] == pytest.approx(0.8) or res["risk_score"] == pytest.approx(0.0)


def test_transaction_anomaly_agent_with_predictor():
    inv = make_invoice_simple()
    pred = DummyPredictor()
    ag = TransactionAnomalyAgent(predictor=pred)
    res = ag.run(inv)
    assert 0.0 <= res["risk_score"] <= 1.0
    assert res["details"] == "model_based"


def test_graph_fraud_agent_default():
    inv = make_invoice_simple()
    ag = GraphFraudAgent()
    res = ag.run(inv)
    assert 0.0 <= res["risk_score"] <= 1.0


def test_graph_fraud_agent_with_builder():
    inv = make_invoice_simple()
    builder = DummyGraphBuilder()
    ag = GraphFraudAgent(graph_builder=builder)
    res = ag.run(inv)
    assert 0.0 <= res["risk_score"] <= 1.0
    assert res["details"] in ("graph_score", "graph_builder_failed")
