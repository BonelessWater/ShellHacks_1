import pytest

from backend.archive.ml_agents import TransactionAnomalyAgent


class DummyPredictor:
    def __call__(self, features):
        # return 0.5 for every feature
        return [0.5 for _ in features]


def test_direct_predictor_usage():
    class Item:
        def __init__(self):
            self.quantity = 1
            self.unit_price = 100
            self.line_total = 100
            self.description = "item"

    class Invoice:
        def __init__(self):
            self.items = [Item()]
            self.total = 100
            self.vendor = "ACME"

    agent = TransactionAnomalyAgent(predictor=DummyPredictor())
    inv = Invoice()
    res = agent.run(inv)
    assert res["details"] == "model_based"
    assert 0.0 <= res["risk_score"] <= 1.0


def test_lazy_predictor_loading(monkeypatch):
    # monkeypatch the create_predictor to return a known predictor
    def fake_create_predictor(model_path=None, scaler_path=None):
        return DummyPredictor()

    monkeypatch.setattr("backend.ml.predictor_utils.create_predictor", fake_create_predictor)

    class Item:
        def __init__(self):
            self.quantity = 2
            self.unit_price = 200
            self.line_total = 400
            self.description = "item"

    class Invoice:
        def __init__(self):
            self.items = [Item()]
            self.total = 400
            self.vendor = "ACME"

    agent = TransactionAnomalyAgent(model_path="./models/model", scaler_path="./models/scaler.pkl")
    inv = Invoice()
    res = agent.run(inv)
    assert res["details"] == "model_based"
    assert 0.0 <= res["risk_score"] <= 1.0
