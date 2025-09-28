from backend.archive import agents
from backend.archive.data_models import Invoice, InvoiceItem


def test_exchange_rate_agent_with_mock_fetcher():
    # Mock fetcher returns a known rate mapping for base EUR to USD
    def mock_fetcher(base):
        return {"USD": 1.1}

    agent = agents.ExchangeRateAgent(fetcher=mock_fetcher)
    inv = Invoice(invoice_id="E1", vendor="ACME", date="2024-01-01", items=[], total=100.0)
    # invoice.currency is missing -> default USD behavior
    inv.currency = "EUR"
    inv.usd_amount = 110.0
    res = agent.run(inv)
    assert isinstance(res, dict)
    assert res.get("risk_score") == 0.0 or res.get("risk_score") >= 0.0
    assert "Exchange rate" in res.get("details", "")


def test_geolocation_agent_with_mock_fetcher():
    # Mock fetcher returns a single result
    def mock_geocode(address, api_key=None):
        return {"results": [{"formatted_address": "1 Test St"}]}

    agent = agents.GeoLocationAgent(fetcher=mock_geocode)
    inv = Invoice(invoice_id="G1", vendor="ACME", date="2024-01-01", items=[], total=0)
    inv.vendor_address = "Test address"
    res = agent.run(inv, api_key="FAKE")
    assert isinstance(res, dict)
    assert res.get("risk_score") == 0.0
    assert res.get("details") == "geocode_ok"
