from backend.api.app import persist_state, load_state, inspect_state, reset_agents
from backend.archive.main_pipeline import get_pipeline
import os


def test_inspect_state_returns_dict():
    # No ADMIN_API_KEY set in env — dev mode allows access
    res = inspect_state()
    assert isinstance(res, dict)


def test_persist_and_load_roundtrip(tmp_path):
    pipeline = get_pipeline()
    coord = pipeline.agent_coordinator
    # Mutate coordinator state a bit
    coord.vendor_agent.add_approved_vendor("X Test Vendor")
    path = tmp_path / "agent_state_test.json"
    ok = persist_state({"path": str(path)})
    assert ok.get("ok") is True or ok is True

    # Reset and load back
    coord.reset_agents()
    assert not ("X Test Vendor" in coord.vendor_agent.approved_vendors)
    ok2 = load_state({"path": str(path)})
    assert ok2.get("ok") is True or ok2 is True
    # After load the vendor should be present
    assert "X Test Vendor" in coord.vendor_agent.approved_vendors


def test_reset_agents():
    pipeline = get_pipeline()
    coord = pipeline.agent_coordinator
    coord.vendor_agent.add_approved_vendor("Another Vendor")
    ok = reset_agents({})
    assert ok.get("ok") is True or ok is True
    # After reset the vendor should be gone
    assert "Another Vendor" not in coord.vendor_agent.approved_vendors
