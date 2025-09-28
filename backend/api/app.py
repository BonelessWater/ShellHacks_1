from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any, Dict

from backend.archive.main_pipeline import (
    run_fraud_detection,
    get_system_status,
    get_pipeline,
    persist_agent_state,
    load_agent_state,
)
from backend.archive.data_models import DataValidator
import os


def _check_admin_key(provided: str | None) -> bool:
    """Simple admin key check: uses ADMIN_API_KEY env var when set."""
    admin_key = os.environ.get("ADMIN_API_KEY")
    if not admin_key:
        # No admin key set in environment -> allow for development
        return True
    return provided == admin_key

app = FastAPI(title="Fraud Detection API")


class InvoicePayload(BaseModel):
    invoice_id: str
    vendor: str
    date: str
    items: list
    total: float
    # Optional fields
    geocode_api_key: str | None = None

    class Config:
        extra = "allow"


@app.get("/health")
def health() -> Dict[str, Any]:
    """Health endpoint returning system status"""
    status = get_system_status()
    return {"status": "ok", "system": status}


@app.post("/analyze")
def analyze_invoice(payload: InvoicePayload):
    try:
        # Accept dicts for direct-call tests or Pydantic models from FastAPI
        if isinstance(payload, dict):
            payload_dict = payload
        else:
            payload_dict = payload.dict()

        # Forward payload; if geocode_api_key present it will be used by GeoLocationAgent
        result = run_fraud_detection(payload_dict, max_iterations=1)
        # run_fraud_detection may return a FraudDetectionResult object or dict
        if hasattr(result, "to_dict"):
            return result.to_dict()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze/agents")
def analyze_per_agent(payload: InvoicePayload):
    """Run per-agent analysis and return their raw outputs.

    This endpoint runs the coordinator's execute_tasks directly with a conservative
    set of tasks to get per-agent outputs quickly. Use `geocode_api_key` in payload
    to supply the geocoding API key for GeoLocationAgent.
    """
    try:
        # Conservative tasks to gather vendor/totals/patterns
        tasks = ["CheckVendor", "CheckTotals", "AnalyzePatterns"]
        if isinstance(payload, dict):
            payload_dict = payload
        else:
            payload_dict = payload.dict()

        coordinator = get_pipeline().agent_coordinator

        # Attach geocode key if provided
        api_key = payload_dict.get("geocode_api_key")
        if api_key:
            payload_dict["_geocode_api_key"] = api_key

        # Convert to Invoice object for coordinator
        invoice_obj = DataValidator.validate_invoice(payload_dict)
        if not invoice_obj:
            raise HTTPException(status_code=400, detail="Invalid invoice payload")

        result = coordinator.execute_tasks(invoice_obj, tasks)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze/ml")
def analyze_ml_agents(payload: InvoicePayload):
    """Run ML/graph agents (TransactionAnomalyAgent, GraphFraudAgent) and return their outputs.

    This endpoint is intentionally separate and conservative: it will not perform any
    network writes or external deployments. If ML agents are not configured it returns
    heuristic outputs or safe defaults.
    """
    try:
        if isinstance(payload, dict):
            payload_dict = payload
        else:
            payload_dict = payload.dict()

        coordinator = get_pipeline().agent_coordinator

        invoice_obj = DataValidator.validate_invoice(payload_dict)
        if not invoice_obj:
            raise HTTPException(status_code=400, detail="Invalid invoice payload")

        results = {}
        # Transaction anomaly agent
        tx_agent = getattr(coordinator, "transaction_agent", None)
        if tx_agent is not None:
            try:
                results["transaction_anomaly"] = tx_agent.run(invoice_obj)
            except Exception as e:
                results["transaction_anomaly"] = {"risk_score": 1.0, "details": f"agent_error:{e}"}
        else:
            results["transaction_anomaly"] = {"risk_score": 0.0, "details": "agent_unavailable"}

        # Graph fraud agent
        graph_agent = getattr(coordinator, "graph_agent", None)
        if graph_agent is not None:
            try:
                results["graph_fraud"] = graph_agent.run(invoice_obj)
            except Exception as e:
                results["graph_fraud"] = {"risk_score": 1.0, "details": f"agent_error:{e}"}
        else:
            results["graph_fraud"] = {"risk_score": 0.0, "details": "agent_unavailable"}

        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/feedback")
def apply_feedback(feedback: dict):
    try:
        # Require admin key in feedback payload for production safety
        admin_key = feedback.get("_admin_key") if isinstance(feedback, dict) else None
        if not _check_admin_key(admin_key):
            raise HTTPException(status_code=403, detail="Forbidden: invalid admin key")

        coordinator = get_pipeline().agent_coordinator
        ok = coordinator.apply_feedback(feedback)
        return {"ok": ok}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/admin/persist")
def persist_state(body: dict):
    admin_key = body.get("_admin_key") if isinstance(body, dict) else None
    if not _check_admin_key(admin_key):
        raise HTTPException(status_code=403, detail="Forbidden: invalid admin key")

    path = body.get("path") if isinstance(body, dict) else None
    ok = persist_agent_state(path)
    return {"ok": bool(ok)}


@app.post("/admin/load")
def load_state(body: dict):
    admin_key = body.get("_admin_key") if isinstance(body, dict) else None
    if not _check_admin_key(admin_key):
        raise HTTPException(status_code=403, detail="Forbidden: invalid admin key")

    path = body.get("path") if isinstance(body, dict) else None
    ok = load_agent_state(path)
    return {"ok": bool(ok)}


@app.get("/admin/state")
def inspect_state(admin_key: str | None = None):
    if not _check_admin_key(admin_key):
        raise HTTPException(status_code=403, detail="Forbidden: invalid admin key")
    coord = get_pipeline().agent_coordinator
    return coord.to_dict()


@app.post("/admin/reset_agents")
def reset_agents(body: dict):
    admin_key = body.get("_admin_key") if isinstance(body, dict) else None
    if not _check_admin_key(admin_key):
        raise HTTPException(status_code=403, detail="Forbidden: invalid admin key")

    ok = get_pipeline().agent_coordinator.reset_agents()
    return {"ok": ok}
