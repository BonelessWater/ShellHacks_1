from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any, Dict

from backend.archive.main_pipeline import run_fraud_detection, get_system_status

app = FastAPI(title="Fraud Detection API")


class InvoicePayload(BaseModel):
    invoice_id: str
    vendor: str
    date: str
    items: list
    total: float


@app.get("/health")
def health() -> Dict[str, Any]:
    """Health endpoint returning system status"""
    status = get_system_status()
    return {"status": "ok", "system": status}


@app.post("/analyze")
def analyze_invoice(payload: InvoicePayload):
    try:
        result = run_fraud_detection(payload.dict(), max_iterations=1)
        # run_fraud_detection may return a FraudDetectionResult object or dict
        if hasattr(result, "to_dict"):
            return result.to_dict()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
