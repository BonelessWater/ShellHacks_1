# backend/api/routes.py
from fastapi import APIRouter, HTTPException
from backend.services.bigquery import bigquery_service
from typing import Dict, Any
import logging
import math
import json

log = logging.getLogger(__name__)

router = APIRouter()

def normalize_invoice_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Return a frontend-friendly invoice shape from BQ records."""
    invoice_id = rec.get('id')
    vendor_name = rec.get('vendor', 'Unknown Vendor')
    amount = rec.get('amount', 0)
    status = rec.get('status', 'processed')
    confidence = rec.get('confidence', 0)
    issues = rec.get('issues', 0)
    date = rec.get('date')
    description = rec.get('description', '')

    return {
        "id": invoice_id,
        "vendor": vendor_name,
        "amount": round(float(amount), 2),
        "status": status,
        "confidence": round(float(confidence), 2),
        "issues": int(issues),
        "date": date,
        "description": description,
    }

@router.get("/invoices")
async def get_invoices(limit: int = 100):
    """Get invoices from BigQuery."""
    try:
        invoices = bigquery_service.get_invoices(limit=limit)
        normalized_invoices = [normalize_invoice_record(inv) for inv in invoices]
        
        log.info(f"Returning {len(normalized_invoices)} invoices.")
        if normalized_invoices:
            log.info(f"Sample invoice: {normalized_invoices[0]}")

        return {"success": True, "invoices": normalized_invoices, "count": len(normalized_invoices)}
    except Exception as e:
        log.error(f"BigQuery query failed: {e}")
        raise HTTPException(status_code=500, detail=f"BigQuery query failed: {str(e)}")

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "Backend API is running"}
