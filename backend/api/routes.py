# backend/api/routes.py
from fastapi import APIRouter, HTTPException
from backend.services.bigquery import bigquery_service
from typing import Dict, Any
import logging
import math
import json
from datetime import datetime, timedelta
import pandas as pd
import sys

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

@router.get("/analytics/money_saved")
async def get_money_saved_analytics():
    """
    Calculates the total amount of money saved per week by catching potentially fraudulent transactions.
    """
    try:
        transactions = bigquery_service.get_transactions_for_analytics()
        if not transactions:
            return {"success": True, "data": []}

        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(transactions)

        # Run fraud detection on each transaction
        def get_risk_score(row):
            return run_fraud_detection(row.to_dict()).get('risk_score', 0)

        df['risk_score'] = df.apply(get_risk_score, axis=1)

        # Filter for non-approved transactions (money saved)
        money_saved_df = df[df['risk_score'] > 0.5]

        # Convert TransactionDT to datetime
        # The TransactionDT is a timedelta from a reference point. Let's assume the reference is the start of the dataset.
        start_date = datetime.fromtimestamp(df['TransactionDT'].min())
        df['transaction_date'] = df['TransactionDT'].apply(lambda x: start_date + timedelta(seconds=x))
        
        # Group by week and sum the amounts
        weekly_savings = money_saved_df.groupby(pd.Grouper(key='transaction_date', freq='W-MON'))['TransactionAmt'].sum().reset_index().sort_values('transaction_date')
        
        # Format for the frontend
        chart_data = {
            "labels": weekly_savings['transaction_date'].dt.strftime('%Y-%m-%d').tolist(),
            "datasets": [{
                "label": "Money Saved",
                "data": weekly_savings['TransactionAmt'].tolist(),
                "backgroundColor": "rgba(75, 192, 192, 0.2)",
                "borderColor": "rgba(75, 192, 192, 1)",
                "borderWidth": 1
            }]
        }

        return {"success": True, "data": chart_data}
    except Exception as e:
        log.error(f"Error in get_money_saved_analytics: {e}", exc_info=True)
        sys.stdout.flush()
        raise HTTPException(status_code=500, detail=f"Failed to calculate money saved analytics: {str(e)}")
