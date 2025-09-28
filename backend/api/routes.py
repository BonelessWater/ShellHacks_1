# backend/api/routes.py
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from datetime import datetime, timedelta
import json
import asyncio
import uuid
import os

# BigQuery manager (lazy import to avoid importing heavy deps at module import time)
def get_bq_manager():
    try:
        from ... import bigquery_config
        return bigquery_config.bq_manager
    except Exception:
        # Try absolute import
        try:
            import bigquery_config
            return bigquery_config.bq_manager
        except Exception:
            return None

# Try to import the ADK agent pipeline (preferred). If unavailable, fall back to
# the simple heuristic scorer below. The project provides a full pipeline at
# `backend.archive.main_pipeline` which exposes `run_fraud_detection` and
# `get_pipeline` helpers.
pipeline_run_fraud_detection = None
pipeline_get = None
try:
    try:
        # Relative import when executed as package
        from ...archive.main_pipeline import run_fraud_detection as _pipeline_run, get_pipeline as _pipeline_get  # type: ignore
    except Exception:
        # Absolute import fallback for different execution contexts
        from backend.archive.main_pipeline import run_fraud_detection as _pipeline_run, get_pipeline as _pipeline_get  # type: ignore

    pipeline_run_fraud_detection = _pipeline_run
    pipeline_get = _pipeline_get
    FRAUD_DETECTION_AVAILABLE = True
except Exception:
    FRAUD_DETECTION_AVAILABLE = False


def run_fraud_detection(payload, max_iterations=1):
    """Call the ADK pipeline run_fraud_detection when available, otherwise use
    a lightweight local mock. The wrapper keeps the rest of the codebase
    calling `run_fraud_detection(...)` unchanged.
    """
    if FRAUD_DETECTION_AVAILABLE and pipeline_run_fraud_detection is not None:
        try:
            # Run with minimal iterations to reduce side-effects and latency
            return pipeline_run_fraud_detection(payload, max_iterations=1)
        except Exception:
            # Fall back to local mock on any pipeline error
            pass

    # Local mock scoring (kept for fallback/testing)
    risk_score = 0.3 + (hash(payload.get("vendor", "")) % 5) * 0.15  # Mock score between 0.3-0.9
    return {
        "risk_score": risk_score,
        "details": f"Analyzed invoice for {payload.get('vendor', 'Unknown')}",
        "agent_results": {
            "vendor_check": {"status": "passed", "risk_score": risk_score * 0.8},
            "amount_check": {"status": "passed", "risk_score": risk_score * 1.2},
            "pattern_analysis": {"status": "flagged" if risk_score > 0.7 else "passed", "risk_score": risk_score}
        }
    }

def get_system_status():
    """Return system status"""
    return {
        "status": "operational",
        "agents": ["vendor_check", "amount_check", "pattern_analysis"],
        "database": "connected",
        "processing_queue": 0
    }

class DataValidator:
    @staticmethod
    def validate_invoice(data):
        """Simple invoice validation"""
        required_fields = ["invoice_id", "vendor", "total"]
        for field in required_fields:
            if field not in data:
                return None
        return data

router = APIRouter()

# Mock database for demonstration - replace with your actual database
mock_invoices = []
mock_analytics = {
    "total_invoices": 0,
    "processed_today": 0,
    "average_processing_time": 2.5,
    "risk_distribution": {"low": 0, "medium": 0, "high": 0, "critical": 0},
    "top_vendors": [],
    "monthly_trends": []
}

# Data models
class InvoiceUploadResponse(BaseModel):
    success: bool
    results: List[Dict[str, Any]]
    message: str

class InvoiceStatus(BaseModel):
    status: str

class VendorInfo(BaseModel):
    name: str
    address: Optional[str] = ""
    phone: Optional[str] = ""
    email: Optional[str] = ""
    tax_id: Optional[str] = ""

class LineItem(BaseModel):
    description: str
    quantity: float
    unit_price: float
    total: float

class InvoiceData(BaseModel):
    invoice_number: str
    vendor: VendorInfo
    invoice_date: str
    due_date: Optional[str] = None
    subtotal: float
    tax_amount: float
    total_amount: float
    payment_terms: Optional[str] = "Net 30"
    purchase_order: Optional[str] = ""
    line_items: List[LineItem] = []
    notes: Optional[str] = ""

# Health and system status endpoints
@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "Backend API is running"}

@router.get("/system/status")
async def system_status():
    """Get system status including agent pipeline status"""
    try:
        status = get_system_status()
        return {
            "status": "healthy",
            "agents_online": True,
            "database_connected": True,
            "processing_queue": 0,
            "system_details": status
        }
    except Exception as e:
        return {
            "status": "degraded",
            "agents_online": False,
            "database_connected": True,  # Assume DB is connected
            "processing_queue": 0,
            "error": str(e)
        }

# Invoice endpoints
@router.post("/invoices/upload", response_model=InvoiceUploadResponse)
async def upload_invoices(
    files: Optional[List[UploadFile]] = File(None),
    invoice_data: Optional[str] = Form(None),
    process_immediately: bool = Form(True)
):
    """Upload and process invoice files or direct invoice data"""
    results = []
    
    try:
        if invoice_data:
            # Process direct invoice data
            data = json.loads(invoice_data)
            result = await process_single_invoice(data)
            results.append(result)
        
        if files:
            # Process uploaded files
            for file in files:
                if file.content_type not in ['application/pdf', 'image/jpeg', 'image/png', 'application/json']:
                    results.append({
                        "filename": file.filename,
                        "success": False,
                        "error": "Unsupported file type"
                    })
                    continue
                
                # For now, create mock invoice data from file
                mock_data = create_mock_invoice_from_file(file.filename)
                result = await process_single_invoice(mock_data)
                result["filename"] = file.filename
                results.append(result)
        
        success_count = sum(1 for r in results if r.get("success"))
        
        return InvoiceUploadResponse(
            success=success_count > 0,
            results=results,
            message=f"Processed {success_count} of {len(results)} invoices successfully"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload processing failed: {str(e)}")

async def process_single_invoice(invoice_data: Dict[str, Any]) -> Dict[str, Any]:
    """Process a single invoice through the fraud detection pipeline"""
    try:
        # Create a unique invoice ID if not provided
        invoice_id = invoice_data.get("invoice_number") or str(uuid.uuid4())
        
        # Prepare data for fraud detection pipeline
        fraud_payload = {
            "invoice_id": invoice_id,
            "vendor": invoice_data.get("vendor", {}).get("name", "Unknown"),
            "date": invoice_data.get("invoice_date", datetime.now().isoformat()),
            "items": invoice_data.get("line_items", []),
            "total": float(invoice_data.get("total_amount", 0))
        }
        
        # Run fraud detection
        fraud_result = run_fraud_detection(fraud_payload, max_iterations=1)
        
        # Format result for frontend
        processed_invoice = {
            "id": invoice_id,
            "invoice_number": invoice_id,
            "vendor": invoice_data.get("vendor", {}),
            "invoice_date": invoice_data.get("invoice_date"),
            "due_date": invoice_data.get("due_date"),
            "subtotal": invoice_data.get("subtotal", 0),
            "tax_amount": invoice_data.get("tax_amount", 0),
            "total_amount": invoice_data.get("total_amount", 0),
            "payment_terms": invoice_data.get("payment_terms", "Net 30"),
            "purchase_order": invoice_data.get("purchase_order", ""),
            "line_items": invoice_data.get("line_items", []),
            "notes": invoice_data.get("notes", ""),
            "processed_date": datetime.now().isoformat(),
            "verification_status": "processed",
            "risk_level": determine_risk_level(fraud_result),
            "confidence_score": extract_confidence_score(fraud_result),
            "verification_results": format_verification_results(fraud_result)
        }
        
        # Store in mock database
        mock_invoices.append(processed_invoice)
        update_analytics(processed_invoice)
        
        return {
            "success": True,
            "invoice": processed_invoice,
            "processing_time": 2.5,
            "fraud_analysis": fraud_result
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "invoice_data": invoice_data
        }

def create_mock_invoice_from_file(filename: str) -> Dict[str, Any]:
    """Create mock invoice data from uploaded file"""
    return {
        "invoice_number": f"INV-{uuid.uuid4().hex[:8].upper()}",
        "vendor": {
            "name": f"Vendor from {filename}",
            "address": "123 Business St",
            "phone": "(555) 123-4567",
            "email": "vendor@example.com"
        },
        "invoice_date": datetime.now().isoformat(),
        "due_date": (datetime.now() + timedelta(days=30)).isoformat(),
        "subtotal": 1000.00,
        "tax_amount": 80.00,
        "total_amount": 1080.00,
        "payment_terms": "Net 30",
        "line_items": [
            {
                "description": "Professional Services",
                "quantity": 1,
                "unit_price": 1000.00,
                "total": 1000.00
            }
        ]
    }

def determine_risk_level(fraud_result) -> str:
    """Determine risk level from fraud detection result"""
    if hasattr(fraud_result, 'risk_score'):
        score = fraud_result.risk_score
    elif isinstance(fraud_result, dict) and 'risk_score' in fraud_result:
        score = fraud_result['risk_score']
    else:
        score = 0.5  # Default medium risk
    
    if score < 0.3:
        return "low"
    elif score < 0.6:
        return "medium"
    elif score < 0.8:
        return "high"
    else:
        return "critical"

def extract_confidence_score(fraud_result) -> float:
    """Extract confidence score from fraud detection result"""
    if hasattr(fraud_result, 'confidence_score'):
        return fraud_result.confidence_score
    elif isinstance(fraud_result, dict) and 'confidence_score' in fraud_result:
        return fraud_result['confidence_score']
    else:
        return 0.85  # Default confidence

def format_verification_results(fraud_result) -> List[Dict[str, Any]]:
    """Format fraud detection results for frontend"""
    results = []
    
    if hasattr(fraud_result, 'agent_results'):
        for agent, result in fraud_result.agent_results.items():
            results.append({
                "agent": agent,
                "status": "passed" if result.get('risk_score', 0) < 0.5 else "flagged",
                "details": result.get('details', ''),
                "risk_score": result.get('risk_score', 0)
            })
    elif isinstance(fraud_result, dict):
        results.append({
            "agent": "fraud_detection",
            "status": "processed",
            "details": "Invoice processed through fraud detection pipeline",
            "risk_score": fraud_result.get('risk_score', 0)
        })
    
    return results

def update_analytics(invoice: Dict[str, Any]):
    """Update analytics with new invoice data"""
    global mock_analytics
    
    mock_analytics["total_invoices"] += 1
    mock_analytics["processed_today"] += 1
    
    # Update risk distribution
    risk_level = invoice.get("risk_level", "medium")
    mock_analytics["risk_distribution"][risk_level] += 1
    
    # Update top vendors
    vendor_name = invoice.get("vendor", {}).get("name", "Unknown")
    vendor_found = False
    for vendor in mock_analytics["top_vendors"]:
        if vendor["name"] == vendor_name:
            vendor["invoice_count"] += 1
            vendor["total_amount"] += invoice.get("total_amount", 0)
            vendor_found = True
            break
    
    if not vendor_found:
        mock_analytics["top_vendors"].append({
            "name": vendor_name,
            "invoice_count": 1,
            "total_amount": invoice.get("total_amount", 0)
        })
    
    # Keep only top 10 vendors
    mock_analytics["top_vendors"] = sorted(
        mock_analytics["top_vendors"],
        key=lambda x: x["invoice_count"],
        reverse=True
    )[:10]

@router.get("/invoices")
async def get_invoices(
    limit: int = 50,
    offset: int = 0,
    status: Optional[str] = None,
    risk_level: Optional[str] = None
):
    """Get invoices with filtering and pagination"""
    # Prefer BigQuery ETL table when available; otherwise return in-memory mock
    try:
        bq = get_bq_manager()
    except Exception:
        bq = None

    if bq:
        # Query the ETL destination table: <project>.training.invoice_training
        table_ref = f"{bq.project_id}.training.invoice_training"
        filters = []
        # Simple vendor filter
        if status:
            # ETL table may not have verification_status; fallback to source filtering when present
            filters.append(f"LOWER(verification_status) = '{status.lower()}'")
        if risk_level:
            filters.append(f"LOWER(risk_level) = '{risk_level.lower()}'")

        where_clause = "WHERE " + " AND ".join(filters) if filters else ""

        sql = f"SELECT * FROM `{table_ref}` {where_clause} LIMIT {limit} OFFSET {offset}"

        try:
            df = bq.query(sql)
            records = []
            for _, row in df.iterrows():
                rec = {}
                for col in df.columns:
                    val = row[col]
                    try:
                        if hasattr(val, 'isoformat'):
                            rec[col] = val.isoformat()
                        else:
                            rec[col] = (val.item() if hasattr(val, 'item') else val)
                    except Exception:
                        rec[col] = str(val)
                records.append(rec)

            total = len(records)
            normalized = [normalize_invoice_record(r) for r in records]
            return {
                "invoices": normalized,
                "total": total,
                "limit": limit,
                "offset": offset,
                "has_more": offset + limit < total
            }
        except Exception:
            # On any BigQuery error, gracefully fall back to mock below
            pass

    # Fallback: in-memory mock invoices
    filtered_invoices = mock_invoices.copy()

    # Apply filters
    if status:
        filtered_invoices = [inv for inv in filtered_invoices if inv.get("verification_status") == status]

    if risk_level:
        filtered_invoices = [inv for inv in filtered_invoices if inv.get("risk_level") == risk_level]

    # Apply pagination
    total_count = len(filtered_invoices)
    paginated_invoices = filtered_invoices[offset:offset + limit]
    # Normalize for frontend
    normalized = [normalize_invoice_record(inv) for inv in paginated_invoices]

    return {
        "invoices": normalized,
        "total": total_count,
        "limit": limit,
        "offset": offset,
        "has_more": offset + limit < total_count
    }


def normalize_invoice_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Return a frontend-friendly invoice shape from BQ or internal records."""
    # Rec may have keys like invoice_id/invoice_number/vendor_name/total_amount
    invoice_id = rec.get('invoice_id') or rec.get('invoice_number') or rec.get('id')
    vendor = None
    vendor_obj = rec.get('vendor')
    if isinstance(vendor_obj, dict):
        # normalize vendor object with common fields
        vendor = vendor_obj.get('name') or vendor_obj.get('vendor_name')
        vendor = vendor or vendor_obj.get('company')
        vendor_normalized = {
            'name': vendor or rec.get('vendor_name') or rec.get('vendor') or 'Unknown Vendor',
            'address': vendor_obj.get('address') or vendor_obj.get('addr') or '',
            'phone': vendor_obj.get('phone') or vendor_obj.get('telephone') or '',
            'email': vendor_obj.get('email') or vendor_obj.get('contact_email') or '',
            'tax_id': vendor_obj.get('tax_id') or vendor_obj.get('tax') or ''
        }
    else:
        vendor_normalized = {
            'name': rec.get('vendor_name') or rec.get('vendor') or 'Unknown Vendor',
            'address': '', 'phone': '', 'email': '', 'tax_id': ''
        }

    amount = rec.get('total_amount') or rec.get('total') or rec.get('amount') or 0
    try:
        amount = float(amount) if amount is not None else 0
    except Exception:
        try:
            amount = float(str(amount))
        except Exception:
            amount = 0

    status = rec.get('verification_status') or rec.get('status') or rec.get('verification_status', 'processed')
    confidence = rec.get('confidence_score') or rec.get('confidence') or 0
    try:
        confidence = float(confidence)
    except Exception:
        confidence = 0

    issues = rec.get('issues')
    if issues is None:
        # derive from verification_results if present
        vres = rec.get('verification_results') or rec.get('agent_results')
        if isinstance(vres, list):
            issues = len(vres)
        elif isinstance(vres, dict):
            issues = len(vres)
        else:
            issues = 0

    date = rec.get('invoice_date') or rec.get('date') or rec.get('processed_date')
    description = rec.get('notes') or rec.get('description') or rec.get('analysis_summary') or ''

    # Normalize line_items if present (BQ repeated RECORD or other shapes)
    raw_items = rec.get('line_items') or rec.get('items') or rec.get('lineItems') or []
    line_items = []
    if isinstance(raw_items, list):
        for it in raw_items:
            if isinstance(it, dict):
                li = {
                    'description': it.get('description') or it.get('desc') or '',
                    'quantity': float(it.get('quantity') or it.get('qty') or 0),
                    'unit_price': float(it.get('unit_price') or it.get('price') or 0),
                    'total': float(it.get('total') or it.get('line_total') or (float(it.get('quantity') or 0) * float(it.get('unit_price') or 0))),
                    'sku': it.get('sku') or it.get('part') or ''
                }
                line_items.append(li)
    # else: if it's a string or other, leave empty list

    return {
        "id": invoice_id,
        "vendor": vendor_normalized,
        # backward-compatible vendor display string for frontend code expecting a string
        "vendor_display": vendor_normalized.get('name'),
        "amount": round(amount, 2),
        "status": status,
        "confidence": round(confidence, 2),
        "issues": issues if isinstance(issues, int) else (int(issues) if str(issues).isdigit() else 0),
        "date": date,
        "description": description,
        "line_items": line_items,
        # pass-through raw record for advanced UI use
        "_raw": rec
    }


@router.get("/invoices/bq")
async def get_invoices_bq(
    limit: int = 50,
    offset: int = 0,
    vendor: Optional[str] = None,
    min_total: Optional[float] = None,
    max_total: Optional[float] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
):
    """Fetch invoices from BigQuery using `bigquery_config.bq_manager`.
    Falls back to the in-memory mock if BQ is unavailable.
    """
    try:
        bq = get_bq_manager()
    except Exception:
        bq = None

    if not bq:
        # BigQuery not available - return the list of invoices (legacy behavior expected by some clients/tests)
        paged = await get_invoices(limit=limit, offset=offset, status=None, risk_level=None)
        return paged.get("invoices", [])

    # Build a basic SQL query based on allowed filters
    dataset_table = "fraud.invoices"  # conservative default; adjust if needed
    filters = []
    if vendor:
        filters.append(f"LOWER(vendor_name) LIKE '%{vendor.lower()}%'")
    if min_total is not None:
        filters.append(f"total_amount >= {float(min_total)}")
    if max_total is not None:
        filters.append(f"total_amount <= {float(max_total)}")
    if start_date:
        filters.append(f"invoice_date >= '{start_date}'")
    if end_date:
        filters.append(f"invoice_date <= '{end_date}'")

    where_clause = "WHERE " + " AND ".join(filters) if filters else ""
    sql = f"SELECT * FROM `{bq.project_id}.{bq.datasets.get('fraud').split('.')[-1]}.invoices` {where_clause} LIMIT {limit} OFFSET {offset}"

    try:
        df = bq.query(sql)
        # Convert dataframe rows to JSON-serializable dicts
        records = []
        for _, row in df.iterrows():
            rec = {}
            for col in df.columns:
                val = row[col]
                # Convert numpy types and pandas timestamps
                try:
                    if hasattr(val, 'isoformat'):
                        rec[col] = val.isoformat()
                    else:
                        rec[col] = (val.item() if hasattr(val, 'item') else val)
                except Exception:
                    rec[col] = str(val)
            records.append(rec)

        total = len(records)
        normalized = [normalize_invoice_record(r) for r in records]
        return {
            "invoices": normalized,
            "total": total,
            "limit": limit,
            "offset": offset,
            "has_more": offset + limit < total
        }
    except Exception:
        # On error, fallback to mock invoices and return the list (legacy expected shape)
        paged = await get_invoices(limit=limit, offset=offset, status=None, risk_level=None)
        return paged.get("invoices", [])


@router.get("/invoices/sample")
async def get_sample_invoices(limit: int = 25, dynamic: bool = False):
    """Return a small privacy-preserving sample for frontend development.

    - If `dynamic` is False (default), the endpoint will serve the static
      file `frontend/public/sample_invoices.json` if it exists.
    - If `dynamic` is True and BigQuery is available, it will query a small
      sample from `training.invoice_training`, sanitize/obfuscate it, and return it.
    """
    static_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'frontend', 'public', 'sample_invoices.json')
    static_path = os.path.normpath(static_path)

    # Serve static sample if present and dynamic not requested
    if not dynamic and os.path.exists(static_path):
        try:
            with open(static_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data[:limit]
        except Exception:
            # fall through to dynamic/dummy
            pass

    # Attempt dynamic sample from BigQuery when requested
    try:
        bq = get_bq_manager()
    except Exception:
        bq = None

    if dynamic and bq:
        try:
            table = f"{bq.project_id}.training.invoice_training"
            sql = f"SELECT * FROM `{table}` LIMIT {limit}`"
            df = bq.query(sql)
            records = []
            for _, row in df.iterrows():
                rec = {}
                for col in df.columns:
                    val = row[col]
                    try:
                        if hasattr(val, 'isoformat'):
                            rec[col] = val.isoformat()
                        else:
                            rec[col] = (val.item() if hasattr(val, 'item') else val)
                    except Exception:
                        rec[col] = str(val)
                records.append(rec)

            # Reuse same obfuscation used by export script
            from scripts.export_sample_invoices import sanitize_record
            sanitized = [sanitize_record(r) for r in records]
            return sanitized
        except Exception:
            pass

    # Last resort: return an empty list or mock invoices limited
    mock_limited = mock_invoices[:limit]
    return [normalize_invoice_record(m) for m in mock_limited]


def _score_invoice_heuristic(inv: Dict[str, Any]) -> Dict[str, Any]:
    """Compute a privacy-preserving heuristic confidence breakdown for an invoice.

    This function deliberately avoids calling external models or persisting data.
    It returns component scores and a combined confidence in [0,1].
    """
    # vendor score: prefer known vendors in analytics top_vendors
    vendor_name = inv.get('vendor', {}) and inv.get('vendor', {}).get('name') or ''
    top_vendors = [v['name'] for v in mock_analytics.get('top_vendors', [])]
    vendor_score = 0.8 if vendor_name in top_vendors and vendor_name else 0.5 if vendor_name else 0.4

    # totals score: compare subtotal+tax to total_amount
    try:
        subtotal = float(inv.get('subtotal') or 0)
        tax = float(inv.get('tax_amount') or 0)
        total = float(inv.get('total_amount') or inv.get('amount') or 0)
    except Exception:
        subtotal = tax = total = 0.0

    if total == 0:
        totals_score = 0.5
    else:
        diff = abs((subtotal + tax) - total)
        rel = diff / (total if total else 1)
        if rel < 0.01:
            totals_score = 0.95
        elif rel < 0.05:
            totals_score = 0.8
        elif rel < 0.15:
            totals_score = 0.6
        else:
            totals_score = 0.2

    # pattern score: more line items usually more reliable; empty line items lowers score
    line_items = inv.get('line_items') or []
    if isinstance(line_items, list) and line_items:
        pattern_score = min(0.9, 0.5 + 0.05 * len(line_items))
    else:
        pattern_score = 0.3

    # Combine scores (weighted)
    confidence = 0.4 * vendor_score + 0.4 * totals_score + 0.2 * pattern_score
    confidence = max(0.0, min(1.0, confidence))

    # Risk classification (lower confidence => higher risk)
    if confidence >= 0.8:
        risk = 'low'
    elif confidence >= 0.6:
        risk = 'medium'
    else:
        risk = 'high'

    return {
        'vendor_score': round(vendor_score, 2),
        'totals_score': round(totals_score, 2),
        'pattern_score': round(pattern_score, 2),
        'confidence': round(confidence, 3),
        'risk': risk,
    }


@router.post('/invoices/score')
async def score_invoices(payload: Dict[str, Any]):
    """Score a single invoice or list of invoices and return confidence breakdowns.

    This endpoint is safe for dev: it does not persist data or train models. If you
    want to run an external model, implement that separately and ensure data
    is obfuscated before sending.
    """
    try:
        items = payload.get('invoices') or payload
        # Accept either a list or a single invoice dict
        if isinstance(items, dict):
            items = [items]

        results = []
        for inv in items:
            # Normalize shape to frontend-friendly record
            rec = normalize_invoice_record(inv) if isinstance(inv, dict) else inv
            # Avoid passing raw data to any external model here
            # Prefer ADK agent pipeline scoring when available
            if FRAUD_DETECTION_AVAILABLE and pipeline_get is not None:
                try:
                    pipeline = pipeline_get()
                    # pipeline.run_detection returns a structured FraudDetectionResult/dict
                    fd_result = pipeline.run_detection(rec, max_iterations=1)
                    # fd_result may be a dict or object with expected fields
                    if isinstance(fd_result, dict):
                        score = {
                            'vendor_score': round(fd_result.get('results', {}).get('vendor', {}).get('confidence', 0.0), 2) if fd_result.get('results') else 0.0,
                            'totals_score': round(fd_result.get('results', {}).get('totals', {}).get('confidence', 0.0), 2) if fd_result.get('results') else 0.0,
                            'pattern_score': round(fd_result.get('results', {}).get('patterns', {}).get('confidence', 0.0), 2) if fd_result.get('results') else 0.0,
                            'confidence': round(fd_result.get('summary', {}).get('confidence_score', fd_result.get('confidence_score', 0.0)), 3) if fd_result.get('summary') or fd_result.get('confidence_score') else 0.0,
                            'risk': fd_result.get('summary', {}).get('fraud_risk', 'unknown') if fd_result.get('summary') else fd_result.get('risk', 'unknown')
                        }
                    else:
                        # For object types, try attribute access
                        try:
                            score = {
                                'vendor_score': round(getattr(fd_result, 'results', {}).get('vendor', {}).get('confidence', 0.0), 2),
                                'totals_score': round(getattr(fd_result, 'results', {}).get('totals', {}).get('confidence', 0.0), 2),
                                'pattern_score': round(getattr(fd_result, 'results', {}).get('patterns', {}).get('confidence', 0.0), 2),
                                'confidence': round(getattr(fd_result, 'summary', {}).confidence_score if getattr(fd_result, 'summary', None) else getattr(fd_result, 'confidence_score', 0.0), 3),
                                'risk': getattr(fd_result, 'summary', {}).fraud_risk if getattr(fd_result, 'summary', None) else getattr(fd_result, 'risk', 'unknown')
                            }
                        except Exception:
                            score = _score_invoice_heuristic(rec)
                except Exception:
                    # On any pipeline error, fall back to heuristic
                    score = _score_invoice_heuristic(rec)
            else:
                score = _score_invoice_heuristic(rec)

            results.append({'invoice_id': rec.get('id') or rec.get('invoice_number') or rec.get('id'), 'score': score})

        return {'results': results}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/invoices/{invoice_id}")
async def get_invoice(invoice_id: str):
    """Get a specific invoice by ID"""
    for invoice in mock_invoices:
        if invoice.get("id") == invoice_id or invoice.get("invoice_number") == invoice_id:
            return {"invoice": invoice}
    
    raise HTTPException(status_code=404, detail="Invoice not found")

@router.patch("/invoices/{invoice_id}/status")
async def update_invoice_status(invoice_id: str, status_update: InvoiceStatus):
    """Update invoice status"""
    for invoice in mock_invoices:
        if invoice.get("id") == invoice_id or invoice.get("invoice_number") == invoice_id:
            invoice["verification_status"] = status_update.status
            return {"success": True, "invoice": invoice}
    
    raise HTTPException(status_code=404, detail="Invoice not found")

# Analytics endpoints
@router.get("/analytics")
async def get_analytics():
    """Get analytics dashboard data"""
    return mock_analytics

# Agent configuration endpoints
@router.get("/agents/config")
async def get_agent_config():
    """Get current agent configuration"""
    return {
        "agents": {
            "vendor_check": {"enabled": True, "confidence": 0.85},
            "amount_check": {"enabled": True, "confidence": 0.90},
            "pattern_analysis": {"enabled": True, "confidence": 0.78}
        },
        "pipeline_status": "active"
    }

@router.put("/agents/config")
async def update_agent_config(config: Dict[str, Any]):
    """Update agent configuration"""
    try:
        # This would update your agent configuration
        # For now, return success
        return {
            "success": True,
            "message": "Agent configuration updated",
            "config": config
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update config: {str(e)}")

# Fraud detection endpoints (existing functionality)
@router.post("/analyze")
async def analyze_invoice_fraud(payload: Dict[str, Any]):
    """Analyze invoice for fraud using existing pipeline"""
    try:
        result = run_fraud_detection(payload, max_iterations=1)
        
        if hasattr(result, "to_dict"):
            return result.to_dict()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
