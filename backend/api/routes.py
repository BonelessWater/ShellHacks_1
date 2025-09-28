# backend/api/routes.py
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from datetime import datetime, timedelta
import json
import asyncio
import uuid

# For now, use simplified fraud detection without complex imports
FRAUD_DETECTION_AVAILABLE = False

def run_fraud_detection(payload, max_iterations=1):
    """Simplified fraud detection that returns mock results"""
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
    filtered_invoices = mock_invoices.copy()
    
    # Apply filters
    if status:
        filtered_invoices = [inv for inv in filtered_invoices if inv.get("verification_status") == status]
    
    if risk_level:
        filtered_invoices = [inv for inv in filtered_invoices if inv.get("risk_level") == risk_level]
    
    # Apply pagination
    total_count = len(filtered_invoices)
    paginated_invoices = filtered_invoices[offset:offset + limit]
    
    return {
        "invoices": paginated_invoices,
        "total": total_count,
        "limit": limit,
        "offset": offset,
        "has_more": offset + limit < total_count
    }

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
