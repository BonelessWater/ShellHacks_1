#!/usr/bin/env python3
"""
Simple FastAPI Backend for Invoice Fraud Detection System
Serves the fraud detection API endpoints for the React frontend
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Invoice Fraud Detection API",
    description="API for analyzing invoices for fraud indicators",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class InvoiceAnalysisRequest(BaseModel):
    invoice_data: Dict[str, Any]

class HealthResponse(BaseModel):
    status: str
    message: str
    backend_connected: bool

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Invoice Fraud Detection API", "status": "running"}

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        message="API is running",
        backend_connected=True
    )

@app.get("/api/health")
async def api_health_check():
    """API health check endpoint"""
    return {
        "status": "healthy",
        "message": "FastAPI API is running",
        "backend_connected": True,
        "fraud_detector": "mock_mode",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/analyze")
async def analyze_invoice(request: InvoiceAnalysisRequest):
    """Analyze invoice for fraud indicators"""
    try:
        # Mock analysis for demo purposes
        invoice_data = request.invoice_data

        # Simulate analysis
        mock_result = {
            "overall_risk_score": 7.5,
            "confidence": 8.0,
            "recommendation": "MANUAL_REVIEW",
            "status": "MEDIUM_RISK",
            "analysis": "Mock fraud analysis completed. Detected potential risk indicators in invoice data.",
            "red_flags": ["HIGH_AMOUNT", "SUSPICIOUS_VENDOR", "WIRE_TRANSFER_ONLY"],
            "agent_results": [
                {
                    "agent": "amount_validator",
                    "risk_score": 8,
                    "confidence": 9,
                    "execution_time": 0.15,
                    "analysis": "Detected unusually round amounts suggesting potential fabrication",
                    "red_flags": ["HIGH_ROUND_AMOUNTS"]
                },
                {
                    "agent": "vendor_validator",
                    "risk_score": 9,
                    "confidence": 8,
                    "execution_time": 0.12,
                    "analysis": "Vendor name contains suspicious patterns commonly used in fraud",
                    "red_flags": ["SUSPICIOUS_VENDOR_NAME"]
                },
                {
                    "agent": "payment_analyzer",
                    "risk_score": 9,
                    "confidence": 9,
                    "execution_time": 0.08,
                    "analysis": "Wire transfer only payment method is a major red flag",
                    "red_flags": ["WIRE_TRANSFER_ONLY"]
                }
            ],
            "agents_used": 3,
            "total_execution_time": 0.35,
            "fraud_indicators": [
                {"type": "amount_anomaly", "severity": "high", "description": "Unusually round amounts"},
                {"type": "vendor_suspicion", "severity": "high", "description": "Suspicious vendor naming pattern"},
                {"type": "payment_risk", "severity": "high", "description": "Wire transfer only requirement"}
            ]
        }

        return JSONResponse(status_code=200, content=mock_result)

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/api/upload")
async def upload_invoice(file: UploadFile = File(...)):
    """Upload and analyze invoice file"""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    try:
        # Read file content
        content = await file.read()

        # Try to decode as text
        try:
            invoice_text = content.decode('utf-8')
        except UnicodeDecodeError:
            raise HTTPException(status_code=400, detail="File must be text-readable")

        # Create analysis request
        request = InvoiceAnalysisRequest(invoice_data={"text": invoice_text, "filename": file.filename})
        return await analyze_invoice(request)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.get("/api/status")
async def get_system_status():
    """Get system status for frontend"""
    return {
        "backend_connected": True,
        "fraud_detector": "mock_mode",
        "api_version": "1.0.0",
        "uptime": "running",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/message")
async def get_message():
    """Basic message endpoint"""
    return {
        "message": "Hello from Invoice Fraud Detection API!",
        "status": "success",
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "main_simple:app",
        host="0.0.0.0",
        port=port,
        reload=False
    )