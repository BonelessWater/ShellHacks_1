#!/usr/bin/env python3
"""
Simple FastAPI Backend for Docker - No complex dependencies
"""

import os
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
    description="Simple API for Docker deployment",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
    return {"message": "Invoice Fraud Detection API - Docker Version", "status": "running"}

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        message="API is running in Docker",
        backend_connected=True
    )

@app.get("/api/health")
async def api_health_check():
    """API health check endpoint"""
    return {
        "status": "healthy",
        "message": "FastAPI API is running in Docker",
        "backend_connected": True,
        "fraud_detector": "mock_mode",
        "timestamp": datetime.now().isoformat(),
        "version": "docker-1.0.0"
    }

@app.post("/api/analyze")
async def analyze_invoice(request: InvoiceAnalysisRequest):
    """Analyze invoice for fraud indicators - Mock version for Docker"""
    try:
        invoice_data = request.invoice_data
        
        # Simple mock analysis
        mock_result = {
            "overall_risk_score": 7.5,
            "confidence": 8.0,
            "recommendation": "MANUAL_REVIEW",
            "status": "MEDIUM_RISK",
            "analysis": "Mock fraud analysis completed in Docker. Detected potential risk indicators.",
            "red_flags": [
                "Unusual invoice amount pattern",
                "Vendor not in approved list",
                "Missing required documentation"
            ],
            "invoice_id": invoice_data.get("invoice_id", "unknown"),
            "processing_time": 1.2,
            "docker_version": True
        }
        
        logger.info(f"Analyzed invoice: {invoice_data.get('invoice_id', 'unknown')}")
        return mock_result
        
    except Exception as e:
        logger.error(f"Error analyzing invoice: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/status")
async def get_status():
    """Get system status"""
    return {
        "status": "running",
        "environment": "docker",
        "python_version": "3.11",
        "fastapi_version": "0.104.1",
        "uptime": "running"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)