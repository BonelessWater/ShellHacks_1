# backend/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, Any
import os
import uvicorn
from datetime import datetime

# Initialize FastAPI app
app = FastAPI(
    title="ShellHacks Invoice API",
    description="Backend API for Invoice Processing System",
    version="1.0.0"
)

# CORS configuration
origins = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class MessageResponse(BaseModel):
    message: str
    timestamp: str
    status: str
    data: Dict[str, Any] = None

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str

# Routes
@app.get("/")
async def root():
    """Root endpoint with welcome message"""
    return MessageResponse(
        message="Welcome to ShellHacks Invoice API! 🚀",
        timestamp=datetime.now().isoformat(),
        status="success",
        data={
            "service": "invoice-processing-api",
            "environment": os.getenv("ENVIRONMENT", "development"),
            "endpoints": [
                "/health",
                "/api/message",
                "/api/invoice/process",
                "/docs"
            ]
        }
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for Docker health checks"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="1.0.0"
    )

@app.get("/api/message", response_model=MessageResponse)
async def get_message():
    """Simple message endpoint to test frontend-backend communication"""
    return MessageResponse(
        message="Hello from Python Backend! 🐍 Your Docker setup is working perfectly!",
        timestamp=datetime.now().isoformat(),
        status="success",
        data={
            "backend_server": "FastAPI",
            "python_version": "3.11",
            "docker_status": "running",
            "message_id": f"msg_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        }
    )

@app.post("/api/invoice/process")
async def process_invoice(invoice_data: dict):
    """Mock endpoint for invoice processing"""
    return MessageResponse(
        message="Invoice processed successfully! (Mock response)",
        timestamp=datetime.now().isoformat(),
        status="success",
        data={
            "invoice_id": f"inv_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "processed_at": datetime.now().isoformat(),
            "status": "processed",
            "received_data": invoice_data
        }
    )

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    return JSONResponse(
        status_code=500,
        content={
            "message": "Internal server error occurred",
            "status": "error",
            "timestamp": datetime.now().isoformat()
        }
    )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )