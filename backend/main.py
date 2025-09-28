# backend/main.py - Updated with file upload support
import os
from pathlib import Path
from typing import List
import json

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="ShellHacks Invoice System", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Routes
@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "Backend is running"}

@app.get("/api/health")
async def api_health_check():
    return {"status": "healthy", "message": "Backend API is running"}

@app.get("/api/message")
async def get_message():
    return {"message": "Hello from ShellHacks backend!", "status": "success"}

@app.get("/")
async def root():
    return {"message": "ShellHacks Invoice System API", "status": "running"}

# File Upload Endpoint
@app.post("/api/invoices/upload")
async def upload_invoice(files: List[UploadFile] = File(...)):
    """
    Handle file upload from FileDropZone component
    Accepts multiple files: PDF, PNG, JPG, JPEG, XLSX, XLS
    """
    try:
        processed_files = []
        
        for file in files:
            # Validate file type
            allowed_types = {
                'application/pdf': '.pdf',
                'image/png': '.png', 
                'image/jpeg': '.jpg',
                'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': '.xlsx',
                'application/vnd.ms-excel': '.xls'
            }
            
            # Check file extension as fallback
            file_ext = Path(file.filename).suffix.lower()
            allowed_extensions = ['.pdf', '.png', '.jpg', '.jpeg', '.xlsx', '.xls']
            
            if file.content_type not in allowed_types and file_ext not in allowed_extensions:
                raise HTTPException(
                    status_code=400, 
                    detail=f"File type not supported: {file.filename}. Allowed types: PDF, PNG, JPG, XLSX, XLS"
                )
            
            # Read file content
            content = await file.read()
            file_size = len(content)
            
            # Process file (placeholder for your fraud detection logic)
            file_result = {
                "filename": file.filename,
                "size": file_size,
                "content_type": file.content_type,
                "status": "processed",
                "fraud_analysis": {
                    "risk_score": 0.25,  # Mock score
                    "risk_level": "low",
                    "confidence": 0.88,
                    "flags": [],
                    "processing_time_ms": 1500
                }
            }
            
            processed_files.append(file_result)
        
        # Return success response
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": f"Successfully processed {len(files)} file(s)",
                "files": processed_files,
                "summary": {
                    "total_files": len(files),
                    "processed": len(processed_files),
                    "failed": 0,
                    "avg_risk_score": sum(f["fraud_analysis"]["risk_score"] for f in processed_files) / len(processed_files)
                }
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"File processing failed: {str(e)}"
        )

# Additional invoice endpoints
@app.get("/api/invoices/{invoice_id}")
async def get_invoice(invoice_id: str):
    """Get invoice analysis results"""
    return {
        "invoice_id": invoice_id,
        "status": "completed",
        "created_at": "2024-01-01T00:00:00Z",
        "analysis": {
            "fraud_score": 0.15,
            "risk_level": "low"
        }
    }

@app.get("/api/system/status")
async def get_system_status():
    return {
        "system": "operational",
        "services": {
            "api": "healthy",
            "upload": "ready",
            "fraud_detection": "ready"
        },
        "uptime": "99.9%",
        "version": "1.0.0"
    }

# Agent configuration endpoints
@app.get("/api/agents/config")
async def get_agent_config():
    return {
        "agents": {
            "fraud_detector": {
                "enabled": True,
                "model": "advanced_v2",
                "threshold": 0.7
            },
            "invoice_parser": {
                "enabled": True,
                "accuracy": 0.95
            }
        }
    }

@app.put("/api/agents/config")
async def update_agent_config(config: dict):
    return {
        "success": True,
        "message": "Agent configuration updated",
        "config": config
    }

# Serve React static files for non-API routes when build exists
frontend_build_path = Path(__file__).parent / "frontend_build"

if frontend_build_path.exists():
    # Mount static files
    app.mount(
        "/static",
        StaticFiles(directory=str(frontend_build_path / "static")),
        name="static",
    )

    # Serve React app for all non-API routes
    @app.get("/{full_path:path}")
    async def serve_react_app(full_path: str):
        # Don't serve React for API routes
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="API endpoint not found")

        # For all other routes, serve React's index.html
        index_file = frontend_build_path / "index.html"
        if index_file.exists():
            return FileResponse(index_file)
        else:
            raise HTTPException(status_code=404, detail="Frontend not found")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)