# backend/main.py - FastAPI backend with file upload support
import os
import shutil
from pathlib import Path
from typing import List
from datetime import datetime

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(
    title="AgentZero API", 
    version="1.0.0",
    description="FastAPI backend for AgentZero invoice processing system"
)

# CORS middleware - Allow Node.js frontend to communicate
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Local Node.js dev server
        "https://agentzero.azurewebsites.net",  # Azure production
        "http://frontend:3000",  # Docker network
        "*"  # Allow all origins for development
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Create uploads directory if it doesn't exist
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "AgentZero FastAPI Backend", 
        "status": "running",
        "version": "1.0.0",
        "docs": "/docs",
        "upload_endpoint": "/api/upload"
    }

# Health check endpoints
@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "message": "FastAPI backend is running",
        "service": "backend-api",
        "port": 8000,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/health")
async def api_health_check():
    return {
        "status": "healthy", 
        "message": "FastAPI API is running",
        "endpoints": ["health", "message", "upload", "invoices"],
        "upload_dir": str(UPLOAD_DIR.absolute()),
        "upload_dir_exists": UPLOAD_DIR.exists()
    }

# Basic API message endpoint
@app.get("/api/message")
async def get_message():
    return {
        "message": "Hello from AgentZero FastAPI backend!", 
        "status": "success",
        "timestamp": datetime.now().isoformat()
    }

# FILE UPLOAD ENDPOINT - Main feature for connecting to frontend
@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload file endpoint for invoice processing
    Accepts PDF, PNG, JPG, JPEG, XLSX, XLS files
    """
    try:
        # Validate file type
        allowed_extensions = {'.pdf', '.png', '.jpg', '.jpeg', '.xlsx', '.xls'}
        file_extension = Path(file.filename).suffix.lower()
        
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"File type {file_extension} not allowed. Allowed types: {', '.join(allowed_extensions)}"
            )
        
        # Create unique filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_filename = f"{timestamp}_{file.filename}"
        file_path = UPLOAD_DIR / safe_filename
        
        # Save file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Get file info
        file_size = file_path.stat().st_size
        file_size_mb = round(file_size / (1024 * 1024), 2)
        
        print(f"✅ File uploaded successfully: {safe_filename} ({file_size_mb}MB)")
        
        # Return success response
        return {
            "success": True,
            "message": f"File '{file.filename}' uploaded successfully!",
            "data": {
                "filename": file.filename,
                "saved_as": safe_filename,
                "size_bytes": file_size,
                "size_mb": file_size_mb,
                "file_type": file_extension,
                "upload_time": datetime.now().isoformat(),
                "file_path": str(file_path)
            },
            "next_steps": "File is ready for processing by fraud detection agents"
        }
        
    except Exception as e:
        print(f"❌ Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

# Multiple file upload endpoint
@app.post("/api/upload/multiple")
async def upload_multiple_files(files: List[UploadFile] = File(...)):
    """
    Upload multiple files at once
    """
    results = []
    errors = []
    
    for file in files:
        try:
            # Use the single file upload logic
            result = await upload_file(file)
            results.append(result)
        except Exception as e:
            errors.append({
                "filename": file.filename,
                "error": str(e)
            })
    
    return {
        "success": len(results) > 0,
        "message": f"Processed {len(files)} files. {len(results)} successful, {len(errors)} failed.",
        "successful_uploads": results,
        "failed_uploads": errors,
        "summary": {
            "total_files": len(files),
            "successful": len(results),
            "failed": len(errors)
        }
    }

# Get uploaded files list
@app.get("/api/uploads")
async def list_uploaded_files():
    """
    List all uploaded files
    """
    try:
        files = []
        for file_path in UPLOAD_DIR.glob("*"):
            if file_path.is_file():
                stat = file_path.stat()
                files.append({
                    "filename": file_path.name,
                    "size_bytes": stat.st_size,
                    "size_mb": round(stat.st_size / (1024 * 1024), 2),
                    "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
                })
        
        return {
            "success": True,
            "files": sorted(files, key=lambda x: x['created'], reverse=True),
            "count": len(files),
            "upload_directory": str(UPLOAD_DIR.absolute())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list files: {str(e)}")

# Invoice processing endpoints (existing)
@app.post("/api/invoices/upload")
async def upload_invoice(invoice_data: dict):
    """
    Process uploaded invoice for fraud detection
    """
    try:
        # Placeholder for your invoice processing logic
        result = {
            "invoice_id": "inv_123456",
            "status": "processed",
            "fraud_score": 0.15,
            "risk_level": "low",
            "analysis": {
                "amount_validated": True,
                "vendor_verified": True,
                "anomalies_detected": [],
                "confidence": 0.85
            },
            "processing_time_ms": 1250
        }
        
        return {
            "success": True,
            "data": result,
            "message": "Invoice processed successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Invoice processing failed: {str(e)}")

@app.get("/api/invoices/{invoice_id}")
async def get_invoice(invoice_id: str):
    """
    Get invoice analysis results
    """
    return {
        "invoice_id": invoice_id,
        "status": "completed",
        "created_at": datetime.now().isoformat(),
        "analysis": {
            "fraud_score": 0.15,
            "risk_level": "low"
        }
    }

# System status endpoint
@app.get("/api/system/status")
async def get_system_status():
    return {
        "system": "operational",
        "services": {
            "api": "healthy",
            "file_upload": "ready",
            "database": "connected",
            "ml_engine": "ready"
        },
        "uptime": "99.9%",
        "version": "1.0.0",
        "upload_directory": str(UPLOAD_DIR.absolute()),
        "upload_dir_exists": UPLOAD_DIR.exists()
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

# FastAPI runs on port 8000 as API-only service
if __name__ == "__main__":
    import uvicorn
    print(f"🚀 Starting FastAPI server...")
    print(f"📁 Upload directory: {UPLOAD_DIR.absolute()}")
    uvicorn.run(app, host="0.0.0.0", port=8000)