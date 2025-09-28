# backend/main.py - FastAPI backend with file upload support
import os
import shutil
import json
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Import your existing fraud detection modules
try:
    from .parallel_llm_agents import ParallelLLMExecutor, LLMTask, get_fraud_detection_agent_configs
    from .agent_definitions import FRAUD_DETECTION_AGENTS
except ImportError:
    # Fallback for direct execution
    from parallel_llm_agents import ParallelLLMExecutor, LLMTask, get_fraud_detection_agent_configs
    from agent_definitions import FRAUD_DETECTION_AGENTS

# Import API routes
try:
    from .api.routes import router as api_router
except ImportError:
    # Fallback for direct execution from backend directory
    from api.routes import router as api_router

# Create upload directory
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

app = FastAPI(title="ShellHacks Invoice System", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router, prefix="/api")

# Basic health check at root level
@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "Backend API is running"}

# Pydantic models for request/response
class InvoiceAnalysisRequest(BaseModel):
    invoice_data: Dict[str, Any]
    analysis_type: Optional[str] = "full"
    priority: Optional[int] = 0

class InvoiceAnalysisResponse(BaseModel):
    invoice_id: str
    overall_risk_score: float
    confidence: float
    status: str
    recommendation: str
    red_flags: List[str]
    agent_results: List[Dict[str, Any]]
    analysis_summary: str
    processing_time: float

# Initialize the parallel LLM fraud detection system
llm_executor = None

async def get_llm_executor():
    """Initialize and return the LLM executor singleton"""
    global llm_executor
    if llm_executor is None:
        llm_executor = ParallelLLMExecutor(max_workers=6)
        # Register all fraud detection agents
        for config in get_fraud_detection_agent_configs():
            llm_executor.add_agent_type(config)
    return llm_executor

# Root endpoint
@app.get("/")
async def root():
    return {"message": "ShellHacks Invoice System API", "status": "running"}

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
        allowed_extensions = {'.pdf', '.png', '.jpg', '.jpeg', '.xlsx', '.xls', '.json'}
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
        
        from main_detector import main as run_detector
        analysis_result = await run_detector(file=str(file_path))

        print(f"📝 Analysis Result: {json.dumps(analysis_result, indent=2)}")

        # Return success response
        return {
            "success": True,
            "message": f"Analysis Result: {json.dumps(analysis_result, indent=2)}",
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

# Enhanced Invoice Analysis with Parallel LLM Agents
@app.post("/api/invoices/analyze")
async def analyze_invoice(request: InvoiceAnalysisRequest, background_tasks: BackgroundTasks):
    """
    Analyze invoice using parallel LLM fraud detection agents
    """
    try:
        start_time = datetime.now()
        executor = await get_llm_executor()

        invoice_data = request.invoice_data
        invoice_id = invoice_data.get('invoice_id', f"inv_{int(datetime.now().timestamp())}")

        print(f"🔍 Starting invoice analysis for: {invoice_id}")

        # Create tasks for all available fraud detection agents
        tasks = []
        agent_types = executor.agent_registry.get_available_agent_types()

        for agent_type in agent_types:
            task = LLMTask(
                task_id=f"{agent_type}_{invoice_id}",
                data=invoice_data,
                agent_names=[agent_type],
                timeout=45.0,
                context={
                    "invoice_id": invoice_id,
                    "analysis_type": request.analysis_type,
                    "priority": request.priority
                }
            )
            tasks.append(task)

        print(f"📋 Created {len(tasks)} analysis tasks")

        # Execute all agents in parallel with synchronization
        results = await executor.execute_tasks_parallel(
            tasks,
            wait_for_all=True,  # Wait for all agents to complete
            aggregate_results=False  # We'll do custom aggregation
        )

        # Aggregate results
        analysis_result = await aggregate_fraud_analysis_results(results, invoice_data, start_time)

        print(f"✅ Analysis completed for {invoice_id}: Risk {analysis_result['overall_risk_score']}/10")

        return {
            "success": True,
            "data": analysis_result,
            "message": f"Invoice {invoice_id} analyzed successfully by {len(results)} agents"
        }

    except Exception as e:
        print(f"❌ Invoice analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Invoice analysis failed: {str(e)}")

async def aggregate_fraud_analysis_results(results, invoice_data, start_time):
    """Aggregate results from all fraud detection agents"""
    successful_results = [r for r in results if r.success]

    invoice_id = invoice_data.get('invoice_id', 'unknown')
    processing_time = (datetime.now() - start_time).total_seconds()

    if not successful_results:
        return {
            "invoice_id": invoice_id,
            "overall_risk_score": 10.0,
            "confidence": 1.0,
            "status": "ANALYSIS_FAILED",
            "recommendation": "MANUAL_REVIEW",
            "red_flags": ["ANALYSIS_FAILURE"],
            "agent_results": [],
            "analysis_summary": "Analysis failed - no agents completed successfully",
            "processing_time": processing_time
        }

    # Extract metrics from successful results
    risk_scores = []
    confidences = []
    all_red_flags = []
    agent_results = []

    for result in successful_results:
        try:
            if isinstance(result.result, dict):
                risk_score = result.result.get('risk_score', 5)
                confidence = result.result.get('confidence', 5)
                red_flags = result.result.get('red_flags', [])
                analysis = result.result.get('analysis', 'No analysis provided')
            elif isinstance(result.result, str):
                # Try to parse JSON string
                try:
                    parsed = json.loads(result.result)
                    risk_score = parsed.get('risk_score', 5)
                    confidence = parsed.get('confidence', 5)
                    red_flags = parsed.get('red_flags', [])
                    analysis = parsed.get('analysis', result.result)
                except json.JSONDecodeError:
                    risk_score = 5
                    confidence = 3
                    red_flags = []
                    analysis = str(result.result)
            else:
                risk_score = 5
                confidence = 3
                red_flags = []
                analysis = "Analysis result format not recognized"

            risk_scores.append(risk_score)
            confidences.append(confidence)
            all_red_flags.extend(red_flags if isinstance(red_flags, list) else [])

            agent_results.append({
                "agent_name": result.agent_name,
                "agent_id": result.agent_id,
                "risk_score": risk_score,
                "confidence": confidence,
                "execution_time": result.execution_time,
                "red_flags": red_flags if isinstance(red_flags, list) else [],
                "analysis": analysis,
                "status": result.status.value if hasattr(result.status, 'value') else str(result.status)
            })

        except Exception as e:
            print(f"⚠️ Error processing result from {result.agent_name}: {e}")
            continue

    # Calculate weighted average (higher confidence = more weight)
    if risk_scores and confidences:
        total_weight = sum(confidences)
        if total_weight > 0:
            weighted_risk = sum(r * c for r, c in zip(risk_scores, confidences)) / total_weight
        else:
            weighted_risk = sum(risk_scores) / len(risk_scores)
        avg_confidence = sum(confidences) / len(confidences)
    else:
        weighted_risk = 5.0
        avg_confidence = 1.0

    # Determine recommendation based on risk score
    if weighted_risk >= 8:
        recommendation = "REJECT"
        status = "HIGH_RISK"
    elif weighted_risk >= 6:
        recommendation = "MANUAL_REVIEW"
        status = "MEDIUM_RISK"
    elif weighted_risk >= 4:
        recommendation = "ADDITIONAL_VERIFICATION"
        status = "LOW_MEDIUM_RISK"
    else:
        recommendation = "APPROVE"
        status = "LOW_RISK"

    # Create summary
    unique_red_flags = list(set(flag for flag in all_red_flags if flag and flag.strip()))

    return {
        "invoice_id": invoice_id,
        "overall_risk_score": round(weighted_risk, 1),
        "confidence": round(avg_confidence, 1),
        "status": status,
        "recommendation": recommendation,
        "red_flags": unique_red_flags,
        "agent_results": agent_results,
        "analysis_summary": f"Analyzed by {len(successful_results)} specialized fraud detection agents. "
                          f"Risk assessment: {status.replace('_', ' ').title()}",
        "processing_time": round(processing_time, 2),
        "metadata": {
            "total_agents_executed": len(results),
            "successful_agents": len(successful_results),
            "failed_agents": len(results) - len(successful_results),
            "unique_red_flags_count": len(unique_red_flags),
            "analysis_timestamp": datetime.now().isoformat()
        }
    }

# Legacy upload endpoint for backward compatibility
@app.post("/api/invoices/upload")
async def upload_invoice(invoice_data: dict):
    """
    Legacy endpoint - redirects to new analysis endpoint
    """
    try:
        request = InvoiceAnalysisRequest(invoice_data=invoice_data)
        result = await analyze_invoice(request, BackgroundTasks())
        return result
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

# Additional FastAPI Endpoints for Enhanced Invoice Processing
# Add these to your existing backend/main.py

from typing import List, Optional
from fastapi import HTTPException, UploadFile, File, Form, BackgroundTasks
from pydantic import BaseModel
import asyncio
import uuid
from datetime import datetime

# Additional Pydantic Models
class UploadStatus(BaseModel):
    file_id: str
    filename: str
    status: str  # 'uploading', 'processing', 'completed', 'failed'
    progress: Optional[int] = 0
    current_step: Optional[str] = None
    error: Optional[str] = None
    processing_time: Optional[float] = None
    result: Optional[dict] = None

class BatchUploadResponse(BaseModel):
    batch_id: str
    total_files: int
    successful_uploads: int
    failed_uploads: int
    upload_statuses: List[UploadStatus]

class InvoiceProcessingStatus(BaseModel):
    invoice_id: str
    status: str
    progress: int
    current_agent: Optional[str] = None
    agents_completed: int
    total_agents: int
    estimated_time_remaining: Optional[float] = None

# In-memory storage for tracking uploads (use Redis in production)
upload_tracking = {}
processing_status = {}

# Enhanced file upload endpoint with progress tracking
@app.post("/api/upload/batch", response_model=BatchUploadResponse)
async def upload_batch_files(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...)
):
    """
    Upload multiple files with progress tracking
    """
    batch_id = str(uuid.uuid4())
    upload_statuses = []
    successful_uploads = 0
    failed_uploads = 0

    for file in files:
        file_id = str(uuid.uuid4())
        
        try:
            # Validate file
            allowed_extensions = {'.pdf', '.png', '.jpg', '.jpeg', '.xlsx', '.xls', '.json'}
            file_extension = Path(file.filename).suffix.lower()
            
            if file_extension not in allowed_extensions:
                status = UploadStatus(
                    file_id=file_id,
                    filename=file.filename,
                    status="failed",
                    error=f"File type {file_extension} not allowed"
                )
                upload_statuses.append(status)
                failed_uploads += 1
                continue

            # Save file
            file_path = UPLOAD_DIR / f"{file_id}_{file.filename}"
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)

            # Create upload status
            status = UploadStatus(
                file_id=file_id,
                filename=file.filename,
                status="completed",
                progress=100,
                processing_time=0.5  # Mock processing time
            )
            
            upload_tracking[file_id] = status
            upload_statuses.append(status)
            successful_uploads += 1

            # Start background processing
            background_tasks.add_task(process_invoice_async, file_id, file_path)

        except Exception as e:
            status = UploadStatus(
                file_id=file_id,
                filename=file.filename,
                status="failed",
                error=str(e)
            )
            upload_statuses.append(status)
            failed_uploads += 1

    return BatchUploadResponse(
        batch_id=batch_id,
        total_files=len(files),
        successful_uploads=successful_uploads,
        failed_uploads=failed_uploads,
        upload_statuses=upload_statuses
    )

# Get upload status
@app.get("/api/upload/status/{file_id}")
async def get_upload_status(file_id: str):
    """
    Get the status of a specific file upload/processing
    """
    if file_id not in upload_tracking:
        raise HTTPException(status_code=404, detail="File not found")
    
    return upload_tracking[file_id]

# Get processing status for an invoice
@app.get("/api/invoices/{invoice_id}/status", response_model=InvoiceProcessingStatus)
async def get_processing_status(invoice_id: str):
    """
    Get real-time processing status for an invoice
    """
    if invoice_id not in processing_status:
        raise HTTPException(status_code=404, detail="Invoice not found")
    
    return processing_status[invoice_id]

# Enhanced invoice analysis with progress tracking
@app.post("/api/invoices/analyze/async")
async def analyze_invoice_async(
    background_tasks: BackgroundTasks,
    request: InvoiceAnalysisRequest
):
    """
    Start asynchronous invoice analysis with progress tracking
    """
    invoice_id = request.invoice_data.get('invoice_id', f"inv_{int(datetime.now().timestamp())}")
    
    # Initialize processing status
    status = InvoiceProcessingStatus(
        invoice_id=invoice_id,
        status="started",
        progress=0,
        agents_completed=0,
        total_agents=6,  # Based on your fraud detection agents
        estimated_time_remaining=30.0
    )
    processing_status[invoice_id] = status
    
    # Start background processing
    background_tasks.add_task(analyze_invoice_background, invoice_id, request)
    
    return {
        "invoice_id": invoice_id,
        "status": "processing_started",
        "message": "Invoice analysis started. Use /api/invoices/{invoice_id}/status to track progress."
    }

# Background task for invoice processing simulation
async def process_invoice_async(file_id: str, file_path: Path):
    """
    Background task to simulate invoice processing with status updates
    """
    try:
        status = upload_tracking[file_id]
        
        # Simulate processing steps
        steps = [
            ("Data extraction", 20),
            ("Fraud detection", 40),
            ("Vendor validation", 60),
            ("Amount verification", 80),
            ("Final analysis", 100)
        ]
        
        for step_name, progress in steps:
            status.status = "processing"
            status.current_step = step_name
            status.progress = progress
            upload_tracking[file_id] = status
            
            # Simulate processing time
            await asyncio.sleep(2)
        
        # Complete processing
        status.status = "completed"
        status.progress = 100
        status.result = {
            "riskScore": 25,  # Mock result
            "confidence": 95,
            "status": "approved"
        }
        upload_tracking[file_id] = status
        
    except Exception as e:
        status.status = "failed"
        status.error = str(e)
        upload_tracking[file_id] = status

# Background task for invoice analysis
async def analyze_invoice_background(invoice_id: str, request: InvoiceAnalysisRequest):
    """
    Background task for invoice analysis with progress updates
    """
    try:
        status = processing_status[invoice_id]
        
        # Simulate agent processing
        agents = [
            "Data Extractor",
            "Fraud Detector", 
            "Vendor Validator",
            "Amount Checker",
            "Duplicate Scanner",
            "Risk Analyzer"
        ]
        
        for i, agent in enumerate(agents):
            status.current_agent = agent
            status.progress = int((i + 1) / len(agents) * 100)
            status.agents_completed = i + 1
            status.estimated_time_remaining = (len(agents) - i - 1) * 3
            processing_status[invoice_id] = status
            
            # Simulate agent processing time
            await asyncio.sleep(3)
        
        # Complete analysis
        status.status = "completed"
        status.progress = 100
        status.current_agent = None
        status.estimated_time_remaining = 0
        processing_status[invoice_id] = status
        
    except Exception as e:
        status.status = "failed"
        processing_status[invoice_id] = status

# Bulk invoice processing endpoint
@app.post("/api/invoices/process/bulk")
async def process_bulk_invoices(
    background_tasks: BackgroundTasks,
    invoice_ids: List[str]
):
    """
    Process multiple invoices in parallel
    """
    batch_id = str(uuid.uuid4())
    
    for invoice_id in invoice_ids:
        # Create mock invoice request
        mock_request = InvoiceAnalysisRequest(
            invoice_data={"invoice_id": invoice_id}
        )
        background_tasks.add_task(analyze_invoice_background, invoice_id, mock_request)
    
    return {
        "batch_id": batch_id,
        "message": f"Started processing {len(invoice_ids)} invoices",
        "invoice_ids": invoice_ids
    }

# Get batch processing status
@app.get("/api/invoices/batch/{batch_id}/status")
async def get_batch_status(batch_id: str):
    """
    Get status of batch processing
    """
    # This would typically query your database
    # For demo, return mock data
    return {
        "batch_id": batch_id,
        "total_invoices": 5,
        "completed": 3,
        "processing": 1,
        "failed": 1,
        "overall_progress": 60
    }

# Enhanced invoice retrieval with filtering
@app.get("/api/invoices")
async def get_invoices(
    status: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    sort_by: str = "date",
    order: str = "desc"
):
    """
    Get invoices with filtering and pagination
    """
    # Mock data - replace with actual database query
    mock_invoices = [
        {
            "id": f"INV-2024-{str(i).zfill(3)}",
            "vendor": f"Vendor {i}",
            "amount": 1000 + (i * 100),
            "status": ["approved", "rejected", "review_required"][i % 3],
            "date": "2024-01-15",
            "confidence": 0.95 - (i * 0.05),
            "risk_score": 20 + (i * 5)
        }
        for i in range(1, 21)
    ]
    
    # Apply filters
    if status:
        mock_invoices = [inv for inv in mock_invoices if inv["status"] == status]
    
    # Apply pagination
    total = len(mock_invoices)
    invoices = mock_invoices[offset:offset + limit]
    
    return {
        "invoices": invoices,
        "total": total,
        "limit": limit,
        "offset": offset,
        "has_more": offset + limit < total
    }

# Simple health check for components
@app.get("/api/upload/health")
async def upload_health():
    """
    Check upload system health
    """
    return {
        "status": "healthy",
        "upload_dir": str(UPLOAD_DIR.absolute()),
        "upload_dir_exists": UPLOAD_DIR.exists(),
        "active_uploads": len(upload_tracking),
        "processing_queue": len(processing_status)
    }

# FastAPI runs on port 8000 as API-only service
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)

