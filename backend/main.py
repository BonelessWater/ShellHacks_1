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

# FastAPI runs on port 8000 as API-only service
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
