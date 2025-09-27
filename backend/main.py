# backend/main.py - Updated to serve React frontend
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import os
from pathlib import Path

app = FastAPI(title="ShellHacks Invoice System", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Routes (your existing backend routes)
@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "message": "Backend is running"}

@app.get("/api/message")
async def get_message():
    return {"message": "Hello from Azure backend!", "status": "success"}

# Add your other API endpoints here...
# @app.post("/api/invoices/upload")
# @app.get("/api/invoices")
# etc.

# Serve React static files
frontend_build_path = Path(__file__).parent / "frontend_build"

# Check if frontend build exists
if frontend_build_path.exists():
    # Mount static files
    app.mount("/static", StaticFiles(directory=str(frontend_build_path / "static")), name="static")
    
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