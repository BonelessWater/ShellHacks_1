# backend/main.py - Updated to serve React frontend
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
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


# API Routes (your existing backend routes)
@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "Backend is running"}

<<<<<<< Updated upstream
@app.get("/api/health")
async def api_health_check():
    return {"status": "healthy", "message": "Backend API is running"}
=======
>>>>>>> Stashed changes

@app.get("/api/message")
async def get_message():
    return {"message": "Hello from ShellHacks backend!", "status": "success"}



# Add your other API endpoints here...
# @app.post("/api/invoices/upload")
# @app.get("/api/invoices")
# etc.

# Root endpoint (simple API response)
@app.get("/")
async def root():
    return {"message": "ShellHacks Invoice System API", "status": "running"}

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
