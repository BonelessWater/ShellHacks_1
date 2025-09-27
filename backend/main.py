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
@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "Backend is running"}

@app.get("/api/health")
async def api_health_check():
    return {"status": "healthy", "message": "Backend API is running"}

@app.get("/api/message")
async def get_message():
    return {"message": "Hello from ShellHacks backend!", "status": "success"}

# Add your other API endpoints here...
# @app.post("/api/invoices/upload")
# @app.get("/api/invoices")
# etc.

# Root endpoint
@app.get("/")
async def root():
    return {"message": "ShellHacks Invoice System API", "status": "running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)