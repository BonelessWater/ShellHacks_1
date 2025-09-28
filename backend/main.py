# backend/main.py - Consolidated main file for ShellHacks Invoice System
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import API routes
from backend.api.routes import router as api_router

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

# Root endpoint
@app.get("/")
async def root():
    return {"message": "ShellHacks Invoice System API", "status": "running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
