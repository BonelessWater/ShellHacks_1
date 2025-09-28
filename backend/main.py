from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os

app = FastAPI(title="AgentZero Backend API", version="1.0.0")

# Import API routes
try:
    from backend.api.routes import router as api_router
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
    allow_origins=["*"],  # In production, specify your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "python-backend"}

# Basic API endpoint
@app.get("/")
async def root():
    return {"message": "AgentZero Python Backend is running"}

# Example data endpoint
@app.get("/data")
async def get_data():
    return {
        "data": [
            {"id": 1, "name": "Item 1"},
            {"id": 2, "name": "Item 2"},
            {"id": 3, "name": "Item 3"}
        ]
    }

# Example POST endpoint
class Item(BaseModel):
    name: str
    description: str = None

@app.post("/items")
async def create_item(item: Item):
    return {"message": f"Created item: {item.name}", "item": item}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)