from fastapi import FastAPI

# Mount only the API router from backend.api.routes to avoid heavy imports in backend.main
try:
    from backend.api.routes import router as api_router
except Exception as e:
    # If importing the package router fails, surface a clear error rather than trying a relative import
    raise ImportError(f"Failed to import backend.api.routes: {e}")

app = FastAPI()
app.include_router(api_router, prefix="/api")

@app.get("/health")
def health():
    return {"status": "healthy", "source": "minimal_backend"}
