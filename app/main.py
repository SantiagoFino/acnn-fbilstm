from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import predictions
from app.core.settings import settings
from app.services.model_loader import ModelLoader
import uvicorn


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="ML Prediction API"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models on startup
@app.on_event("startup")
async def startup_event():
    """Load ML models when the application starts"""
    model_loader = ModelLoader()
    await model_loader.load_all_models()
    app.state.model_loader = model_loader

# Include routers
app.include_router(predictions.router, prefix="/api/v1", tags=["predictions"])

@app.get("/")
async def root():
    return {"message": "Prediction API is running", "version": settings.VERSION}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": settings.PROJECT_NAME}

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )
