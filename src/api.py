"""FastAPI backend for SansRAG."""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from config.settings import L1_REG_LAMBDA, L2_REG_LAMBDA, RRF_TOP_K
from src.service import SansRAGService


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=RRF_TOP_K, ge=1, le=50)
    l1_lambda: float = Field(default=L1_REG_LAMBDA, ge=0.0, le=0.5)
    l2_lambda: float = Field(default=L2_REG_LAMBDA, ge=0.0, le=0.1)
    regularization: Literal["combined", "l1", "l2", "none"] = "combined"


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=RRF_TOP_K, ge=1, le=50)
    l1_lambda: float = Field(default=L1_REG_LAMBDA, ge=0.0, le=0.5)
    l2_lambda: float = Field(default=L2_REG_LAMBDA, ge=0.0, le=0.1)
    regularization: Literal["combined", "l1", "l2", "none"] = "combined"


class DockerRequest(BaseModel):
    action: Literal["start", "stop", "status"]


# Global service instance
_service_instance: SansRAGService | None = None


async def get_service() -> SansRAGService:
    """Get the service instance created during app startup."""
    if _service_instance is None:
        raise RuntimeError("Service not initialized. App startup failed.")
    return _service_instance


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage app lifecycle: startup and shutdown."""
    global _service_instance
    # Startup
    try:
        _service_instance = SansRAGService()
        print("✓ SansRAGService initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize SansRAGService: {e}")
        raise
    
    yield
    
    # Shutdown
    _service_instance = None


app = FastAPI(
    title="SansRAG API",
    description="Dual-DB evidence retrieval and Sanskrit answer generation API.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health(service: SansRAGService = Depends(get_service)):
    return service.health()


@app.get("/api/stats")
async def stats(service: SansRAGService = Depends(get_service)):
    return service.stats()


@app.post("/api/ask")
async def ask(request: AskRequest, service: SansRAGService = Depends(get_service)):
    return service.ask(
        query=request.query,
        top_k=request.top_k,
        l1_lambda=request.l1_lambda,
        l2_lambda=request.l2_lambda,
        regularization=request.regularization,
    )


@app.post("/api/search")
async def search(request: SearchRequest, service: SansRAGService = Depends(get_service)):
    return service.search(
        query=request.query,
        top_k=request.top_k,
        l1_lambda=request.l1_lambda,
        l2_lambda=request.l2_lambda,
        regularization=request.regularization,
    )


@app.post("/api/docker")
async def docker(request: DockerRequest, service: SansRAGService = Depends(get_service)):
    return service.docker(request.action)


frontend_dist = ROOT_DIR / "frontend" / "dist"
if frontend_dist.exists():
    app.mount("/assets", StaticFiles(directory=frontend_dist / "assets"), name="assets")

    @app.get("/")
    async def index():
        return FileResponse(frontend_dist / "index.html")
