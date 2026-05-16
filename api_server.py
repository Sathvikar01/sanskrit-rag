"""FastAPI server for SRAG - Sanskrit RAG with Knowledge Graph."""

import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pathlib import Path

from main import SRAGPipeline
from src.utils.config import Config

app = FastAPI(title="SRAG API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = None


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    query: str
    query_iast: str
    query_devanagari: str
    concepts: list[str]
    answer: str
    verses_cited: list[str]
    top_verses: list[dict]
    pipeline_confidence: dict


@app.on_event("startup")
def startup():
    global pipeline
    config = Config()
    pipeline = SRAGPipeline(config)
    pipeline.preprocess()
    pipeline.build_indices()
    try:
        pipeline._get_graph_retriever()
    except Exception:
        pass


@app.post("/api/query", response_model=QueryResponse)
def query_endpoint(req: QueryRequest):
    result = pipeline.query(req.query, use_api=True)
    return QueryResponse(
        query=result["query"],
        query_iast=result["query_iast"],
        query_devanagari=result["query_devanagari"],
        concepts=result["concepts_extracted"],
        answer=result["answer"],
        verses_cited=result["verses_cited"],
        top_verses=result["top_verses"],
        pipeline_confidence=result["pipeline_confidence"],
    )


@app.get("/api/health")
def health():
    return {"status": "ok", "chunks": len(pipeline.chunks) if pipeline else 0}


dist_dir = Path(__file__).parent / "web" / "dist"
if dist_dir.exists():
    app.mount("/assets", StaticFiles(directory=str(dist_dir / "assets")), name="assets")

    @app.get("/{full_path:path}")
    def serve_spa(full_path: str):
        file_path = dist_dir / full_path
        if file_path.exists() and file_path.is_file():
            return FileResponse(str(file_path))
        return FileResponse(str(dist_dir / "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
