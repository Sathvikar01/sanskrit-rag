"""FastAPI server for SRAG - Sanskrit RAG with Knowledge Graph."""

import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pathlib import Path
from typing import Optional

from src.utils.config import Config
from src.utils.logger import logger

pipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    config = Config()

    use_langgraph = config.get("langgraph.enabled", True)

    if use_langgraph:
        from src.langchain_components.graph import SRAGGraphPipeline
        pipeline = SRAGGraphPipeline(config)
        logger.info("Using LangGraph pipeline")
    else:
        from main import SRAGPipeline
        pipeline = SRAGPipeline(config)
        logger.info("Using standard pipeline")

    pipeline.preprocess()
    pipeline.build_indices()
    try:
        if hasattr(pipeline, '_get_graph_retriever'):
            pipeline._get_graph_retriever()
            logger.info("Neo4j graph retriever connected")
    except Exception as e:
        logger.error(f"Neo4j connection failed: {e}. Graph retrieval will be unavailable.")

    logger.info("SRAG pipeline ready")
    yield
    pipeline.close()


app = FastAPI(title="SansRAG API", version="0.3.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str
    toggles: Optional[dict] = None
    normalize: Optional[str] = None  # "none", "minmax", "l2", "zscore"


class QueryResponse(BaseModel):
    query: str
    query_iast: str
    query_devanagari: str
    concepts: list[str]
    answer: str
    verses_cited: list[str]
    top_verses: list[dict]
    pipeline_confidence: dict
    query_type: str = ""
    intermediate: dict = {}
    commentaries: dict = {}
    normalize_method: str = "none"


@app.post("/api/query", response_model=QueryResponse)
def query_endpoint(req: QueryRequest):
    old_method = None
    if req.normalize and hasattr(pipeline, 'reranker'):
        old_method = pipeline.reranker.normalize_method
        pipeline.reranker.normalize_method = req.normalize

    result = pipeline.query(req.query, use_api=True, toggles=req.toggles)

    if old_method is not None:
        pipeline.reranker.normalize_method = old_method

    return QueryResponse(
        query=result["query"],
        query_iast=result["query_iast"],
        query_devanagari=result["query_devanagari"],
        concepts=result["concepts_extracted"],
        answer=result["answer"],
        verses_cited=result["verses_cited"],
        top_verses=result["top_verses"],
        pipeline_confidence=result["pipeline_confidence"],
        query_type=result.get("query_type", ""),
        intermediate=result.get("intermediate", {}),
        commentaries=result.get("commentaries", {}),
        normalize_method=req.normalize or "none",
    )


@app.get("/api/health")
def health():
    neo4j_ok = False
    if pipeline and hasattr(pipeline, '_graph_connected'):
        neo4j_ok = pipeline._graph_connected
    return {
        "status": "ok",
        "chunks": len(pipeline.chunks) if pipeline else 0,
        "pipeline_type": "langgraph" if hasattr(pipeline, 'graph') else "standard",
        "neo4j_connected": neo4j_ok,
    }


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
