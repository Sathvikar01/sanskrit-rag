"""State definitions for the SRAG LangGraph pipeline."""

from typing import TypedDict


class SRAGState(TypedDict):
    """State for the SRAG pipeline graph."""

    query: str
    query_iast: str
    query_devanagari: str
    concepts: list[str]
    language: str
    query_type: str

    vector_results: list[dict]
    graph_results: list[dict]
    bm25_results: list[dict]
    fused_results: list[dict]
    reranked_results: list[dict]

    answer: str
    citations: list[str]
    confidence: dict

    verse_ref_detected: bool
    verse_ref: str

    iteration: int
    should_expand: bool
    error: str
