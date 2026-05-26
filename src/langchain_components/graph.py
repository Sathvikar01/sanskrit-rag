"""LangGraph state machine for the SRAG pipeline.

This implements the full RAG pipeline as a LangGraph state machine with:
- Query processing (IAST conversion + concept extraction)
- Adaptive retrieval (vector + graph + BM25 with dynamic weights)
- Linguistic re-ranking with 9 features
- Iterative query expansion on low confidence
- Answer generation with MiMo
"""

import re
import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from langgraph.graph import END, StateGraph

from src.langchain_components.state import SRAGState
from src.preprocessing.chunker import Chunk, load_chunks
from src.preprocessing.concept_extractor import ConceptExtractor
from src.reranking.adaptive_reranker import detect_query_type
from src.reranking.linguistic_reranker import LinguisticReranker
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.graph_retriever import GraphRetriever
from src.retrieval.hybrid_fusion import HybridRetriever
from src.retrieval.vector_store import VectorStore
from src.generation.generator import AnswerGenerator
from src.generation.query_processor import QueryProcessor
from src.storage.commentary_store import CommentaryStore
from src.utils.config import Config
from src.utils.logger import logger


# Default toggle state
DEFAULT_TOGGLES = {"vector": True, "graph": True, "bm25": True}

VERSE_REF_PATTERN = re.compile(r'(?:BhG|BG)\s+(\d+\.\d+)', re.IGNORECASE)


def extract_verse_refs_from_text(text: str) -> list[str]:
    """Extract unique verse references like 'BhG 2.47' from text."""
    seen = set()
    refs = []
    for m in VERSE_REF_PATTERN.finditer(text):
        ref = f"BhG {m.group(1)}"
        if ref not in seen:
            seen.add(ref)
            refs.append(ref)
    return refs


class SRAGGraphPipeline:
    """SRAG pipeline implemented as a LangGraph state machine."""

    def __init__(self, config: Config = None):
        if config is None:
            config = Config()

        self.config = config
        self.query_processor = QueryProcessor(config)
        self.vector_store = VectorStore(config)
        self.bm25_retriever = BM25Retriever()
        self.hybrid_retriever = HybridRetriever(config)
        self.reranker = LinguisticReranker(config)
        self.generator = AnswerGenerator(config)
        self.concept_extractor = ConceptExtractor()
        self.commentary_store = CommentaryStore()

        self.chunks: list[Chunk] = []
        self.chunk_map: dict[str, Chunk] = {}

        self._graph_retriever = None
        self._graph_connected = False

        self._toggles = DEFAULT_TOGGLES.copy()
        self._intermediate: dict = {}

        self.max_iterations = config.get("langgraph.max_iterations", 2)
        self.confidence_threshold = config.get("langgraph.confidence_threshold", 0.3)
        self.expand_on_low_conf = config.get("langgraph.expand_query_on_low_confidence", True)

        self._retrieval_only = False

        self.graph = self._build_graph()

    def _get_graph_retriever(self) -> GraphRetriever:
        if self._graph_retriever is None:
            self._graph_retriever = GraphRetriever(self.config)
        if not self._graph_connected:
            try:
                self._graph_retriever.connect()
                self._graph_connected = True
            except Exception as e:
                logger.warning(f"Graph connection failed: {e}")
        return self._graph_retriever

    def preprocess(self, force: bool = False):
        from pathlib import Path
        chunks_path = Path(self.config.get("data.chunks_file", "data/processed/chunks.jsonl"))
        if chunks_path.exists():
            logger.info("Loading existing chunks...")
            self.chunks = load_chunks(chunks_path)
        self.chunk_map = {c.chunk_id: c for c in self.chunks}

    def build_indices(self):
        from pathlib import Path
        if not self.chunks:
            raise ValueError("No chunks loaded. Run preprocess() first.")

        faiss_path = Path(self.config.get("data.faiss_index"))
        metadata_path = Path(self.config.get("data.faiss_metadata"))

        if faiss_path.exists() and metadata_path.exists():
            logger.info("Loading existing FAISS index...")
            self.vector_store.load(faiss_path, metadata_path)
        else:
            logger.info("Building FAISS index...")
            self.vector_store.build_index(self.chunks, use_devanagari=True, verse_only=True)
            self.vector_store.save(faiss_path, metadata_path)

        logger.info("Building BM25 index...")
        self.bm25_retriever.build_index(self.chunks, use_lemmas=True)

    def _node_process_query(self, state: SRAGState) -> dict:
        """Process the user query: detect language, extract IAST, concepts."""
        query = state["query"]
        iteration = state.get("iteration", 0)

        if iteration == 0:
            processed = self.query_processor.process_query(query)
            return {
                "query_iast": processed.query_iast,
                "query_devanagari": processed.query_devanagari,
                "concepts": processed.concepts,
                "language": processed.language_detected,
                "query_type": detect_query_type(processed.query_iast, processed.concepts),
                "iteration": 0,
            }
        else:
            # Re-extract concepts from the query to find related ones
            query_iast = state.get("query_iast", "")
            found = self.concept_extractor.extract_from_text(query_iast)
            related = [fc["concept"].name_iast for fc in found]
            expanded_concepts = list(set(state.get("concepts", []) + related))
            return {
                "concepts": expanded_concepts,
                "iteration": iteration + 1,
            }

    def _node_verse_ref_retrieval(self, state: SRAGState) -> dict:
        """Check query for BhG X.Y / BG X.Y pattern.

        If detected, retrieve the verse directly from Neo4j and skip the full
        retrieval pipeline (vector + BM25 + graph combined).
        """
        query = state["query"]
        query_iast = state.get("query_iast", "")
        refs = extract_verse_refs_from_text(query)
        if not refs:
            refs = extract_verse_refs_from_text(query_iast)

        if refs:
            vid = refs[0]
            try:
                graph = self._get_graph_retriever()
                vr = graph.search_by_verse_ref(vid, top_k=1)
                if vr:
                    logger.info(f"Rule-based verse_ref node: '{vid}' found, skipping full retrieval")
                    vr[0]["exact_verse_match"] = True
                    self._intermediate["verse_ref_detected"] = True
                    self._intermediate["verse_ref"] = vid
                    self._intermediate["reranked_results"] = [
                        {"chunk_id": r.get("chunk_id"), "verse_ref": r.get("verse_ref", ""),
                         "chunk_type": r.get("chunk_type", "")}
                        for r in vr
                    ]
                    return {
                        "fused_results": vr,
                        "reranked_results": vr,
                        "verse_ref_detected": True,
                        "verse_ref": vid,
                    }
            except Exception as e:
                logger.warning(f"Verse ref retrieval failed for '{vid}': {e}")

        return {"verse_ref_detected": False, "verse_ref": ""}

    def _route_after_verse_check(self, state: SRAGState) -> str:
        """Route to generate if verse ref was found, else to normal retrieval."""
        if state.get("verse_ref_detected", False):
            return "generate"
        return "retrieve"

    def _node_retrieve(self, state: SRAGState) -> dict:
        """Run vector, graph, and BM25 retrieval with toggle support."""
        query_iast = state["query_iast"]
        query_devanagari = state["query_devanagari"]
        concepts = state.get("concepts", [])

        # Vector
        vector_results = []
        if self._toggles.get("vector", True):
            vector_results = self.vector_store.search(query_devanagari, top_k=50)

        # Graph
        graph_results = []
        if self._toggles.get("graph", True):
            try:
                graph_retriever = self._get_graph_retriever()
                graph_results = graph_retriever.search_combined(query_iast, concepts, top_k=50)
            except Exception as e:
                logger.warning(f"Graph retrieval failed: {e}")

        # BM25
        bm25_results = []
        if self._toggles.get("bm25", True):
            bm25_results = self.bm25_retriever.search(query_iast, top_k=50)

        # Track intermediate results
        self._intermediate["vector_results"] = [
            {"chunk_id": r["chunk_id"], "score": round(r.get("score", 0), 4), "verse_ref": r.get("verse_ref", "")}
            for r in vector_results[:10]
        ]
        self._intermediate["graph_results"] = [
            {"chunk_id": r["chunk_id"], "score": round(r.get("score", 0), 4), "verse_ref": r.get("verse_ref", "")}
            for r in graph_results[:10]
        ]
        self._intermediate["bm25_results"] = [
            {"chunk_id": r["chunk_id"], "score": round(r.get("score", 0), 4), "verse_ref": r.get("verse_ref", "")}
            for r in bm25_results[:10]
        ]

        return {
            "vector_results": vector_results,
            "graph_results": graph_results,
            "bm25_results": bm25_results,
        }

    def _node_fuse(self, state: SRAGState) -> dict:
        """Fuse retrieval results with adaptive weights."""
        query_type = state.get("query_type", "general_medium")

        fused = self.hybrid_retriever.fuse_results(
            state["vector_results"],
            state["graph_results"],
            state["bm25_results"],
            top_k=50,
            query_type=query_type,
        )

        self._intermediate["fused_results"] = [
            {"chunk_id": r["chunk_id"], "score": round(r.get("rrf_score", 0), 4),
             "verse_ref": r.get("verse_ref", ""), "sources": r.get("sources", []),
             "vector_score": round(r.get("vector_score", 0), 4),
             "graph_score": round(r.get("graph_score", 0), 4),
             "bm25_score": round(r.get("bm25_score", 0), 4)}
            for r in fused[:20]
        ]

        return {"fused_results": fused}

    def _node_rerank(self, state: SRAGState) -> dict:
        """Re-rank candidates with linguistic features."""
        reranked = self.reranker.rerank(
            query_iast=state["query_iast"],
            concepts=state.get("concepts", []),
            candidates=state["fused_results"],
            all_chunks=self.chunks,
            chunk_map=self.chunk_map,
        )

        avg_confidence = 0.0
        if reranked:
            confidences = [
                r.get("confidence", {}).get("overall_confidence", 0)
                for r in reranked
            ]
            avg_confidence = sum(confidences) / len(confidences)

        self._intermediate["reranked_results"] = [
            {"chunk_id": r.get("chunk_id", ""), "verse_ref": r.get("verse_ref", ""),
             "confidence": round(r.get("confidence", {}).get("overall_confidence", 0), 4),
             "final_score": round(r.get("final_score", 0), 4),
             "chunk_type": r.get("chunk_type", "")}
            for r in reranked[:10]
        ]

        return {
            "reranked_results": reranked,
            "confidence": {
                "avg_reranking_confidence": avg_confidence,
                "top_score": reranked[0]["final_score"] if reranked else 0,
            },
            "should_expand": not self._retrieval_only and avg_confidence < self.confidence_threshold,
        }

    def _node_generate(self, state: SRAGState) -> dict:
        """Generate the answer using MiMo (no-op when retrieval_only=True)."""
        if self._retrieval_only:
            return {}

        result = self.generator.generate(
            query=state["query"],
            reranked_results=state["reranked_results"],
            concepts=state.get("concepts", []),
        )

        return {
            "answer": result.answer,
            "citations": result.verses_cited,
            "confidence": {
                **state.get("confidence", {}),
                "generation_confidence": result.generation_confidence,
                "overall_confidence": result.pipeline_confidence.get("overall_confidence", 0),
            },
        }

    def _should_expand(self, state: SRAGState) -> str:
        """Decide whether to expand query or generate answer."""
        if not self.expand_on_low_conf:
            return "generate"

        iteration = state.get("iteration", 0)
        should_expand = state.get("should_expand", False)

        if should_expand and iteration < self.max_iterations:
            logger.info(
                f"Low confidence ({state.get('confidence', {}).get('avg_reranking_confidence', 0):.3f}), "
                f"expanding query (iteration {iteration + 1}/{self.max_iterations})"
            )
            return "expand"
        return "generate"

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state machine."""
        workflow = StateGraph(SRAGState)

        workflow.add_node("process_query", self._node_process_query)
        workflow.add_node("verse_ref_retrieval", self._node_verse_ref_retrieval)
        workflow.add_node("retrieve", self._node_retrieve)
        workflow.add_node("fuse", self._node_fuse)
        workflow.add_node("rerank", self._node_rerank)
        workflow.add_node("generate", self._node_generate)

        workflow.set_entry_point("process_query")
        workflow.add_edge("process_query", "verse_ref_retrieval")
        workflow.add_conditional_edges(
            "verse_ref_retrieval",
            self._route_after_verse_check,
            {"generate": "generate", "retrieve": "retrieve"},
        )
        workflow.add_edge("retrieve", "fuse")
        workflow.add_edge("fuse", "rerank")

        workflow.add_conditional_edges(
            "rerank",
            self._should_expand,
            {
                "expand": "process_query",
                "generate": "generate",
            },
        )

        workflow.add_edge("generate", END)

        return workflow.compile()

    def query(self, user_query: str, use_api: bool = True, toggles: dict | None = None, retrieval_only: bool = False) -> dict:
        """Process a query through the LangGraph pipeline.

        Args:
            user_query: The user's question.
            use_api: Whether to use MiMo API for query processing.
            toggles: Dict with 'vector', 'graph', 'bm25' bools.
            retrieval_only: Skip generation (no-op) and disable query expansion.

        Returns:
            Complete response dictionary.
        """
        if toggles is not None:
            self._toggles = toggles
        else:
            self._toggles = DEFAULT_TOGGLES.copy()

        self._retrieval_only = retrieval_only
        self._intermediate = {}

        initial_state: SRAGState = {
            "query": user_query,
            "query_iast": "",
            "query_devanagari": "",
            "concepts": [],
            "language": "",
            "query_type": "general_medium",
            "vector_results": [],
            "graph_results": [],
            "bm25_results": [],
            "fused_results": [],
            "reranked_results": [],
            "verse_ref_detected": False,
            "verse_ref": "",
            "answer": "",
            "citations": [],
            "confidence": {},
            "iteration": 0,
            "should_expand": False,
            "error": "",
        }

        final_state = self.graph.invoke(initial_state)

        # Fetch commentaries for top verses
        commentaries = {}
        try:
            if not self.commentary_store.conn:
                self.commentary_store.connect()
            verse_refs = []
            for r in final_state.get("reranked_results", [])[:5]:
                ref = r.get("verse_ref", "")
                if ref and ref not in verse_refs:
                    verse_refs.append(ref)
            if verse_refs:
                commentaries = self.commentary_store.get_commentaries_for_verses(verse_refs)
        except Exception as e:
            logger.warning(f"Commentary store failed: {e}")

        reranked_results = final_state.get("reranked_results", [])
        fused_results = final_state.get("fused_results", [])

        return {
            "query": user_query,
            "query_iast": final_state.get("query_iast", ""),
            "query_devanagari": final_state.get("query_devanagari", ""),
            "concepts_extracted": final_state.get("concepts", []),
            "query_type": final_state.get("query_type", ""),
            "answer": final_state.get("answer", ""),
            "verses_cited": final_state.get("citations", []),
            "reranked_results": reranked_results,
            "fused_results": fused_results,
            "top_verses": [
                {
                    "ref": r.get("verse_ref"),
                    "text_iast": r.get("text_iast", "")[:200],
                    "confidence": r.get("confidence", {}).get("overall_confidence", 0),
                }
                for r in reranked_results[:5]
            ],
            "pipeline_confidence": final_state.get("confidence", {}),
            "iterations": final_state.get("iteration", 0) + 1,
            "intermediate": self._intermediate,
            "commentaries": {
                ref: [
                    {"commentator": c["commentator"], "text": c["text_iast"][:300]}
                    for c in comms
                ]
                for ref, comms in commentaries.items()
            },
        }

    def close(self):
        if self._graph_retriever and self._graph_connected:
            self._graph_retriever.close()
            self._graph_connected = False
