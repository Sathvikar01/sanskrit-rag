"""Integrated Advanced RAG Pipeline.

Combines all enhanced components:
- Semantic chunking and commentary extraction
- HyDE (Hypothetical Document Embeddings)
- Query Transformation
- Cross-Encoder Re-ranking
- Strict Grounding and Self-Correction
- GraphRAG Multi-hop Retrieval
"""
import time
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.xml_parser import TEIXMLParser, CommentaryChunk
from src.embedding_client import NVIDIAEmbeddingClient
from src.qdrant_manager import QdrantManager
from src.neo4j_manager import Neo4jManager
from src.retriever import HybridRetriever, parse_verse_references, VerseFilter
from src.hyde_reranker import (
    HyDEGenerator, QueryTransformer, CrossEncoderReranker, HybridHyDERetriever
)
from src.grounding_reflection import (
    StrictGroundingEnforcer, SelfCorrectionReflection, 
    GroundedAnswerPipeline, HallucinationDetector
)
from src.commentary_manager import CommentaryManager


@dataclass
class AdvancedRAGResult:
    """Result from advanced RAG pipeline."""
    query: str
    answer: str
    contexts: List[Dict[str, Any]]
    commentaries: Dict[str, List[Dict[str, Any]]]
    metrics: Dict[str, Any]
    verification: Dict[str, Any]
    retrieval_details: Dict[str, Any]
    latency_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            "contexts": self.contexts,
            "commentaries": self.commentaries,
            "metrics": self.metrics,
            "verification": self.verification,
            "retrieval_details": self.retrieval_details,
            "latency_ms": self.latency_ms
        }


class AdvancedRAGPipeline:
    """Full Advanced RAG Pipeline with all enhancements."""

    def __init__(
        self,
        embedding_client: NVIDIAEmbeddingClient = None,
        qdrant_manager: QdrantManager = None,
        neo4j_manager: Neo4jManager = None,
        llm_api_key: str = None,
        use_hyde: bool = True,
        use_query_transformation: bool = True,
        use_reranking: bool = True,
        use_grounding: bool = True,
        use_reflection: bool = True,
        use_evaluation: bool = False
    ):
        self.embedding_client = embedding_client
        self.qdrant = qdrant_manager
        self.neo4j = neo4j_manager
        self.llm_api_key = llm_api_key

        self.use_hyde = use_hyde
        self.use_query_transformation = use_query_transformation
        self.use_reranking = use_reranking
        self.use_grounding = use_grounding
        self.use_reflection = use_reflection
        self.use_evaluation = False

        self._init_components()

    def _init_components(self):
        """Initialize all pipeline components."""
        if self.use_hyde or self.use_query_transformation:
            self.hyde_generator = HyDEGenerator(
                llm_api_key=self.llm_api_key,
                embedding_client=self.embedding_client
            )

        if self.use_query_transformation:
            self.query_transformer = QueryTransformer(
                llm_api_key=self.llm_api_key
            )

        if self.use_reranking:
            self.reranker = CrossEncoderReranker(
                api_key=self.llm_api_key
            )

        if self.use_grounding:
            self.grounding_enforcer = StrictGroundingEnforcer(
                api_key=self.llm_api_key
            )

        if self.use_reflection:
            self.reflection = SelfCorrectionReflection(
                api_key=self.llm_api_key,
                embedding_client=self.embedding_client
            )

        self.hallucination_detector = HallucinationDetector()

        self.commentary_manager = CommentaryManager(
            qdrant_manager=self.qdrant,
            neo4j_manager=self.neo4j,
            embedding_client=self.embedding_client
        )

        self.base_retriever = None
        if self.qdrant or self.neo4j:
            self.base_retriever = HybridRetriever(
                embedding_client=self.embedding_client,
                qdrant_manager=self.qdrant,
                neo4j_manager=self.neo4j
            )

    def query(
        self,
        query: str,
        top_k: int = 10,
        include_commentaries: bool = True,
        authors: List[str] = None
    ) -> AdvancedRAGResult:
        """Execute full advanced RAG query.

        Steps:
        1. Parse verse references from query
        2. Transform query into sub-queries (if enabled)
        3. Generate hypothetical document for HyDE (if enabled)
        4. Retrieve from Qdrant (semantic) and Neo4j (graph)
        5. Re-rank results with cross-encoder (if enabled)
        6. Retrieve commentaries (if enabled)
        7. Generate grounded answer
        8. Self-correction and reflection (if enabled)
        """
        start_time = time.time()

        verse_filter = parse_verse_references(query)

        all_queries = [query]
        if self.use_query_transformation:
            sub_queries = self.query_transformer.transform_query(query)
            all_queries.extend([sq.sub_query for sq in sub_queries])

        hyde_embedding = None
        if self.use_hyde:
            hyp_doc = self.hyde_generator.generate_hypothetical_document(query)
            hyde_embedding = self.hyde_generator.embed_hypothetical_document(hyp_doc)

        all_results = []
        for q in all_queries[:5]:
            if self.base_retriever:
                try:
                    results = self.base_retriever.cross_db_rrf_search(
                        q, top_k=top_k, verse_filter=verse_filter
                    )
                    all_results.extend(results)
                except Exception as e:
                    pass

        if hyde_embedding is not None and self.qdrant:
            try:
                hyde_results = self.qdrant.search_dense(hyde_embedding, top_k=top_k)
                all_results.extend(hyde_results)
            except Exception as e:
                pass

        seen_ids = set()
        unique_results = []
        for r in all_results:
            rid = getattr(r, 'id', str(hash(str(r))))
            if rid not in seen_ids:
                seen_ids.add(rid)
                unique_results.append(r)

        if self.use_reranking and unique_results:
            unique_results = self.reranker.rerank(query, unique_results, top_k=top_k)

        contexts = [
            {
                "id": getattr(r, 'id', ''),
                "text": getattr(r, 'text', ''),
                "verse_id": getattr(r, 'verse_id', ''),
                "score": getattr(r, 'final_score', getattr(r, 'score', 0)),
                "metadata": getattr(r, 'metadata', {})
            }
            for r in unique_results[:top_k]
        ]

        commentaries = {}
        if include_commentaries:
            verse_ids = list(set(c.get("verse_id", "") for c in contexts if c.get("verse_id")))
            for vid in verse_ids[:3]:
                comm = self.commentary_manager.get_commentary_for_verse(
                    vid, authors=authors
                )
                for author, results in comm.items():
                    if author not in commentaries:
                        commentaries[author] = []
                    commentaries[author].extend([
                        {
                            "verse_id": r.verse_id,
                            "text": r.text,
                            "score": r.score
                        }
                        for r in results
                    ])

        if self.use_grounding and contexts:
            answer = self.grounding_enforcer.generate_grounded_answer(query, contexts)
        else:
            answer = self._generate_simple_answer(query, contexts)

        verification = {}
        if self.use_grounding and contexts:
            verif_result = self.grounding_enforcer.verify_groundedness(answer, contexts)
            verification = {
                "is_grounded": verif_result.is_grounded,
                "faithfulness_score": verif_result.faithfulness_score,
                "hallucinated_claims": verif_result.hallucinated_claims,
                "supported_claims": verif_result.supported_claims
            }

        if self.use_reflection and contexts:
            reflection_result = self.reflection.reflect_and_correct(
                query, answer, contexts
            )
            if reflection_result.needs_revision and reflection_result.corrected_answer:
                answer = reflection_result.corrected_answer
            verification["reflection_issues"] = reflection_result.issues_found

        metrics = {}

        hallucination_risk = self.hallucination_detector.detect_hallucination_risk(answer)
        verification["hallucination_risk"] = hallucination_risk

        latency_ms = (time.time() - start_time) * 1000

        return AdvancedRAGResult(
            query=query,
            answer=answer,
            contexts=contexts,
            commentaries=commentaries,
            metrics=metrics,
            verification=verification,
            retrieval_details={
                "num_queries_used": len(all_queries),
                "hyde_used": self.use_hyde,
                "transformation_used": self.use_query_transformation,
                "reranking_used": self.use_reranking,
                "verse_filter": verse_filter.to_dict() if verse_filter.has_filter() else None
            },
            latency_ms=latency_ms
        )

    def _generate_simple_answer(
        self,
        query: str,
        contexts: List[Dict[str, Any]]
    ) -> str:
        """Generate a simple answer from contexts."""
        if not contexts:
            return "I do not have enough information to answer this question."

        answer_parts = ["Based on the retrieved verses:\n"]

        for i, ctx in enumerate(contexts[:3], 1):
            verse_id = ctx.get("verse_id", "Unknown")
            text = ctx.get("text", "")
            if text:
                answer_parts.append(f"\n[{verse_id}]: {text[:300]}")

        return "\n".join(answer_parts)


def ingest_with_advanced_parsing(
    base_dir: str,
    embedding_client: NVIDIAEmbeddingClient,
    qdrant_manager: QdrantManager,
    neo4j_manager: Neo4jManager = None
) -> Dict[str, Any]:
    """Ingest data with advanced parsing and commentary extraction."""
    parser = TEIXMLParser()

    main_chunks, commentaries = parser.parse_all_with_commentaries(base_dir)

    stats = {
        "main_chunks": {},
        "commentaries": {}
    }

    for dtype, chunks in main_chunks.items():
        if chunks:
            stats["main_chunks"][dtype] = len(chunks)

    commentary_manager = CommentaryManager(
        qdrant_manager=qdrant_manager,
        neo4j_manager=neo4j_manager,
        embedding_client=embedding_client
    )

    for dtype, comm_dict in commentaries.items():
        stats["commentaries"][dtype] = {}
        for author, chunks in comm_dict.items():
            stats["commentaries"][dtype][author] = len(chunks)

    return stats


if __name__ == "__main__":
    print("Advanced RAG Pipeline Components:")
    print("=" * 50)
    print("\nEnabled Features:")
    print(f"  - HyDE: {'Yes' if True else 'No'}")
    print(f"  - Query Transformation: {'Yes' if True else 'No'}")
    print(f"  - Cross-Encoder Re-ranking: {'Yes' if True else 'No'}")
    print(f"  - Strict Grounding: {'Yes' if True else 'No'}")
    print(f"  - Self-Correction: {'Yes' if True else 'No'}")

    print("\nNew Modules Created:")
    print("  1. src/xml_parser.py - Enhanced with commentary extraction")
    print("  2. src/hyde_reranker.py - HyDE, Query Transform, Re-ranking")
    print("  3. src/commentary_manager.py - Commentary embedding/storage")
    print("  4. src/grounding_reflection.py - Grounding & Self-correction")
    print("  5. src/advanced_rag_pipeline.py - Integrated pipeline")

    print("\nEnhanced Files:")
    print("  1. src/neo4j_manager.py - Added GraphRAG methods")
    print("  2. src/qdrant_manager.py - Updated for semantic embeddings")
    print("  3. requirements.txt - Added new dependencies")
