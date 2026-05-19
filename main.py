"""SRAG: Sanskrit RAG with Graph-Enhanced Linguistic Re-ranking.

End-to-end pipeline for querying the Bhagavad Gita.
"""

import argparse
import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from src.generation.generator import AnswerGenerator
from src.generation.query_processor import QueryProcessor
from src.preprocessing.chunker import Chunk, create_all_chunks, load_chunks, save_chunks
from src.preprocessing.graph_builder import GraphBuilder
from src.preprocessing.graph_import import save_graph_import_data
from src.preprocessing.xml_parser import XMLParser
from src.reranking.linguistic_reranker import LinguisticReranker
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.graph_retriever import GraphRetriever
from src.retrieval.hybrid_fusion import HybridRetriever
from src.retrieval.vector_store import VectorStore
from src.storage.commentary_store import CommentaryStore
from src.utils.config import Config
from src.utils.logger import logger


# Default toggle state for retrieval methods
DEFAULT_TOGGLES = {
    "vector": True,
    "graph": True,
    "bm25": True,
}


class SRAGPipeline:
    """End-to-end SRAG pipeline."""

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
        self.commentary_store = CommentaryStore()

        self.chunks: list[Chunk] = []
        self.chunk_map: dict[str, Chunk] = {}

        self._graph_retriever = None
        self._graph_connected = False

    def _get_graph_retriever(self) -> GraphRetriever:
        """Get or create graph retriever with connection."""
        if self._graph_retriever is None:
            self._graph_retriever = GraphRetriever(self.config)
        if not self._graph_connected:
            self._graph_retriever.connect()
            self._graph_connected = True
        return self._graph_retriever

    def preprocess(self, force: bool = False):
        """Run the preprocessing pipeline."""
        chunks_path = Path(self.config.get("data.chunks_file", "data/processed/chunks.jsonl"))
        graph_dir = Path(self.config.get("data.graph_import_dir", "data/processed/graph_import"))

        if chunks_path.exists() and not force:
            logger.info("Loading existing chunks...")
            self.chunks = load_chunks(chunks_path)
        else:
            logger.info("Running preprocessing pipeline...")
            parser = XMLParser(
                main_xml=self.config.get("data.xml_main"),
                morpho_xml=self.config.get("data.xml_morpho"),
                segmentation_xml=self.config.get("data.xml_segmentation"),
            )
            verses, morpho, segmentation = parser.parse_all()
            self.chunks = create_all_chunks(verses, morpho, segmentation)
            save_chunks(self.chunks, chunks_path)
            save_graph_import_data(graph_dir, verses, self.chunks)
            logger.info("Preprocessing complete!")

        self.chunk_map = {c.chunk_id: c for c in self.chunks}

    def build_graph(self, clear: bool = True):
        """Build the Neo4j knowledge graph."""
        import_dir = Path(self.config.get("data.graph_import_dir", "data/processed/graph_import"))
        with GraphBuilder(self.config) as builder:
            builder.build_from_files(import_dir, clear=clear)
            stats = builder.get_stats()
            logger.info(f"Graph stats: {stats}")

    def build_indices(self, use_devanagari: bool = True):
        """Build vector and BM25 indices."""
        if not self.chunks:
            raise ValueError("No chunks loaded. Run preprocess() first.")

        faiss_path = Path(self.config.get("data.faiss_index"))
        metadata_path = Path(self.config.get("data.faiss_metadata"))

        if faiss_path.exists() and metadata_path.exists():
            logger.info("Loading existing FAISS index...")
            self.vector_store.load(faiss_path, metadata_path)
        else:
            logger.info("Building FAISS index...")
            self.vector_store.build_index(self.chunks, use_devanagari=use_devanagari, verse_only=True)
            self.vector_store.save(faiss_path, metadata_path)

        logger.info("Building BM25 index...")
        self.bm25_retriever.build_index(self.chunks, use_lemmas=True)

    def retrieve(
        self,
        query_iast: str,
        query_devanagari: str,
        concepts: list[str],
        top_k: int = 50,
        query_type: str = "general_medium",
        toggles: dict | None = None,
    ) -> tuple[list[dict], dict]:
        """Retrieve candidates using hybrid retrieval.

        Args:
            toggles: Dict with 'vector', 'graph', 'bm25' bools to enable/disable methods.

        Returns:
            Tuple of (fused_results, intermediate_results).
        """
        if toggles is None:
            toggles = DEFAULT_TOGGLES.copy()

        intermediate = {}

        # Vector retrieval
        vector_results = []
        if toggles.get("vector", True):
            vector_results = self.vector_store.search(query_devanagari, top_k=top_k)
            intermediate["vector_results"] = [
                {"chunk_id": r["chunk_id"], "score": round(r.get("score", 0), 4), "verse_ref": r.get("verse_ref", "")}
                for r in vector_results[:10]
            ]
        else:
            intermediate["vector_results"] = []

        # Graph retrieval
        graph_results = []
        if toggles.get("graph", True):
            try:
                graph_retriever = self._get_graph_retriever()
                graph_results = graph_retriever.search_combined(query_iast, concepts, top_k=top_k)
            except Exception as e:
                logger.warning(f"Graph retrieval failed: {e}")
            intermediate["graph_results"] = [
                {"chunk_id": r["chunk_id"], "score": round(r.get("score", 0), 4), "verse_ref": r.get("verse_ref", "")}
                for r in graph_results[:10]
            ]
        else:
            intermediate["graph_results"] = []

        # BM25 retrieval
        bm25_results = []
        if toggles.get("bm25", True):
            bm25_results = self.bm25_retriever.search(query_iast, top_k=top_k)
            intermediate["bm25_results"] = [
                {"chunk_id": r["chunk_id"], "score": round(r.get("score", 0), 4), "verse_ref": r.get("verse_ref", "")}
                for r in bm25_results[:10]
            ]
        else:
            intermediate["bm25_results"] = []

        # Fusion
        fused = self.hybrid_retriever.fuse_results(
            vector_results, graph_results, bm25_results, top_k=top_k,
            query_type=query_type,
        )
        intermediate["fused_results"] = [
            {"chunk_id": r["chunk_id"], "score": round(r.get("rrf_score", 0), 4),
             "verse_ref": r.get("verse_ref", ""), "sources": r.get("sources", []),
             "vector_score": round(r.get("vector_score", 0), 4),
             "graph_score": round(r.get("graph_score", 0), 4),
             "bm25_score": round(r.get("bm25_score", 0), 4)}
            for r in fused[:20]
        ]

        return fused, intermediate

    def _fetch_commentaries(self, reranked: list[dict]) -> dict:
        """Fetch commentaries from SQLite for top reranked verses."""
        try:
            if not self.commentary_store.conn:
                self.commentary_store.connect()
        except Exception as e:
            logger.warning(f"Commentary store connection failed: {e}")
            return {}

        verse_refs = []
        for r in reranked[:5]:
            ref = r.get("verse_ref", "")
            if ref and ref not in verse_refs:
                verse_refs.append(ref)

        if not verse_refs:
            return {}

        return self.commentary_store.get_commentaries_for_verses(verse_refs)

    def query(
        self,
        user_query: str,
        use_api: bool = True,
        toggles: dict | None = None,
    ) -> dict:
        """Process a user query end-to-end."""
        if use_api:
            processed = self.query_processor.process_query(user_query)
        else:
            processed = self.query_processor.process_query_local(user_query)

        from src.reranking.adaptive_reranker import detect_query_type
        query_type = detect_query_type(processed.query_iast, processed.concepts)

        logger.info(
            f"Query processed: iast='{processed.query_iast[:50]}...', "
            f"concepts={processed.concepts}, type={query_type}"
        )

        candidates, intermediate = self.retrieve(
            processed.query_iast,
            processed.query_devanagari,
            processed.concepts,
            query_type=query_type,
            toggles=toggles,
        )

        reranked = self.reranker.rerank(
            query_iast=processed.query_iast,
            concepts=processed.concepts,
            candidates=candidates,
            all_chunks=self.chunks,
            chunk_map=self.chunk_map,
        )

        # Add reranked top-10 to intermediate
        intermediate["reranked_results"] = [
            {"chunk_id": r.get("chunk_id", ""), "verse_ref": r.get("verse_ref", ""),
             "confidence": round(r.get("confidence", {}).get("overall_confidence", 0), 4),
             "final_score": round(r.get("final_score", 0), 4),
             "chunk_type": r.get("chunk_type", "")}
            for r in reranked[:10]
        ]

        # Fetch commentaries for top verses
        commentaries = self._fetch_commentaries(reranked)

        result = self.generator.generate(
            query=user_query,
            reranked_results=reranked,
            concepts=processed.concepts,
        )

        response = {
            "query": user_query,
            "query_iast": processed.query_iast,
            "query_devanagari": processed.query_devanagari,
            "concepts_extracted": processed.concepts,
            "query_type": query_type,
            "answer": result.answer,
            "verses_cited": result.verses_cited,
            "top_verses": [
                {
                    "ref": r.get("verse_ref"),
                    "text_iast": r.get("text_iast", "")[:200],
                    "confidence": r.get("confidence", {}).get("overall_confidence", 0),
                }
                for r in reranked[:5]
            ],
            "pipeline_confidence": result.pipeline_confidence,
            "model_used": result.model_used,
            "intermediate": intermediate,
            "commentaries": {
                ref: [
                    {"commentator": c["commentator"], "text": c["text_iast"][:300]}
                    for c in comms
                ]
                for ref, comms in commentaries.items()
            },
        }

        return response

    def close(self):
        """Clean up resources."""
        if self._graph_retriever and self._graph_connected:
            self._graph_retriever.close()
            self._graph_connected = False


def main():
    """Main entry point for SRAG pipeline."""
    parser = argparse.ArgumentParser(
        description="SRAG: Sanskrit RAG with Graph-Enhanced Linguistic Re-ranking"
    )

    parser.add_argument(
        "command",
        choices=["preprocess", "build-graph", "build-indices", "query", "serve"],
        help="Command to execute",
    )
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Query to process",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reprocessing",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Use local fallback for query processing (no API calls)",
    )
    parser.add_argument(
        "--langgraph",
        action="store_true",
        help="Use LangGraph pipeline instead of standard pipeline",
    )

    args = parser.parse_args()

    config = Config(args.config)

    if args.langgraph and args.command == "query":
        from src.langchain_components.graph import SRAGGraphPipeline
        pipeline = SRAGGraphPipeline(config)
    else:
        pipeline = SRAGPipeline(config)

    try:
        if args.command == "preprocess":
            pipeline.preprocess(force=args.force)

        elif args.command == "build-graph":
            pipeline.preprocess()
            pipeline.build_graph(clear=args.force)

        elif args.command == "build-indices":
            pipeline.preprocess()
            pipeline.build_indices()

        elif args.command == "query":
            if not args.query:
                print("Error: --query is required for 'query' command")
                sys.exit(1)

            pipeline.preprocess()
            pipeline.build_indices()

            try:
                if hasattr(pipeline, '_get_graph_retriever'):
                    pipeline._get_graph_retriever()
                    logger.info("Neo4j graph retriever connected")
            except Exception as e:
                logger.error(f"Neo4j connection failed: {e}. Graph retrieval will be unavailable.")

            result = pipeline.query(args.query, use_api=not args.local)

            print("\n" + "=" * 80)
            print(f"QUERY: {result['query']}")
            print(f"IAST: {result['query_iast']}")
            print(f"CONCEPTS: {', '.join(result['concepts_extracted'])}")
            if 'query_type' in result:
                print(f"QUERY TYPE: {result['query_type']}")
            print("=" * 80)
            print(f"\nANSWER:\n{result['answer']}")
            print(f"\nVERSES CITED: {', '.join(result['verses_cited'])}")
            print("\nTOP VERSES:")
            for v in result["top_verses"]:
                print(f"  {v['ref']} (confidence: {v['confidence']:.2f})")
            print(f"\nPIPELINE CONFIDENCE: {result['pipeline_confidence']}")

        elif args.command == "serve":
            print("Server mode not yet implemented. Use 'query' command.")

    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
