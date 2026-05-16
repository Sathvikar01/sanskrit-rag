"""SRAG: Sanskrit RAG with Graph-Enhanced Linguistic Re-ranking.

End-to-end pipeline for querying the Bhagavad Gita.
"""

import argparse
import json
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
from src.preprocessing.iast_devanagari import get_converter
from src.preprocessing.xml_parser import XMLParser
from src.reranking.linguistic_reranker import LinguisticReranker
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.graph_retriever import GraphRetriever
from src.retrieval.hybrid_fusion import HybridRetriever
from src.retrieval.vector_store import VectorStore
from src.utils.config import Config
from src.utils.logger import logger


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
        """Run the preprocessing pipeline.

        Args:
            force: Whether to force reprocessing even if chunks exist.
        """
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
        """Build the Neo4j knowledge graph.

        Args:
            clear: Whether to clear existing graph first.
        """
        import_dir = Path(self.config.get("data.graph_import_dir", "data/processed/graph_import"))

        with GraphBuilder(self.config) as builder:
            builder.build_from_files(import_dir, clear=clear)
            stats = builder.get_stats()
            logger.info(f"Graph stats: {stats}")

    def build_indices(self, use_devanagari: bool = True):
        """Build vector and BM25 indices.

        Args:
            use_devanagari: Whether to use Devanagari for vector embeddings.
        """
        if not self.chunks:
            raise ValueError("No chunks loaded. Run preprocess() first.")

        faiss_path = Path(self.config.get("data.faiss_index"))
        metadata_path = Path(self.config.get("data.faiss_metadata"))

        if faiss_path.exists() and metadata_path.exists():
            logger.info("Loading existing FAISS index...")
            self.vector_store.load(faiss_path, metadata_path)
        else:
            logger.info("Building FAISS index...")
            self.vector_store.build_index(self.chunks, use_devanagari=use_devanagari, verse_only=False)
            self.vector_store.save(faiss_path, metadata_path)

        logger.info("Building BM25 index...")
        self.bm25_retriever.build_index(self.chunks, use_lemmas=True)

    def retrieve(
        self,
        query_iast: str,
        query_devanagari: str,
        concepts: list[str],
        top_k: int = 50,
    ) -> list[dict]:
        """Retrieve candidates using hybrid retrieval.

        Args:
            query_iast: Query in IAST.
            query_devanagari: Query in Devanagari.
            concepts: Extracted concept names.
            top_k: Number of candidates per method.

        Returns:
            Fused retrieval results.
        """
        vector_results = self.vector_store.search(query_devanagari, top_k=top_k)

        graph_retriever = self._get_graph_retriever()
        graph_results = graph_retriever.search_combined(query_iast, concepts, top_k=top_k)

        bm25_results = self.bm25_retriever.search(query_iast, top_k=top_k)

        fused = self.hybrid_retriever.fuse_results(
            vector_results, graph_results, bm25_results, top_k=top_k
        )

        return fused

    def query(
        self,
        user_query: str,
        use_api: bool = True,
    ) -> dict:
        """Process a user query end-to-end.

        Args:
            user_query: The user's question.
            use_api: Whether to use MiMo API for query processing.

        Returns:
            Complete response dictionary.
        """
        if use_api:
            processed = self.query_processor.process_query(user_query)
        else:
            processed = self.query_processor.process_query_local(user_query)

        logger.info(
            f"Query processed: iast='{processed.query_iast[:50]}...', "
            f"concepts={processed.concepts}"
        )

        candidates = self.retrieve(
            processed.query_iast,
            processed.query_devanagari,
            processed.concepts,
        )

        reranked = self.reranker.rerank(
            query_iast=processed.query_iast,
            concepts=processed.concepts,
            candidates=candidates,
            all_chunks=self.chunks,
            chunk_map=self.chunk_map,
        )

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

    args = parser.parse_args()

    config = Config(args.config)
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
                pipeline._get_graph_retriever()
            except Exception as e:
                logger.warning(f"Graph connection failed: {e}. Continuing without graph.")

            result = pipeline.query(args.query, use_api=not args.local)

            print("\n" + "=" * 80)
            print(f"QUERY: {result['query']}")
            print(f"IAST: {result['query_iast']}")
            print(f"CONCEPTS: {', '.join(result['concepts_extracted'])}")
            print("=" * 80)
            print(f"\nANSWER:\n{result['answer']}")
            print(f"\nVERSES CITED: {', '.join(result['verses_cited'])}")
            print(f"\nTOP VERSES:")
            for v in result["top_verses"]:
                print(f"  {v['ref']} (confidence: {v['confidence']:.2f})")
            print(f"\nPIPELINE CONFIDENCE: {result['pipeline_confidence']}")

        elif args.command == "serve":
            print("Server mode not yet implemented. Use 'query' command.")

    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
