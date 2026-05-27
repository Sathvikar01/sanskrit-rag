"""Main Pipeline Orchestration for SansRAG - Sanskrit Text Retrieval System.

BGE-M3 Embeddings | Neo4j Graph DB | Gemini LLM | RRF Reranking | Citation-Backed Answers
"""
import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from config.settings import (
    COLLECTION_NAMES,
    TEST_QUERIES,
    L1_REG_LAMBDA,
    L2_REG_LAMBDA,
    MAX_CHUNKS_FOR_TESTING,
    RRF_TOP_K
)
from src.commentary_manager import CommentaryManager
from src.xml_parser import TEIXMLParser
from src.embedding_client import NVIDIAEmbeddingClient
from src.gemini_client import GeminiClient
from src.qdrant_manager import QdrantManager
from src.neo4j_manager import Neo4jManager
from src.retriever import HybridRetriever, RegularizedRetriever, HybridSearchResult
from src.answer_generator import AnswerGenerator, AnswerResult
from src.verse_db import EXPECTED_BHAGAVAD_GITA_VERSE_COUNT, VerseDatabase, ingest_xml_to_sqlite


class Colors:
    """ANSI color codes for terminal output."""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class SansRAGPipeline:
    """Main pipeline for Sanskrit text embedding and retrieval.
    
    Workflow:
    1. User Query Input
    2. Translation to IAST Format (Gemini LLM)
    3. ByT5-Sanskrit Processing (normalization)
    4. Dual Retrieval Paths (Vector DB + Graph DB)
    5. HybridRAG Fusion Ranking (RRF)
    6. Graph DB Metadata Fetch
    7. Retrieve Full Verse Text
    8. LLM Answer Generation (Gemini)
    9. Citation-Backed Answer Output
    """
    
    def __init__(
        self,
        data_dir: str = None,
        l1_lambda: float = L1_REG_LAMBDA,
        l2_lambda: float = L2_REG_LAMBDA,
        adaptive: bool = True,
        verbose: bool = True,
        top_k: int = RRF_TOP_K
    ):
        self.data_dir = Path(data_dir) if data_dir else ROOT_DIR
        self.verbose = verbose
        self.top_k = top_k
        
        self.parser = TEIXMLParser()
        self.embedder = NVIDIAEmbeddingClient()
        self.gemini = GeminiClient()
        self.qdrant = QdrantManager()
        self.neo4j = Neo4jManager()
        self.verse_db = VerseDatabase()
        self._init_verse_db()
        self.commentary_manager = CommentaryManager(
            qdrant_manager=self.qdrant,
            neo4j_manager=self.neo4j,
            embedding_client=self.embedder,
        )
        
        if adaptive:
            self.retriever = RegularizedRetriever(
                embedding_client=self.embedder,
                qdrant_manager=self.qdrant,
                neo4j_manager=self.neo4j,
                l1_lambda=l1_lambda,
                l2_lambda=l2_lambda,
                adaptive=adaptive
            )
        else:
            self.retriever = HybridRetriever(
                embedding_client=self.embedder,
                qdrant_manager=self.qdrant,
                neo4j_manager=self.neo4j,
                l1_lambda=l1_lambda,
                l2_lambda=l2_lambda
            )
        
        self.answer_generator = AnswerGenerator(
            gemini_client=self.gemini,
            retriever=self.retriever,
            qdrant_manager=self.qdrant,
            neo4j_manager=self.neo4j,
            verse_db=self.verse_db,
            top_k=top_k
        )
        
        self._initialized = False
        self._ingested = False

    def _init_verse_db(self):
        self.verse_db.connect()
        stats = self.verse_db.get_stats()
        if stats["total_verses"] == 0 or stats["total_verses"] < EXPECTED_BHAGAVAD_GITA_VERSE_COUNT:
            xml_path = ROOT_DIR / "dataset.xml"
            if xml_path.exists():
                ingest_xml_to_sqlite(str(xml_path), self.verse_db.db_path)
                self.verse_db.connect()
    
    def log(self, message: str, color: str = None):
        """Print log message with optional color."""
        if self.verbose:
            if color:
                print(f"{color}{message}{Colors.END}")
            else:
                print(message)
    
    def start_neo4j(self) -> bool:
        """Start Neo4j Docker container."""
        self.log("\n" + "=" * 60, Colors.CYAN)
        self.log("Starting Neo4j Docker container...", Colors.HEADER)
        self.log("=" * 60, Colors.CYAN)
        
        try:
            result = subprocess.run(
                ["docker", "run", "-d", "-p", "7474:7474", "-p", "7687:7687",
                 "-e", "NEO4J_AUTH=neo4j/password", "neo4j"],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                self.log("Neo4j container started successfully", Colors.GREEN)
                return True
            else:
                self.log(f"Docker error: {result.stderr}", Colors.RED)
                return False
        except FileNotFoundError:
            self.log("Error: docker not found. Please install Docker.", Colors.RED)
            return False
    
    def connect_qdrant(self, max_retries: int = 5, delay: int = 5) -> bool:
        """Connect to Qdrant (local mode, no Docker needed)."""
        import time

        self.log("\nConnecting to Qdrant...", Colors.HEADER)

        for attempt in range(max_retries):
            if self.qdrant.connect():
                self.log(f"Connected to Qdrant successfully", Colors.GREEN)
                return True

            if attempt < max_retries - 1:
                self.log(f"Connection attempt {attempt + 1} failed. Retrying in {delay}s...", Colors.YELLOW)
                time.sleep(delay)

        self.log("Failed to connect to Qdrant after maximum retries", Colors.RED)
        return False

    def connect_neo4j(self, max_retries: int = 5, delay: int = 5) -> bool:
        """Connect to Neo4j server with retries."""
        import time

        self.log("\nConnecting to Neo4j...", Colors.HEADER)

        for attempt in range(max_retries):
            if self.neo4j.connect():
                self.log(f"Connected to Neo4j successfully", Colors.GREEN)
                return True

            if attempt < max_retries - 1:
                self.log(f"Connection attempt {attempt + 1} failed. Retrying in {delay}s...", Colors.YELLOW)
                time.sleep(delay)

        self.log("Failed to connect to Neo4j after maximum retries", Colors.RED)
        return False

    def check_llm_quota(self) -> bool:
        """Pre-check LLM availability to avoid query timeouts."""
        self.log("\nChecking LLM availability...", Colors.HEADER)
        if self.gemini.pre_check_quota():
            self.log("LLM available", Colors.GREEN)
            return True
        else:
            self.log("LLM quota exhausted or unavailable, answers will be context-only", Colors.YELLOW)
            return False
    
    def ingest_datasets(self) -> bool:
        """Parse, embed, and store all datasets."""
        self.log("\n" + "=" * 60, Colors.CYAN)
        self.log("INGESTING DATASETS", Colors.HEADER)
        self.log("=" * 60, Colors.CYAN)
        
        self.log("\n[1/4] Parsing XML datasets...", Colors.BLUE)
        chunks = self.parser.parse_all_datasets(str(self.data_dir))
        raw_commentaries = {}
        raw_xml_path = self.data_dir / "dataset.xml"
        if raw_xml_path.exists():
            self.log("  Parsing raw commentary source (dataset.xml)...", Colors.YELLOW)
            _, raw_commentaries = self.parser.parse_with_commentaries(str(raw_xml_path), "raw")
        
        if not chunks:
            self.log("No chunks extracted from datasets", Colors.RED)
            return False
        
        total_chunks = sum(len(c) for c in chunks.values())
        self.log(f"Extracted {total_chunks} total chunks from {len(chunks)} datasets", Colors.GREEN)
        
        self.log("\n[2/4] Creating Qdrant collection (seg_lemma)...", Colors.BLUE)
        self.qdrant.create_collection(COLLECTION_NAMES["seg_lemma"])
        self.commentary_manager.create_commentary_collections(drop_if_exists=False)
        
        self.log("\n[2/4] Creating Neo4j schema (lemma_morph)...", Colors.BLUE)
        self.neo4j.create_schema()
        
        self.log(f"Created vector store + graph store", Colors.GREEN)
        
        self.log("\n[3/4] Generating BGE-M3 embeddings...", Colors.BLUE)
        all_embeddings = {}
        
        for dtype, chunk_list in chunks.items():
            self.log(f"\n  Embedding {dtype} ({len(chunk_list)} chunks)...", Colors.YELLOW)
            if MAX_CHUNKS_FOR_TESTING > 0:
                chunk_list = chunk_list[:MAX_CHUNKS_FOR_TESTING]
                self.log(f"  (Limited to {MAX_CHUNKS_FOR_TESTING} chunks for testing)", Colors.YELLOW)
            embeddings = self.embedder.embed_chunks(chunk_list, show_progress=True)
            all_embeddings[dtype] = embeddings
            self.log(f"  Generated {len(embeddings)} embeddings for {dtype}", Colors.GREEN)

        commentary_embeddings = {}
        if raw_commentaries:
            self.log("\n  Embedding raw commentator collections...", Colors.YELLOW)
            commentary_embeddings = self.commentary_manager.embed_commentary_chunks(
                raw_commentaries,
                source_dataset="dataset.xml",
                text_variant="raw",
                show_progress=True,
            )
        
        self.log("\n[4/4] Inserting embeddings...", Colors.BLUE)
        
        # Insert seg_lemma into Qdrant
        seg_embs = all_embeddings.get("seg_lemma", [])
        if seg_embs:
            self.log(f"\n  Inserting {len(seg_embs)} into Qdrant (seg_lemma)...", Colors.YELLOW)
            count = self.qdrant.insert_embeddings(COLLECTION_NAMES["seg_lemma"], seg_embs)
            self.log(f"  Inserted {count} into Qdrant", Colors.GREEN)
        
        # Insert lemma_morph into Neo4j
        morph_embs = all_embeddings.get("lemma_morph", [])
        if morph_embs:
            self.log(f"\n  Inserting {len(morph_embs)} into Neo4j (lemma_morph)...", Colors.YELLOW)
            count = self.neo4j.insert_embeddings(morph_embs)
            self.log(f"  Inserted {count} into Neo4j", Colors.GREEN)

        if commentary_embeddings:
            stored_counts = self.commentary_manager.store_commentary_embeddings(
                commentary_embeddings,
                batch_size=100,
                show_progress=True,
            )
            for author, count in stored_counts.items():
                self.log(
                    f"  Inserted {count} commentary chunks into {author}",
                    Colors.GREEN,
                )
        
        self._ingested = True
        self.log("\n" + "=" * 60, Colors.GREEN)
        self.log("INGESTION COMPLETE", Colors.BOLD)
        self.log("=" * 60, Colors.GREEN)
        
        return True
    
    def search(
        self,
        query: str,
        collection: str = None,
        top_k: int = 10,
        regularization: str = "combined"
    ) -> Dict[str, List[HybridSearchResult]]:
        """Search across collections."""
        if collection:
            results = {
                collection: self.retriever.hybrid_search(
                    query, COLLECTION_NAMES.get(collection, collection),
                    top_k=top_k,
                    regularization=regularization
                )
            }
        else:
            results = self.retriever.search_all_collections(
                query, top_k=top_k, regularization=regularization
            )
        
        return results
    
    def search_rrf(
        self,
        query: str,
        top_k: int = 10,
        k_rrf: int = 60,
        include_bm25: bool = True,
        regularization: str = "combined"
    ) -> List[HybridSearchResult]:
        """
        Cross-database RRF reranked search.
        Retrieves from BOTH Qdrant (vector) and Neo4j (graph),
        then fuses results using Reciprocal Rank Fusion.
        """
        return self.retriever.cross_db_rrf_search(
            query,
            top_k=top_k,
            k_rrf=k_rrf,
            include_bm25=include_bm25,
            regularization=regularization
        )
    
    def ask(self, query: str) -> AnswerResult:
        """Full HybridRAG pipeline: IAST translation -> dual retrieval -> RRF -> citation-backed answer."""
        return self.answer_generator.generate_answer(query)
    
    def display_answer(self, result: AnswerResult):
        """Pretty-print a citation-backed answer."""
        self.log("\n" + "=" * 70, Colors.CYAN)
        self.log(f"Query: \"{result.query}\"", Colors.BOLD + Colors.HEADER)
        self.log(f"IAST: \"{result.iast_query}\"", Colors.CYAN)
        self.log(f"Normalized: \"{result.normalized_query}\"", Colors.CYAN)
        self.log("=" * 70, Colors.CYAN)
        
        self.log(f"\nLatency: {result.latency_ms:.2f}ms", Colors.YELLOW)
        self.log(f"Sources: Qdrant={result.sources.get('qdrant_verses', 0)}, "
                 f"Neo4j={result.sources.get('neo4j_verses', 0)}, "
                 f"Total={result.sources.get('total_retrieved', 0)}", Colors.YELLOW)
        
        self.log(f"\n{Colors.BOLD}Answer:{Colors.END}\n", Colors.GREEN)
        self.log(result.answer, Colors.END)
        
        if result.citations:
            self.log(f"\n{Colors.BOLD}Citations:{Colors.END}", Colors.BLUE)
            for i, c in enumerate(result.citations, 1):
                vid = c.get("verse_id", "N/A")
                text = c.get("text", "")[:100]
                score = c.get("score", 0.0)
                source = c.get("source", "Unknown")
                self.log(f"  [{i}] {vid} (Score: {score:.4f}, Source: {source}): {text}...", Colors.CYAN)
        
        self.log("\n" + "=" * 70, Colors.CYAN)
    
    def save_answer(
        self,
        result: AnswerResult,
        output_dir: str = None
    ) -> str:
        """Save answer with citations to JSON."""
        output_dir = Path(output_dir) if output_dir else ROOT_DIR / "results"
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"answer_{timestamp}.json"
        filepath = output_dir / filename
        
        output = {
            "query": result.query,
            "iast_query": result.iast_query,
            "normalized_query": result.normalized_query,
            "timestamp": datetime.now().isoformat(),
            "latency_ms": result.latency_ms,
            "answer": result.answer,
            "citations": result.citations,
            "sources": result.sources,
            "retrieval_stats": result.retrieval_stats
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        self.log(f"\nSaved answer to: {filepath}", Colors.GREEN)
        return str(filepath)
    
    def display_rrf_results(self, results: List[HybridSearchResult], query: str):
        """Pretty-print RRF reranked search results."""
        self.log("\n" + "=" * 70, Colors.CYAN)
        self.log(f"RRF Reranked Query: \"{query}\"", Colors.BOLD + Colors.HEADER)
        self.log("=" * 70, Colors.CYAN)
        
        reg_params = self.retriever.get_regularization_params() if hasattr(self.retriever, 'get_regularization_params') else {}
        
        if reg_params:
            self.log(f"Regularization - L1: {reg_params.get('l1_lambda', 0):.4f}, L2: {reg_params.get('l2_lambda', 0):.4f}", Colors.YELLOW)
        
        if not results:
            self.log("No results found.", Colors.RED)
            return
        
        rank = 1
        for r in results:
            sources = r.metadata.get('sources', {})
            source_tags = []
            if sources.get('qdrant'):
                source_tags.append("Qdrant")
            if sources.get('neo4j'):
                source_tags.append("Neo4j")
            source_str = ", ".join(source_tags) if source_tags else "Unknown"
            
            self.log(f"\n  [{rank}] ID: {r.id} | RRF Score: {r.final_score:.4f}", Colors.BOLD)
            self.log(f"      Sources: {source_str}", Colors.CYAN)
            self.log(f"      Verse: {r.verse_id or 'N/A'}", Colors.CYAN)
            self.log(f"      Text: {r.text[:80]}{'...' if len(r.text) > 80 else ''}", Colors.END)
            self.log(f"      Breakdown: D={r.dense_score:.3f} | S={r.sparse_score:.3f} | "
                     f"C={r.colbert_score:.3f} | BM25={r.bm25_score:.3f}", Colors.YELLOW)
            rank += 1
        
        self.log("\n" + "=" * 70, Colors.CYAN)
    
    def display_results(self, results: Dict[str, List[HybridSearchResult]], query: str):
        """Pretty-print search results to console."""
        self.log("\n" + "=" * 70, Colors.CYAN)
        self.log(f"Query: \"{query}\"", Colors.BOLD + Colors.HEADER)
        self.log("=" * 70, Colors.CYAN)
        
        reg_params = self.retriever.get_regularization_params() if hasattr(self.retriever, 'get_regularization_params') else {}
        
        if reg_params:
            self.log(f"Regularization - L1: {reg_params.get('l1_lambda', 0):.4f}, L2: {reg_params.get('l2_lambda', 0):.4f}", Colors.YELLOW)
        
        total_results = sum(len(r) for r in results.values())
        
        if total_results == 0:
            self.log("No results found.", Colors.RED)
            return
        
        rank = 1
        for dtype, result_list in results.items():
            if result_list:
                self.log(f"\n[{dtype.upper()}]", Colors.BLUE + Colors.BOLD)
                
                for r in result_list:
                    self.log(f"\n  [{rank}] ID: {r.id} | Score: {r.final_score:.4f}", Colors.BOLD)
                    self.log(f"      Verse: {r.verse_id or 'N/A'}", Colors.CYAN)
                    self.log(f"      Text: {r.text[:80]}{'...' if len(r.text) > 80 else ''}", Colors.END)
                    self.log(f"      Breakdown: D={r.dense_score:.3f} | S={r.sparse_score:.3f} | "
                             f"C={r.colbert_score:.3f} | BM25={r.bm25_score:.3f}", Colors.YELLOW)
                    rank += 1
        
        self.log("\n" + "=" * 70, Colors.CYAN)
    
    def save_rrf_results(
        self,
        results: List[HybridSearchResult],
        query: str,
        output_dir: str = None
    ) -> str:
        """Save RRF results to JSON file."""
        output_dir = Path(output_dir) if output_dir else ROOT_DIR / "results"
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"rrf_query_{timestamp}.json"
        filepath = output_dir / filename
        
        reg_params = {}
        if hasattr(self.retriever, 'get_regularization_params'):
            reg_params = self.retriever.get_regularization_params()
        
        output = {
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "method": "cross_db_rrf",
            "regularization": {
                "l1_lambda": reg_params.get('l1_lambda', L1_REG_LAMBDA),
                "l2_lambda": reg_params.get('l2_lambda', L2_REG_LAMBDA)
            },
            "total_results": len(results),
            "results": [r.to_dict() for r in results]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        self.log(f"\nSaved RRF results to: {filepath}", Colors.GREEN)
        return str(filepath)
    
    def save_results(
        self,
        results: Dict[str, List[HybridSearchResult]],
        query: str,
        output_dir: str = None
    ) -> str:
        """Save results to JSON file."""
        output_dir = Path(output_dir) if output_dir else ROOT_DIR / "results"
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"query_{timestamp}.json"
        filepath = output_dir / filename
        
        reg_params = {}
        if hasattr(self.retriever, 'get_regularization_params'):
            reg_params = self.retriever.get_regularization_params()
        
        output = {
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "regularization": {
                "l1_lambda": reg_params.get('l1_lambda', L1_REG_LAMBDA),
                "l2_lambda": reg_params.get('l2_lambda', L2_REG_LAMBDA)
            },
            "total_results": sum(len(r) for r in results.values()),
            "results": {
                dtype: [r.to_dict() for r in result_list]
                for dtype, result_list in results.items()
            }
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        self.log(f"\nSaved to: {filepath}", Colors.GREEN)
        return str(filepath)
    
    def run_test_queries(self) -> Dict[str, Any]:
        """Run predefined test queries and return metrics."""
        self.log("\n" + "=" * 60, Colors.CYAN)
        self.log("RUNNING TEST QUERIES", Colors.HEADER)
        self.log("=" * 60, Colors.CYAN)
        
        metrics = {
            "queries": [],
            "total_latency_ms": 0,
            "avg_latency_ms": 0
        }
        
        import time
        
        for query in TEST_QUERIES:
            start = time.time()
            results = self.search(query, top_k=5)
            latency_ms = (time.time() - start) * 1000
            
            total_results = sum(len(r) for r in results.values())
            
            query_metrics = {
                "query": query,
                "latency_ms": latency_ms,
                "total_results": total_results,
                "results_by_collection": {
                    dtype: len(r) for dtype, r in results.items()
                }
            }
            
            metrics["queries"].append(query_metrics)
            metrics["total_latency_ms"] += latency_ms
            
            self.log(f"\n  Query: \"{query}\"", Colors.BLUE)
            self.log(f"    Latency: {latency_ms:.2f}ms", Colors.YELLOW)
            self.log(f"    Results: {total_results}", Colors.GREEN)
        
        metrics["avg_latency_ms"] = metrics["total_latency_ms"] / len(TEST_QUERIES)
        
        self.log("\n" + "-" * 60, Colors.CYAN)
        self.log(f"Average Latency: {metrics['avg_latency_ms']:.2f}ms", Colors.BOLD + Colors.GREEN)
        self.log("-" * 60, Colors.CYAN)
        
        return metrics
    
    def repl_loop(self):
        """Interactive REPL for search queries."""
        self.log("\n" + "=" * 70, Colors.GREEN + Colors.BOLD)
        self.log("SansRAG Interactive REPL", Colors.HEADER + Colors.BOLD)
        self.log("BGE-M3 Embeddings | Neo4j Vector DB | L1/L2 Regularization", Colors.CYAN)
        self.log("=" * 70, Colors.GREEN + Colors.BOLD)
        
        self.log("\nCommands:", Colors.BLUE)
        self.log("  <query>       - Full HybridRAG: IAST + dual retrieval + RRF + Gemini answer", Colors.END)
        self.log("  :rrf <query>  - Cross-database RRF reranked search only", Colors.END)
        self.log("  :set l1 0.02  - Set L1 lambda parameter", Colors.END)
        self.log("  :set l2 0.002 - Set L2 lambda parameter", Colors.END)
        self.log("  :weights      - Show current retrieval weights", Colors.END)
        self.log("  :stats        - Show regularization parameters", Colors.END)
        self.log("  :test         - Run test queries", Colors.END)
        self.log("  :quit         - Exit", Colors.END)
        self.log("")
        
        while True:
            try:
                user_input = input(f"\n{Colors.BOLD}>>> {Colors.END}").strip()
            except (EOFError, KeyboardInterrupt):
                self.log("\nGoodbye!", Colors.GREEN)
                break
            
            if not user_input:
                continue
            
            if user_input == ":quit":
                self.log("Goodbye!", Colors.GREEN)
                break
            
            elif user_input.startswith(":set l1"):
                try:
                    new_l1 = float(user_input.split()[-1])
                    self.retriever.l1_lambda = new_l1
                    self.log(f"L1 lambda set to {new_l1}", Colors.GREEN)
                except (ValueError, IndexError):
                    self.log("Usage: :set l1 <value>", Colors.RED)
            
            elif user_input.startswith(":set l2"):
                try:
                    new_l2 = float(user_input.split()[-1])
                    self.retriever.l2_lambda = new_l2
                    self.log(f"L2 lambda set to {new_l2}", Colors.GREEN)
                except (ValueError, IndexError):
                    self.log("Usage: :set l2 <value>", Colors.RED)
            
            elif user_input == ":weights":
                weights = self.retriever.weights
                self.log(f"\nRetrieval Weights:", Colors.BLUE)
                self.log(f"  Dense:   {weights.dense}", Colors.END)
                self.log(f"  Sparse:  {weights.sparse}", Colors.END)
                self.log(f"  Colbert: {weights.colbert}", Colors.END)
                self.log(f"  BM25:    {weights.bm25}", Colors.END)
            
            elif user_input == ":stats":
                if hasattr(self.retriever, 'get_regularization_params'):
                    params = self.retriever.get_regularization_params()
                    self.log(f"\nRegularization Parameters:", Colors.BLUE)
                    self.log(f"  L1 Lambda: {params.get('l1_lambda', 'N/A')}", Colors.END)
                    self.log(f"  L2 Lambda: {params.get('l2_lambda', 'N/A')}", Colors.END)
                else:
                    self.log(f"\nL1 Lambda: {self.retriever.l1_lambda}", Colors.END)
                    self.log(f"L2 Lambda: {self.retriever.l2_lambda}", Colors.END)
            
            elif user_input == ":test":
                self.run_test_queries()
            
            elif user_input.startswith(":rrf"):
                rrf_query = user_input[4:].strip()
                if rrf_query:
                    rrf_results = self.search_rrf(rrf_query, top_k=10)
                    self.display_rrf_results(rrf_results, rrf_query)
                    self.save_rrf_results(rrf_results, rrf_query)
                else:
                    self.log("Usage: :rrf <query>", Colors.RED)
            
            else:
                answer = self.ask(user_input)
                self.display_answer(answer)
                self.save_answer(answer)
    
    def run(self, ingest: bool = False, test: bool = False, interactive: bool = True):
        """Run the complete pipeline."""
        qdrant_ok = self.connect_qdrant()
        neo4j_ok = self.connect_neo4j()
        
        if not qdrant_ok and not neo4j_ok:
            self.log("Neither Qdrant nor Neo4j connected. Exiting.", Colors.RED)
            return False

        if not neo4j_ok:
            self.log("Neo4j not connected, continuing without graph store", Colors.YELLOW)
            self.retriever._neo4j_available = False

        self.check_llm_quota()

        if ingest:
            if not self.ingest_datasets():
                return False

        if test:
            self.run_test_queries()

        if interactive:
            self.repl_loop()

        return True
    
    def cleanup(self):
        """Cleanup resources."""
        self.qdrant.disconnect()
        self.neo4j.disconnect()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="SansRAG - Sanskrit Text Retrieval System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --ingest                    # Ingest datasets and start REPL
  python main.py --test                      # Run test queries
  python main.py --ingest --test --interactive  # Full pipeline
  python main.py --l1 0.02 --l2 0.002        # Custom regularization params
        """
    )
    
    parser.add_argument(
        "--ingest", "-i",
        action="store_true",
        help="Ingest datasets into Milvus"
    )
    
    parser.add_argument(
        "--test", "-t",
        action="store_true",
        help="Run test queries"
    )
    
    parser.add_argument(
        "--interactive", "-r",
        action="store_true",
        default=False,
        help="Start interactive REPL (default: False)"
    )
    
    parser.add_argument(
        "--l1",
        type=float,
        default=L1_REG_LAMBDA,
        help=f"L1 regularization lambda (default: {L1_REG_LAMBDA})"
    )
    
    parser.add_argument(
        "--l2",
        type=float,
        default=L2_REG_LAMBDA,
        help=f"L2 regularization lambda (default: {L2_REG_LAMBDA})"
    )
    
    parser.add_argument(
        "--no-adaptive",
        action="store_true",
        help="Disable adaptive regularization tuning"
    )
    
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(ROOT_DIR),
        help="Directory containing XML datasets"
    )
    
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress verbose output"
    )
    
    args = parser.parse_args()
    
    pipeline = SansRAGPipeline(
        data_dir=args.data_dir,
        l1_lambda=args.l1,
        l2_lambda=args.l2,
        adaptive=not args.no_adaptive,
        verbose=not args.quiet
    )
    
    try:
        pipeline.run(
            ingest=args.ingest,
            test=args.test,
            interactive=args.interactive
        )
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        pipeline.cleanup()


if __name__ == "__main__":
    main()
