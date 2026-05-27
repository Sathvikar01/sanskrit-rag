"""Shared SansRAG service used by API and UI layers."""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from config.settings import COLLECTION_NAMES, L1_REG_LAMBDA, L2_REG_LAMBDA, RRF_TOP_K
from src.answer_generator import AnswerGenerator
from src.commentary_manager import COMMENTARY_CONFIG, CommentaryManager
from src.embedding_client import NVIDIAEmbeddingClient
from src.gemini_client import NVIDIA_LLM_Client
from src.neo4j_manager import Neo4jManager
from src.qdrant_manager import QdrantManager
from src.retriever import RegularizedRetriever, parse_verse_references
from src.verse_db import EXPECTED_BHAGAVAD_GITA_VERSE_COUNT, VerseDatabase, ingest_xml_to_sqlite
from src.xml_parser import TEIXMLParser
from src.entity_lexicon import expand_query_with_aliases
from src.evidence_reranker import RerankContext
from src.query_intent import classify_query_intent


def manage_docker(action: str) -> str:
    """Start, stop, or inspect the Docker Compose services."""
    compose_file = str(ROOT_DIR / "docker-compose.yml")
    try:
        if action == "start":
            result = subprocess.run(
                ["docker", "compose", "-f", compose_file, "up", "-d"],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode == 0:
                return "Docker containers started (Qdrant + Neo4j)."
            return f"Failed to start Docker:\n{result.stderr}"

        if action == "stop":
            result = subprocess.run(
                ["docker", "compose", "-f", compose_file, "down"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            return "Docker containers stopped." if result.returncode == 0 else f"Failed to stop:\n{result.stderr}"

        if action == "status":
            result = subprocess.run(
                ["docker", "compose", "-f", compose_file, "ps"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result.stdout if result.stdout.strip() else "No containers running."
    except FileNotFoundError:
        return "Docker not found. Please install Docker Desktop."
    except subprocess.TimeoutExpired:
        return "Docker command timed out."

    return f"Unsupported Docker action: {action}"


class SansRAGService:
    """Application service that coordinates stores, retrieval, and answer generation."""

    def __init__(self):
        self.parser = TEIXMLParser()
        self.embedder = NVIDIAEmbeddingClient()
        self.llm = NVIDIA_LLM_Client()
        self.qdrant = QdrantManager()
        self.neo4j = Neo4jManager()
        self.verse_db = VerseDatabase()
        self.commentary_manager = CommentaryManager(
            qdrant_manager=self.qdrant,
            embedding_client=self.embedder,
        )

        self.retriever: Optional[RegularizedRetriever] = None
        self.answer_generator: Optional[AnswerGenerator] = None
        self.qdrant_ok = False
        self.neo4j_ok = False
        self.sqlite_ok = False
        self._connected = False

    def _qdrant_search_ready(self) -> bool:
        """Whether the main Qdrant search collection is present and queryable."""
        if not (self.qdrant_ok and self.qdrant and self.qdrant.collection_exists(COLLECTION_NAMES["seg_lemma"])):
            return False
        try:
            return int(self.qdrant.get_collection_stats(COLLECTION_NAMES["seg_lemma"]).get("row_count", 0) or 0) > 0
        except Exception:
            return False

    def _load_cached_embeddings_for_dataset(self, dataset_type: str) -> List[Any]:
        """Load cached embeddings for a specific dataset type from local .npy cache files."""
        cache_dir = ROOT_DIR / "cache"
        if not cache_dir.exists():
            return []

        embeddings = []
        for cache_file in sorted(cache_dir.glob("*.npy")):
            try:
                data = np.load(str(cache_file), allow_pickle=True).item()
            except Exception:
                continue

            metadata = data.get("metadata", {}) or {}
            if metadata.get("dataset_type") != dataset_type:
                continue

            sparse_vector = {
                int(key): float(value)
                for key, value in (data.get("sparse_vector") or {}).items()
            }
            embeddings.append(
                self.embedder._embedding_from_dict(
                    {
                        **data,
                        "id": data.get("id") or cache_file.stem,
                        "metadata": {
                            **metadata,
                            "original_id": metadata.get("original_id") or data.get("id") or cache_file.stem,
                        },
                        "sparse_vector": sparse_vector,
                    }
                )
            )

        return embeddings

    def _init_neo4j_store(self) -> None:
        """Ensure the lemma-morph dataset is available inside Neo4j."""
        xml_path = ROOT_DIR / "dataset.lemma-morphosyntax.xml"
        if not xml_path.exists() or not self.neo4j_ok:
            return

        try:
            stats = self.neo4j.get_collection_stats(refresh=True)
            if stats.get("chunk_count", 0) > 0:
                return

            self.neo4j.create_schema(drop_if_exists=False)

            cached_embeddings = self._load_cached_embeddings_for_dataset("lemma_morph")
            if cached_embeddings:
                print(f"Bootstrapping Neo4j from cached lemma_morph embeddings ({len(cached_embeddings)} chunks)...")
                self.neo4j.insert_embeddings(
                    cached_embeddings,
                    batch_size=100,
                    show_progress=False,
                )
                self.neo4j.get_collection_stats(refresh=True)
                return

            print("Cached lemma_morph embeddings not found; parsing dataset.lemma-morphosyntax.xml for Neo4j bootstrap...")
            chunks = self.parser.parse_file(str(xml_path), "lemma_morph")
            if not chunks:
                return

            embeddings = self.embedder.embed_chunks(chunks, show_progress=False)
            self.neo4j.insert_embeddings(
                embeddings,
                batch_size=100,
                show_progress=False,
            )
            self.neo4j.get_collection_stats(refresh=True)
        except Exception as exc:
            print(f"Neo4j initialization warning: {exc}")

    def _init_qdrant_store(self) -> None:
        """Ensure the segmented+lemmatized dataset is available inside Qdrant."""
        xml_path = ROOT_DIR / "dataset.segmentation-lemma.xml"
        if not xml_path.exists() or not self.qdrant_ok:
            return

        collection_name = COLLECTION_NAMES["seg_lemma"]

        try:
            if self.qdrant.collection_exists(collection_name):
                stats = self.qdrant.get_collection_stats(collection_name)
                if stats.get("row_count", 0) > 0:
                    return
            else:
                self.qdrant.create_collection(collection_name, drop_if_exists=False)

            cached_embeddings = self._load_cached_embeddings_for_dataset("seg_lemma")
            if cached_embeddings:
                print(f"Bootstrapping Qdrant from cached seg_lemma embeddings ({len(cached_embeddings)} chunks)...")
                self.qdrant.insert_embeddings(
                    collection_name,
                    cached_embeddings,
                    batch_size=256,
                    show_progress=False,
                )
                self.qdrant.load_collection(collection_name)
                return

            print("Cached seg_lemma embeddings not found; parsing dataset.segmentation-lemma.xml for Qdrant bootstrap...")
            chunks = self.parser.parse_file(str(xml_path), "seg_lemma")
            if not chunks:
                return

            embeddings = self.embedder.embed_chunks(chunks, show_progress=False)
            self.qdrant.insert_embeddings(
                collection_name,
                embeddings,
                batch_size=256,
                show_progress=False,
            )
            self.qdrant.load_collection(collection_name)
        except Exception as exc:
            print(f"Qdrant initialization warning: {exc}")

    def _init_verse_db(self) -> None:
        self.verse_db.connect()
        stats = self.verse_db.get_stats()
        if stats["total_verses"] == 0 or stats["total_verses"] < EXPECTED_BHAGAVAD_GITA_VERSE_COUNT:
            xml_path = ROOT_DIR / "dataset.xml"
            if xml_path.exists():
                ingest_xml_to_sqlite(str(xml_path), self.verse_db.db_path)
                self.verse_db.connect()
        self.sqlite_ok = True

    def _init_commentary_store(self) -> None:
        xml_path = ROOT_DIR / "dataset.xml"
        if not xml_path.exists() or not self.qdrant_ok:
            return
        try:
            self.commentary_manager.ensure_commentary_ingested(
                str(xml_path),
                force_reingest=False,
                show_progress=False,
            )
        except Exception as exc:
            print(f"Commentary initialization warning: {exc}")

    def connect(self) -> bool:
        """Connect all available components and build the retrieval pipeline."""
        if self._connected:
            return True

        self.qdrant_ok = self.qdrant.connect()
        self.neo4j_ok = self.neo4j.connect()
        self._init_verse_db()
        self._init_qdrant_store()
        self._init_neo4j_store()
        self._init_commentary_store()

        self.retriever = RegularizedRetriever(
            embedding_client=self.embedder,
            qdrant_manager=self.qdrant if self.qdrant_ok else None,
            neo4j_manager=self.neo4j if self.neo4j_ok else None,
            l1_lambda=L1_REG_LAMBDA,
            l2_lambda=L2_REG_LAMBDA,
            adaptive=True,
            llm_client=self.llm,
        )
        self.retriever._qdrant_available = self._qdrant_search_ready()
        self.retriever._neo4j_available = self.neo4j_ok

        self.answer_generator = AnswerGenerator(
            gemini_client=self.llm,
            retriever=self.retriever,
            qdrant_manager=self.qdrant if self.qdrant_ok else None,
            neo4j_manager=self.neo4j if self.neo4j_ok else None,
            verse_db=self.verse_db,
            top_k=RRF_TOP_K,
        )

        self.llm.pre_check_quota()
        self._connected = True
        return True

    def health(self) -> Dict[str, Any]:
        self.connect()
        return {
            "ok": True,
            "components": {
                "qdrant": self.qdrant_ok,
                "neo4j": self.neo4j_ok,
                "sqlite": self.sqlite_ok,
                "llm": self.llm.is_available(),
            },
        }

    def stats(self) -> Dict[str, Any]:
        self.connect()
        stats: Dict[str, Any] = {
            "components": self.health()["components"],
            "qdrant": {},
            "neo4j": {},
            "sqlite": {},
            "commentary": {},
            "architecture": {
                "qdrant": "Semantic vector retrieval with dense, sparse, and BM25 search.",
                "neo4j": "Verse-level and multi-hop graph retrieval over lemma/morphology data.",
                "sqlite": "Canonical original verse lookup used for final answer evidence.",
            },
        }

        if self.qdrant_ok:
            try:
                stats["qdrant"] = {
                    **self.qdrant.get_collection_stats(),
                    "search_ready": self._qdrant_search_ready(),
                }
                stats["commentary"] = self.commentary_manager.get_commentary_collection_stats()
            except Exception as exc:
                stats["qdrant"] = {"error": str(exc)}

        if self.neo4j_ok:
            try:
                stats["neo4j"] = self.neo4j.get_collection_stats()
            except Exception as exc:
                stats["neo4j"] = {"error": str(exc)}

        if self.sqlite_ok:
            try:
                stats["sqlite"] = self.verse_db.get_stats()
            except Exception as exc:
                stats["sqlite"] = {"error": str(exc)}

        if self.answer_generator is not None:
            stats["cache"] = self.answer_generator.cache.status()

        for author, config in COMMENTARY_CONFIG.items():
            stats["commentary"].setdefault(author, 0)
            stats["commentary"].setdefault(f"{author}_display", config["display_name"])

        return stats

    def ask(
        self,
        query: str,
        top_k: int = RRF_TOP_K,
        l1_lambda: float = L1_REG_LAMBDA,
        l2_lambda: float = L2_REG_LAMBDA,
        regularization: str = "combined",
    ) -> Dict[str, Any]:
        self.connect()
        if not query.strip():
            return {"ok": False, "error": "Please enter a search query."}

        assert self.retriever is not None
        assert self.answer_generator is not None

        self.retriever.l1_lambda = l1_lambda
        self.retriever.l2_lambda = l2_lambda
        self.answer_generator.top_k = int(top_k)

        result = self.answer_generator.generate_answer(
            query=query,
            regularization=regularization,
        )
        data = result.to_dict()
        data["ok"] = True
        data["has_evidence"] = bool(
            data.get("evidence", {}).get("canonical_verses")
            or data.get("evidence", {}).get("supporting_chunks")
        )
        return data

    def search(
        self,
        query: str,
        top_k: int = RRF_TOP_K,
        l1_lambda: float = L1_REG_LAMBDA,
        l2_lambda: float = L2_REG_LAMBDA,
        regularization: str = "combined",
    ) -> Dict[str, Any]:
        self.connect()
        if not query.strip():
            return {"ok": False, "error": "Please enter a search query."}

        assert self.retriever is not None
        assert self.answer_generator is not None

        self.retriever.l1_lambda = l1_lambda
        self.retriever.l2_lambda = l2_lambda
        verse_filter = parse_verse_references(query)
        query_expansion = expand_query_with_aliases(query)
        query_intent = classify_query_intent(
            query,
            verse_filter=verse_filter,
            entities=query_expansion.get("entities", []),
        ).to_dict()
        retrieval_query = str(query_expansion.get("expanded_query") or query)
        results = self.retriever.cross_db_rrf_search(
            retrieval_query,
            top_k=int(top_k) * 2,
            include_bm25=True,
            regularization=regularization,
            verse_filter=verse_filter,
        )
        initial_verse_ids = list(dict.fromkeys(result.verse_id for result in results if result.verse_id))
        commentary_verse_ids = self.answer_generator._sqlite_commentary_verse_ids(initial_verse_ids + (verse_filter.verse_ids or []))
        results = self.answer_generator.evidence_reranker.rerank(
            results,
            RerankContext(
                query=query,
                verse_filter=verse_filter,
                query_intent=query_intent,
                entities=query_expansion.get("entities", []),
                commentary_verse_ids=commentary_verse_ids,
            ),
            top_k=int(top_k),
        )

        verse_ids = list(dict.fromkeys((verse_filter.verse_ids or []) + [result.verse_id for result in results if result.verse_id]))
        canonical_verses = self.answer_generator._resolve_canonical_verses(verse_ids, results)
        commentary_matches = self.answer_generator._retrieve_commentary_matches(
            query=query,
            verse_filter=verse_filter,
            retrieved_verses=canonical_verses,
        )
        db_status = self.answer_generator._build_db_status(results)

        return {
            "ok": True,
            "query": query,
            "expanded_query": retrieval_query,
            "method": "cross_db_rrf",
            "query_intent": query_intent,
            "entities": query_expansion.get("entities", []),
            "verse_filter": verse_filter.to_dict() if verse_filter.has_filter() else None,
            "db_status": db_status,
            "total_results": len(results),
            "results": [result.to_dict() for result in results],
            "canonical_verses": canonical_verses,
            "commentary_matches": commentary_matches,
        }

    def docker(self, action: str) -> Dict[str, Any]:
        return {
            "ok": True,
            "action": action,
            "output": manage_docker(action),
        }
