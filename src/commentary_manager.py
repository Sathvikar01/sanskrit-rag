"""Commentary embedding, storage, and retrieval helpers."""
from __future__ import annotations

import hashlib
import re
import unicodedata
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import DENSE_DIM, MAX_TEXT_LENGTH
from src.xml_parser import TEIXMLParser


COMMENTARY_CONFIG = {
    "vishwanatha": {
        "collection_name": "commentary_vishwanatha",
        "display_name": "Vishwanatha",
        "description": "Commentary by Vishwanatha Cakravarti.",
        "aliases": ["vishwanatha", "visvanatha", "viswanatha", "visvanathah"],
    },
    "shreedhara": {
        "collection_name": "commentary_shreedhara",
        "display_name": "Shreedhara",
        "description": "Commentary by Shreedhara Svami.",
        "aliases": ["shreedhara", "sridhara", "shridhara", "sridharah", "sridharah"],
    },
    "baladeva": {
        "collection_name": "commentary_baladeva",
        "display_name": "Baladeva",
        "description": "Commentary by Baladeva Vidyabhusana.",
        "aliases": ["baladeva", "baladev", "baladevah"],
    },
}


def _ascii_fold(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text or "")
    ascii_text = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return ascii_text.lower()


def normalize_author_key(text: str) -> Optional[str]:
    """Map commentary labels from all datasets to stable author keys."""
    folded = _ascii_fold(text)
    folded = folded.replace("|", " ").replace("_", " ").replace("-", " ")
    folded = re.sub(r"[^a-z\s]", " ", folded)
    folded = re.sub(r"\s+", " ", folded).strip()

    if not folded:
        return None

    for author_key, config in COMMENTARY_CONFIG.items():
        for alias in config["aliases"]:
            if folded == alias or folded.startswith(f"{alias} "):
                return author_key

    return None


def get_author_display_name(author_key: str) -> str:
    return COMMENTARY_CONFIG.get(author_key, {}).get("display_name", author_key.title())


@dataclass
class CommentaryEmbedding:
    """Embedding container for commentary text."""

    id: str
    text: str
    author: str
    verse_id: str
    dense_vector: np.ndarray
    sparse_vector: Dict[int, float]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "author": self.author,
            "verse_id": self.verse_id,
            "dense_vector": self.dense_vector.tolist()
            if hasattr(self.dense_vector, "tolist")
            else list(self.dense_vector),
            "sparse_vector": self.sparse_vector,
            "metadata": self.metadata,
        }


@dataclass
class CommentarySearchResult:
    """Result from commentary search."""

    id: str
    text: str
    author: str
    verse_id: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VerseCommentaryMatch:
    """Top commentary match to display for a verse."""

    verse_id: str
    commentary_id: str
    author_key: str
    author_display_name: str
    text: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verse_id": self.verse_id,
            "commentary_id": self.commentary_id,
            "author_key": self.author_key,
            "author_display_name": self.author_display_name,
            "text": self.text,
            "score": self.score,
            "metadata": self.metadata,
        }


class CommentaryManager:
    """Manager for commentator-specific embedding and retrieval."""

    def __init__(
        self,
        qdrant_manager: Any = None,
        neo4j_manager: Any = None,
        embedding_client: Any = None,
    ):
        self.qdrant = qdrant_manager
        self.neo4j = neo4j_manager
        self.embedding_client = embedding_client

    def commentary_collection_names(self) -> List[str]:
        return [config["collection_name"] for config in COMMENTARY_CONFIG.values()]

    def create_commentary_collections(self, drop_if_exists: bool = False) -> Dict[str, bool]:
        """Create separate Qdrant collections for each commentator."""
        results = {}

        for author, config in COMMENTARY_CONFIG.items():
            if self.qdrant and hasattr(self.qdrant, "create_collection"):
                try:
                    success = self.qdrant.create_collection(
                        collection_name=config["collection_name"],
                        drop_if_exists=drop_if_exists,
                    )
                    results[author] = success
                except Exception as exc:
                    print(f"Failed to create collection for {author}: {exc}")
                    results[author] = False
            else:
                results[author] = False

        return results

    def get_commentary_collection_stats(self) -> Dict[str, int]:
        stats = {}
        if not self.qdrant or not hasattr(self.qdrant, "get_collection_stats"):
            return stats

        for author, config in COMMENTARY_CONFIG.items():
            coll_stats = self.qdrant.get_collection_stats(config["collection_name"])
            stats[author] = int(coll_stats.get("row_count", 0))
        return stats

    def _collections_populated(self) -> bool:
        if not self.qdrant or not hasattr(self.qdrant, "collection_exists"):
            return False

        for config in COMMENTARY_CONFIG.values():
            collection_name = config["collection_name"]
            if not self.qdrant.collection_exists(collection_name):
                return False
            coll_stats = self.qdrant.get_collection_stats(collection_name)
            if int(coll_stats.get("row_count", 0)) <= 0:
                return False
        return True

    def ensure_commentary_ingested(
        self,
        xml_path: str,
        force_reingest: bool = False,
        show_progress: bool = True,
    ) -> Dict[str, int]:
        """Ensure raw commentary text is embedded into commentator collections."""
        if not self.qdrant:
            return {}

        self.create_commentary_collections(drop_if_exists=force_reingest)

        if not force_reingest and self._collections_populated():
            return self.get_commentary_collection_stats()

        parser = TEIXMLParser()
        _, commentaries = parser.parse_with_commentaries(xml_path, "raw")

        embeddings_by_author = self.embed_commentary_chunks(
            commentary_chunks={author: commentaries.get(author, []) for author in COMMENTARY_CONFIG},
            source_dataset="dataset.xml",
            text_variant="raw",
            show_progress=show_progress,
        )
        self.store_commentary_embeddings(
            embeddings_by_author,
            batch_size=100,
            show_progress=show_progress,
        )
        return self.get_commentary_collection_stats()

    def embed_commentary_chunks(
        self,
        commentary_chunks: Dict[str, List[Any]],
        source_dataset: str = "dataset.xml",
        text_variant: str = "raw",
        show_progress: bool = True,
    ) -> Dict[str, List[CommentaryEmbedding]]:
        """Embed commentary chunks separately for each author."""
        embeddings_by_author: Dict[str, List[CommentaryEmbedding]] = {}

        for author, chunks in commentary_chunks.items():
            if not chunks:
                continue

            author_embeddings = []
            if show_progress:
                print(f"Embedding {len(chunks)} commentary chunks for {author}...")

            for index, chunk in enumerate(chunks):
                if self.embedding_client:
                    emb_result = self.embedding_client.embed_query(chunk.text, allow_fallback=False)
                    dense_vector = emb_result.dense_vector
                    sparse_vector = emb_result.sparse_vector
                else:
                    np.random.seed(hash(chunk.text) % (2**31))
                    dense_vector = np.random.randn(DENSE_DIM).astype(np.float32)
                    sparse_vector = {}

                metadata = {
                    **chunk.metadata,
                    "author_key": author,
                    "author_display_name": get_author_display_name(author),
                    "verse_id": chunk.verse_id,
                    "chapter": chunk.chapter,
                    "verse_num": chunk.verse_num,
                    "source_dataset": source_dataset,
                    "text_variant": text_variant,
                    "is_commentary": True,
                }

                author_embeddings.append(
                    CommentaryEmbedding(
                        id=chunk.id,
                        text=chunk.text[:MAX_TEXT_LENGTH],
                        author=author,
                        verse_id=chunk.verse_id,
                        dense_vector=dense_vector,
                        sparse_vector=sparse_vector,
                        metadata=metadata,
                    )
                )

                if show_progress and (index + 1) % 20 == 0:
                    print(f"  Embedded {index + 1}/{len(chunks)} for {author}")

            embeddings_by_author[author] = author_embeddings

        return embeddings_by_author

    def store_commentary_embeddings(
        self,
        embeddings_by_author: Dict[str, List[CommentaryEmbedding]],
        batch_size: int = 100,
        show_progress: bool = True,
    ) -> Dict[str, int]:
        """Store commentary embeddings in commentator collections."""
        stored_counts = {}

        for author, embeddings in embeddings_by_author.items():
            if not embeddings:
                continue

            collection_name = COMMENTARY_CONFIG[author]["collection_name"]
            try:
                stored_counts[author] = self._insert_to_qdrant(
                    collection_name=collection_name,
                    embeddings=embeddings,
                    batch_size=batch_size,
                    show_progress=show_progress,
                )
            except Exception as exc:
                print(f"Failed to store embeddings for {author}: {exc}")
                stored_counts[author] = 0

        return stored_counts

    def _insert_to_qdrant(
        self,
        collection_name: str,
        embeddings: List[CommentaryEmbedding],
        batch_size: int,
        show_progress: bool,
    ) -> int:
        """Insert commentary embeddings into a Qdrant collection."""
        if not self.qdrant or not hasattr(self.qdrant, "client"):
            return 0

        try:
            from qdrant_client.models import PointStruct, SparseVector
        except ImportError:
            return 0

        total_inserted = 0

        for start in range(0, len(embeddings), batch_size):
            batch = embeddings[start : start + batch_size]
            points = []

            for embedding in batch:
                dense_vec = (
                    embedding.dense_vector.tolist()
                    if hasattr(embedding.dense_vector, "tolist")
                    else list(embedding.dense_vector)
                )
                vector_payload: Dict[str, Any] = {"": dense_vec}

                if embedding.sparse_vector:
                    vector_payload["text_sparse"] = SparseVector(
                        indices=list(embedding.sparse_vector.keys()),
                        values=list(embedding.sparse_vector.values()),
                    )

                payload = {
                    "text": embedding.text,
                    "commentary_id": embedding.id,
                    "dataset_type": "commentary",
                    "author": embedding.author,
                    "verse_id": embedding.verse_id,
                    "metadata": {
                        **embedding.metadata,
                        "original_commentary_id": embedding.id,
                    },
                }
                points.append(
                    PointStruct(
                        id=self._qdrant_point_id(embedding.id),
                        vector=vector_payload,
                        payload=payload,
                    )
                )

            self.qdrant.client.upsert(collection_name=collection_name, points=points)
            total_inserted += len(batch)

            if show_progress and total_inserted % 50 == 0:
                print(f"  Stored {total_inserted}/{len(embeddings)} for {collection_name}")

        return total_inserted

    def _qdrant_point_id(self, raw_id: str) -> str:
        """Convert local commentary IDs into a Qdrant-safe UUID string."""
        try:
            return str(uuid.UUID(str(raw_id)))
        except (TypeError, ValueError, AttributeError):
            return str(uuid.uuid5(uuid.NAMESPACE_URL, f"commentary:{raw_id}"))

    def search_commentary(
        self,
        query: str,
        authors: Optional[List[str]] = None,
        verse_ids: Optional[List[str]] = None,
        top_k: int = 5,
    ) -> Dict[str, List[CommentarySearchResult]]:
        """Search commentaries across specified authors."""
        authors = authors or list(COMMENTARY_CONFIG.keys())
        results_by_author: Dict[str, List[CommentarySearchResult]] = {}

        if not query or not query.strip():
            return self.get_commentary_for_verses(verse_ids or [], authors=authors, limit_per_author=top_k)

        if self.embedding_client:
            query_embedding = self.embedding_client.embed_query(query)
            if not (query_embedding.metadata or {}).get("dense_available", True):
                return self.get_commentary_for_verses(verse_ids or [], authors=authors, limit_per_author=top_k)
            query_vector = query_embedding.dense_vector
        else:
            np.random.seed(hash(query) % (2**31))
            query_vector = np.random.randn(DENSE_DIM).astype(np.float32)

        verse_filter = {"verse_ids": verse_ids} if verse_ids else None

        for author in authors:
            if author not in COMMENTARY_CONFIG:
                continue

            collection_name = COMMENTARY_CONFIG[author]["collection_name"]
            search_results = []
            if self.qdrant and hasattr(self.qdrant, "search_dense"):
                try:
                    search_results = self.qdrant.search_dense(
                        query_vector,
                        top_k=top_k,
                        verse_filter=verse_filter,
                        collection_name=collection_name,
                    )
                except Exception as exc:
                    print(f"Search failed for {author}: {exc}")

            results_by_author[author] = [
                CommentarySearchResult(
                    id=str(result.id),
                    text=result.text,
                    author=author,
                    verse_id=result.verse_id,
                    score=result.score,
                    metadata=result.metadata,
                )
                for result in search_results
            ]

        return results_by_author

    def get_commentary_for_verses(
        self,
        verse_ids: List[str],
        authors: Optional[List[str]] = None,
        limit_per_author: int = 10,
    ) -> Dict[str, List[CommentarySearchResult]]:
        """Directly retrieve commentaries for verse IDs from all author collections."""
        authors = authors or list(COMMENTARY_CONFIG.keys())
        results: Dict[str, List[CommentarySearchResult]] = {}

        if not verse_ids or not self.qdrant or not hasattr(self.qdrant, "client"):
            return {author: [] for author in authors}

        try:
            from qdrant_client.models import FieldCondition, Filter, MatchAny
        except ImportError:
            return {author: [] for author in authors}

        verse_filter = Filter(
            must=[FieldCondition(key="verse_id", match=MatchAny(any=verse_ids))]
        )

        for author in authors:
            collection_name = COMMENTARY_CONFIG[author]["collection_name"]
            try:
                points, _ = self.qdrant.client.scroll(
                    collection_name=collection_name,
                    scroll_filter=verse_filter,
                    limit=limit_per_author,
                    with_payload=True,
                )
            except Exception as exc:
                print(f"Failed to fetch commentary for {author}: {exc}")
                results[author] = []
                continue

            results[author] = [
                CommentarySearchResult(
                    id=(point.payload or {}).get("commentary_id")
                    or (point.payload or {}).get("metadata", {}).get("original_commentary_id")
                    or str(point.id),
                    text=(point.payload or {}).get("text", ""),
                    author=author,
                    verse_id=(point.payload or {}).get("verse_id", ""),
                    score=1.0,
                    metadata=(point.payload or {}).get("metadata", {}),
                )
                for point in points
            ]

        return results

    def get_best_matches(
        self,
        query: str,
        verse_ids: List[str],
        authors: Optional[List[str]] = None,
        top_k_per_author: int = 5,
    ) -> List[VerseCommentaryMatch]:
        """Return the top commentary overall for each requested verse."""
        ordered_verse_ids = []
        seen = set()
        for verse_id in verse_ids:
            if verse_id and verse_id not in seen:
                seen.add(verse_id)
                ordered_verse_ids.append(verse_id)

        if not ordered_verse_ids:
            return []

        authors = authors or list(COMMENTARY_CONFIG.keys())
        results_by_author = self.search_commentary(
            query=query,
            authors=authors,
            verse_ids=ordered_verse_ids,
            top_k=max(top_k_per_author, len(ordered_verse_ids)),
        )

        best_by_verse: Dict[str, VerseCommentaryMatch] = {}
        for author, results in results_by_author.items():
            for result in results:
                if result.verse_id not in seen:
                    continue

                match = VerseCommentaryMatch(
                    verse_id=result.verse_id,
                    commentary_id=result.id,
                    author_key=author,
                    author_display_name=get_author_display_name(author),
                    text=result.text,
                    score=float(result.score),
                    metadata=result.metadata,
                )
                existing = best_by_verse.get(result.verse_id)
                if existing is None or match.score > existing.score:
                    best_by_verse[result.verse_id] = match

        missing_verse_ids = [verse_id for verse_id in ordered_verse_ids if verse_id not in best_by_verse]

        if missing_verse_ids:
            direct_results = self.get_commentary_for_verses(
                missing_verse_ids,
                authors=authors,
                limit_per_author=max(top_k_per_author, 1),
            )
            for author, results in direct_results.items():
                for result in results:
                    if result.verse_id in best_by_verse:
                        continue
                    best_by_verse[result.verse_id] = VerseCommentaryMatch(
                        verse_id=result.verse_id,
                        commentary_id=result.id,
                        author_key=author,
                        author_display_name=get_author_display_name(author),
                        text=result.text,
                        score=float(result.score),
                        metadata=result.metadata,
                    )

        return [best_by_verse[verse_id] for verse_id in ordered_verse_ids if verse_id in best_by_verse]


class CommentaryGraphManager:
    """Neo4j graph manager for commentary relationships."""

    def __init__(self, neo4j_manager: Any = None):
        self.neo4j = neo4j_manager

    def create_commentary_schema(self) -> bool:
        """Create Neo4j schema for commentary relationships."""
        if not self.neo4j or not hasattr(self.neo4j, "_run"):
            return False

        self.neo4j._run(
            """
            CREATE CONSTRAINT commentary_id_unique IF NOT EXISTS
            FOR (c:Commentary) REQUIRE c.id IS UNIQUE
            """
        )
        self.neo4j._run(
            """
            CREATE CONSTRAINT author_name_unique IF NOT EXISTS
            FOR (a:Author) REQUIRE a.name IS UNIQUE
            """
        )
        self.neo4j._run(
            """
            CREATE VECTOR INDEX commentary_embeddings IF NOT EXISTS
            FOR (c:Commentary) ON (c.embedding)
            OPTIONS {
                indexConfig: {
                    `vector.dimensions`: 1024,
                    `vector.similarity_function`: 'cosine'
                }
            }
            """
        )
        return True

    def store_commentary_graph(
        self, embeddings_by_author: Dict[str, List[CommentaryEmbedding]]
    ) -> int:
        """Store commentary chunks as graph nodes."""
        if not self.neo4j or not hasattr(self.neo4j, "_run"):
            return 0

        total_stored = 0
        for author, embeddings in embeddings_by_author.items():
            for embedding in embeddings:
                self.neo4j._run(
                    """
                    MERGE (a:Author {name: $author})
                    MERGE (c:Commentary {id: $id})
                    SET c.text = $text,
                        c.verse_id = $verse_id,
                        c.embedding = $embedding,
                        c.author = $author
                    MERGE (v:Verse {id: $verse_id})
                    MERGE (c)-[:COMMENTATES_ON]->(v)
                    MERGE (a)-[:WROTE]->(c)
                    """,
                    {
                        "author": author,
                        "id": embedding.id,
                        "text": embedding.text,
                        "verse_id": embedding.verse_id,
                        "embedding": embedding.dense_vector.tolist()
                        if hasattr(embedding.dense_vector, "tolist")
                        else list(embedding.dense_vector),
                    },
                )
                total_stored += 1

        return total_stored
