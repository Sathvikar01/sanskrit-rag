"""Neo4j Graph Database Manager for Sanskrit Text Retrieval.

Stores lemmatised + segmented + morphosyntactic data as a graph
with vector search capabilities for dense embeddings.

Retrieval modes:
- Verse-level: Direct lookup by verse_id, chapter/verse range filtering
- Multi-hop: Query terms → Lemma → Word → Chunk → Related Lemmas
- Morphosyntax: Filter by grammatical features (Case, Gender, Number, etc.)
- GraphRAG: Concept-based retrieval with semantic expansion
- Cross-reference: Follow relationships between verses and concepts
"""
import time
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import (
    NEO4J_URI,
    NEO4J_USER,
    NEO4J_PASSWORD,
    DENSE_DIM,
    MAX_TEXT_LENGTH,
)

try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    print("Warning: neo4j not installed. Run: pip install neo4j")


@dataclass
class SearchResult:
    """Container for search results."""
    id: str
    text: str
    score: float
    dataset_type: str
    verse_id: Optional[str]
    metadata: Dict[str, Any]


class Neo4jManager:
    """Manager for Neo4j graph database operations.
    
    Provides verse-level and multi-hop graph retrieval for morphosyntax-tagged data.
    """
    
    def __init__(
        self,
        uri: str = NEO4J_URI,
        user: str = NEO4J_USER,
        password: str = NEO4J_PASSWORD
    ):
        self.uri = uri
        self.user = user
        self.password = password
        self._driver = None
        self._connected = False
        self._stats_cache: Optional[Dict[str, Any]] = None
    
    def connect(self) -> bool:
        """Connect to Neo4j server."""
        if not NEO4J_AVAILABLE:
            print("Neo4j not available")
            return False

        try:
            self._driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password),
            )
            self._driver.verify_connectivity()
            self._connected = True
            self._invalidate_stats_cache()
            print(f"Connected to Neo4j at {self.uri}")
            return True
        except Exception as e:
            print(f"Failed to connect to Neo4j: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from Neo4j."""
        if self._driver:
            self._driver.close()
            self._connected = False
            self._invalidate_stats_cache()

    def _invalidate_stats_cache(self) -> None:
        self._stats_cache = None

    def _existing_labels(self) -> set[str]:
        """Return the set of labels currently present in the graph."""
        if not self._connected or not self._driver:
            return set()
        try:
            with self._driver.session() as session:
                return {
                    record["label"]
                    for record in session.run("CALL db.labels() YIELD label RETURN label")
                }
        except Exception as exc:
            print(f"Neo4j label inspection error: {exc}")
            return set()

    def has_graph_data(self, refresh: bool = False) -> bool:
        """Return whether the graph currently has chunk nodes available for search."""
        stats = self.get_collection_stats(refresh=refresh)
        return bool(stats.get("chunk_count", 0))
    
    def _run(self, query: str, params: Dict = None) -> Any:
        """Execute a Cypher query and return all records with error handling."""
        if not self._connected or not self._driver:
            return []
        try:
            with self._driver.session() as session:
                result = session.run(query, params or {})
                return list(result)
        except Exception as e:
            print(f"Neo4j query error: {e}")
            print(f"Query: {query[:200]}...")
            return []
    
    def create_schema(self, drop_if_exists: bool = True) -> bool:
        """Create graph schema, constraints, and vector index."""
        if not NEO4J_AVAILABLE or not self._connected:
            return False
        
        if drop_if_exists:
            self._run("MATCH (n) DETACH DELETE n")
            print("Cleared existing graph data")
        
        self._run("CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE")
        self._run("CREATE CONSTRAINT lemma_text_unique IF NOT EXISTS FOR (l:Lemma) REQUIRE l.text IS UNIQUE")
        self._run("CREATE CONSTRAINT word_form_unique IF NOT EXISTS FOR (w:Word) REQUIRE w.form IS UNIQUE")
        
        self._run(f"""
            CREATE VECTOR INDEX chunk_embeddings IF NOT EXISTS
            FOR (c:Chunk) ON (c.embedding)
            OPTIONS {{
                indexConfig: {{
                    `vector.dimensions`: {DENSE_DIM},
                    `vector.similarity_function`: 'cosine'
                }}
            }}
        """)
        
        print("Created Neo4j schema: constraints + vector index")
        self._invalidate_stats_cache()
        return True
    
    def insert_embeddings(
        self,
        embeddings: List[Any],
        batch_size: int = 100,
        show_progress: bool = True
    ) -> int:
        """Insert embeddings as graph nodes with relationships using batched UNWIND."""
        if not NEO4J_AVAILABLE or not self._connected:
            return 0
        
        total_inserted = 0
        
        for i in range(0, len(embeddings), batch_size):
            batch = embeddings[i:i + batch_size]
            
            chunk_data = []
            all_words = []
            
            for e in batch:
                text = e.text[:MAX_TEXT_LENGTH - 1] if len(e.text) >= MAX_TEXT_LENGTH else e.text
                dense_vec = e.dense_vector.tolist() if hasattr(e.dense_vector, 'tolist') else list(e.dense_vector)
                dataset_type = e.metadata.get("dataset_type", "lemma_morph")
                verse_id = e.metadata.get("verse_id", "")
                line_number = e.metadata.get("line_number", 0)
                original_id = e.metadata.get("original_id", e.id)
                
                words = self._parse_morphosyntax(text)[0]
                
                chunk_data.append({
                    "id": e.id,
                    "text": text,
                    "embedding": dense_vec,
                    "dataset_type": dataset_type,
                    "verse_id": verse_id,
                    "line_number": line_number,
                    "original_id": original_id
                })
                
                for w in words:
                    all_words.append({
                        "chunk_id": e.id,
                        "form": w["form"],
                        "lemma": w.get("lemma", w["form"]),
                        "position": w["position"],
                        "features": w.get("features", {})
                    })
            
            if chunk_data:
                self._run("""
                    UNWIND $chunks AS c
                    MERGE (chunk:Chunk {id: c.id})
                    SET chunk.text = c.text,
                        chunk.embedding = c.embedding,
                        chunk.dataset_type = c.dataset_type,
                        chunk.verse_id = c.verse_id,
                        chunk.line_number = c.line_number,
                        chunk.original_id = c.original_id
                """, {"chunks": chunk_data})
            
            if all_words:
                self._run("""
                    UNWIND $words AS w
                    MATCH (c:Chunk {id: w.chunk_id})
                    MERGE (word:Word {form: w.form})
                    MERGE (l:Lemma {text: w.lemma})
                    MERGE (c)-[:CONTAINS {position: w.position}]->(word)
                    MERGE (word)-[:HAS_LEMMA]->(l)
                    SET word += w.features
                """, {"words": all_words})
            
            total_inserted += len(batch)
            
            if show_progress and total_inserted % 500 == 0:
                print(f"  Inserted {total_inserted}/{len(embeddings)} chunks")
        
        print(f"Total inserted: {total_inserted}")
        self._invalidate_stats_cache()
        return total_inserted
    
    def _parse_morphosyntax(self, text: str) -> Tuple[List[Dict], List[str], List[Dict]]:
        """Parse morphosyntactic annotations from text.
        
        Expected format: word_Case=X|Gender=Y|Number=Z lemma_Case=A|...
        """
        words = []
        tokens = text.split()
        
        for pos, token in enumerate(tokens):
            if "_" in token and ("Case=" in token or "Gender=" in token):
                parts = token.rsplit("_", 1)
                form = parts[0].replace("_", " ")
                feat_str = parts[1] if len(parts) > 1 else ""
                
                features = {}
                lemma = form
                for feat in feat_str.split("|"):
                    if "=" in feat:
                        key, val = feat.split("=", 1)
                        features[key] = val
                    elif feat and not feat.startswith("_"):
                        lemma = feat
                
                words.append({
                    "form": form,
                    "lemma": lemma,
                    "features": features,
                    "position": pos
                })
            else:
                clean = token.replace("_", " ").strip()
                if clean:
                    words.append({
                        "form": clean,
                        "lemma": clean,
                        "features": {},
                        "position": pos
                    })
        
        return words, [w["lemma"] for w in words], [w["features"] for w in words]
    
    # ============================================================
    # VERSE-LEVEL RETRIEVAL (Primary mode for graph DB)
    # ============================================================
    
    def search_by_verse_id(self, verse_id: str) -> List[SearchResult]:
        """Direct lookup by exact verse_id. Returns full verse with graph context."""
        if not NEO4J_AVAILABLE or not self._connected or not self.has_graph_data():
            return []
        
        result = self._run("""
            MATCH (c:Chunk {verse_id: $verse_id})
            OPTIONAL MATCH (c)-[:CONTAINS]->(w:Word)-[:HAS_LEMMA]->(l:Lemma)
            RETURN c.id AS id,
                   c.text AS text,
                   c.dataset_type AS dataset_type,
                   c.verse_id AS verse_id,
                   c.line_number AS line_number,
                   collect(DISTINCT l.text) AS lemmas,
                   collect(DISTINCT w.form) AS word_forms,
                   count(DISTINCT w) AS word_count
        """, {"verse_id": verse_id})
        
        search_results = []
        if result:
            for record in result:
                search_results.append(SearchResult(
                    id=record["id"],
                    text=record["text"] or "",
                    score=1.0,
                    dataset_type=record["dataset_type"] or "",
                    verse_id=record["verse_id"] or "",
                    metadata={
                        "line_number": record["line_number"],
                        "lemmas": record["lemmas"] or [],
                        "word_forms": record["word_forms"] or [],
                        "word_count": record["word_count"] or 0,
                        "retrieval_mode": "verse_exact"
                    }
                ))
        
        return search_results
    
    def search_by_verse_range(self, chapter: int, verse_start: int, verse_end: int = None) -> List[SearchResult]:
        """Retrieve all chunks within a verse range of a chapter."""
        if not NEO4J_AVAILABLE or not self._connected or not self.has_graph_data():
            return []
        
        if verse_end is None:
            verse_end = verse_start
        
        result = self._run("""
            MATCH (c:Chunk)
            WHERE c.verse_id STARTS WITH ('BhG ' + toString($chapter) + '.')
            AND toInteger(split(c.verse_id, '.')[1]) >= $verse_start
            AND toInteger(split(c.verse_id, '.')[1]) <= $verse_end
            OPTIONAL MATCH (c)-[:CONTAINS]->(w:Word)-[:HAS_LEMMA]->(l:Lemma)
            RETURN c.id AS id,
                   c.text AS text,
                   c.dataset_type AS dataset_type,
                   c.verse_id AS verse_id,
                   c.line_number AS line_number,
                   collect(DISTINCT l.text) AS lemmas,
                   count(DISTINCT w) AS word_count
            ORDER BY toInteger(split(c.verse_id, '.')[1])
        """, {"chapter": chapter, "verse_start": verse_start, "verse_end": verse_end})
        
        search_results = []
        if result:
            for record in result:
                search_results.append(SearchResult(
                    id=record["id"],
                    text=record["text"] or "",
                    score=0.9,
                    dataset_type=record["dataset_type"] or "",
                    verse_id=record["verse_id"] or "",
                    metadata={
                        "line_number": record["line_number"],
                        "lemmas": record["lemmas"] or [],
                        "word_count": record["word_count"] or 0,
                        "retrieval_mode": "verse_range"
                    }
                ))
        
        return search_results
    
    def search_by_chapter(self, chapter: int, top_k: int = 20) -> List[SearchResult]:
        """Retrieve all chunks from a specific chapter."""
        if not NEO4J_AVAILABLE or not self._connected or not self.has_graph_data():
            return []
        
        result = self._run("""
            MATCH (c:Chunk)
            WHERE c.verse_id STARTS WITH ('BhG ' + toString($chapter) + '.')
            OPTIONAL MATCH (c)-[:CONTAINS]->(w:Word)-[:HAS_LEMMA]->(l:Lemma)
            RETURN c.id AS id,
                   c.text AS text,
                   c.dataset_type AS dataset_type,
                   c.verse_id AS verse_id,
                   c.line_number AS line_number,
                   collect(DISTINCT l.text) AS lemmas,
                   count(DISTINCT w) AS word_count
            ORDER BY toInteger(split(c.verse_id, '.')[1])
            LIMIT $top_k
        """, {"chapter": chapter, "top_k": top_k})
        
        search_results = []
        if result:
            for record in result:
                search_results.append(SearchResult(
                    id=record["id"],
                    text=record["text"] or "",
                    score=0.8,
                    dataset_type=record["dataset_type"] or "",
                    verse_id=record["verse_id"] or "",
                    metadata={
                        "line_number": record["line_number"],
                        "lemmas": record["lemmas"] or [],
                        "word_count": record["word_count"] or 0,
                        "retrieval_mode": "chapter"
                    }
                ))
        
        return search_results
    
    # ============================================================
    # MULTI-HOP GRAPH TRAVERSAL
    # ============================================================
    
    def search_multi_hop(
        self,
        query_lemmas: List[str],
        top_k: int = 10,
        max_hops: int = 2
    ) -> List[SearchResult]:
        """Multi-hop graph traversal: Query terms → Lemma → Word → Chunk → Related Lemmas.
        
        Hop 1: Match query lemmas to Lemma nodes
        Hop 2: Traverse to Word nodes and Chunk nodes
        Hop 3 (optional): Find related lemmas via shared chunks
        """
        if not NEO4J_AVAILABLE or not self._connected or not query_lemmas or not self.has_graph_data():
            return []
        
        if max_hops == 1:
            query = """
                MATCH (l:Lemma)
                WHERE l.text IN $query_lemmas
                MATCH (l)<-[:HAS_LEMMA]-(w:Word)<-[:CONTAINS]-(c:Chunk)
                RETURN c.id AS id,
                       c.text AS text,
                       c.dataset_type AS dataset_type,
                       c.verse_id AS verse_id,
                       c.line_number AS line_number,
                       count(DISTINCT w) AS match_count,
                       collect(DISTINCT l.text) AS matched_lemmas
                ORDER BY match_count DESC
                LIMIT $top_k
            """
        else:
            query = """
                MATCH (l:Lemma)
                WHERE l.text IN $query_lemmas
                MATCH (l)<-[:HAS_LEMMA]-(w:Word)<-[:CONTAINS]-(c:Chunk)
                OPTIONAL MATCH (c)-[:CONTAINS]->(w2)-[:HAS_LEMMA]->(l2)
                WHERE NOT l2.text IN $query_lemmas
                RETURN c.id AS id,
                       c.text AS text,
                       c.dataset_type AS dataset_type,
                       c.verse_id AS verse_id,
                       c.line_number AS line_number,
                       count(DISTINCT w) AS direct_match_count,
                       count(DISTINCT l2) AS related_lemma_count,
                       collect(DISTINCT l.text) AS matched_lemmas,
                       collect(DISTINCT l2.text)[0..10] AS related_lemmas
                ORDER BY (direct_match_count * 2 + related_lemma_count) DESC
                LIMIT $top_k
            """
        
        result = self._run(query, {"query_lemmas": query_lemmas, "top_k": top_k})
        
        search_results = []
        if result:
            for record in result:
                direct_count = record.get("direct_match_count", record.get("match_count", 0))
                related_count = record.get("related_lemma_count", 0)
                score = min(1.0, (direct_count * 2 + related_count) / (len(query_lemmas) * 2))
                
                search_results.append(SearchResult(
                    id=record["id"],
                    text=record["text"] or "",
                    score=round(score, 4),
                    dataset_type=record["dataset_type"] or "",
                    verse_id=record["verse_id"] or "",
                    metadata={
                        "line_number": record["line_number"],
                        "matched_lemmas": record.get("matched_lemmas", []),
                        "related_lemmas": record.get("related_lemmas", []),
                        "direct_matches": direct_count,
                        "related_matches": related_count,
                        "retrieval_mode": "multi_hop",
                        "hops": max_hops
                    }
                ))
        
        return search_results
    
    def search_by_lemma(self, lemma: str, top_k: int = 10) -> List[SearchResult]:
        """Search chunks containing a specific lemma."""
        if not NEO4J_AVAILABLE or not self._connected or not self.has_graph_data():
            return []
        
        result = self._run("""
            MATCH (l:Lemma {text: $lemma})<-[:HAS_LEMMA]-(w:Word)<-[:CONTAINS]-(c:Chunk)
            RETURN DISTINCT c.id AS id,
                   c.text AS text,
                   c.dataset_type AS dataset_type,
                   c.verse_id AS verse_id,
                   c.line_number AS line_number,
                   count(w) AS lemma_count
            ORDER BY lemma_count DESC
            LIMIT $top_k
        """, {"lemma": lemma, "top_k": top_k})
        
        search_results = []
        if result:
            for record in result:
                search_results.append(SearchResult(
                    id=record["id"],
                    text=record["text"],
                    score=float(record["lemma_count"]),
                    dataset_type=record["dataset_type"] or "",
                    verse_id=record["verse_id"] or "",
                    metadata={"line_number": record["line_number"], "lemma_matches": record["lemma_count"]}
                ))
        
        return search_results
    
    def search_by_feature(
        self,
        feature_key: str,
        feature_value: str,
        top_k: int = 10
    ) -> List[SearchResult]:
        """Search chunks containing words with specific morphological feature."""
        if not NEO4J_AVAILABLE or not self._connected or not self.has_graph_data():
            return []
        
        result = self._run(f"""
            MATCH (w:Word {{{feature_key}: $feature_value}})<-[:CONTAINS]-(c:Chunk)
            RETURN DISTINCT c.id AS id,
                   c.text AS text,
                   c.dataset_type AS dataset_type,
                   c.verse_id AS verse_id,
                   c.line_number AS line_number
            LIMIT $top_k
        """, {"feature_value": feature_value, "top_k": top_k})
        
        search_results = []
        if result:
            for record in result:
                search_results.append(SearchResult(
                    id=record["id"],
                    text=record["text"],
                    score=1.0,
                    dataset_type=record["dataset_type"] or "",
                    verse_id=record["verse_id"] or "",
                    metadata={"line_number": record["line_number"]}
                ))
        
        return search_results
    
    # ============================================================
    # VECTOR SEARCH (fallback when graph traversal not applicable)
    # ============================================================
    
    def search_dense(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        **kwargs
    ) -> List[SearchResult]:
        """Search using dense vectors via Neo4j vector index."""
        if not NEO4J_AVAILABLE or not self._connected or not self.has_graph_data():
            return []
        
        query_list = query_vector.tolist() if hasattr(query_vector, 'tolist') else list(query_vector)
        
        result = self._run("""
            CALL db.index.vector.queryNodes('chunk_embeddings', $top_k, $query_vector)
            YIELD node, score
            RETURN node.id AS id,
                   node.text AS text,
                   score AS score,
                   node.dataset_type AS dataset_type,
                   node.verse_id AS verse_id,
                   node.line_number AS line_number,
                   node.original_id AS original_id
        """, {"query_vector": query_list, "top_k": top_k})
        
        search_results = []
        if result:
            for record in result:
                search_results.append(SearchResult(
                    id=record["id"],
                    text=record["text"],
                    score=record["score"],
                    dataset_type=record["dataset_type"] or "",
                    verse_id=record["verse_id"] or "",
                    metadata={
                        "line_number": record["line_number"],
                        "original_id": record["original_id"]
                    }
                ))
        
        return search_results
    
    def search_sparse(
        self,
        query_sparse: Dict[int, float],
        top_k: int = 10
    ) -> List[SearchResult]:
        """Not applicable for Neo4j - returns empty."""
        return []
    
    def search_colbert(
        self,
        query_colbert: np.ndarray,
        top_k: int = 10,
        nprobe: int = 16
    ) -> List[SearchResult]:
        """Not applicable for Neo4j - returns empty."""
        return []
    
    def bm25_search(
        self,
        query_terms: List[str],
        top_k: int = 10
    ) -> List[SearchResult]:
        """Search by matching word forms in the graph."""
        if not NEO4J_AVAILABLE or not self._connected or not self.has_graph_data():
            return []
        
        results = []
        for term in query_terms:
            term_results = self._run("""
                MATCH (w:Word)
                WHERE toLower(w.form) CONTAINS toLower($term)
                MATCH (w)<-[:CONTAINS]-(c:Chunk)
                RETURN DISTINCT c.id AS id,
                       c.text AS text,
                       c.dataset_type AS dataset_type,
                       c.verse_id AS verse_id,
                       c.line_number AS line_number
                LIMIT $top_k
            """, {"term": term, "top_k": top_k})
            
            if term_results:
                for record in term_results:
                    results.append(SearchResult(
                        id=record["id"],
                        text=record["text"],
                        score=1.0,
                        dataset_type=record["dataset_type"] or "",
                        verse_id=record["verse_id"] or "",
                        metadata={"line_number": record["line_number"]}
                    ))
        
        return results[:top_k]
    
    # ============================================================
    # METADATA & STATS
    # ============================================================
    
    def get_verse_metadata(self, verse_id: str) -> Dict[str, Any]:
        """Get full graph metadata for a specific verse."""
        if not NEO4J_AVAILABLE or not self._connected or not self.has_graph_data():
            return {}
        
        result = self._run("""
            MATCH (c:Chunk {verse_id: $verse_id})
            OPTIONAL MATCH (c)-[:CONTAINS]->(w:Word)-[:HAS_LEMMA]->(l:Lemma)
            RETURN c.id AS id,
                   c.text AS text,
                   c.verse_id AS verse_id,
                   collect(DISTINCT l.text) AS lemmas,
                   collect(DISTINCT w.form) AS word_forms,
                   collect(DISTINCT w.Case) AS cases,
                   collect(DISTINCT w.Gender) AS genders,
                   count(DISTINCT w) AS word_count,
                   count(DISTINCT l) AS lemma_count
        """, {"verse_id": verse_id})
        
        if result and result[0]:
            record = result[0]
            return {
                "id": record["id"],
                "text": record["text"] or "",
                "verse_id": record["verse_id"] or "",
                "lemmas": record["lemmas"] or [],
                "word_forms": record["word_forms"] or [],
                "cases": list(set(record["cases"] or [])),
                "genders": list(set(record["genders"] or [])),
                "word_count": record["word_count"] or 0,
                "lemma_count": record["lemma_count"] or 0
            }
        return {}
    
    def load_collection(self, collection_name: str = None):
        """No-op for Neo4j."""
        if self._connected:
            print("Neo4j graph database ready for search")
    
    def get_collection_stats(self, collection_name: str = None, refresh: bool = False) -> Dict[str, Any]:
        """Get graph statistics."""
        if not NEO4J_AVAILABLE or not self._connected:
            return {}

        if self._stats_cache is not None and not refresh:
            return dict(self._stats_cache)

        labels = self._existing_labels()
        stats = {
            "name": "neo4j_sanskrit",
            "chunk_count": 0,
            "word_count": 0,
            "lemma_count": 0,
        }

        if not labels:
            self._stats_cache = stats
            return dict(stats)

        with self._driver.session() as session:
            if "Chunk" in labels:
                stats["chunk_count"] = session.run(
                    "MATCH (c:Chunk) RETURN count(c) AS count"
                ).single()["count"] or 0
            if "Word" in labels:
                stats["word_count"] = session.run(
                    "MATCH (w:Word) RETURN count(w) AS count"
                ).single()["count"] or 0
            if "Lemma" in labels:
                stats["lemma_count"] = session.run(
                    "MATCH (l:Lemma) RETURN count(l) AS count"
                ).single()["count"] or 0

        self._stats_cache = stats
        return dict(stats)

    # ============================================================
    # GRAPHRAG: CONCEPT-BASED RETRIEVAL WITH SEMANTIC EXPANSION
    # ============================================================

    def search_graph_rag(
        self,
        query: str,
        query_lemmas: List[str],
        top_k: int = 10,
        max_hops: int = 3,
        include_related_concepts: bool = True
    ) -> List[SearchResult]:
        """GraphRAG retrieval: Find verses through concept expansion.

        Steps:
        1. Match query lemmas to Lemma nodes
        2. Find all verses containing these lemmas
        3. Expand to related concepts via shared edges
        4. Return ranked results with graph context
        """
        if not NEO4J_AVAILABLE or not self._connected or not query_lemmas or not self.has_graph_data():
            return []

        query = """
        MATCH (l:Lemma)
        WHERE l.text IN $query_lemmas
        MATCH (l)<-[:HAS_LEMMA]-(w:Word)<-[:CONTAINS]-(c:Chunk)

        WITH c, count(DISTINCT l) AS direct_match_count
        WHERE direct_match_count > 0

        OPTIONAL MATCH (c)-[:CONTAINS]->(w2:Word)-[:HAS_LEMMA]->(l2)
        WHERE NOT l2.text IN $query_lemmas

        WITH c, direct_match_count,
             count(DISTINCT l2) AS related_concept_count,
             collect(DISTINCT l2.text)[0..20] AS related_lemmas

        OPTIONAL MATCH (c)-[:REFERENCES]->(ref:Chunk)
        OPTIONAL MATCH (c)<-[:REFERENCES]-(refBy:Chunk)

        WITH c, direct_match_count, related_concept_count, related_lemmas,
             count(ref) AS outgoing_refs, count(refBy) AS incoming_refs

        RETURN c.id AS id,
               c.text AS text,
               c.dataset_type AS dataset_type,
               c.verse_id AS verse_id,
               c.line_number AS line_number,
               direct_match_count,
               related_concept_count,
               related_lemmas,
               outgoing_refs + incoming_refs AS total_references,
               (direct_match_count * 3 + related_concept_count * 0.5 + outgoing_refs + incoming_refs) AS graph_score
        ORDER BY graph_score DESC
        LIMIT $top_k
        """

        result = self._run(query, {
            "query_lemmas": query_lemmas,
            "top_k": top_k
        })

        search_results = []
        if result:
            for record in result:
                graph_score = record.get("graph_score", 0)
                max_possible = len(query_lemmas) * 3 + 20
                normalized_score = min(1.0, graph_score / max_possible) if max_possible > 0 else 0.5

                search_results.append(SearchResult(
                    id=record["id"],
                    text=record["text"] or "",
                    score=round(normalized_score, 4),
                    dataset_type=record["dataset_type"] or "",
                    verse_id=record["verse_id"] or "",
                    metadata={
                        "line_number": record["line_number"],
                        "direct_matches": record.get("direct_match_count", 0),
                        "related_concepts": record.get("related_lemmas", []),
                        "total_references": record.get("total_references", 0),
                        "retrieval_mode": "graph_rag",
                        "graph_score": graph_score
                    }
                ))

        return search_results

    def search_concept_expansion(
        self,
        concept: str,
        top_k: int = 10,
        expansion_depth: int = 2
    ) -> List[SearchResult]:
        """Expand from a concept to find related verses.

        Given a concept (lemma), find all verses containing it,
        then expand to verses containing related concepts.
        """
        if not NEO4J_AVAILABLE or not self._connected or not self.has_graph_data():
            return []

        if expansion_depth == 1:
            query = """
            MATCH (l:Lemma {text: $concept})
            MATCH (l)<-[:HAS_LEMMA]-(w:Word)<-[:CONTAINS]-(c:Chunk)
            RETURN DISTINCT c.id AS id, c.text AS text,
                   c.dataset_type AS dataset_type, c.verse_id AS verse_id,
                   c.line_number AS line_number, count(w) AS match_count
            ORDER BY match_count DESC
            LIMIT $top_k
            """
        else:
            query = """
            MATCH (l:Lemma {text: $concept})
            MATCH (l)<-[:HAS_LEMMA]-(w:Word)<-[:CONTAINS]-(c:Chunk)

            WITH c, count(w) AS direct_count
            WHERE direct_count > 0

            OPTIONAL MATCH (c)-[:CONTAINS]->(w2:Word)-[:HAS_LEMMA]->(l2)
            WHERE l2.text <> $concept

            WITH c, direct_count, l2, count(DISTINCT w2) AS l2_count
            ORDER BY l2_count DESC
            LIMIT $top_k

            RETURN c.id AS id, c.text AS text,
                   c.dataset_type AS dataset_type, c.verse_id AS verse_id,
                   c.line_number AS line_number,
                   direct_count,
                   collect(DISTINCT l2.text)[0..10] AS expanded_concepts
            ORDER BY direct_count DESC
            """

        result = self._run(query, {"concept": concept, "top_k": top_k})

        search_results = []
        if result:
            for record in result:
                search_results.append(SearchResult(
                    id=record["id"],
                    text=record["text"] or "",
                    score=float(record.get("direct_count", 1)) / 10.0,
                    dataset_type=record["dataset_type"] or "",
                    verse_id=record["verse_id"] or "",
                    metadata={
                        "line_number": record["line_number"],
                        "expanded_concepts": record.get("expanded_concepts", []),
                        "retrieval_mode": "concept_expansion"
                    }
                ))

        return search_results

    def search_cross_references(
        self,
        verse_id: str,
        top_k: int = 10
    ) -> List[SearchResult]:
        """Find verses that reference or are referenced by this verse.

        Uses the REFERENCES relationship to navigate between related verses.
        """
        if not NEO4J_AVAILABLE or not self._connected or not self.has_graph_data():
            return []

        query = """
        MATCH (c:Chunk {verse_id: $verse_id})
        OPTIONAL MATCH (c)-[:REFERENCES]->(ref:Chunk)
        OPTIONAL MATCH (c)<-[:REFERENCES]-(refBy:Chunk)

        WITH COALESCE(ref, refBy) AS related
        WHERE related IS NOT NULL

        OPTIONAL MATCH (related)-[:CONTAINS]->(w:Word)-[:HAS_LEMMA]->(l:Lemma)

        RETURN DISTINCT related.id AS id,
               related.text AS text,
               related.dataset_type AS dataset_type,
               related.verse_id AS verse_id,
               related.line_number AS line_number,
               collect(DISTINCT l.text)[0..20] AS shared_lemmas
        LIMIT $top_k
        """

        result = self._run(query, {"verse_id": verse_id, "top_k": top_k})

        search_results = []
        if result:
            for record in result:
                search_results.append(SearchResult(
                    id=record["id"],
                    text=record["text"] or "",
                    score=0.8,
                    dataset_type=record["dataset_type"] or "",
                    verse_id=record["verse_id"] or "",
                    metadata={
                        "line_number": record["line_number"],
                        "shared_lemmas": record.get("shared_lemmas", []),
                        "retrieval_mode": "cross_reference"
                    }
                ))

        return search_results

    def search_semantic_neighbors(
        self,
        query_vector: np.ndarray,
        verse_id: str,
        top_k: int = 10,
        neighbor_hop: int = 1
    ) -> List[SearchResult]:
        """Find semantically similar verses that are also graph neighbors.

        Combines vector similarity with graph proximity.
        """
        if not NEO4J_AVAILABLE or not self._connected or not self.has_graph_data():
            return []

        query_list = query_vector.tolist() if hasattr(query_vector, 'tolist') else list(query_vector)

        query = """
        MATCH (source:Chunk {verse_id: $verse_id})
        MATCH (source)-[:CONTAINS]->(w:Word)-[:HAS_LEMMA]->(l:Lemma)
        MATCH (l)<-[:HAS_LEMMA]-(w2:Word)<-[:CONTAINS]-(neighbor:Chunk)
        WHERE neighbor.verse_id <> $verse_id

        WITH DISTINCT neighbor
        CALL db.index.vector.queryNodes('chunk_embeddings', $top_k * 2, $query_vector)
        YIELD node, score
        WHERE node.id = neighbor.id

        OPTIONAL MATCH (neighbor)-[:CONTAINS]->(nw:Word)-[:HAS_LEMMA]->(nl:Lemma)

        RETURN neighbor.id AS id,
               neighbor.text AS text,
               neighbor.dataset_type AS dataset_type,
               neighbor.verse_id AS verse_id,
               neighbor.line_number AS line_number,
               score,
               collect(DISTINCT nl.text)[0..15] AS neighbor_lemmas
        ORDER BY score DESC
        LIMIT $top_k
        """

        result = self._run(query, {
            "verse_id": verse_id,
            "query_vector": query_list,
            "top_k": top_k
        })

        search_results = []
        if result:
            for record in result:
                search_results.append(SearchResult(
                    id=record["id"],
                    text=record["text"] or "",
                    score=record["score"],
                    dataset_type=record["dataset_type"] or "",
                    verse_id=record["verse_id"] or "",
                    metadata={
                        "line_number": record["line_number"],
                        "neighbor_lemmas": record.get("neighbor_lemmas", []),
                        "retrieval_mode": "semantic_neighbor"
                    }
                ))

        return search_results

    def get_concept_graph(
        self,
        concepts: List[str],
        max_depth: int = 2
    ) -> Dict[str, Any]:
        """Get the subgraph connecting given concepts.

        Returns nodes and edges for visualization.
        """
        if not NEO4J_AVAILABLE or not self._connected or not self.has_graph_data():
            return {"nodes": [], "edges": []}

        query = """
        MATCH (l:Lemma)
        WHERE l.text IN $concepts
        CALL apoc.path.subgraphAll(l, {
            maxDepth: $max_depth,
            relationshipFilter: 'HAS_LEMMA|CONTAINS',
            labelFilter: '+Lemma,+Word,+Chunk'
        })
        YIELD nodes, relationships

        WITH nodes, relationships
        UNWIND nodes AS n
        WITH collect(DISTINCT {
            id: elementId(n),
            labels: labels(n),
            text: COALESCE(n.text, n.form, n.verse_id)
        }) AS node_list, relationships

        UNWIND relationships AS r
        WITH node_list, collect({
            source: elementId(startNode(r)),
            target: elementId(endNode(r)),
            type: type(r)
        }) AS edge_list

        RETURN node_list AS nodes, edge_list AS edges
        """

        result = self._run(query, {
            "concepts": concepts,
            "max_depth": max_depth
        })

        if result and result[0]:
            return {
                "nodes": result[0].get("nodes", []),
                "edges": result[0].get("edges", [])
            }

        return {"nodes": [], "edges": []}


def create_all_collections(manager: Neo4jManager) -> Dict[str, str]:
    """Create Neo4j schema."""
    if manager.create_schema():
        return {"seg_lemma": "neo4j_sanskrit"}
    return {}


if __name__ == "__main__":
    manager = Neo4jManager()
    
    if manager.connect():
        manager.create_schema()
        stats = manager.get_collection_stats()
        print(f"Stats: {stats}")
        manager.disconnect()
