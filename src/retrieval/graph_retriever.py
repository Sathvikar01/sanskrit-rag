"""Graph retriever using Neo4j for knowledge graph traversal."""


import re

from neo4j import GraphDatabase

from src.utils.config import Config
from src.utils.logger import logger


def normalize_verse_ref(ref: str) -> str:
    """Normalize a verse reference to 'BhG X.Y' format.

    Handles: 'BG 2.47', 'BhG 2.47', 'bhagavad gita 2.47', '2.47'.
    """
    ref = ref.strip()
    m = re.search(r'(\d+\.\d+)', ref)
    if not m:
        return ref
    return f"BhG {m.group(1)}"


class GraphRetriever:
    """Retrieve relevant passages using Neo4j graph queries."""

    def __init__(self, config: Config = None):
        if config is None:
            config = Config()

        self.uri = config.neo4j_uri
        self.user = config.neo4j_user
        self.password = config.neo4j_password
        self.driver = None

    def connect(self):
        """Establish connection to Neo4j."""
        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        self.driver.verify_connectivity()
        logger.info("GraphRetriever connected to Neo4j")

    def close(self):
        """Close the Neo4j driver."""
        if self.driver:
            self.driver.close()

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def search_fulltext(self, query: str, top_k: int = 50) -> list[dict]:
        """Search verses using full-text index.

        Args:
            query: IAST text query.
            top_k: Maximum results to return.

        Returns:
            List of dicts with chunk_id, score, rank.
        """
        cypher_query = """
        CALL db.index.fulltext.queryNodes('verse_text_ft', $search_text)
        YIELD node, score
        RETURN node.ref AS ref, node.text_iast AS text, score
        ORDER BY score DESC
        LIMIT $top_k
        """

        with self.driver.session() as session:
            result = session.run(cypher_query, search_text=query, top_k=top_k)
            results = []
            for rank, record in enumerate(result, 1):
                results.append(
                    {
                        "chunk_id": f"{record['ref'].replace(' ', '_')}_verse",
                        "score": float(record["score"]),
                        "rank": rank,
                        "ref": record["ref"],
                        "verse_ref": record["ref"],
                        "text": record["text"],
                    }
                )

        logger.info(f"Full-text search returned {len(results)} results")
        return results

    def search_by_concepts(
        self,
        concepts: list[str],
        top_k: int = 50,
    ) -> list[dict]:
        """Search verses by concept names.

        Args:
            concepts: List of concept name_iast values.
            top_k: Maximum results to return.

        Returns:
            List of dicts with chunk_id, score, rank.
        """
        cypher_query = """
        MATCH (v:Verse)-[r:MENTIONS_CONCEPT]->(c:Concept)
        WHERE c.name_iast IN $concepts
        WITH v, count(c) AS concept_overlap,
             sum(r.confidence) AS total_confidence
        OPTIONAL MATCH (v)--(other)
        WITH v, concept_overlap, total_confidence, count(other) AS degree_centrality
        RETURN v.ref AS ref, v.text_iast AS text,
               concept_overlap, total_confidence, degree_centrality,
               (concept_overlap * 0.4 + total_confidence * 0.4 + degree_centrality * 0.002) AS score
        ORDER BY score DESC
        LIMIT $top_k
        """

        with self.driver.session() as session:
            result = session.run(
                cypher_query,
                concepts=concepts,
                top_k=top_k,
            )
            results = []
            for rank, record in enumerate(result, 1):
                results.append(
                    {
                        "chunk_id": f"{record['ref'].replace(' ', '_')}_verse",
                        "score": float(record["score"]),
                        "rank": rank,
                        "ref": record["ref"],
                        "verse_ref": record["ref"],
                        "text": record["text"],
                        "concept_overlap": record["concept_overlap"],
                    }
                )

        logger.info(f"Concept search returned {len(results)} results")
        return results

    def search_commentary_consensus(
        self,
        concepts: list[str],
        top_k: int = 50,
    ) -> list[dict]:
        """Find verses with multi-commentary support for concepts.

        Args:
            concepts: List of concept name_iast values.
            top_k: Maximum results to return.

        Returns:
            List of dicts with chunk_id, score, rank.
        """
        cypher_query = """
        MATCH (v:Verse)-[:HAS_COMMENTARY]->(com:Commentary)-[:DISCUSSES_CONCEPT]->(c:Concept)
        WHERE c.name_iast IN $concepts
        WITH v, collect(DISTINCT com.commentator) AS commentators, c
        WHERE size(commentators) >= 2
        RETURN v.ref AS ref, v.text_iast AS text,
               size(commentators) AS commentary_count
        ORDER BY commentary_count DESC
        LIMIT $top_k
        """

        with self.driver.session() as session:
            result = session.run(cypher_query, concepts=concepts, top_k=top_k)
            results = []
            for rank, record in enumerate(result, 1):
                score = record["commentary_count"] / 3.0
                results.append(
                    {
                        "chunk_id": f"{record['ref'].replace(' ', '_')}_verse",
                        "score": score,
                        "rank": rank,
                        "ref": record["ref"],
                        "verse_ref": record["ref"],
                        "text": record["text"],
                        "commentary_count": record["commentary_count"],
                    }
                )

        logger.info(f"Commentary consensus search returned {len(results)} results")
        return results

    def search_concept_neighborhood(
        self,
        concepts: list[str],
        hop_distance: int = 2,
        top_k: int = 50,
    ) -> list[dict]:
        """Search verses through concept neighborhood traversal.

        Args:
            concepts: Starting concept names.
            hop_distance: Number of hops in concept graph (1 or 2).
            top_k: Maximum results to return.

        Returns:
            List of dicts with chunk_id, score, rank.
        """
        cypher_query = f"""
        MATCH (c:Concept)-[:RELATED_TO*1..{hop_distance}]-(related:Concept)
        WHERE c.name_iast IN $concepts
        MATCH (v:Verse)-[:MENTIONS_CONCEPT]->(related)
        RETURN DISTINCT v.ref AS ref, v.text_iast AS text,
               related.name_iast AS matched_concept,
               c.name_iast AS query_concept
        LIMIT $top_k
        """

        with self.driver.session() as session:
            result = session.run(cypher_query, concepts=concepts, top_k=top_k)
            results = []
            seen_refs = set()
            for rank, record in enumerate(result, 1):
                ref = record["ref"]
                if ref in seen_refs:
                    continue
                seen_refs.add(ref)
                results.append(
                    {
                        "chunk_id": f"{ref.replace(' ', '_')}_verse",
                        "score": 1.0 / rank,
                        "rank": len(results) + 1,
                        "ref": ref,
                        "verse_ref": ref,
                        "text": record["text"],
                        "matched_concept": record["matched_concept"],
                    }
                )

        logger.info(f"Concept neighborhood search returned {len(results)} results")
        return results

    def search_by_verse_ref(self, verse_ref: str, top_k: int = 10) -> list[dict]:
        """Search verses by exact verse reference (e.g. 'BhG 2.47').

        Uses exact match (not CONTAINS) and normalizes the input ref
        to handle format variations (BG → BhG, extra whitespace, etc.).

        Args:
            verse_ref: Verse reference string to search for.
            top_k: Maximum results to return.

        Returns:
            List of dicts with chunk_id, score, rank, verse_ref, graph_score, chunk_type.
        """
        normalized = normalize_verse_ref(verse_ref)
        cypher_query = """
        MATCH (v:Verse)
        WHERE v.ref = $verse_ref
        RETURN v.ref AS ref, v.text_iast AS text, v.text_devanagari AS text_devanagari
        LIMIT $top_k
        """

        with self.driver.session() as session:
            result = session.run(cypher_query, verse_ref=normalized, top_k=top_k)
            results = []
            for rank, record in enumerate(result, 1):
                ref = record["ref"]
                results.append(
                    {
                        "chunk_id": f"{ref.replace(' ', '_')}_verse",
                        "score": 100.0,
                        "graph_score": 100.0,
                        "rank": rank,
                        "verse_ref": ref,
                        "text_iast": record["text"],
                        "text_devanagari": record["text_devanagari"],
                        "sources": ["graph"],
                        "chunk_type": "verse",
                    }
                )

        logger.info(f"Verse ref search for '{verse_ref}' (normalized: '{normalized}') returned {len(results)} results")
        return results

    def search_combined(
        self,
        query_text: str,
        concepts: list[str],
        top_k: int = 50,
    ) -> list[dict]:
        """Combined graph search using multiple strategies.

        Args:
            query_text: IAST query text for full-text search.
            concepts: Extracted concept names.
            top_k: Maximum results per strategy.

        Returns:
            Merged and deduplicated results.
        """
        all_results = {}

        ft_results = self.search_fulltext(query_text, top_k)
        for r in ft_results:
            cid = r["chunk_id"]
            if cid not in all_results or r["score"] > all_results[cid]["score"]:
                all_results[cid] = r

        if concepts:
            concept_results = self.search_by_concepts(concepts, top_k)
            for r in concept_results:
                cid = r["chunk_id"]
                if cid not in all_results or r["score"] > all_results[cid]["score"]:
                    all_results[cid] = r

            neighborhood_results = self.search_concept_neighborhood(concepts, 2, top_k)
            for r in neighborhood_results:
                cid = r["chunk_id"]
                if cid not in all_results or r["score"] > all_results[cid]["score"]:
                    all_results[cid] = r

        results = sorted(all_results.values(), key=lambda x: x["score"], reverse=True)
        for i, r in enumerate(results):
            r["rank"] = i + 1

        return results[:top_k]
