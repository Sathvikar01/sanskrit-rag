"""Answer Generator Pipeline with citation-backed responses.

Workflow:
1. User Query Input
2. Language Detection (skip IAST for English queries)
3. Optional IAST Translation (for Sanskrit queries)
4. Verse Reference Parsing
5. Dual Retrieval Paths (Neo4j Graph + Qdrant Vector)
6. HybridRAG Fusion Ranking (RRF)
7. LLM Re-ranking
8. LLM Answer Generation (with retrieved text + metadata)
9. Citation-Backed Markdown Answer Output
10. Semantic Consistency Check (optional retry)
"""
import re
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import RRF_TOP_K, L1_REG_LAMBDA, L2_REG_LAMBDA, SEMANTIC_CONSISTENCY_THRESHOLD, MAX_CONSISTENCY_RETRIES
from src.commentary_manager import CommentaryManager
from src.gemini_client import NVIDIA_LLM_Client
from src.retriever import HybridRetriever, HybridSearchResult, parse_verse_references, VerseFilter
from src.qdrant_manager import QdrantManager
from src.neo4j_manager import Neo4jManager
from src.verse_db import VerseDatabase
from src.answer_templates import get_answer_template
from src.entity_lexicon import expand_query_with_aliases
from src.evidence_cache import EvidenceCache
from src.evidence_reranker import EvidenceReranker, RerankContext
from src.query_intent import classify_query_intent
from src.text_quality import clean_text, score_text_quality


ENGLISH_STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "shall", "can", "need", "dare",
    "ought", "used", "to", "of", "in", "for", "on", "with", "at", "by",
    "from", "as", "into", "through", "during", "before", "after", "above",
    "below", "between", "under", "again", "further", "then", "once", "here",
    "there", "when", "where", "why", "how", "all", "each", "few", "more",
    "most", "other", "some", "such", "no", "nor", "not", "only", "own",
    "same", "so", "than", "too", "very", "just", "because", "but", "and",
    "or", "if", "while", "about", "what", "which", "who", "whom", "this",
    "that", "these", "those", "i", "me", "my", "myself", "we", "our",
    "ours", "ourselves", "you", "your", "yours", "yourself", "he", "him",
    "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself",
    "they", "them", "their", "theirs", "themselves", "am", "also",
}


def is_english_query(query: str) -> bool:
    """Detect if a query is primarily English."""
    words = re.findall(r'\b[a-zA-Z]+\b', query.lower())
    if not words:
        return False
    english_count = sum(1 for w in words if w in ENGLISH_STOP_WORDS or len(w) <= 4)
    return english_count / len(words) > 0.5


@dataclass
class AnswerResult:
    """Container for a complete answer with citations."""
    answer: str
    query: str
    iast_query: str
    normalized_query: str
    citations: List[Dict[str, Any]] = field(default_factory=list)
    sources: Dict[str, Any] = field(default_factory=dict)
    retrieval_stats: Dict[str, Any] = field(default_factory=dict)
    commentary_matches: List[Dict[str, Any]] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0
    consistency_score: float = 0.0
    retrieval_passes: int = 1
    verse_filter: Optional[Dict[str, Any]] = None
    query_intent: Optional[Dict[str, Any]] = None
    confidence: float = 0.0
    evidence_quality: Dict[str, Any] = field(default_factory=dict)
    abstention_reason: str = ""
    explicit_references: List[str] = field(default_factory=list)
    cache: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "query": self.query,
            "iast_query": self.iast_query,
            "normalized_query": self.normalized_query,
            "citations": self.citations,
            "sources": self.sources,
            "retrieval_stats": self.retrieval_stats,
            "commentary_matches": self.commentary_matches,
            "evidence": self.evidence,
            "latency_ms": self.latency_ms,
            "consistency_score": self.consistency_score,
            "retrieval_passes": self.retrieval_passes,
            "verse_filter": self.verse_filter,
            "query_intent": self.query_intent,
            "confidence": self.confidence,
            "evidence_quality": self.evidence_quality,
            "abstention_reason": self.abstention_reason,
            "explicit_references": self.explicit_references,
            "cache": self.cache,
        }


class AnswerGenerator:
    """Orchestrates the full HybridRAG pipeline with citation-backed answers."""

    def __init__(
        self,
        gemini_client: NVIDIA_LLM_Client,
        retriever: HybridRetriever,
        qdrant_manager: QdrantManager = None,
        neo4j_manager: Neo4jManager = None,
        verse_db: VerseDatabase = None,
        top_k: int = RRF_TOP_K
    ):
        self.gemini = gemini_client
        self.retriever = retriever
        self.qdrant = qdrant_manager
        self.neo4j = neo4j_manager
        self.verse_db = verse_db
        self.top_k = top_k
        self.commentary_manager = CommentaryManager(
            qdrant_manager=qdrant_manager,
            embedding_client=getattr(retriever, "embedding_client", None),
        )
        self.evidence_reranker = EvidenceReranker()
        self.cache = EvidenceCache()

    def _retrieve_commentary_matches(
        self,
        query: str,
        verse_filter: VerseFilter,
        retrieved_verses: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Retrieve commentary from canonical SQLite first, then Qdrant if available."""
        if verse_filter.has_filter() and verse_filter.verse_ids:
            candidate_verse_ids = verse_filter.verse_ids
        else:
            candidate_verse_ids = []
            for verse in retrieved_verses:
                verse_id = verse.get("verse_id")
                if verse_id and verse_id not in candidate_verse_ids:
                    candidate_verse_ids.append(verse_id)
                if len(candidate_verse_ids) >= 5:
                    break

        if not candidate_verse_ids:
            return []

        matches = self._sqlite_commentary_matches(candidate_verse_ids)

        if self.commentary_manager and self.qdrant:
            try:
                qdrant_matches = self.commentary_manager.get_best_matches(query, candidate_verse_ids)
                matches.extend(match.to_dict() for match in qdrant_matches)
            except Exception as exc:
                print(f"Commentary retrieval warning: {exc}")

        best_by_key: Dict[tuple, Dict[str, Any]] = {}
        for match in matches:
            verse_id = match.get("verse_id", "")
            author = match.get("author_display_name") or match.get("author_key") or match.get("commentary_author", "")
            key = (verse_id, author)
            existing = best_by_key.get(key)
            if existing is None or float(match.get("score", 0.0) or 0.0) > float(existing.get("score", 0.0) or 0.0):
                best_by_key[key] = match

        ordered = []
        for verse_id in candidate_verse_ids:
            verse_matches = [match for key, match in best_by_key.items() if key[0] == verse_id]
            verse_matches.sort(key=lambda item: float(item.get("score", 0.0) or 0.0), reverse=True)
            ordered.extend(verse_matches[:2])

        return ordered[: max(8, len(candidate_verse_ids) * 2)]

    def _sqlite_commentary_matches(self, verse_ids: List[str]) -> List[Dict[str, Any]]:
        """Fetch commentary saved with canonical SQLite verses."""
        if not self.verse_db or not verse_ids:
            return []

        matches = []
        for verse in self.verse_db.get_verses_by_ids(verse_ids):
            for index, commentary in enumerate(verse.get("commentaries", []) or []):
                text = clean_text(commentary.get("text", ""))
                if not text:
                    continue
                quality = score_text_quality(text)
                commentator = commentary.get("commentator", "Commentary")
                matches.append({
                    "verse_id": verse.get("verse_id", ""),
                    "commentary_id": f"sqlite-{verse.get('verse_id', '').replace(' ', '_')}-{index}",
                    "author_key": str(commentator).lower().replace(" ", "_"),
                    "author_display_name": commentator,
                    "commentary_author": commentator,
                    "commentary_source": "SQLite",
                    "commentary_score": 1.0,
                    "text": text,
                    "score": 1.0,
                    "quality_score": quality["quality_score"],
                    "filtered_reason": quality["filtered_reason"],
                    "metadata": {
                        "source": "SQLite",
                        "source_type": "canonical_commentary",
                        "quality_score": quality["quality_score"],
                        "filtered_reason": quality["filtered_reason"],
                    },
                })
        return matches

    def _sqlite_commentary_verse_ids(self, verse_ids: List[str]) -> set:
        """Return verse IDs that have SQLite commentary available."""
        if not self.verse_db or not verse_ids:
            return set()
        available = set()
        for verse in self.verse_db.get_verses_by_ids(verse_ids):
            if verse.get("commentaries"):
                available.add(verse.get("verse_id"))
        return available

    def _direct_commentary_verse_ids(
        self,
        query: str,
        query_intent: Dict[str, Any],
        top_k: int = 5,
    ) -> List[str]:
        """Let commentary search nominate verse IDs for commentary-heavy queries."""
        profile = query_intent.get("retrieval_profile", {}) if query_intent else {}
        if float(profile.get("commentary_priority", 0.0) or 0.0) < 0.9:
            return []
        if not (self.commentary_manager and self.qdrant):
            return []

        try:
            results_by_author = self.commentary_manager.search_commentary(query=query, top_k=top_k)
        except Exception as exc:
            print(f"Direct commentary search warning: {exc}")
            return []

        ranked = []
        for results in results_by_author.values():
            for result in results:
                if result.verse_id:
                    ranked.append((float(result.score or 0.0), result.verse_id))

        ranked.sort(reverse=True)
        return list(dict.fromkeys(verse_id for _, verse_id in ranked))[:top_k]

    def _build_db_status(self, rrf_results: List[HybridSearchResult]) -> Dict[str, Any]:
        """Summarize which retrieval stores were available and contributed evidence."""
        qdrant_attempted = self.qdrant is not None
        neo4j_attempted = self.neo4j is not None
        qdrant_available = bool(qdrant_attempted and getattr(self.retriever, "_qdrant_available", True))
        neo4j_available = bool(neo4j_attempted and getattr(self.retriever, "_neo4j_available", True))

        qdrant_count = 0
        neo4j_count = 0
        for result in rrf_results:
            sources = result.metadata.get("sources", {}) if result.metadata else {}
            if sources.get("qdrant"):
                qdrant_count += 1
            if sources.get("neo4j"):
                neo4j_count += 1

        return {
            "qdrant": {
                "attempted": qdrant_attempted,
                "available": qdrant_available,
                "contributed": qdrant_count > 0,
                "candidate_count": qdrant_count,
            },
            "neo4j": {
                "attempted": neo4j_attempted,
                "available": neo4j_available,
                "contributed": neo4j_count > 0,
                "candidate_count": neo4j_count,
            },
        }

    def _supporting_chunks(
        self,
        rrf_results: List[HybridSearchResult],
        max_chunks: int = 10,
    ) -> List[Dict[str, Any]]:
        """Convert reranked chunks into serializable support evidence."""
        chunks = []
        for rank, result in enumerate(rrf_results[:max_chunks], 1):
            quality = score_text_quality(result.text)
            chunks.append({
                "rank": rank,
                "id": result.id,
                "verse_id": result.verse_id,
                "text": quality["text"],
                "score": result.final_score,
                "dataset_type": result.dataset_type,
                "sources": result.metadata.get("sources", {}) if result.metadata else {},
                "quality_score": quality["quality_score"],
                "filtered_reason": quality["filtered_reason"],
                "metadata": {
                    **(result.metadata or {}),
                    "quality_score": quality["quality_score"],
                    "filtered_reason": quality["filtered_reason"],
                },
            })
        return chunks

    def _resolve_canonical_verses(
        self,
        verse_ids: List[str],
        rrf_results: List[HybridSearchResult],
    ) -> List[Dict[str, Any]]:
        """Resolve reranked verse IDs to original SQLite verses, with DB chunks as fallback."""
        if not verse_ids:
            return []

        score_by_verse: Dict[str, float] = {}
        support_by_verse: Dict[str, List[Dict[str, Any]]] = {}
        fallback_by_verse: Dict[str, Dict[str, Any]] = {}

        for result in rrf_results:
            if not result.verse_id:
                continue

            score_by_verse[result.verse_id] = max(
                score_by_verse.get(result.verse_id, 0.0),
                float(result.final_score),
            )
            support_by_verse.setdefault(result.verse_id, []).append({
                "id": result.id,
                "text": result.text,
                "score": result.final_score,
                "dataset_type": result.dataset_type,
                "sources": result.metadata.get("sources", {}) if result.metadata else {},
            })

            existing = fallback_by_verse.get(result.verse_id)
            if existing is None or result.final_score > existing.get("score", 0.0):
                fallback_by_verse[result.verse_id] = {
                    "verse_id": result.verse_id,
                    "text": result.text,
                    "source": "RRF fallback",
                    "score": float(result.final_score),
                    "metadata": {
                        **(result.metadata or {}),
                        "canonical_source": "retrieval_chunk",
                    },
                }

        sqlite_by_id: Dict[str, Dict[str, Any]] = {}
        if self.verse_db:
            try:
                for verse in self.verse_db.get_verses_by_ids(verse_ids):
                    sqlite_by_id[verse.get("verse_id", "")] = verse
            except Exception as exc:
                print(f"Error fetching canonical verses from SQLite: {exc}")

        canonical_verses = []
        missing_ids = []

        for verse_id in verse_ids[:self.top_k]:
            support = support_by_verse.get(verse_id, [])
            score = score_by_verse.get(verse_id, 0.0)
            sqlite_verse = sqlite_by_id.get(verse_id)

            if sqlite_verse:
                lines = sqlite_verse.get("lines") or []
                text = "\n".join(lines).strip() or sqlite_verse.get("sanskrit_text", "")
                quality = score_text_quality(text)
                canonical_verses.append({
                    "verse_id": verse_id,
                    "text": quality["text"],
                    "source": "SQLite",
                    "score": score,
                    "quality_score": quality["quality_score"],
                    "filtered_reason": quality["filtered_reason"],
                    "is_canonical": True,
                    "metadata": {
                        "chapter": sqlite_verse.get("chapter"),
                        "verse_num": sqlite_verse.get("verse_num"),
                        "speaker": sqlite_verse.get("speaker", ""),
                        "word_count": sqlite_verse.get("word_count", 0),
                        "canonical_source": "sqlite",
                        "is_canonical": True,
                        "quality_score": quality["quality_score"],
                        "filtered_reason": quality["filtered_reason"],
                        "supporting_chunks": support[:3],
                    },
                })
                continue

            fallback = fallback_by_verse.get(verse_id)
            if fallback:
                quality = score_text_quality(fallback.get("text", ""))
                fallback["text"] = quality["text"]
                fallback["quality_score"] = quality["quality_score"]
                fallback["filtered_reason"] = quality["filtered_reason"]
                fallback["is_canonical"] = False
                fallback["metadata"] = {
                    **fallback.get("metadata", {}),
                    "is_canonical": False,
                    "quality_score": quality["quality_score"],
                    "filtered_reason": quality["filtered_reason"],
                    "supporting_chunks": support[:3],
                }
                canonical_verses.append(fallback)
            else:
                missing_ids.append(verse_id)

        for verse in self._fetch_verse_text_from_qdrant(missing_ids):
            if verse["verse_id"] not in {v["verse_id"] for v in canonical_verses}:
                quality = score_text_quality(verse.get("text", ""))
                verse["text"] = quality["text"]
                verse["quality_score"] = quality["quality_score"]
                verse["filtered_reason"] = quality["filtered_reason"]
                verse["is_canonical"] = False
                verse["metadata"] = {
                    **verse.get("metadata", {}),
                    "canonical_source": "qdrant_fallback",
                    "is_canonical": False,
                    "quality_score": quality["quality_score"],
                    "filtered_reason": quality["filtered_reason"],
                }
                canonical_verses.append(verse)

        still_missing = [
            verse_id for verse_id in missing_ids
            if verse_id not in {v["verse_id"] for v in canonical_verses}
        ]
        for verse in self._fetch_verse_text_from_neo4j(still_missing):
            if verse["verse_id"] not in {v["verse_id"] for v in canonical_verses}:
                quality = score_text_quality(verse.get("text", ""))
                verse["text"] = quality["text"]
                verse["quality_score"] = quality["quality_score"]
                verse["filtered_reason"] = quality["filtered_reason"]
                verse["is_canonical"] = False
                verse["metadata"] = {
                    **verse.get("metadata", {}),
                    "canonical_source": "neo4j_fallback",
                    "is_canonical": False,
                    "quality_score": quality["quality_score"],
                    "filtered_reason": quality["filtered_reason"],
                }
                canonical_verses.append(verse)

        canonical_verses.sort(
            key=lambda verse: verse_ids.index(verse["verse_id"])
            if verse.get("verse_id") in verse_ids
            else len(verse_ids)
        )
        return canonical_verses[:self.top_k]

    def _fetch_verse_text_from_qdrant(self, verse_ids: List[str]) -> List[Dict[str, Any]]:
        """Fetch full verse text from Qdrant using verse IDs."""
        verses = []
        if not self.qdrant or not verse_ids:
            return verses

        try:
            from qdrant_client.models import Filter, FieldCondition, MatchAny
            seen = set()
            for vid in verse_ids[:self.top_k]:
                if vid in seen:
                    continue
                seen.add(vid)
                results = self.qdrant.client.query_points(
                    collection_name="sansr_seg_lemma",
                    query_filter=Filter(
                        must=[FieldCondition(key="verse_id", match=MatchAny(any=[vid]))]
                    ),
                    limit=3,
                    with_payload=True
                ).points

                for point in results:
                    payload = point.payload or {}
                    verses.append({
                        "verse_id": vid,
                        "text": payload.get("text", ""),
                        "source": "Qdrant",
                        "score": point.score if hasattr(point, 'score') else 0.0,
                        "metadata": payload.get("metadata", {})
                    })
        except Exception as e:
            print(f"Error fetching verse text from Qdrant: {e}")

        return verses

    def _fetch_verse_text_from_neo4j(self, verse_ids: List[str]) -> List[Dict[str, Any]]:
        """Fetch full verse text from Neo4j using verse IDs."""
        verses = []
        if not self.neo4j or not verse_ids:
            return verses

        try:
            for vid in verse_ids[:self.top_k]:
                records = self.neo4j._run("""
                    MATCH (c:Chunk {verse_id: $verse_id})
                    OPTIONAL MATCH (c)-[:CONTAINS]->(w:Word)-[:HAS_LEMMA]->(l:Lemma)
                    RETURN c.id AS id, c.text AS text, c.dataset_type AS dataset_type,
                           collect(DISTINCT l.text) AS lemmas
                """, {"verse_id": vid})

                for record in records:
                    verses.append({
                        "verse_id": vid,
                        "text": record["text"] or "",
                        "source": "Neo4j",
                        "score": 1.0,
                        "metadata": {
                            "id": record["id"],
                            "dataset_type": record["dataset_type"],
                            "lemmas": record.get("lemmas", []),
                        }
                    })
        except Exception as e:
            print(f"Error fetching verse text from Neo4j: {e}")

        return verses

    def _fetch_graph_metadata(self, verse_ids: List[str]) -> List[Dict[str, Any]]:
        """Fetch structured metadata from Neo4j graph."""
        metadata = []
        if not self.neo4j or not verse_ids:
            return metadata

        try:
            for vid in verse_ids[:self.top_k]:
                records = self.neo4j._run("""
                    MATCH (c:Chunk {verse_id: $verse_id})-[:CONTAINS]->(w:Word)-[:HAS_LEMMA]->(l:Lemma)
                    RETURN c.verse_id AS verse_id,
                           collect(DISTINCT l.text) AS lemmas,
                           collect(DISTINCT w.form) AS word_forms,
                           count(DISTINCT w) AS word_count
                """, {"verse_id": vid})

                for record in records:
                    metadata.append({
                        "verse_id": record["verse_id"] or vid,
                        "lemmas": record["lemmas"] or [],
                        "word_forms": record["word_forms"] or [],
                        "word_count": record["word_count"] or 0
                    })
        except Exception as e:
            print(f"Error fetching graph metadata: {e}")

        return metadata

    def _format_answer_markdown(self, answer_text: str, citations: List[Dict]) -> str:
        """Format the answer as proper Markdown with citations."""
        if not answer_text:
            return "*No answer generated.*"

        lines = answer_text.strip().split("\n")
        formatted = []
        in_list = False

        for line in lines:
            stripped = line.strip()

            if stripped.startswith("[Citation") and "]" in stripped:
                match = re.match(r'\[Citation (\d+)\]\s*\(ID:\s*(.*?),\s*Source:\s*(.*?),\s*Score:\s*([\d.]+)\)', stripped)
                if match:
                    num, vid, source, score = match.groups()
                    formatted.append(f"\n> **Citation {num}** — `{vid}` (Source: {source}, Score: {score})\n")
                    continue

            if stripped.startswith("[Meta"):
                formatted.append(f"\n*{stripped}*\n")
                continue

            if re.match(r'^\d+\.\s', stripped) or stripped.startswith('- '):
                if not in_list:
                    formatted.append("")
                    in_list = True
                formatted.append(stripped)
            else:
                if in_list and stripped:
                    formatted.append("")
                    in_list = False
                if stripped:
                    formatted.append(stripped)
                else:
                    formatted.append("")

        result = "\n".join(formatted).strip()

        if citations:
            result += "\n\n---\n\n### References\n"
            seen = set()
            for i, c in enumerate(citations[:5], 1):
                vid = c.get("verse_id", "Unknown")
                if vid in seen:
                    continue
                seen.add(vid)
                text = c.get("text", "")[:150]
                result += f"\n**[{i}] `{vid}`**\n{text}\n"

        return result

    def _score_evidence_quality(
        self,
        canonical_verses: List[Dict[str, Any]],
        supporting_chunks: List[Dict[str, Any]],
        commentary_matches: List[Dict[str, Any]],
        db_status: Dict[str, Any],
        explicit_references: List[str],
        rrf_results: List[HybridSearchResult],
    ) -> Dict[str, Any]:
        """Compute answer confidence and abstention metadata from evidence strength."""
        verse_quality_scores = [
            float(verse.get("quality_score", score_text_quality(verse.get("text", ""))["quality_score"]))
            for verse in canonical_verses
        ]
        support_quality_scores = [
            float(chunk.get("quality_score", score_text_quality(chunk.get("text", ""))["quality_score"]))
            for chunk in supporting_chunks
        ]
        all_quality = verse_quality_scores or support_quality_scores or [0.0]
        avg_quality = sum(all_quality) / len(all_quality)

        explicit_hits = [
            verse_id for verse_id in explicit_references
            if verse_id in {verse.get("verse_id") for verse in canonical_verses}
        ]
        contributed_dbs = sum(
            1 for status in db_status.values()
            if isinstance(status, dict) and status.get("contributed")
        )

        confidence = 0.0
        if canonical_verses:
            confidence += 0.25
        elif supporting_chunks:
            confidence += 0.1
        if explicit_references:
            confidence += 0.2 * (len(explicit_hits) / max(len(explicit_references), 1))
        if commentary_matches:
            confidence += 0.15
        confidence += min(0.15, contributed_dbs * 0.075)
        confidence += 0.2 * avg_quality
        if rrf_results:
            top_score = max(float(result.final_score or 0.0) for result in rrf_results[:5])
            confidence += min(0.05, top_score * 0.05)

        confidence = round(max(0.0, min(1.0, confidence)), 4)
        abstention_reason = ""
        if not canonical_verses and not supporting_chunks:
            abstention_reason = "no_usable_evidence"
        elif confidence < 0.22:
            abstention_reason = "low_confidence_evidence"
        elif avg_quality < 0.25 and not explicit_references:
            abstention_reason = "low_quality_evidence"

        return {
            "confidence": confidence,
            "abstention_reason": abstention_reason,
            "evidence_quality": {
                "average_quality_score": round(avg_quality, 4),
                "canonical_verse_count": len(canonical_verses),
                "supporting_chunk_count": len(supporting_chunks),
                "commentary_count": len(commentary_matches),
                "dbs_contributed": contributed_dbs,
                "explicit_reference_count": len(explicit_references),
                "explicit_reference_hits": explicit_hits,
            },
        }

    def _abstention_answer(self, query: str, reason: str, canonical_verses: List[Dict[str, Any]]) -> str:
        """Return a grounded no-answer message without asking the LLM to guess."""
        if canonical_verses:
            verse_ids = ", ".join(verse.get("verse_id", "Unknown") for verse in canonical_verses[:3])
            return (
                "I found limited evidence, but not enough to answer confidently from the supplied "
                f"verses/commentary. Relevant verse evidence found: {verse_ids}."
            )
        if reason == "no_usable_evidence":
            return (
                "No answer could be generated because neither Qdrant nor Neo4j returned usable "
                "supporting evidence, and no canonical SQLite verse was matched."
            )
        return "The available evidence is too weak to answer without guessing."

    def generate_answer(
        self,
        query: str,
        reference_answer: str = None,
        regularization: str = "combined",
        max_retries: int = MAX_CONSISTENCY_RETRIES,
    ) -> AnswerResult:
        """Full pipeline with evidence-first routing, reranking, and confidence."""
        start_time = time.time()

        verse_filter = parse_verse_references(query)
        explicit_references = verse_filter.verse_ids or []

        expansion_key = self.cache.build_key({"query": query, "kind": "query_expansion"})
        query_expansion = self.cache.get("query_expansion", expansion_key)
        cache_status = {
            "query_expansion": "hit" if query_expansion else "miss",
            "rerank_bundle": "miss",
            "llm_answer": "miss",
        }
        if not query_expansion:
            query_expansion = expand_query_with_aliases(query)
            self.cache.set("query_expansion", expansion_key, query_expansion)

        entities = query_expansion.get("entities", [])
        query_intent = classify_query_intent(query, verse_filter=verse_filter, entities=entities).to_dict()

        is_english = is_english_query(query)
        normalized_query = str(query_expansion.get("expanded_query") or query)

        if is_english:
            iast_query = query
        else:
            iast_query = query
            if self.gemini.is_available():
                try:
                    iast_query = self.gemini.translate_to_iast(query)
                    if not iast_query:
                        iast_query = query
                except Exception:
                    iast_query = query
                normalized_query = f"{iast_query}\n{query_expansion.get('expanded_query') or query}"
                if self.gemini.is_available():
                    try:
                        normalized_query = self.gemini.normalize_with_byt5(iast_query)
                        if not normalized_query:
                            normalized_query = iast_query
                    except Exception:
                        normalized_query = iast_query
                if query_expansion.get("aliases_added"):
                    normalized_query = f"{normalized_query}\nRelated aliases: {', '.join(query_expansion['aliases_added'])}"

        retrieval_pass = 0
        best_result = None
        best_consistency = -1.0

        total_attempts = max_retries if reference_answer else 1

        for attempt in range(total_attempts):
            retrieval_pass = attempt + 1

            if attempt > 0:
                normalized_query = f"{query} {reference_answer}" if reference_answer else query

            rrf_results = self.retriever.cross_db_rrf_search(
                normalized_query,
                top_k=max(self.top_k * 2, self.top_k + attempt * 5),
                include_bm25=True,
                regularization=regularization,
                verse_filter=verse_filter if attempt == 0 else None,
            )

            initial_verse_ids = list(dict.fromkeys(r.verse_id for r in rrf_results if r.verse_id))
            direct_commentary_verse_ids = self._direct_commentary_verse_ids(
                query=query,
                query_intent=query_intent,
                top_k=min(self.top_k, 5),
            )
            commentary_verse_ids = self._sqlite_commentary_verse_ids(
                initial_verse_ids + explicit_references + direct_commentary_verse_ids
            )
            rerank_key = self.cache.build_key({
                "query": normalized_query,
                "intent": query_intent.get("intent"),
                "candidate_ids": [result.id for result in rrf_results],
                "explicit_references": explicit_references,
            })
            cached_rerank = self.cache.get("rerank_bundle", rerank_key)
            if cached_rerank:
                cache_status["rerank_bundle"] = "hit"
                score_by_id = cached_rerank.get("scores", {})
                order = cached_rerank.get("ordered_ids", [])
                for result in rrf_results:
                    if result.id in score_by_id:
                        result.final_score = score_by_id[result.id]
                rrf_results.sort(key=lambda item: order.index(item.id) if item.id in order else len(order))
                rrf_results = rrf_results[:self.top_k]
            else:
                rrf_results = self.evidence_reranker.rerank(
                    rrf_results,
                    RerankContext(
                        query=query,
                        verse_filter=verse_filter,
                        query_intent=query_intent,
                        entities=entities,
                        commentary_verse_ids=commentary_verse_ids,
                    ),
                    top_k=self.top_k,
                )
                self.cache.set("rerank_bundle", rerank_key, {
                    "ordered_ids": [result.id for result in rrf_results],
                    "scores": {result.id: result.final_score for result in rrf_results},
                })

            verse_ids = list(dict.fromkeys(
                explicit_references
                + [r.verse_id for r in rrf_results if r.verse_id]
                + direct_commentary_verse_ids
            ))
            db_status = self._build_db_status(rrf_results)
            supporting_chunks = self._supporting_chunks(rrf_results)

            graph_metadata = []
            if self.neo4j and verse_ids:
                graph_metadata = self._fetch_graph_metadata(verse_ids)

            retrieved_verses = self._resolve_canonical_verses(verse_ids, rrf_results)
            commentary_matches = self._retrieve_commentary_matches(
                query=query,
                verse_filter=verse_filter,
                retrieved_verses=retrieved_verses,
            )

            retrieval_metadata = {
                "db_status": db_status,
                "rrf_results": len(rrf_results),
                "unique_verses": len(verse_ids),
                "top_verse_ids": verse_ids[:self.top_k],
                "commentary_candidates": len(commentary_matches),
                "direct_commentary_verse_ids": direct_commentary_verse_ids,
                "retrieval_mode": query_intent.get("intent") or ("verse_level" if verse_filter.has_filter() else "multi_hop"),
                "top_scores": [r.final_score for r in rrf_results[:5]],
                "query_intent": query_intent,
                "entities": entities,
                "explicit_references": explicit_references,
                "cache": cache_status,
            }

            quality_info = self._score_evidence_quality(
                canonical_verses=retrieved_verses,
                supporting_chunks=supporting_chunks,
                commentary_matches=commentary_matches,
                db_status=db_status,
                explicit_references=explicit_references,
                rrf_results=rrf_results,
            )
            confidence = quality_info["confidence"]
            abstention_reason = quality_info["abstention_reason"]
            evidence_quality = quality_info["evidence_quality"]

            if abstention_reason == "no_usable_evidence" or (abstention_reason and not explicit_references):
                answer_text = self._abstention_answer(query, abstention_reason, retrieved_verses)
                citations = []
                latency_ms = (time.time() - start_time) * 1000
                best_result = AnswerResult(
                    answer=answer_text,
                    query=query,
                    iast_query=iast_query,
                    normalized_query=normalized_query,
                    citations=citations,
                    sources={
                        "qdrant_verses": 0,
                        "neo4j_verses": 0,
                        "sqlite_verses": 0,
                        "graph_metadata": 0,
                        "commentary_matches": 0,
                        "total_retrieved": len(retrieved_verses),
                        "db_status": db_status,
                    },
                    retrieval_stats=retrieval_metadata,
                    commentary_matches=commentary_matches,
                    evidence={
                        "canonical_verses": retrieved_verses,
                        "supporting_chunks": supporting_chunks,
                        "commentary_matches": commentary_matches,
                        "graph_metadata": graph_metadata,
                        "db_status": db_status,
                        "query_expansion": query_expansion,
                    },
                    latency_ms=latency_ms,
                    consistency_score=0.0,
                    retrieval_passes=retrieval_pass,
                    verse_filter=verse_filter.to_dict() if verse_filter.has_filter() else None,
                    query_intent=query_intent,
                    confidence=confidence,
                    evidence_quality=evidence_quality,
                    abstention_reason=abstention_reason,
                    explicit_references=explicit_references,
                    cache=cache_status,
                )
                break

            if self.gemini.is_available():
                llm_key = self.cache.build_key({
                    "query": query,
                    "model": getattr(self.gemini, "model_name", "unknown"),
                    "intent": query_intent.get("intent"),
                    "verse_ids": [verse.get("verse_id") for verse in retrieved_verses],
                    "commentary_ids": [match.get("commentary_id") for match in commentary_matches],
                    "template": query_intent.get("intent"),
                })
                answer_data = self.cache.get("llm_answer", llm_key)
                if answer_data:
                    cache_status["llm_answer"] = "hit"
                else:
                    answer_data = self.gemini.generate_answer(
                        query=query,
                        iast_query=iast_query,
                        retrieved_verses=retrieved_verses,
                        metadata=graph_metadata,
                        commentary_matches=commentary_matches,
                        supporting_chunks=supporting_chunks,
                        retrieval_metadata=retrieval_metadata,
                        answer_template=get_answer_template(str(query_intent.get("intent", ""))),
                        query_intent=query_intent,
                        entities=entities,
                        confidence=confidence,
                    )
                    self.cache.set("llm_answer", llm_key, answer_data)
                answer_text = answer_data["answer"]
                citations = answer_data["citations"]
            else:
                verse_context = "\n\n".join([
                    f"[{i+1}] {v.get('verse_id', 'Unknown')}\n{v['text']}"
                    for i, v in enumerate(retrieved_verses)
                ])
                commentary_context = "\n\n".join([
                    f"[{m.get('verse_id', 'Unknown')}] {m.get('author_display_name', 'Commentary')}: {m.get('text', '')}"
                    for m in commentary_matches
                ])
                answer_text = f"Based on the retrieved verses:\n\n{verse_context}"
                if commentary_context:
                    answer_text += f"\n\nRelated commentary:\n\n{commentary_context}"
                citations = [
                    {
                        "verse_id": v.get("verse_id", ""),
                        "source": v.get("source", ""),
                        "score": v.get("score", 0.0),
                        "text": v.get("text", "")[:200],
                    }
                    for v in retrieved_verses
                ]

            answer_text = self._format_answer_markdown(answer_text, citations)

            latency_ms = (time.time() - start_time) * 1000

            consistency_score = 0.0
            if reference_answer and self.gemini.is_available():
                try:
                    from src.embedding_client import NVIDIAEmbeddingClient
                    emb_client = self.retriever.embedding_client
                    ref_emb = emb_client.embed_query(reference_answer)
                    gen_emb = emb_client.embed_query(answer_text)
                    import numpy as np
                    v1 = np.array(ref_emb.dense_vector)
                    v2 = np.array(gen_emb.dense_vector)
                    norm1 = np.linalg.norm(v1)
                    norm2 = np.linalg.norm(v2)
                    if norm1 > 0 and norm2 > 0:
                        consistency_score = float(np.dot(v1, v2) / (norm1 * norm2))
                except Exception:
                    pass

            if consistency_score > best_consistency:
                best_consistency = consistency_score
                best_result = AnswerResult(
                    answer=answer_text,
                    query=query,
                    iast_query=iast_query,
                    normalized_query=normalized_query,
                    citations=citations,
                    sources={
                        "qdrant_verses": db_status["qdrant"]["candidate_count"],
                        "neo4j_verses": db_status["neo4j"]["candidate_count"],
                        "sqlite_verses": sum(1 for v in retrieved_verses if v.get("source") == "SQLite"),
                        "graph_metadata": len(graph_metadata),
                        "commentary_matches": len(commentary_matches),
                        "total_retrieved": len(retrieved_verses),
                        "db_status": db_status,
                    },
                    retrieval_stats=retrieval_metadata,
                    commentary_matches=commentary_matches,
                    evidence={
                        "canonical_verses": retrieved_verses,
                        "supporting_chunks": supporting_chunks,
                        "commentary_matches": commentary_matches,
                        "graph_metadata": graph_metadata,
                        "db_status": db_status,
                        "query_expansion": query_expansion,
                    },
                    latency_ms=latency_ms,
                    consistency_score=round(consistency_score, 4),
                    retrieval_passes=retrieval_pass,
                    verse_filter=verse_filter.to_dict() if verse_filter.has_filter() else None,
                    query_intent=query_intent,
                    confidence=confidence,
                    evidence_quality=evidence_quality,
                    abstention_reason=abstention_reason,
                    explicit_references=explicit_references,
                    cache=cache_status,
                )

            if consistency_score >= SEMANTIC_CONSISTENCY_THRESHOLD:
                break

        if best_result is None:
            best_result = AnswerResult(
                answer="No answer could be generated.",
                query=query,
                iast_query=iast_query,
                normalized_query=normalized_query,
                commentary_matches=[],
                latency_ms=(time.time() - start_time) * 1000,
                query_intent=query_intent,
                confidence=0.0,
                evidence_quality={},
                abstention_reason="generation_failed",
                explicit_references=explicit_references,
                cache=cache_status,
            )

        return best_result
