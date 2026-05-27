"""Deterministic evidence reranking after dual-DB RRF fusion."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set

from src.retriever import HybridSearchResult, VerseFilter
from src.text_quality import score_text_quality


STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with",
    "what", "why", "how", "who", "which", "does", "did", "is", "are", "was",
    "were", "about", "according", "verse", "chapter", "bg", "bhg", "gita",
}


@dataclass
class RerankContext:
    """Inputs used by the deterministic reranker."""

    query: str
    verse_filter: VerseFilter
    query_intent: Dict[str, Any]
    entities: List[Dict[str, Any]]
    commentary_verse_ids: Optional[Set[str]] = None


def _terms(text: str) -> Set[str]:
    return {
        token
        for token in re.findall(r"[a-zA-Z\u0900-\u097F]+", (text or "").lower())
        if len(token) > 2 and token not in STOPWORDS
    }


class EvidenceReranker:
    """Feature-based reranker that preserves explicit references."""

    def rerank(
        self,
        candidates: List[HybridSearchResult],
        context: RerankContext,
        top_k: int,
    ) -> List[HybridSearchResult]:
        if not candidates:
            return []

        explicit_ids = set(context.verse_filter.verse_ids or [])
        query_terms = _terms(context.query)
        entity_terms = set()
        for entity in context.entities:
            entity_terms.add(str(entity.get("canonical", "")).lower())
            entity_terms.update(str(alias).lower() for alias in entity.get("aliases", [])[:8])

        commentary_verse_ids = context.commentary_verse_ids or set()
        profile = context.query_intent.get("retrieval_profile", {}) if context.query_intent else {}
        commentary_priority = float(profile.get("commentary_priority", 0.5))

        for index, candidate in enumerate(candidates):
            candidate_terms = _terms(candidate.text)
            lexical_overlap = len(query_terms & candidate_terms) / max(len(query_terms), 1)
            entity_overlap = len(entity_terms & candidate_terms) / max(len(entity_terms), 1) if entity_terms else 0.0
            quality = score_text_quality(candidate.text)
            sources = candidate.metadata.get("sources", {}) if candidate.metadata else {}
            source_diversity = sum(1 for contributed in sources.values() if contributed) / 2.0
            explicit_bonus = 1.0 if candidate.verse_id in explicit_ids else 0.0
            commentary_bonus = commentary_priority if candidate.verse_id in commentary_verse_ids else 0.0

            rerank_score = (
                0.42 * float(candidate.final_score or 0.0)
                + 0.2 * lexical_overlap
                + 0.16 * explicit_bonus
                + 0.08 * entity_overlap
                + 0.06 * source_diversity
                + 0.05 * float(quality["quality_score"])
                + 0.03 * commentary_bonus
            )

            candidate.metadata = {
                **(candidate.metadata or {}),
                "reranker": {
                    "original_rank": index + 1,
                    "original_score": candidate.final_score,
                    "deterministic_score": round(rerank_score, 6),
                    "lexical_overlap": round(lexical_overlap, 4),
                    "entity_overlap": round(entity_overlap, 4),
                    "explicit_reference": bool(explicit_bonus),
                    "commentary_available": candidate.verse_id in commentary_verse_ids,
                    "source_diversity": round(source_diversity, 4),
                    "quality_score": quality["quality_score"],
                    "filtered_reason": quality["filtered_reason"],
                },
                "quality_score": quality["quality_score"],
                "filtered_reason": quality["filtered_reason"],
            }
            candidate.final_score = float(rerank_score)

        usable = [
            candidate
            for candidate in candidates
            if not candidate.metadata.get("filtered_reason") or candidate.verse_id in explicit_ids
        ]
        usable.sort(
            key=lambda item: (
                0 if item.verse_id in explicit_ids else 1,
                -float(item.final_score or 0.0),
            )
        )
        return usable[:top_k]
