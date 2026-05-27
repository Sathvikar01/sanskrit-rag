"""Deterministic query intent routing for SansRAG evidence assembly."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.retriever import VerseFilter


@dataclass(frozen=True)
class QueryIntent:
    """A lightweight intent classification used to tune retrieval and prompting."""

    intent: str
    confidence: float
    reasons: List[str] = field(default_factory=list)
    retrieval_profile: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent": self.intent,
            "confidence": self.confidence,
            "reasons": self.reasons,
            "retrieval_profile": self.retrieval_profile,
        }


INTENT_PROFILES: Dict[str, Dict[str, Any]] = {
    "explicit_verse_lookup": {
        "graph_weight": 0.7,
        "vector_weight": 0.3,
        "commentary_priority": 0.55,
        "strict_abstention": False,
    },
    "commentary_question": {
        "graph_weight": 0.35,
        "vector_weight": 0.65,
        "commentary_priority": 1.0,
        "strict_abstention": True,
    },
    "comparison_question": {
        "graph_weight": 0.45,
        "vector_weight": 0.55,
        "commentary_priority": 0.65,
        "strict_abstention": True,
    },
    "character_entity_question": {
        "graph_weight": 0.55,
        "vector_weight": 0.45,
        "commentary_priority": 0.7,
        "strict_abstention": True,
    },
    "source_inspection": {
        "graph_weight": 0.5,
        "vector_weight": 0.5,
        "commentary_priority": 0.8,
        "strict_abstention": True,
    },
    "summary_explanation": {
        "graph_weight": 0.4,
        "vector_weight": 0.6,
        "commentary_priority": 0.75,
        "strict_abstention": True,
    },
    "theme_concept_question": {
        "graph_weight": 0.45,
        "vector_weight": 0.55,
        "commentary_priority": 0.65,
        "strict_abstention": True,
    },
}


def classify_query_intent(
    query: str,
    verse_filter: Optional[VerseFilter] = None,
    entities: Optional[List[Dict[str, Any]]] = None,
) -> QueryIntent:
    """Classify the query into a small set of retrieval/prompt intents."""
    text = (query or "").lower()
    reasons: List[str] = []
    entities = entities or []

    if verse_filter and verse_filter.has_filter():
        reasons.append("explicit verse reference detected")
        return QueryIntent(
            "explicit_verse_lookup",
            0.98,
            reasons,
            INTENT_PROFILES["explicit_verse_lookup"],
        )

    if re.search(r"\b(commentary|commentator|purport|according to|interpretation|explains?)\b", text):
        reasons.append("commentary or purport language detected")
        return QueryIntent(
            "commentary_question",
            0.88,
            reasons,
            INTENT_PROFILES["commentary_question"],
        )

    if re.search(r"\b(compare|contrast|difference|similarity|versus| vs |between .+ and )\b", text):
        reasons.append("comparison language detected")
        return QueryIntent(
            "comparison_question",
            0.86,
            reasons,
            INTENT_PROFILES["comparison_question"],
        )

    if re.search(r"\b(citation|cite|source|which verse|where does|reference)\b", text):
        reasons.append("source inspection language detected")
        return QueryIntent(
            "source_inspection",
            0.84,
            reasons,
            INTENT_PROFILES["source_inspection"],
        )

    if entities and re.search(r"\b(who|why.*name|known as|called|meaning of name|what is .* name)\b", text):
        reasons.append("entity/name question detected")
        return QueryIntent(
            "character_entity_question",
            0.82,
            reasons,
            INTENT_PROFILES["character_entity_question"],
        )

    if re.search(r"\b(summarize|summary|explain|meaning|teach|teaches|lesson|message)\b", text):
        reasons.append("summary or explanation language detected")
        return QueryIntent(
            "summary_explanation",
            0.78,
            reasons,
            INTENT_PROFILES["summary_explanation"],
        )

    if entities or re.search(r"\b(dharma|karma|yoga|bhakti|atman|self|soul|duty|action|devotion)\b", text):
        reasons.append("theme or concept terms detected")
        return QueryIntent(
            "theme_concept_question",
            0.72,
            reasons,
            INTENT_PROFILES["theme_concept_question"],
        )

    reasons.append("defaulted to conceptual retrieval")
    return QueryIntent(
        "theme_concept_question",
        0.55,
        reasons,
        INTENT_PROFILES["theme_concept_question"],
    )
