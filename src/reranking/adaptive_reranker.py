"""Adaptive re-ranker with dynamic weight profiles based on query type."""

import re


def detect_query_type(query_iast: str, concepts: list[str]) -> str:
    """Detect the type of query to determine optimal retrieval weights.

    Args:
        query_iast: Query text in IAST.
        concepts: Extracted concept names.

    Returns:
        Query type string.
    """
    words = query_iast.split()
    word_count = len(words)
    has_concepts = len(concepts) > 0

    definition_patterns = [
        r'\bwhat\s+is\b', r'\bwho\s+is\b', r'\bdefine\b', r'\bmeaning\s+of\b',
        r'\bexplain\b', r'\bdescribe\b', r'\bwhat\s+does\b', r'\bwhat\s+are\b',
    ]
    is_definition = any(re.search(p, query_iast.lower()) for p in definition_patterns)

    if word_count <= 8 and is_definition:
        return "concept_short"
    elif word_count <= 8:
        return "factual_short"
    elif word_count > 18:
        return "complex_long"
    elif has_concepts:
        return "concept_medium"
    else:
        return "general_medium"


# Dynamic weight profiles for re-ranking
# Key insight: vector > graph > bm25 as user requested
# But BM25 gets more weight for long queries where exact terms matter
ADAPTIVE_WEIGHT_PROFILES = {
    "concept_short": {
        "score_vector": 0.45,
        "score_graph": 0.20,
        "score_bm25": 0.05,
        "score_lemma": 0.12,
        "score_morph": 0.08,
        "score_compound": 0.03,
        "score_commentary": 0.04,
        "score_concept": 0.02,
        "score_graph_centrality": 0.01,
    },
    "factual_short": {
        "score_vector": 0.35,
        "score_graph": 0.30,
        "score_bm25": 0.08,
        "score_lemma": 0.10,
        "score_morph": 0.07,
        "score_compound": 0.03,
        "score_commentary": 0.04,
        "score_concept": 0.02,
        "score_graph_centrality": 0.01,
    },
    "complex_long": {
        "score_vector": 0.35,
        "score_graph": 0.15,
        "score_bm25": 0.20,
        "score_lemma": 0.12,
        "score_morph": 0.08,
        "score_compound": 0.04,
        "score_commentary": 0.04,
        "score_concept": 0.01,
        "score_graph_centrality": 0.01,
    },
    "concept_medium": {
        "score_vector": 0.40,
        "score_graph": 0.22,
        "score_bm25": 0.08,
        "score_lemma": 0.12,
        "score_morph": 0.08,
        "score_compound": 0.03,
        "score_commentary": 0.04,
        "score_concept": 0.02,
        "score_graph_centrality": 0.01,
    },
    "general_medium": {
        "score_vector": 0.40,
        "score_graph": 0.18,
        "score_bm25": 0.12,
        "score_lemma": 0.12,
        "score_morph": 0.08,
        "score_compound": 0.03,
        "score_commentary": 0.04,
        "score_concept": 0.02,
        "score_graph_centrality": 0.01,
    },
}

DEFAULT_WEIGHTS = {
    "score_vector": 0.40,
    "score_graph": 0.20,
    "score_bm25": 0.10,
    "score_lemma": 0.12,
    "score_morph": 0.08,
    "score_compound": 0.03,
    "score_commentary": 0.04,
    "score_concept": 0.02,
    "score_graph_centrality": 0.01,
}


def get_adaptive_weights(query_type: str) -> dict[str, float]:
    """Get adaptive re-ranking weights based on query type.

    Args:
        query_type: One of concept_short, factual_short, complex_long,
                   concept_medium, general_medium.

    Returns:
        Dictionary of feature weights.
    """
    weights = ADAPTIVE_WEIGHT_PROFILES.get(query_type, DEFAULT_WEIGHTS)
    total = sum(weights.values())
    return {k: v / total for k, v in weights.items()}


def describe_query_type(query_type: str) -> str:
    """Get human-readable description of query type."""
    descriptions = {
        "concept_short": "Short definitional query (vector-heavy)",
        "factual_short": "Short factual query (graph-heavy)",
        "complex_long": "Long complex query (BM25 boost for specificity)",
        "concept_medium": "Medium concept query (balanced)",
        "general_medium": "General medium query (balanced)",
    }
    return descriptions.get(query_type, "Unknown query type")
