"""Feature extractors for the linguistic re-ranker."""

import re
from dataclasses import dataclass

from src.preprocessing.chunker import Chunk
from src.preprocessing.morpho_extractor import (
    MorphologicalProfile,
    compute_morpho_similarity,
)


def extract_lemma_overlap_score(
    query_lemmas: set[str],
    doc_lemmas: set[str],
) -> float:
    """Compute lemma overlap score between query and document.

    score_lemma = |query_lemmas ∩ doc_lemmas| / |query_lemmas|

    Args:
        query_lemmas: Set of lemmas from the query.
        doc_lemmas: Set of lemmas from the document.

    Returns:
        Overlap score between 0 and 1.
    """
    if not query_lemmas:
        return 0.0

    overlap = query_lemmas.intersection(doc_lemmas)
    return len(overlap) / len(query_lemmas)


def extract_morphological_alignment_score(
    query_morpho: MorphologicalProfile,
    doc_morpho: MorphologicalProfile,
) -> float:
    """Compute morphological feature alignment score.

    Compares distributions of grammatical features (case, gender, number,
    tense, mood) between query and document.

    Args:
        query_morpho: Morphological profile of the query.
        doc_morpho: Morphological profile of the document.

    Returns:
        Alignment score between 0 and 1.
    """
    return compute_morpho_similarity(query_morpho, doc_morpho)


def extract_compound_match_score(
    query_tokens: list[str],
    doc_tokens: list[str],
) -> float:
    """Compute compound decomposition match score.

    Checks if compound parts from the query appear in the document.

    Args:
        query_tokens: Compound parts from the query.
        doc_tokens: Compound parts from the document.

    Returns:
        Match score between 0 and 1.
    """
    if not query_tokens:
        return 0.0

    query_set = set(t.lower() for t in query_tokens)
    doc_set = set(t.lower() for t in doc_tokens)

    overlap = query_set.intersection(doc_set)
    return len(overlap) / len(query_set) if query_set else 0.0


def extract_commentary_consensus_score(
    chunk: Chunk,
    all_chunks: list[Chunk],
) -> float:
    """Compute commentary consensus score.

    Higher scores for verses where multiple commentators discuss the same theme.

    Args:
        chunk: The chunk being scored.
        all_chunks: All available chunks.

    Returns:
        Consensus score between 0 and 1.
    """
    verse_ref = chunk.verse_ref
    commentary_count = sum(
        1 for c in all_chunks
        if c.verse_ref == verse_ref and c.chunk_type == "commentary"
    )

    return min(commentary_count / 3.0, 1.0)


def extract_concept_overlap_score(
    query_concepts: set[str],
    doc_concepts: set[str],
) -> float:
    """Compute concept overlap score.

    Args:
        query_concepts: Set of concept names from the query.
        doc_concepts: Set of concept names from the document.

    Returns:
        Overlap score between 0 and 1.
    """
    if not query_concepts:
        return 0.0

    overlap = query_concepts.intersection(doc_concepts)
    return len(overlap) / len(query_concepts)


def extract_graph_centrality_score(
    degree_centrality: int,
    max_degree: int = 100,
) -> float:
    """Normalize graph degree centrality to [0, 1].

    Args:
        degree_centrality: Raw degree centrality value.
        max_degree: Expected maximum degree for normalization.

    Returns:
        Normalized centrality score.
    """
    return min(degree_centrality / max_degree, 1.0)


def tokenize_iast(text: str) -> list[str]:
    """Simple IAST tokenizer."""
    text = text.lower()
    text = re.sub(r'[।॥,;:!?.\-—\(\)\[\]]', ' ', text)
    tokens = text.split()
    return [t for t in tokens if len(t) > 1]


def extract_lemmas_from_text(text: str) -> set[str]:
    """Extract lemmas from text using simple normalization.

    For a more sophisticated approach, use the segmentation data.
    """
    tokens = tokenize_iast(text)
    lemmas = set()

    for token in tokens:
        lemmas.add(token)
        if token.endswith("aḥ"):
            lemmas.add(token[:-1])
        elif token.endswith("am"):
            lemmas.add(token[:-2])
        elif token.endswith("āḥ"):
            lemmas.add(token[:-2] + "a")
        elif token.endswith("aiḥ"):
            lemmas.add(token[:-3])
        elif token.endswith("ebhyaḥ"):
            lemmas.add(token[:-6])

    return lemmas


@dataclass
class ReRankingFeatures:
    """Feature vector for re-ranking a candidate."""

    score_vector: float = 0.0
    score_graph: float = 0.0
    score_bm25: float = 0.0
    score_lemma: float = 0.0
    score_morph: float = 0.0
    score_compound: float = 0.0
    score_commentary: float = 0.0
    score_concept: float = 0.0
    score_graph_centrality: float = 0.0

    def to_vector(self) -> list[float]:
        """Convert to feature vector."""
        return [
            self.score_vector,
            self.score_graph,
            self.score_bm25,
            self.score_lemma,
            self.score_morph,
            self.score_compound,
            self.score_commentary,
            self.score_concept,
            self.score_graph_centrality,
        ]

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary."""
        return {
            "score_vector": self.score_vector,
            "score_graph": self.score_graph,
            "score_bm25": self.score_bm25,
            "score_lemma": self.score_lemma,
            "score_morph": self.score_morph,
            "score_compound": self.score_compound,
            "score_commentary": self.score_commentary,
            "score_concept": self.score_concept,
            "score_graph_centrality": self.score_graph_centrality,
        }
