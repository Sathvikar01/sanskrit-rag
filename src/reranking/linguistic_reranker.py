"""Linguistic re-ranker for SRAG - Novel contribution."""

import math
from typing import Optional

from src.preprocessing.chunker import Chunk
from src.preprocessing.concept_extractor import ConceptExtractor
from src.preprocessing.morpho_extractor import (
    MorphologicalProfile,
    build_morphological_profile,
)
from src.reranking.adaptive_reranker import (
    detect_query_type,
    get_adaptive_weights,
    describe_query_type,
)
from src.reranking.confidence import PipelineConfidence, sigmoid_normalize
from src.reranking.feature_extractors import (
    ReRankingFeatures,
    extract_commentary_consensus_score,
    extract_compound_match_score,
    extract_concept_overlap_score,
    extract_graph_centrality_score,
    extract_lemma_overlap_score,
    extract_morphological_alignment_score,
    extract_lemmas_from_text,
    tokenize_iast,
)
from src.utils.config import Config
from src.utils.logger import logger


class LinguisticReranker:
    """Novel linguistic re-ranker for Sanskrit text retrieval.

    This re-ranker combines multiple signals:
    1. Vector retrieval confidence
    2. Graph retrieval relevance
    3. BM25 keyword matching
    4. Lemma overlap (normalized forms)
    5. Morphological feature alignment (case, gender, tense, mood)
    6. Compound decomposition matching
    7. Commentary consensus (multi-commentary agreement)
    8. Concept overlap from knowledge graph
    9. Graph centrality (verse connectivity)

    Supports adaptive weighting based on query type.
    """

    def __init__(self, config: Config = None):
        if config is None:
            config = Config()

        self.config_weights = config.get("reranking.weights", {})
        self.top_n = config.get("reranking.top_n", 5)
        self.use_adaptive = config.get("reranking.adaptive", True)
        self.confidence = PipelineConfidence()
        self.concept_extractor = ConceptExtractor()

        self.default_weights = {
            "score_vector": self.config_weights.get("score_vector", 0.20),
            "score_graph": self.config_weights.get("score_graph", 0.15),
            "score_bm25": self.config_weights.get("score_bm25", 0.10),
            "score_lemma": self.config_weights.get("score_lemma", 0.15),
            "score_morph": self.config_weights.get("score_morph", 0.15),
            "score_compound": self.config_weights.get("score_compound", 0.05),
            "score_commentary": self.config_weights.get("score_commentary", 0.10),
            "score_concept": self.config_weights.get("score_concept", 0.05),
            "score_graph_centrality": self.config_weights.get("score_graph_centrality", 0.05),
        }

        logger.info(
            f"LinguisticReranker initialized (adaptive={self.use_adaptive}, "
            f"default weights: {self.default_weights})"
        )

    def _get_sanskrit_morpho_hint(self, token: str) -> str:
        """Guess morphological case from Sanskrit suffix patterns.

        This is a heuristic - not a full morphological analyzer.
        """
        token = token.lower()

        if token.endswith("aḥ") or token.endswith("āḥ"):
            return "Nom"
        elif token.endswith("am") or token.endswith("aṃ"):
            return "Acc"
        elif token.endswith("ena") or token.endswith("eṇa"):
            return "Ins"
        elif token.endswith("āya"):
            return "Dat"
        elif token.endswith("āt") or token.endswith("asmāt"):
            return "Abl"
        elif token.endswith("asya") or token.endswith("sya"):
            return "Gen"
        elif token.endswith("e") or token.endswith("asi"):
            return "Loc"
        elif token.endswith("au") or token.endswith("āḥ"):
            return "Nom"
        elif token.endswith("aiḥ") or token.endswith("ebhiḥ"):
            return "Ins"
        elif token.endswith("ebhyaḥ"):
            return "Dat"
        elif token.endswith("ānām"):
            return "Gen"
        elif token.endswith("eṣu"):
            return "Loc"
        else:
            return "Nom"

    def _extract_query_features(
        self,
        query_iast: str,
        concepts: list[str],
    ) -> dict:
        """Extract linguistic features from the query.

        Args:
            query_iast: Query text in IAST.
            concepts: Extracted concept names.

        Returns:
            Dictionary with query features.
        """
        lemmas = extract_lemmas_from_text(query_iast)
        tokens = tokenize_iast(query_iast)
        morpho_lines = []

        for token in tokens:
            case_hint = self._get_sanskrit_morpho_hint(token)
            morpho_lines.append(f"{token}_Case={case_hint}")

        query_morpho = build_morphological_profile(morpho_lines) if morpho_lines else MorphologicalProfile()

        return {
            "lemmas": lemmas,
            "tokens": tokens,
            "morpho_profile": query_morpho,
            "concepts": set(concepts),
        }

    def compute_features(
        self,
        query_features: dict,
        chunk: Chunk,
        retrieval_result: dict,
        all_chunks: list[Chunk],
        chunk_map: dict[str, Chunk],
    ) -> ReRankingFeatures:
        """Compute all re-ranking features for a candidate chunk.

        Args:
            query_features: Extracted query features.
            chunk: The candidate chunk.
            retrieval_result: Retrieval scores from hybrid fusion.
            all_chunks: All chunks for context.
            chunk_map: Map of chunk_id to Chunk for lookups.

        Returns:
            ReRankingFeatures with all computed scores.
        """
        features = ReRankingFeatures()

        features.score_vector = retrieval_result.get("vector_score", retrieval_result.get("score", 0.0))
        features.score_graph = retrieval_result.get("graph_score", 0.0)
        features.score_bm25 = retrieval_result.get("bm25_score", 0.0)

        doc_lemmas = set(chunk.lemmas) if chunk.lemmas else extract_lemmas_from_text(chunk.text_iast)
        features.score_lemma = extract_lemma_overlap_score(
            query_features["lemmas"], doc_lemmas
        )

        doc_morpho_lines = chunk.morpho_features if chunk.morpho_features else []
        if doc_morpho_lines:
            doc_morpho = build_morphological_profile(doc_morpho_lines)
            features.score_morph = extract_morphological_alignment_score(
                query_features["morpho_profile"], doc_morpho
            )

        doc_tokens = chunk.surface_forms if chunk.surface_forms else tokenize_iast(chunk.text_iast)
        features.score_compound = extract_compound_match_score(
            query_features["tokens"], doc_tokens
        )

        features.score_commentary = extract_commentary_consensus_score(chunk, all_chunks)

        doc_concepts = set()
        found = self.concept_extractor.extract_from_text(chunk.text_iast)
        for fc in found:
            doc_concepts.add(fc["concept"].name_iast)
        features.score_concept = extract_concept_overlap_score(
            query_features["concepts"], doc_concepts
        )

        graph_centrality = retrieval_result.get("degree_centrality", 0)
        features.score_graph_centrality = extract_graph_centrality_score(graph_centrality)

        return features

    def compute_final_score(
        self,
        features: ReRankingFeatures,
        weights: dict[str, float],
    ) -> float:
        """Compute weighted final score from features.

        Args:
            features: Computed re-ranking features.
            weights: Feature weights to use.

        Returns:
            Final weighted score.
        """
        feature_dict = features.to_dict()
        score = sum(
            weights.get(key, 0) * value
            for key, value in feature_dict.items()
        )
        return score

    def rerank(
        self,
        query_iast: str,
        concepts: list[str],
        candidates: list[dict],
        all_chunks: list[Chunk],
        chunk_map: dict[str, Chunk],
    ) -> list[dict]:
        """Re-rank candidate chunks using linguistic features.

        Args:
            query_iast: Query text in IAST.
            concepts: Extracted concept names.
            candidates: Candidate results from hybrid retrieval.
            all_chunks: All chunks for context.
            chunk_map: Map of chunk_id to Chunk.

        Returns:
            Re-ranked results with features and confidence.
        """
        query_type = detect_query_type(query_iast, concepts)

        if self.use_adaptive:
            weights = get_adaptive_weights(query_type)
        else:
            weights = self.default_weights

        logger.info(
            f"Re-ranking: query_type={query_type} "
            f"({describe_query_type(query_type)}), "
            f"vector_weight={weights.get('score_vector', 0):.2f}, "
            f"graph_weight={weights.get('score_graph', 0):.2f}, "
            f"bm25_weight={weights.get('score_bm25', 0):.2f}"
        )

        query_features = self._extract_query_features(query_iast, concepts)

        reranked = []
        for candidate in candidates:
            chunk_id = candidate["chunk_id"]
            chunk = chunk_map.get(chunk_id)

            if chunk is None:
                logger.warning(f"Chunk not found: {chunk_id}")
                continue

            features = self.compute_features(
                query_features, chunk, candidate, all_chunks, chunk_map
            )

            final_score = self.compute_final_score(features, weights)

            retrieval_conf = self.confidence.compute_retrieval_confidence(
                candidate.get("rrf_score", 0),
                len(candidate.get("sources", [])),
            )
            reranking_conf = self.confidence.compute_reranking_confidence(final_score)
            pipeline_conf = self.confidence.compute_pipeline_confidence(
                retrieval_conf, reranking_conf, 0.0
            )

            result = {
                "chunk_id": chunk_id,
                "verse_ref": chunk.verse_ref,
                "text_iast": chunk.text_iast,
                "text_devanagari": chunk.text_devanagari,
                "chunk_type": chunk.chunk_type,
                "commentator": chunk.commentator,
                "final_score": final_score,
                "features": features.to_dict(),
                "confidence": pipeline_conf,
                "sources": candidate.get("sources", []),
                "query_type": query_type,
            }
            reranked.append(result)

        reranked.sort(key=lambda x: x["final_score"], reverse=True)

        for i, r in enumerate(reranked):
            r["rank"] = i + 1

        if reranked:
            logger.info(
                f"Re-ranked {len(reranked)} candidates, "
                f"top score: {reranked[0]['final_score']:.4f}"
            )
        else:
            logger.warning("No results after re-ranking")

        return reranked[:self.top_n]
