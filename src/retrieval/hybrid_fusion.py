"""Hybrid retrieval fusion combining vector, BM25, and graph results."""

import math
from typing import Optional

from src.utils.config import Config
from src.utils.logger import logger


def reciprocal_rank_fusion(
    ranked_lists: list[list[dict]],
    k: int = 60,
) -> list[dict]:
    """Fuse multiple ranked lists using Reciprocal Rank Fusion.

    Args:
        ranked_lists: List of ranked result lists, each with chunk_id and score.
        k: RRF parameter (default 60).

    Returns:
        Fused and re-ranked list of results.
    """
    rrf_scores: dict[str, float] = {}
    source_map: dict[str, list[str]] = {}
    data_map: dict[str, dict] = {}

    source_names = ["vector", "graph", "bm25"]

    for list_idx, ranked_list in enumerate(ranked_lists):
        source = source_names[list_idx] if list_idx < len(source_names) else f"source_{list_idx}"

        for rank, result in enumerate(ranked_list, 1):
            chunk_id = result["chunk_id"]
            rrf_score = 1.0 / (k + rank)

            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + rrf_score

            if chunk_id not in source_map:
                source_map[chunk_id] = []
            source_map[chunk_id].append(source)

            if chunk_id not in data_map:
                data_map[chunk_id] = result.copy()

    fused = []
    for chunk_id, score in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True):
        result = data_map[chunk_id].copy()
        result["rrf_score"] = score
        result["sources"] = source_map[chunk_id]
        result["retrieval_confidence"] = score
        fused.append(result)

    for i, r in enumerate(fused):
        r["rank"] = i + 1

    return fused


def weighted_fusion(
    ranked_lists: list[list[dict]],
    weights: list[float] = None,
    normalize_scores: bool = True,
) -> list[dict]:
    """Fuse multiple ranked lists using weighted score combination.

    Args:
        ranked_lists: List of ranked result lists.
        weights: Weights for each retrieval method.
        normalize_scores: Whether to normalize scores to [0, 1].

    Returns:
        Fused and re-ranked list of results.
    """
    if weights is None:
        weights = [0.4, 0.35, 0.25]

    if len(weights) != len(ranked_lists):
        raise ValueError("Number of weights must match number of ranked lists")

    if normalize_scores:
        normalized_lists = []
        for ranked_list in ranked_lists:
            if not ranked_list:
                normalized_lists.append([])
                continue

            scores = [r.get("score", 0) for r in ranked_list]
            min_s = min(scores) if scores else 0
            max_s = max(scores) if scores else 1
            range_s = max_s - min_s

            normalized = []
            for result in ranked_list:
                r = result.copy()
                if range_s > 0:
                    r["normalized_score"] = (result.get("score", 0) - min_s) / range_s
                else:
                    r["normalized_score"] = 1.0
                normalized.append(r)
            normalized_lists.append(normalized)
    else:
        normalized_lists = ranked_lists

    weighted_scores: dict[str, float] = {}
    source_map: dict[str, list[str]] = {}
    data_map: dict[str, dict] = {}

    source_names = ["vector", "graph", "bm25"]

    for list_idx, (weight, ranked_list) in enumerate(zip(weights, normalized_lists)):
        source = source_names[list_idx] if list_idx < len(source_names) else f"source_{list_idx}"

        for result in ranked_list:
            chunk_id = result["chunk_id"]
            score = result.get("normalized_score", result.get("score", 0))

            weighted_scores[chunk_id] = weighted_scores.get(chunk_id, 0.0) + weight * score

            if chunk_id not in source_map:
                source_map[chunk_id] = []
            source_map[chunk_id].append(source)

            if chunk_id not in data_map:
                data_map[chunk_id] = result.copy()

    fused = []
    for chunk_id, score in sorted(weighted_scores.items(), key=lambda x: x[1], reverse=True):
        result = data_map[chunk_id].copy()
        result["weighted_score"] = score
        result["sources"] = source_map[chunk_id]
        result["retrieval_confidence"] = score
        fused.append(result)

    for i, r in enumerate(fused):
        r["rank"] = i + 1

    return fused


def sigmoid_normalize(score: float, temperature: float = 1.0) -> float:
    """Normalize a score using sigmoid function."""
    return 1.0 / (1.0 + math.exp(-score / temperature))


class HybridRetriever:
    """Hybrid retrieval combining vector, BM25, and graph retrieval."""

    def __init__(self, config: Config = None):
        if config is None:
            config = Config()

        self.fusion_method = config.get("retrieval.fusion_method", "rrf")
        self.rrf_k = config.get("retrieval.rrf_k", 60)
        self.top_k = config.get("retrieval.vector_top_k", 50)

    def fuse_results(
        self,
        vector_results: list[dict],
        graph_results: list[dict],
        bm25_results: list[dict],
        top_k: int = None,
    ) -> list[dict]:
        """Fuse results from all three retrieval methods.

        Args:
            vector_results: Results from vector retrieval.
            graph_results: Results from graph retrieval.
            bm25_results: Results from BM25 retrieval.
            top_k: Maximum results to return.

        Returns:
            Fused and re-ranked results.
        """
        if top_k is None:
            top_k = self.top_k

        ranked_lists = [vector_results, graph_results, bm25_results]

        if self.fusion_method == "rrf":
            fused = reciprocal_rank_fusion(ranked_lists, k=self.rrf_k)
        elif self.fusion_method == "weighted":
            fused = weighted_fusion(ranked_lists)
        else:
            logger.warning(f"Unknown fusion method: {self.fusion_method}, using RRF")
            fused = reciprocal_rank_fusion(ranked_lists, k=self.rrf_k)

        logger.info(
            f"Hybrid fusion: {len(vector_results)} vector + "
            f"{len(graph_results)} graph + {len(bm25_results)} BM25 → "
            f"{len(fused)} fused results"
        )

        return fused[:top_k]
