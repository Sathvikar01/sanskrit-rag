"""Confidence scoring and calibration for SRAG pipeline."""

import math
from typing import Optional

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from src.utils.logger import logger


def min_max_normalize(scores: list[float]) -> list[float]:
    """Normalize scores to [0, 1] using min-max scaling."""
    if not scores:
        return []

    min_s = min(scores)
    max_s = max(scores)
    range_s = max_s - min_s

    if range_s == 0:
        return [1.0] * len(scores)

    return [(s - min_s) / range_s for s in scores]


def sigmoid_normalize(score: float, temperature: float = 1.0) -> float:
    """Normalize a score using sigmoid function."""
    return 1.0 / (1.0 + math.exp(-score / temperature))


def z_score_normalize(scores: list[float]) -> list[float]:
    """Normalize scores using z-score normalization."""
    if not scores:
        return []

    arr = np.array(scores)
    mean = arr.mean()
    std = arr.std()

    if std == 0:
        return [0.0] * len(scores)

    return list((arr - mean) / std)


def rank_normalize(scores: list[float], k: int = 60) -> list[float]:
    """Normalize scores based on rank positions (RRF-style)."""
    if not scores:
        return []

    sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    ranks = [0] * len(scores)

    for rank, idx in enumerate(sorted_indices, 1):
        ranks[idx] = rank

    return [1.0 / (k + rank) for rank in ranks]


class ConfidenceCalibrator:
    """Calibrate confidence scores to true probabilities."""

    def __init__(self, method: str = "platt"):
        self.method = method
        self.calibrator = None
        self.is_fitted = False

    def fit(self, scores: np.ndarray, labels: np.ndarray):
        """Fit the calibrator on training data.

        Args:
            scores: Raw confidence scores.
            labels: Binary labels (1 = relevant, 0 = not relevant).
        """
        if self.method == "platt":
            self.calibrator = LogisticRegression()
            self.calibrator.fit(scores.reshape(-1, 1), labels)
        elif self.method == "isotonic":
            self.calibrator = IsotonicRegression(out_of_bounds="clip")
            self.calibrator.fit(scores, labels)
        else:
            raise ValueError(f"Unknown calibration method: {self.method}")

        self.is_fitted = True
        logger.info(f"Confidence calibrator fitted with {self.method} method")

    def calibrate(self, scores: np.ndarray) -> np.ndarray:
        """Calibrate raw scores to probabilities.

        Args:
            scores: Raw confidence scores.

        Returns:
            Calibrated probabilities.
        """
        if not self.is_fitted:
            logger.warning("Calibrator not fitted, returning raw scores")
            return scores

        if self.method == "platt":
            return self.calibrator.predict_proba(scores.reshape(-1, 1))[:, 1]
        elif self.method == "isotonic":
            return self.calibrator.predict(scores)

        return scores


class PipelineConfidence:
    """Compute end-to-end pipeline confidence scores."""

    def __init__(
        self,
        retrieval_weight: float = 0.3,
        reranking_weight: float = 0.5,
        generation_weight: float = 0.2,
    ):
        self.retrieval_weight = retrieval_weight
        self.reranking_weight = reranking_weight
        self.generation_weight = generation_weight

    def compute_retrieval_confidence(
        self,
        rrf_score: float,
        source_count: int,
    ) -> float:
        """Compute retrieval confidence from fusion score.

        Args:
            rrf_score: Reciprocal rank fusion score.
            source_count: Number of retrieval sources that found this result.

        Returns:
            Retrieval confidence between 0 and 1.
        """
        source_bonus = min(source_count / 3.0, 1.0) * 0.2
        return min(rrf_score + source_bonus, 1.0)

    def compute_reranking_confidence(
        self,
        weighted_score: float,
    ) -> float:
        """Compute re-ranking confidence from feature-weighted score.

        Args:
            weighted_score: Weighted sum of re-ranking features.

        Returns:
            Re-ranking confidence between 0 and 1.
        """
        return sigmoid_normalize(weighted_score, temperature=0.5)

    def compute_generation_confidence(
        self,
        llm_confidence: Optional[float] = None,
        citation_count: int = 0,
        max_citations: int = 5,
    ) -> float:
        """Compute generation confidence.

        Args:
            llm_confidence: LLM's self-reported confidence (if available).
            citation_count: Number of citations in the generated answer.
            max_citations: Expected maximum citations.

        Returns:
            Generation confidence between 0 and 1.
        """
        if llm_confidence is not None:
            citation_bonus = min(citation_count / max_citations, 1.0) * 0.2
            return min(llm_confidence + citation_bonus, 1.0)

        citation_score = min(citation_count / max_citations, 1.0)
        return citation_score

    def compute_pipeline_confidence(
        self,
        retrieval_confidence: float,
        reranking_confidence: float,
        generation_confidence: float,
    ) -> dict[str, float]:
        """Compute overall pipeline confidence.

        Args:
            retrieval_confidence: Confidence from retrieval stage.
            reranking_confidence: Confidence from re-ranking stage.
            generation_confidence: Confidence from generation stage.

        Returns:
            Dictionary with individual and overall confidence scores.
        """
        overall = (
            self.retrieval_weight * retrieval_confidence
            + self.reranking_weight * reranking_confidence
            + self.generation_weight * generation_confidence
        )

        return {
            "retrieval_confidence": round(retrieval_confidence, 4),
            "reranking_confidence": round(reranking_confidence, 4),
            "generation_confidence": round(generation_confidence, 4),
            "overall_confidence": round(overall, 4),
        }
