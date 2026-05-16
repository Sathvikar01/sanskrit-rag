"""Tests for SRAG re-ranking modules."""

import pytest

from src.reranking.feature_extractors import (
    ReRankingFeatures,
    extract_lemma_overlap_score,
    extract_compound_match_score,
    extract_concept_overlap_score,
    extract_graph_centrality_score,
    tokenize_iast,
    extract_lemmas_from_text,
)
from src.reranking.confidence import (
    min_max_normalize,
    sigmoid_normalize,
    PipelineConfidence,
)


class TestFeatureExtractors:
    """Test feature extraction functions."""

    def test_lemma_overlap_full(self):
        query = {"dharma", "karma"}
        doc = {"dharma", "karma", "yoga", "bhakti"}
        score = extract_lemma_overlap_score(query, doc)
        assert score == 1.0

    def test_lemma_overlap_partial(self):
        query = {"dharma", "karma", "moksha"}
        doc = {"dharma", "yoga"}
        score = extract_lemma_overlap_score(query, doc)
        assert abs(score - 1 / 3) < 0.01

    def test_lemma_overlap_empty(self):
        score = extract_lemma_overlap_score(set(), {"dharma"})
        assert score == 0.0

    def test_compound_match(self):
        query = ["dharma", "kṣetra"]
        doc = ["dharma", "kṣetra", "kuru"]
        score = extract_compound_match_score(query, doc)
        assert score == 1.0

    def test_concept_overlap(self):
        query = {"dharma", "karma"}
        doc = {"dharma", "yoga"}
        score = extract_concept_overlap_score(query, doc)
        assert score == 0.5

    def test_graph_centrality(self):
        score = extract_graph_centrality_score(50, max_degree=100)
        assert score == 0.5

    def test_tokenize_iast(self):
        tokens = tokenize_iast("dharma-kṣetre kuru-kṣetre")
        assert "dharma" in tokens or "kṣetre" in tokens

    def test_extract_lemmas(self):
        lemmas = extract_lemmas_from_text("dharma-kṣetre kuru-kṣetre")
        assert "dharma" in lemmas
        assert "kṣetre" in lemmas


class TestConfidence:
    """Test confidence scoring functions."""

    def test_min_max_normalize(self):
        scores = [0.1, 0.5, 0.9]
        normalized = min_max_normalize(scores)
        assert normalized[0] == 0.0
        assert normalized[-1] == 1.0

    def test_sigmoid_normalize(self):
        score = sigmoid_normalize(0)
        assert score == 0.5

        score = sigmoid_normalize(10)
        assert score > 0.9

        score = sigmoid_normalize(-10)
        assert score < 0.1

    def test_pipeline_confidence(self):
        pc = PipelineConfidence()
        result = pc.compute_pipeline_confidence(0.8, 0.9, 0.7)
        assert "overall_confidence" in result
        assert 0 <= result["overall_confidence"] <= 1

    def test_retrieval_confidence(self):
        pc = PipelineConfidence()
        conf = pc.compute_retrieval_confidence(0.5, 3)
        assert conf > 0.5


class TestReRankingFeatures:
    """Test ReRankingFeatures dataclass."""

    def test_to_vector(self):
        features = ReRankingFeatures(
            score_vector=0.9,
            score_lemma=0.8,
        )
        vec = features.to_vector()
        assert len(vec) == 9
        assert vec[0] == 0.9

    def test_to_dict(self):
        features = ReRankingFeatures()
        d = features.to_dict()
        assert "score_vector" in d
        assert "score_lemma" in d
