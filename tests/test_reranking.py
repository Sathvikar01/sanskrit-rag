"""Tests for SRAG re-ranking modules."""


from src.reranking.feature_extractors import (
    ReRankingFeatures,
    extract_lemma_overlap_score,
    extract_compound_match_score,
    extract_concept_overlap_score,
    extract_graph_centrality_score,
    tokenize_iast,
    extract_lemmas_from_text,
    normalize_features_minmax,
    normalize_features_l2,
    normalize_features_zscore,
    normalize_feature_matrix,
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


class TestFeatureNormalization:
    """Test feature normalization methods."""

    def _create_test_features(self):
        """Create test features with different scales."""
        return [
            ReRankingFeatures(score_vector=0.9, score_bm25=10.0, score_lemma=0.8),
            ReRankingFeatures(score_vector=0.3, score_bm25=2.0, score_lemma=0.1),
            ReRankingFeatures(score_vector=0.6, score_bm25=5.0, score_lemma=0.5),
        ]

    def test_minmax_normalizes_to_0_1(self):
        features = self._create_test_features()
        normalized = normalize_features_minmax(features)
        for f in normalized:
            for val in f.to_vector():
                assert 0.0 <= val <= 1.0

    def test_minmax_preserves_order(self):
        features = self._create_test_features()
        normalized = normalize_features_minmax(features)
        # Original vector scores: 0.9, 0.3, 0.6 -> normalized: 1.0, 0.0, 0.5
        assert normalized[0].score_vector > normalized[2].score_vector > normalized[1].score_vector

    def test_minmax_handles_constant_feature(self):
        features = [
            ReRankingFeatures(score_vector=0.5, score_bm25=3.0),
            ReRankingFeatures(score_vector=0.5, score_bm25=7.0),
        ]
        normalized = normalize_features_minmax(features)
        # Constant feature: (0.5 - 0.5) / 1.0 = 0.0
        assert normalized[0].score_vector == 0.0
        assert normalized[1].score_vector == 0.0

    def test_l2_normalizes_to_unit_length(self):
        features = self._create_test_features()
        normalized = normalize_features_l2(features)
        import math
        for f in normalized:
            vec = f.to_vector()
            norm = math.sqrt(sum(v ** 2 for v in vec))
            assert abs(norm - 1.0) < 0.01

    def test_l2_handles_zero_vector(self):
        features = [ReRankingFeatures()]
        normalized = normalize_features_l2(features)
        # Zero vector should remain zero
        assert all(v == 0.0 for v in normalized[0].to_vector())

    def test_zscore_normalizes(self):
        features = self._create_test_features()
        normalized = normalize_features_zscore(features)
        # Z-score normalized features should be non-negative (shifted)
        for f in normalized:
            for val in f.to_vector():
                assert val >= 0.0

    def test_zscore_handles_constant_feature(self):
        features = [
            ReRankingFeatures(score_vector=0.5, score_bm25=3.0),
            ReRankingFeatures(score_vector=0.5, score_bm25=7.0),
        ]
        normalized = normalize_features_zscore(features)
        # Constant feature should be normalized to 0 (shifted from negative)
        assert normalized[0].score_vector == normalized[1].score_vector

    def test_normalize_feature_matrix_none(self):
        features = self._create_test_features()
        result = normalize_feature_matrix(features, "none")
        assert result is features

    def test_normalize_feature_matrix_minmax(self):
        features = self._create_test_features()
        result = normalize_feature_matrix(features, "minmax")
        assert len(result) == len(features)
        for f in result:
            for val in f.to_vector():
                assert 0.0 <= val <= 1.0

    def test_normalize_feature_matrix_l2(self):
        features = self._create_test_features()
        result = normalize_feature_matrix(features, "l2")
        assert len(result) == len(features)
        import math
        for f in result:
            vec = f.to_vector()
            norm = math.sqrt(sum(v ** 2 for v in vec))
            assert abs(norm - 1.0) < 0.01

    def test_normalize_feature_matrix_zscore(self):
        features = self._create_test_features()
        result = normalize_feature_matrix(features, "zscore")
        assert len(result) == len(features)
        for f in result:
            for val in f.to_vector():
                assert val >= 0.0

    def test_normalize_empty_list(self):
        assert normalize_features_minmax([]) == []
        assert normalize_features_l2([]) == []
        assert normalize_features_zscore([]) == []

    def test_normalize_single_candidate(self):
        features = [ReRankingFeatures(score_vector=0.9, score_bm25=10.0)]
        normalized = normalize_features_minmax(features)
        # Single candidate: all features should be 0.0 (min == max, range = 1, (v - min) / 1 = 0)
        assert normalized[0].score_vector == 0.0
