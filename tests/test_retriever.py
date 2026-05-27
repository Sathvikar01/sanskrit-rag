"""Tests for Hybrid Retriever with L1/L2 Regularization."""
import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retriever import (
    HybridRetriever,
    HybridSearchResult,
    RerankerWeights,
    RegularizedRetriever
)
from src.embedding_client import EmbeddingResult
from src.neo4j_manager import SearchResult


class TestHybridSearchResult(unittest.TestCase):
    """Test cases for HybridSearchResult dataclass."""
    
    def test_creation(self):
        result = HybridSearchResult(
            id="test123",
            text="dharma kṣetra",
            final_score=0.95,
            dense_score=0.92,
            sparse_score=0.88,
            colbert_score=0.90,
            bm25_score=0.85,
            dataset_type="raw",
            verse_id="BhG 1.1",
            metadata={"key": "value"}
        )
        
        self.assertEqual(result.id, "test123")
        self.assertEqual(result.final_score, 0.95)
        self.assertEqual(result.dense_score, 0.92)
    
    def test_to_dict(self):
        result = HybridSearchResult(
            id="test",
            text="text",
            final_score=0.9,
            dataset_type="raw"
        )
        
        d = result.to_dict()
        
        self.assertIn("id", d)
        self.assertIn("text", d)
        self.assertIn("final_score", d)
        self.assertIn("dense_score", d)
        self.assertIn("sparse_score", d)
        self.assertIn("colbert_score", d)
        self.assertIn("bm25_score", d)
        self.assertIn("exact_score", d)


class TestRerankerWeights(unittest.TestCase):
    """Test cases for RerankerWeights."""
    
    def test_default_weights(self):
        weights = RerankerWeights()
        
        self.assertEqual(weights.dense, 1.0)
        self.assertEqual(weights.sparse, 0.8)
        self.assertEqual(weights.colbert, 0.9)
        self.assertEqual(weights.bm25, 0.7)
    
    def test_to_array(self):
        weights = RerankerWeights(dense=1.0, sparse=0.8, colbert=0.9, bm25=0.7)
        
        arr = weights.to_array()
        
        np.testing.assert_array_equal(arr, np.array([1.0, 0.8, 0.9, 0.7]))
    
    def test_normalize(self):
        weights = RerankerWeights(dense=2.0, sparse=2.0, colbert=0.0, bm25=0.0)
        
        normalized = weights.normalize()
        
        self.assertEqual(normalized.dense, 0.5)
        self.assertEqual(normalized.sparse, 0.5)
        self.assertEqual(normalized.colbert, 0.0)
        self.assertEqual(normalized.bm25, 0.0)
    
    def test_normalize_zero_weights(self):
        weights = RerankerWeights(dense=0.0, sparse=0.0, colbert=0.0, bm25=0.0)
        
        normalized = weights.normalize()
        
        self.assertEqual(normalized.dense, 0.0)


class TestHybridRetriever(unittest.TestCase):
    """Test cases for HybridRetriever class."""
    
    def setUp(self):
        self.mock_embedder = Mock()
        self.mock_qdrant = Mock()
        self.mock_neo4j = Mock()
        self.retriever = HybridRetriever(
            embedding_client=self.mock_embedder,
            qdrant_manager=self.mock_qdrant,
            neo4j_manager=self.mock_neo4j
        )
    
    def test_initialization(self):
        self.assertIsNotNone(self.retriever.embedding_client)
        self.assertIsNotNone(self.retriever.qdrant_manager)
        self.assertEqual(self.retriever.l1_lambda, 0.01)
        self.assertEqual(self.retriever.l2_lambda, 0.001)
    
    def test_embed_query(self):
        mock_result = EmbeddingResult(
            id="query",
            text="dharma",
            dense_vector=np.array([0.1] * 1024),
            sparse_vector={1: 0.5},
            colbert_vectors=np.array([[0.1] * 128] * 128),
            metadata={}
        )
        self.mock_embedder.embed_query.return_value = mock_result
        
        result = self.retriever.embed_query("dharma")
        
        self.mock_embedder.embed_query.assert_called_once_with("dharma")
        self.assertEqual(result.text, "dharma")
    
    def test_rrf_fusion(self):
        result1 = [SearchResult(id="a", text="t1", score=0.9, dataset_type="raw", verse_id=None, metadata={}),
                   SearchResult(id="b", text="t2", score=0.8, dataset_type="raw", verse_id=None, metadata={})]
        result2 = [SearchResult(id="b", text="t2", score=0.95, dataset_type="raw", verse_id=None, metadata={}),
                   SearchResult(id="c", text="t3", score=0.7, dataset_type="raw", verse_id=None, metadata={})]
        
        weights = np.array([1.0, 1.0])
        fused = self.retriever.rrf_fusion([result1, result2], weights, k=60)
        
        self.assertIn("b", fused)
        self.assertGreater(fused["b"], 0)
    
    def test_normalize_scores(self):
        scores = {"a": 0.9, "b": 0.5, "c": 0.1}
        
        normalized = self.retriever.normalize_scores(scores)
        
        self.assertAlmostEqual(normalized["a"], 1.0, places=2)
        self.assertAlmostEqual(normalized["c"], 0.0, places=2)
    
    def test_normalize_scores_empty(self):
        normalized = self.retriever.normalize_scores({})
        self.assertEqual(normalized, {})
    
    def test_normalize_scores_all_same(self):
        scores = {"a": 0.5, "b": 0.5}
        
        normalized = self.retriever.normalize_scores(scores)
        
        self.assertEqual(normalized["a"], 0.5)
        self.assertEqual(normalized["b"], 0.5)

    def test_cross_db_search_skips_dense_when_query_embedding_falls_back(self):
        fallback_embedding = EmbeddingResult(
            id="query",
            text="karma yoga",
            dense_vector=np.zeros(1024, dtype=np.float32),
            sparse_vector={1: 1.0, 2: 0.5},
            colbert_vectors=np.zeros((128, 128), dtype=np.float32),
            metadata={"dense_available": False},
        )
        qdrant = Mock()
        qdrant.search_sparse.return_value = [
            SearchResult(
                id="doc-1",
                text="karma-yoga text",
                score=0.8,
                dataset_type="seg_lemma",
                verse_id="BhG 2.47",
                metadata={},
            )
        ]
        qdrant.search_by_verse_ids.return_value = []
        qdrant.bm25_search.return_value = [
            SearchResult(
                id="doc-1",
                text="karma-yoga text",
                score=0.7,
                dataset_type="seg_lemma",
                verse_id="BhG 2.47",
                metadata={},
            )
        ]
        qdrant.search_dense = Mock(side_effect=AssertionError("dense search should be skipped"))

        retriever = HybridRetriever(
            embedding_client=Mock(embed_query=Mock(return_value=fallback_embedding)),
            qdrant_manager=qdrant,
            neo4j_manager=None,
        )

        results = retriever.cross_db_rrf_search("karma yoga", top_k=5)

        qdrant.search_sparse.assert_called_once()
        qdrant.bm25_search.assert_called_once()
        qdrant.search_dense.assert_not_called()
        self.assertEqual(len(results), 1)
        self.assertTrue(results[0].metadata["sources"]["qdrant"])

    def test_cross_db_search_uses_exact_qdrant_verse_lookup_when_filter_present(self):
        fallback_embedding = EmbeddingResult(
            id="query",
            text="Explain BG 2.47",
            dense_vector=np.zeros(1024, dtype=np.float32),
            sparse_vector={1: 1.0},
            colbert_vectors=np.zeros((128, 128), dtype=np.float32),
            metadata={"dense_available": False},
        )
        qdrant = Mock()
        qdrant.search_dense.return_value = []
        qdrant.search_sparse.return_value = []
        qdrant.bm25_search.return_value = []
        qdrant.search_by_verse_ids.return_value = [
            SearchResult(
                id="doc-47",
                text="karma yoga exact verse chunk",
                score=1.0,
                dataset_type="seg_lemma",
                verse_id="BhG 2.47",
                metadata={"retrieval_mode": "verse_filter_exact"},
            )
        ]

        retriever = HybridRetriever(
            embedding_client=Mock(embed_query=Mock(return_value=fallback_embedding)),
            qdrant_manager=qdrant,
            neo4j_manager=None,
        )

        results = retriever.cross_db_rrf_search("Explain BG 2.47", top_k=5)

        qdrant.search_by_verse_ids.assert_called_once_with(["BhG 2.47"], top_k=10)
        self.assertEqual(len(results), 1)
        self.assertGreater(results[0].exact_score, 0)
        self.assertTrue(results[0].metadata["sources"]["qdrant"])
        self.assertIn("exact_verse", results[0].metadata["qdrant_modes"])

    def test_qdrant_collection_override_is_threaded_through_hybrid_search(self):
        fallback_embedding = EmbeddingResult(
            id="query",
            text="karma yoga",
            dense_vector=np.zeros(1024, dtype=np.float32),
            sparse_vector={1: 1.0},
            colbert_vectors=np.zeros((128, 128), dtype=np.float32),
            metadata={"dense_available": False},
        )
        qdrant = Mock()
        qdrant.search_sparse.return_value = [
            SearchResult(
                id="doc-1",
                text="karma yoga",
                score=0.8,
                dataset_type="raw",
                verse_id="BhG 2.47",
                metadata={},
            )
        ]
        qdrant.bm25_search.return_value = []
        qdrant.search_by_verse_ids.return_value = []

        retriever = HybridRetriever(
            embedding_client=Mock(embed_query=Mock(return_value=fallback_embedding)),
            qdrant_manager=qdrant,
            neo4j_manager=None,
        )

        retriever.hybrid_search("karma yoga", "sansr_raw", top_k=5)

        self.assertEqual(qdrant.search_sparse.call_args.kwargs["collection_name"], "sansr_raw")
        self.assertEqual(qdrant.bm25_search.call_args.kwargs["collection_name"], "sansr_raw")

    def test_cross_db_search_fuses_same_verse_across_qdrant_modes(self):
        fallback_embedding = EmbeddingResult(
            id="query",
            text="karma yoga",
            dense_vector=np.zeros(1024, dtype=np.float32),
            sparse_vector={1: 1.0},
            colbert_vectors=np.zeros((128, 128), dtype=np.float32),
            metadata={"dense_available": False},
        )
        qdrant = Mock()
        qdrant.search_sparse.return_value = [
            SearchResult(
                id="sparse-doc",
                text="sparse evidence",
                score=0.8,
                dataset_type="seg_lemma",
                verse_id="BhG 2.47",
                metadata={},
            )
        ]
        qdrant.bm25_search.return_value = [
            SearchResult(
                id="bm25-doc",
                text="bm25 evidence",
                score=0.9,
                dataset_type="seg_lemma",
                verse_id="BhG 2.47",
                metadata={},
            )
        ]
        qdrant.search_by_verse_ids.return_value = []

        retriever = HybridRetriever(
            embedding_client=Mock(embed_query=Mock(return_value=fallback_embedding)),
            qdrant_manager=qdrant,
            neo4j_manager=None,
        )

        results = retriever.cross_db_rrf_search("karma yoga", top_k=5)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].verse_id, "BhG 2.47")
        self.assertGreater(results[0].sparse_score, 0)
        self.assertGreater(results[0].bm25_score, 0)
        self.assertEqual(results[0].metadata["fusion_level"], "verse")


class TestL1RegularizationInRetriever(unittest.TestCase):
    """Test L1 regularization in retriever context."""
    
    def setUp(self):
        self.mock_embedder = Mock()
        self.mock_qdrant = Mock()
        self.mock_neo4j = Mock()
        self.retriever = HybridRetriever(
            embedding_client=self.mock_embedder,
            qdrant_manager=self.mock_qdrant,
            neo4j_manager=self.mock_neo4j,
            l1_lambda=0.05
        )
    
    def test_l1_regularization_applied(self):
        scores = np.array([0.9, 0.8, 0.7])
        
        regularized = self.retriever.apply_l1_regularization(scores)
        
        l1_penalty = 0.05 * np.sum(np.abs(self.retriever.weights.to_array()))
        expected = scores - l1_penalty
        
        np.testing.assert_array_almost_equal(regularized, expected)
    
    def test_l1_penalty_reduces_scores(self):
        scores = np.array([0.9, 0.8, 0.7])
        
        regularized = self.retriever.apply_l1_regularization(scores)
        
        self.assertTrue(np.all(regularized < scores))


class TestL2RegularizationInRetriever(unittest.TestCase):
    """Test L2 regularization in retriever context."""
    
    def setUp(self):
        self.mock_embedder = Mock()
        self.mock_qdrant = Mock()
        self.mock_neo4j = Mock()
        self.retriever = HybridRetriever(
            embedding_client=self.mock_embedder,
            qdrant_manager=self.mock_qdrant,
            neo4j_manager=self.mock_neo4j,
            l2_lambda=0.01
        )
    
    def test_l2_regularization_applied(self):
        scores = np.array([0.9, 0.8, 0.7])
        
        regularized = self.retriever.apply_l2_regularization(scores)
        
        l2_penalty = 0.01 * np.sum(self.retriever.weights.to_array() ** 2)
        expected = scores - l2_penalty
        
        np.testing.assert_array_almost_equal(regularized, expected)
    
    def test_l2_penalty_smaller_than_l1(self):
        scores = np.array([0.9, 0.8, 0.7])
        
        l1_result = HybridRetriever(
            self.mock_embedder, self.mock_qdrant, l1_lambda=0.01
        ).apply_l1_regularization(scores)
        
        l2_result = HybridRetriever(
            self.mock_embedder, self.mock_qdrant, l2_lambda=0.01
        ).apply_l2_regularization(scores)
        
        self.assertTrue(np.all(l2_result > l1_result))


class TestCombinedRegularization(unittest.TestCase):
    """Test combined L1/L2 regularization."""
    
    def setUp(self):
        self.mock_embedder = Mock()
        self.mock_qdrant = Mock()
        self.mock_neo4j = Mock()
        self.retriever = HybridRetriever(
            embedding_client=self.mock_embedder,
            qdrant_manager=self.mock_qdrant,
            neo4j_manager=self.mock_neo4j,
            l1_lambda=0.02,
            l2_lambda=0.005
        )
    
    def test_combined_reduces_more_than_individual(self):
        scores = np.array([0.9, 0.8, 0.7])
        weights = self.retriever.weights.to_array()
        
        combined = self.retriever.apply_combined_regularization(scores, weights)
        l1_only = self.retriever.apply_l1_regularization(scores)
        l2_only = self.retriever.apply_l2_regularization(scores)
        
        l1_penalty = self.retriever.l1_lambda * np.sum(np.abs(weights))
        l2_penalty = self.retriever.l2_lambda * np.sum(weights ** 2)
        combined_penalty = l1_penalty + l2_penalty
        
        self.assertGreater(combined_penalty, l1_penalty)
        self.assertGreater(combined_penalty, l2_penalty)


class TestRegularizedRetriever(unittest.TestCase):
    """Test cases for RegularizedRetriever with adaptive tuning."""
    
    def setUp(self):
        self.mock_embedder = Mock()
        self.mock_qdrant = Mock()
        self.mock_neo4j = Mock()
        self.retriever = RegularizedRetriever(
            embedding_client=self.mock_embedder,
            qdrant_manager=self.mock_qdrant,
            neo4j_manager=self.mock_neo4j,
            l1_lambda=0.01,
            l2_lambda=0.001,
            adaptive=True
        )
    
    def test_adaptive_initialization(self):
        self.assertTrue(self.retriever.adaptive)
        self.assertEqual(len(self.retriever._score_history), 0)
    
    def test_update_regularization_low_variance(self):
        for i in range(15):
            self.retriever._score_history.append({
                'l1_lambda': 0.01,
                'l2_lambda': 0.001,
                'performance': 0.8 + 0.01 * (i % 3)
            })
        
        initial_l1 = self.retriever.l1_lambda
        initial_l2 = self.retriever.l2_lambda
        
        self.retriever.update_regularization({}, 0.85)
        
        self.assertLess(self.retriever.l1_lambda, initial_l1)
        self.assertLess(self.retriever.l2_lambda, initial_l2)
    
    def test_update_regularization_high_variance(self):
        for i in range(15):
            self.retriever._score_history.append({
                'l1_lambda': 0.01,
                'l2_lambda': 0.001,
                'performance': 0.5 + 0.2 * np.sin(i)
            })
        
        initial_l1 = self.retriever.l1_lambda
        initial_l2 = self.retriever.l2_lambda
        
        self.retriever.update_regularization({}, 0.6)
        
        self.assertGreater(self.retriever.l1_lambda, initial_l1)
        self.assertGreater(self.retriever.l2_lambda, initial_l2)
    
    def test_regularization_params_clipped(self):
        self.retriever.l1_lambda = 0.5
        self.retriever.l2_lambda = 0.5

        for _ in range(11):
            self.retriever.update_regularization({}, 0.7)

        self.assertLessEqual(self.retriever.l1_lambda, 0.1)
        self.assertLessEqual(self.retriever.l2_lambda, 0.01)
    
    def test_get_regularization_params(self):
        params = self.retriever.get_regularization_params()
        
        self.assertIn('l1_lambda', params)
        self.assertIn('l2_lambda', params)
        self.assertEqual(params['l1_lambda'], 0.01)
        self.assertEqual(params['l2_lambda'], 0.001)


class TestGetRetrievalStats(unittest.TestCase):
    """Test cases for retrieval statistics."""
    
    def setUp(self):
        self.mock_embedder = Mock()
        self.mock_qdrant = Mock()
        self.mock_neo4j = Mock()
        self.retriever = HybridRetriever(
            embedding_client=self.mock_embedder,
            qdrant_manager=self.mock_qdrant,
            neo4j_manager=self.mock_neo4j
        )
    
    def test_stats_empty_results(self):
        stats = self.retriever.get_retrieval_stats([])
        
        self.assertEqual(stats, {})
    
    def test_stats_calculated(self):
        results = [
            HybridSearchResult(id="1", text="t1", final_score=0.9, dense_score=0.9, sparse_score=0.8, colbert_score=0.85, bm25_score=0.75),
            HybridSearchResult(id="2", text="t2", final_score=0.8, dense_score=0.85, sparse_score=0.75, colbert_score=0.8, bm25_score=0.7)
        ]
        
        stats = self.retriever.get_retrieval_stats(results)
        
        self.assertEqual(stats["total_results"], 2)
        self.assertAlmostEqual(stats["avg_final_score"], 0.85, places=2)
        self.assertIn("avg_dense_score", stats)
        self.assertIn("avg_sparse_score", stats)
        self.assertIn("avg_colbert_score", stats)
        self.assertIn("avg_bm25_score", stats)


if __name__ == "__main__":
    unittest.main()
