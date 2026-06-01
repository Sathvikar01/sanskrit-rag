"""Tests for NVIDIA Embedding Client."""
import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from pathlib import Path
import tempfile
import os

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embedding_client import (
    NVIDIAEmbeddingClient,
    EmbeddingResult,
    compute_l1_regularization,
    compute_l2_regularization,
    apply_regularization
)


class TestEmbeddingResult(unittest.TestCase):
    """Test cases for EmbeddingResult dataclass."""
    
    def test_creation(self):
        result = EmbeddingResult(
            id="test123",
            text="dharma kṣetra",
            dense_vector=np.random.randn(1024).astype(np.float32),
            sparse_vector={1: 0.5, 2: 0.3},
            colbert_vectors=np.random.randn(128, 128).astype(np.float32),
            metadata={"verse": "BhG 1.1"}
        )
        
        self.assertEqual(result.id, "test123")
        self.assertEqual(result.dense_vector.shape, (1024,))
        self.assertEqual(len(result.sparse_vector), 2)
        self.assertEqual(result.colbert_vectors.shape, (128, 128))
    
    def test_to_dict(self):
        result = EmbeddingResult(
            id="test",
            text="test text",
            dense_vector=np.array([0.1, 0.2]),
            sparse_vector={1: 0.5},
            colbert_vectors=np.array([[0.1, 0.2]]),
            metadata={}
        )
        
        d = result.to_dict()
        self.assertIn("id", d)
        self.assertIn("text", d)
        self.assertIn("dense_vector", d)
        self.assertIn("sparse_vector", d)
        self.assertIn("colbert_vectors", d)


class TestL1Regularization(unittest.TestCase):
    """Test cases for L1 regularization."""
    
    def test_compute_l1_zero_weights(self):
        weights = np.zeros(10)
        penalty = compute_l1_regularization(weights)
        self.assertEqual(penalty, 0.0)
    
    def test_compute_l1_positive_weights(self):
        weights = np.array([1.0, 2.0, 3.0])
        penalty = compute_l1_regularization(weights)
        expected = 1.0 + 2.0 + 3.0
        self.assertEqual(penalty, expected)
    
    def test_compute_l1_mixed_weights(self):
        weights = np.array([-1.0, 2.0, -3.0])
        penalty = compute_l1_regularization(weights)
        expected = abs(-1.0) + abs(2.0) + abs(-3.0)
        self.assertEqual(penalty, expected)
    
    def test_l1_reduces_scores(self):
        scores = np.array([1.0, 0.8, 0.9])
        weights = np.array([1.0, 0.8, 0.9])
        regularized = apply_regularization(scores, weights, l1_lambda=0.1, l2_lambda=0.0)
        
        l1_penalty = 0.1 * np.sum(np.abs(weights))
        expected = scores - l1_penalty
        
        np.testing.assert_array_almost_equal(regularized, expected)


class TestL2Regularization(unittest.TestCase):
    """Test cases for L2 regularization."""
    
    def test_compute_l2_zero_weights(self):
        weights = np.zeros(10)
        penalty = compute_l2_regularization(weights)
        self.assertEqual(penalty, 0.0)
    
    def test_compute_l2_positive_weights(self):
        weights = np.array([1.0, 2.0, 3.0])
        penalty = compute_l2_regularization(weights)
        expected = 1.0**2 + 2.0**2 + 3.0**2
        self.assertEqual(penalty, expected)
    
    def test_l2_smooths_distribution(self):
        weights = np.array([10.0, 0.1, 0.1])
        penalty = compute_l2_regularization(weights)
        
        weights_uniform = np.array([3.0, 3.0, 3.0])
        penalty_uniform = compute_l2_regularization(weights_uniform)
        
        self.assertGreater(penalty, penalty_uniform)
    
    def test_l2_reduces_scores(self):
        scores = np.array([1.0, 0.8, 0.9])
        weights = np.array([1.0, 0.8, 0.9])
        regularized = apply_regularization(scores, weights, l1_lambda=0.0, l2_lambda=0.01)
        
        l2_penalty = 0.01 * np.sum(weights ** 2)
        expected = scores - l2_penalty
        
        np.testing.assert_array_almost_equal(regularized, expected)


class TestCombinedRegularization(unittest.TestCase):
    """Test cases for combined L1/L2 regularization."""
    
    def test_combined_penalty(self):
        weights = np.array([1.0, 2.0, 3.0])
        scores = np.array([0.9, 0.8, 0.7])
        
        l1_penalty = np.sum(np.abs(weights))
        l2_penalty = np.sum(weights ** 2)
        
        combined = apply_regularization(scores, weights, l1_lambda=0.01, l2_lambda=0.001)
        
        expected_penalty = 0.01 * l1_penalty + 0.001 * l2_penalty
        expected = scores - expected_penalty
        
        np.testing.assert_array_almost_equal(combined, expected)
    
    def test_regularization_preserves_order(self):
        scores = np.array([0.9, 0.8, 0.7, 0.6])
        weights = np.ones(4) * 0.5
        
        regularized = apply_regularization(scores, weights, l1_lambda=0.01, l2_lambda=0.001)
        
        original_order = np.argsort(scores)[::-1]
        regularized_order = np.argsort(regularized)[::-1]
        
        np.testing.assert_array_equal(original_order, regularized_order)


class TestNVIDIAEmbeddingClient(unittest.TestCase):
    """Test cases for NVIDIAEmbeddingClient."""
    
    def test_sparse_embedding_generation(self):
        client = NVIDIAEmbeddingClient()
        texts = ["dharma kṣetra", "kuru kṣetra"]
        
        sparse_embeddings = client._get_sparse_embeddings(texts)
        
        self.assertEqual(len(sparse_embeddings), 2)
        self.assertIsInstance(sparse_embeddings[0], dict)
    
    def test_colbert_embedding_shape(self):
        client = NVIDIAEmbeddingClient()
        texts = ["dharma kṣetra kuru"]
        
        colbert_embeddings = client._get_colbert_embeddings(texts)
        
        self.assertEqual(len(colbert_embeddings), 1)
        self.assertEqual(len(colbert_embeddings[0]), 128)
        self.assertEqual(len(colbert_embeddings[0][0]), 128)
    
    def test_colbert_padding(self):
        client = NVIDIAEmbeddingClient()
        texts = ["dharma"]
        
        colbert_embeddings = client._get_colbert_embeddings(texts)
        
        self.assertEqual(len(colbert_embeddings[0]), 128)
    
    @patch('src.embedding_client.requests.Session')
    def test_get_cache_key(self, mock_session):
        client = NVIDIAEmbeddingClient()
        
        key1 = client._get_cache_key("test text")
        key2 = client._get_cache_key("test text")
        key3 = client._get_cache_key("different text")
        query_key = client._get_cache_key("test text", input_type="query")
        
        self.assertEqual(key1, key2)
        self.assertNotEqual(key1, key3)
        self.assertNotEqual(key1, query_key)

    def test_cache_key_separates_embedding_backends(self):
        local_client = NVIDIAEmbeddingClient(backend="local", local_model_name="local-test-model")
        nvidia_client = NVIDIAEmbeddingClient(backend="nvidia", model="nvidia-test-model")

        self.assertNotEqual(
            local_client._get_cache_key("test text", input_type="query"),
            nvidia_client._get_cache_key("test text", input_type="query"),
        )

    def test_local_backend_uses_sentence_transformer_dense_embeddings(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            client = NVIDIAEmbeddingClient(
                backend="local",
                local_model_name="fake-local-model",
                local_batch_size=2,
                cache_dir=tmpdir,
            )
            fake_model = Mock()
            fake_model.encode.return_value = np.ones((2, 1024), dtype=np.float32)
            texts = [f"dharma yoga {Path(tmpdir).name}", f"karma yoga {Path(tmpdir).name}"]

            with patch.object(client, "_get_local_model", return_value=fake_model):
                results = client.get_embeddings_batch(texts, input_type="query")

        fake_model.encode.assert_called_once()
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].dense_vector.shape, (1024,))
        self.assertEqual(results[0].metadata["embedding_backend"], "local")
        self.assertEqual(results[0].metadata["embedding_model"], "fake-local-model")
        self.assertEqual(results[0].metadata["embedding_cache_version"], "v2")

    def test_stale_cache_from_other_backend_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            client = NVIDIAEmbeddingClient(
                backend="local",
                local_model_name="fake-local-model",
                cache_dir=tmpdir,
            )
            stale = EmbeddingResult(
                id="old",
                text="test text",
                dense_vector=np.ones(1024, dtype=np.float32),
                sparse_vector={1: 1.0},
                colbert_vectors=np.zeros((128, 128), dtype=np.float32),
                metadata={
                    "embedding_cache_version": "v2",
                    "embedding_backend": "nvidia",
                    "embedding_model": "nvidia/nv-embedqa-e5-v5",
                    "embedding_input_type": "query",
                    "dense_dim": 1024,
                },
            )
            cache_key = client._get_cache_key("test text", input_type="query")
            client._save_to_cache(stale, cache_key)

            self.assertIsNone(client._load_from_cache(cache_key, input_type="query"))

    def test_local_colbert_path_does_not_call_nvidia(self):
        client = NVIDIAEmbeddingClient(backend="local", local_model_name="fake-local-model")
        client.session.post = Mock(side_effect=AssertionError("NVIDIA should not be called"))

        colbert = client._get_colbert_embeddings(["dharma yoga"])

        client.session.post.assert_not_called()
        self.assertEqual(len(colbert), 1)
        self.assertEqual(len(colbert[0]), 128)

    def test_embed_query_requests_query_embedding(self):
        client = NVIDIAEmbeddingClient()
        result = EmbeddingResult(
            id="query",
            text="karma yoga",
            dense_vector=np.ones(1024, dtype=np.float32),
            sparse_vector={1: 1.0},
            colbert_vectors=np.zeros((128, 128), dtype=np.float32),
            metadata={},
        )

        with patch.object(client, "get_embeddings_batch", return_value=[result]) as mocked:
            embedded = client.embed_query("karma yoga")

        mocked.assert_called_once_with(["karma yoga"], input_type="query")
        self.assertTrue(embedded.metadata["dense_available"])

    def test_embed_query_falls_back_to_local_sparse_on_api_failure(self):
        client = NVIDIAEmbeddingClient()

        with patch.object(client, "get_embeddings_batch", side_effect=Exception("403 forbidden")):
            result = client.embed_query("karma yoga")

        self.assertEqual(result.text, "karma yoga")
        self.assertEqual(result.id, client._get_cache_key("karma yoga", input_type="query")[:16])
        self.assertEqual(result.dense_vector.shape, (1024,))
        self.assertFalse(result.metadata["dense_available"])
        self.assertEqual(result.metadata["embedding_fallback"], "local_sparse_only")
        self.assertTrue(result.sparse_vector)


class TestEmbeddingCaching(unittest.TestCase):
    """Test cases for embedding caching."""
    
    def test_cache_key_consistency(self):
        client = NVIDIAEmbeddingClient()
        
        text = "dharma kṣetra"
        key1 = client._get_cache_key(text)
        key2 = client._get_cache_key(text)
        
        self.assertEqual(key1, key2)
    
    def test_cache_key_different_for_different_texts(self):
        client = NVIDIAEmbeddingClient()
        
        key1 = client._get_cache_key("dharma")
        key2 = client._get_cache_key("karma")
        
        self.assertNotEqual(key1, key2)
    
    def test_cache_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            client = NVIDIAEmbeddingClient(cache_dir=tmpdir)
            
            result = EmbeddingResult(
                id="test",
                text="test text",
                dense_vector=np.ones(1024, dtype=np.float32),
                sparse_vector={1: 0.5},
                colbert_vectors=np.zeros((128, 128), dtype=np.float32),
                metadata={
                    "embedding_cache_version": "v2",
                    "embedding_backend": client.backend,
                    "embedding_model": client.model,
                    "embedding_input_type": "passage",
                    "normalize_embeddings": client.local_normalize_embeddings,
                    "dense_dim": 1024,
                }
            )
            
            cache_key = "test_key"
            client._save_to_cache(result, cache_key)
            
            loaded = client._load_from_cache(cache_key, input_type="passage")
            
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.id, "test")


if __name__ == "__main__":
    unittest.main()
