"""Tests for Qdrant Manager and Vector Operations."""
import unittest
import json
import tempfile
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.qdrant_manager import (
    BM25_INDEX_VERSION,
    QdrantManager,
    SearchResult,
    create_all_collections
)


class TestSearchResult(unittest.TestCase):
    """Test cases for SearchResult dataclass."""
    
    def test_creation(self):
        result = SearchResult(
            id="test123",
            text="dharma kṣetra",
            score=0.95,
            dataset_type="raw",
            verse_id="BhG 1.1",
            metadata={"key": "value"}
        )
        
        self.assertEqual(result.id, "test123")
        self.assertEqual(result.score, 0.95)
        self.assertEqual(result.dataset_type, "raw")
    
    def test_defaults(self):
        result = SearchResult(
            id="test",
            text="text",
            score=0.5,
            dataset_type="raw",
            verse_id=None,
            metadata={}
        )
        
        self.assertIsNone(result.verse_id)
        self.assertEqual(result.metadata, {})


class TestQdrantManager(unittest.TestCase):
    """Test cases for QdrantManager class."""
    
    @patch('src.qdrant_manager.QDRANT_AVAILABLE', False)
    def test_initialization_without_qdrant(self):
        manager = QdrantManager()
        self.assertFalse(manager._connected)
    
    @patch('src.qdrant_manager.QDRANT_AVAILABLE', True)
    @patch('src.qdrant_manager.QdrantClient')
    def test_connect_success(self, mock_client):
        manager = QdrantManager()
        mock_instance = Mock()
        mock_instance.get_collections.return_value = Mock(collections=[])
        mock_client.return_value = mock_instance
        
        result = manager.connect()
        
        self.assertGreaterEqual(mock_client.call_count, 1)
        self.assertTrue(result)
    
    @patch('src.qdrant_manager.QDRANT_AVAILABLE', True)
    @patch('src.qdrant_manager.QdrantClient')
    def test_create_collection(self, mock_client):
        manager = QdrantManager()
        mock_instance = Mock()
        mock_instance.get_collections.return_value = Mock(collections=[])
        mock_client.return_value = mock_instance
        manager.connect()
        
        manager.create_collection("test_collection")
        
        mock_instance.create_collection.assert_called_once()
    
    @patch('src.qdrant_manager.QDRANT_AVAILABLE', False)
    def test_search_returns_empty_without_qdrant(self):
        manager = QdrantManager()
        
        results = manager.search_dense(np.array([0.1] * 1024))
        self.assertEqual(results, [])
        
        results = manager.search_sparse({1: 0.5})
        self.assertEqual(results, [])
        
        results = manager.search_colbert(np.array([[0.1] * 128]))
        self.assertEqual(results, [])

    @patch('src.qdrant_manager.QDRANT_AVAILABLE', True)
    def test_search_dense_uses_collection_override(self):
        manager = QdrantManager()
        manager._connected = True
        manager.collection_exists = Mock(return_value=True)
        manager._client = Mock()
        manager._client.query_points.return_value = Mock(points=[])

        manager.search_dense(
            np.array([0.1] * 1024),
            top_k=3,
            collection_name="commentary_baladeva",
        )

        self.assertEqual(
            manager._client.query_points.call_args.kwargs["collection_name"],
            "commentary_baladeva",
        )

    @patch('src.qdrant_manager.QDRANT_AVAILABLE', True)
    def test_bm25_search_uses_collection_override(self):
        manager = QdrantManager()
        manager._connected = True
        manager.collection_exists = Mock(return_value=True)
        manager._client = Mock()
        manager._bm25_states["commentary_vishwanatha"] = {
            "term_freq": {"dharma": {"doc-1": 2}},
            "idf": {"dharma": 1.0},
            "total_docs": 1,
            "avg_doc_len": 10.0,
            "loaded": True,
        }
        manager._client.retrieve.return_value = [
            Mock(id="doc-1", payload={"text": "dharma", "dataset_type": "commentary", "verse_id": "BhG 1.1", "metadata": {}})
        ]

        results = manager.bm25_search(
            ["dharma"],
            top_k=1,
            collection_name="commentary_vishwanatha",
        )

        self.assertEqual(
            manager._client.retrieve.call_args.kwargs["collection_name"],
            "commentary_vishwanatha",
        )
        self.assertEqual(len(results), 1)


class TestBM25Search(unittest.TestCase):
    """Test cases for BM25 search functionality."""
    
    @patch('src.qdrant_manager.QDRANT_AVAILABLE', False)
    def test_bm25_search_without_qdrant(self):
        manager = QdrantManager()
        
        results = manager.bm25_search(["dharma", "kṣetra"])
        self.assertEqual(results, [])


    def test_stale_bm25_cache_without_metadata_is_ignored(self):
        manager = QdrantManager()

        with tempfile.TemporaryDirectory() as tmpdir:
            index_file = Path(tmpdir) / "bm25_index.json"
            index_file.write_text(
                json.dumps({
                    "term_freq": {"dharma": {"doc-1": 1}},
                    "idf": {"dharma": 1.0},
                    "doc_lengths": {"doc-1": 1},
                    "total_docs": 1,
                    "avg_doc_len": 1.0,
                }),
                encoding="utf-8",
            )

            loaded = manager._load_bm25_cache(index_file, "sansr_seg_lemma", expected_points=1)

        self.assertIsNone(loaded)

    def test_bm25_cache_loads_only_when_collection_count_matches(self):
        manager = QdrantManager()

        with tempfile.TemporaryDirectory() as tmpdir:
            index_file = Path(tmpdir) / "bm25_index.json"
            index_file.write_text(
                json.dumps({
                    "_meta": {
                        "bm25_index_version": BM25_INDEX_VERSION,
                        "collection_name": "sansr_seg_lemma",
                        "source_point_count": 1,
                    },
                    "term_freq": {"dharma": {"doc-1": 1}},
                    "idf": {"dharma": 1.0},
                    "doc_lengths": {"doc-1": 1},
                    "total_docs": 1,
                    "avg_doc_len": 1.0,
                }),
                encoding="utf-8",
            )

            valid = manager._load_bm25_cache(index_file, "sansr_seg_lemma", expected_points=1)
            stale = manager._load_bm25_cache(index_file, "sansr_seg_lemma", expected_points=2)

        self.assertIsNotNone(valid)
        self.assertIsNone(stale)


class TestCollectionStats(unittest.TestCase):
    """Test cases for collection statistics."""
    
    @patch('src.qdrant_manager.QDRANT_AVAILABLE', False)
    def test_stats_without_qdrant(self):
        manager = QdrantManager()
        
        stats = manager.get_collection_stats("test")
        self.assertEqual(stats, {})


class TestInsertEmbeddings(unittest.TestCase):
    """Test cases for embedding insertion."""
    
    @patch('src.qdrant_manager.QDRANT_AVAILABLE', False)
    def test_insert_without_qdrant(self):
        manager = QdrantManager()
        
        mock_embeddings = [Mock(
            id="test",
            text="test",
            dense_vector=np.array([0.1] * 1024),
            sparse_vector={1: 0.5},
            colbert_vectors=np.array([[0.1] * 128]),
            metadata={"dataset_type": "raw"}
        )]
        
        count = manager.insert_embeddings("test", mock_embeddings)
        self.assertEqual(count, 0)


class TestQdrantExactVerseLookup(unittest.TestCase):
    @patch('src.qdrant_manager.QDRANT_AVAILABLE', True)
    def test_search_by_verse_ids_uses_collection_override(self):
        manager = QdrantManager()
        manager._connected = True
        manager.collection_exists = Mock(return_value=True)
        manager._client = Mock()
        manager._client.scroll.return_value = ([
            Mock(
                id="doc-1",
                payload={
                    "text": "karma yoga",
                    "dataset_type": "seg_lemma",
                    "verse_id": "BhG 2.47",
                    "metadata": {"line_number": 1},
                },
            )
        ], None)

        results = manager.search_by_verse_ids(
            ["BhG 2.47"],
            collection_name="commentary_baladeva",
        )

        self.assertEqual(
            manager._client.scroll.call_args.kwargs["collection_name"],
            "commentary_baladeva",
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].metadata["retrieval_mode"], "verse_filter_exact")


class TestBM25VerseFilter(unittest.TestCase):
    @patch('src.qdrant_manager.QDRANT_AVAILABLE', True)
    def test_bm25_search_applies_verse_filter_to_retrieved_points(self):
        manager = QdrantManager()
        manager._connected = True
        manager.collection_exists = Mock(return_value=True)
        manager._client = Mock()
        manager._bm25_states["sansr_seg_lemma"] = {
            "term_freq": {"karma": {"doc-1": 2, "doc-2": 1}},
            "idf": {"karma": 1.0},
            "total_docs": 2,
            "avg_doc_len": 10.0,
            "loaded": True,
        }
        manager._client.retrieve.return_value = [
            Mock(id="doc-1", payload={"text": "karma", "dataset_type": "seg_lemma", "verse_id": "BhG 2.46", "metadata": {}}),
            Mock(id="doc-2", payload={"text": "karma", "dataset_type": "seg_lemma", "verse_id": "BhG 2.47", "metadata": {}}),
        ]

        results = manager.bm25_search(
            ["karma"],
            top_k=2,
            verse_filter={"verse_ids": ["BhG 2.47"]},
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].verse_id, "BhG 2.47")


if __name__ == "__main__":
    unittest.main()
