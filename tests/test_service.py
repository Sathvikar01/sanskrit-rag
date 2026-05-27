"""Tests for SansRAG service initialization behavior."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embedding_client import EmbeddingResult
from src.service import SansRAGService


class TestSansRAGServiceNeo4jBootstrap(unittest.TestCase):
    def test_init_neo4j_store_bootstraps_from_cached_embeddings_when_graph_empty(self):
        service = SansRAGService()
        service.neo4j_ok = True
        service.neo4j = Mock()
        service.neo4j.get_collection_stats.return_value = {"chunk_count": 0}

        cached_embedding = EmbeddingResult(
            id="chunk-1",
            text="lemma morph text",
            dense_vector=np.zeros(1024, dtype=np.float32),
            sparse_vector={1: 1.0},
            colbert_vectors=np.zeros((128, 128), dtype=np.float32),
            metadata={"dataset_type": "lemma_morph"},
        )
        service._load_cached_embeddings_for_dataset = Mock(return_value=[cached_embedding])

        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir) / "dataset.lemma-morphosyntax.xml"
            dataset_path.write_text("<TEI/>", encoding="utf-8")

            with patch("src.service.ROOT_DIR", Path(tmpdir)):
                service._init_neo4j_store()

        service.neo4j.create_schema.assert_called_once_with(drop_if_exists=False)
        service.neo4j.insert_embeddings.assert_called_once_with(
            [cached_embedding],
            batch_size=100,
            show_progress=False,
        )


class TestSansRAGServiceQdrantBootstrap(unittest.TestCase):
    def test_init_qdrant_store_bootstraps_from_cached_embeddings_when_collection_missing(self):
        service = SansRAGService()
        service.qdrant_ok = True
        service.qdrant = Mock()
        service.qdrant.collection_exists.return_value = False

        cached_embedding = EmbeddingResult(
            id="chunk-1",
            text="seg lemma text",
            dense_vector=np.zeros(1024, dtype=np.float32),
            sparse_vector={1: 1.0},
            colbert_vectors=np.zeros((128, 128), dtype=np.float32),
            metadata={"dataset_type": "seg_lemma"},
        )
        service._load_cached_embeddings_for_dataset = Mock(return_value=[cached_embedding])

        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir) / "dataset.segmentation-lemma.xml"
            dataset_path.write_text("<TEI/>", encoding="utf-8")

            with patch("src.service.ROOT_DIR", Path(tmpdir)):
                service._init_qdrant_store()

        service.qdrant.create_collection.assert_called_once_with("sansr_seg_lemma", drop_if_exists=False)
        service.qdrant.insert_embeddings.assert_called_once_with(
            "sansr_seg_lemma",
            [cached_embedding],
            batch_size=256,
            show_progress=False,
        )
        service.qdrant.load_collection.assert_called_once_with("sansr_seg_lemma")

    def test_qdrant_search_ready_requires_main_collection(self):
        service = SansRAGService()
        service.qdrant_ok = True
        service.qdrant = Mock()
        service.qdrant.collection_exists.return_value = False
        self.assertFalse(service._qdrant_search_ready())

        service.qdrant.collection_exists.return_value = True
        service.qdrant.get_collection_stats.return_value = {"row_count": 0}
        self.assertFalse(service._qdrant_search_ready())

        service.qdrant.get_collection_stats.return_value = {"row_count": 12}
        self.assertTrue(service._qdrant_search_ready())


if __name__ == "__main__":
    unittest.main()
