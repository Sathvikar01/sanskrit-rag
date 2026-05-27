"""Tests for commentary ingestion, retrieval, and UI formatting."""
import unittest
from unittest.mock import MagicMock, Mock

import numpy as np

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.answer_generator import AnswerGenerator
from src.commentary_manager import (
    COMMENTARY_CONFIG,
    CommentaryEmbedding,
    CommentaryManager,
    CommentarySearchResult,
)
from src.embedding_client import EmbeddingResult
from src.retriever import VerseFilter
from src.ui import SansRAGUI


class DummyChunk:
    def __init__(self, chunk_id, text, verse_id, chapter, verse_num, metadata=None):
        self.id = chunk_id
        self.text = text
        self.verse_id = verse_id
        self.chapter = chapter
        self.verse_num = verse_num
        self.metadata = metadata or {}


class DummyMatch:
    def __init__(self, payload):
        self.payload = payload

    def to_dict(self):
        return self.payload


class TestCommentaryManager(unittest.TestCase):
    def setUp(self):
        self.mock_qdrant = Mock()
        self.mock_qdrant.client = MagicMock()
        self.mock_embedder = Mock()
        self.mock_embedder.embed_query.return_value = EmbeddingResult(
            id="query",
            text="query",
            dense_vector=np.array([0.1] * 1024, dtype=np.float32),
            sparse_vector={1: 1.0},
            colbert_vectors=np.array([[0.1] * 128], dtype=np.float32),
            metadata={},
        )
        self.manager = CommentaryManager(
            qdrant_manager=self.mock_qdrant,
            embedding_client=self.mock_embedder,
        )

    def test_embed_commentary_chunks_adds_required_metadata(self):
        chunk = DummyChunk(
            chunk_id="c1",
            text="Baladeva commentary text",
            verse_id="BhG 1.1",
            chapter=1,
            verse_num=1,
            metadata={"line_number": 4},
        )

        embedded = self.manager.embed_commentary_chunks(
            {"baladeva": [chunk]},
            source_dataset="dataset.xml",
            text_variant="raw",
            show_progress=False,
        )

        self.assertIn("baladeva", embedded)
        metadata = embedded["baladeva"][0].metadata
        self.assertEqual(metadata["author_key"], "baladeva")
        self.assertEqual(metadata["author_display_name"], "Baladeva")
        self.assertEqual(metadata["verse_id"], "BhG 1.1")
        self.assertEqual(metadata["chapter"], 1)
        self.assertEqual(metadata["verse_num"], 1)
        self.assertEqual(metadata["source_dataset"], "dataset.xml")
        self.assertEqual(metadata["text_variant"], "raw")
        self.assertTrue(metadata["is_commentary"])

    def test_store_commentary_embeddings_routes_to_author_collection(self):
        embedding = CommentaryEmbedding(
            id="comm1",
            text="Vishwanatha text",
            author="vishwanatha",
            verse_id="BhG 2.47",
            dense_vector=np.array([0.1] * 1024, dtype=np.float32),
            sparse_vector={},
            metadata={"author_key": "vishwanatha"},
        )
        self.manager._insert_to_qdrant = Mock(return_value=1)

        counts = self.manager.store_commentary_embeddings(
            {"vishwanatha": [embedding]},
            show_progress=False,
        )

        self.assertEqual(counts["vishwanatha"], 1)
        self.manager._insert_to_qdrant.assert_called_once()
        self.assertEqual(
            self.manager._insert_to_qdrant.call_args.kwargs["collection_name"],
            COMMENTARY_CONFIG["vishwanatha"]["collection_name"],
        )

    def test_insert_to_qdrant_uses_uuid_point_ids_and_preserves_original_commentary_id(self):
        embedding = CommentaryEmbedding(
            id="baladeva_BhG_2-47_L1_custom",
            text="Baladeva text",
            author="baladeva",
            verse_id="BhG 2.47",
            dense_vector=np.array([0.1] * 1024, dtype=np.float32),
            sparse_vector={1: 1.0},
            metadata={"author_key": "baladeva"},
        )

        count = self.manager._insert_to_qdrant(
            collection_name=COMMENTARY_CONFIG["baladeva"]["collection_name"],
            embeddings=[embedding],
            batch_size=10,
            show_progress=False,
        )

        self.assertEqual(count, 1)
        upsert_kwargs = self.mock_qdrant.client.upsert.call_args.kwargs
        point = upsert_kwargs["points"][0]
        self.assertNotEqual(str(point.id), embedding.id)
        self.assertEqual(point.payload["commentary_id"], embedding.id)
        self.assertEqual(point.payload["metadata"]["original_commentary_id"], embedding.id)

    def test_search_commentary_falls_back_to_direct_retrieval_when_dense_query_embedding_is_unavailable(self):
        self.mock_embedder.embed_query.return_value = EmbeddingResult(
            id="query",
            text="karma yoga",
            dense_vector=np.zeros(1024, dtype=np.float32),
            sparse_vector={1: 1.0},
            colbert_vectors=np.zeros((128, 128), dtype=np.float32),
            metadata={"dense_available": False},
        )
        self.manager.get_commentary_for_verses = Mock(
            return_value={"baladeva": [CommentarySearchResult(id="c1", text="Direct hit", author="baladeva", verse_id="BhG 2.47", score=1.0, metadata={})]}
        )

        results = self.manager.search_commentary(
            "karma yoga",
            authors=["baladeva"],
            verse_ids=["BhG 2.47"],
            top_k=3,
        )

        self.manager.get_commentary_for_verses.assert_called_once_with(
            ["BhG 2.47"],
            authors=["baladeva"],
            limit_per_author=3,
        )
        self.assertEqual(results["baladeva"][0].verse_id, "BhG 2.47")

    def test_get_best_matches_returns_top_one_overall_per_verse(self):
        self.manager.search_commentary = Mock(
            return_value={
                "baladeva": [
                    CommentarySearchResult(
                        id="b1",
                        text="Baladeva on 1.1",
                        author="baladeva",
                        verse_id="BhG 1.1",
                        score=0.81,
                        metadata={},
                    )
                ],
                "shreedhara": [
                    CommentarySearchResult(
                        id="s1",
                        text="Shreedhara on 1.1",
                        author="shreedhara",
                        verse_id="BhG 1.1",
                        score=0.92,
                        metadata={},
                    )
                ],
                "vishwanatha": [
                    CommentarySearchResult(
                        id="v1",
                        text="Vishwanatha on 2.47",
                        author="vishwanatha",
                        verse_id="BhG 2.47",
                        score=0.77,
                        metadata={},
                    )
                ],
            }
        )

        matches = self.manager.get_best_matches(
            "duty and action",
            ["BhG 1.1", "BhG 2.47"],
        )

        self.assertEqual(len(matches), 2)
        self.assertEqual(matches[0].verse_id, "BhG 1.1")
        self.assertEqual(matches[0].author_key, "shreedhara")
        self.assertEqual(matches[1].verse_id, "BhG 2.47")
        self.assertEqual(matches[1].author_key, "vishwanatha")

    def test_get_best_matches_falls_back_to_direct_retrieval(self):
        self.manager.search_commentary = Mock(
            return_value={"baladeva": [], "shreedhara": [], "vishwanatha": []}
        )
        self.manager.get_commentary_for_verses = Mock(
            return_value={
                "baladeva": [
                    CommentarySearchResult(
                        id="b1",
                        text="Baladeva direct retrieval",
                        author="baladeva",
                        verse_id="BhG 18.66",
                        score=1.0,
                        metadata={},
                    )
                ],
                "shreedhara": [],
                "vishwanatha": [],
            }
        )

        matches = self.manager.get_best_matches("", ["BhG 18.66"])

        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].author_key, "baladeva")
        self.assertEqual(matches[0].verse_id, "BhG 18.66")


class TestAnswerGeneratorCommentarySelection(unittest.TestCase):
    def setUp(self):
        retriever = Mock()
        retriever.embedding_client = Mock()
        self.generator = AnswerGenerator(
            gemini_client=Mock(),
            retriever=retriever,
            qdrant_manager=Mock(),
            neo4j_manager=None,
            top_k=5,
        )
        self.generator.commentary_manager = Mock()

    def test_explicit_verse_filter_uses_requested_verse_ids_only(self):
        self.generator.commentary_manager.get_best_matches.return_value = [
            DummyMatch({"verse_id": "BhG 2.47", "author_display_name": "Baladeva"})
        ]
        verse_filter = VerseFilter(
            chapter=2,
            verse_start=47,
            verse_end=47,
            verse_ids=["BhG 2.47"],
            raw_match="BG 2.47",
        )

        matches = self.generator._retrieve_commentary_matches(
            query="Explain BG 2.47",
            verse_filter=verse_filter,
            retrieved_verses=[{"verse_id": "BhG 1.1"}, {"verse_id": "BhG 2.47"}],
        )

        self.generator.commentary_manager.get_best_matches.assert_called_once_with(
            "Explain BG 2.47",
            ["BhG 2.47"],
        )
        self.assertEqual(matches[0]["verse_id"], "BhG 2.47")

    def test_non_verse_query_uses_top_five_distinct_verse_ids(self):
        self.generator.commentary_manager.get_best_matches.return_value = []
        verse_filter = VerseFilter()
        retrieved = [{"verse_id": f"BhG 1.{index}"} for index in [1, 2, 2, 3, 4, 5, 6]]

        self.generator._retrieve_commentary_matches(
            query="karma yoga",
            verse_filter=verse_filter,
            retrieved_verses=retrieved,
        )

        self.generator.commentary_manager.get_best_matches.assert_called_once_with(
            "karma yoga",
            ["BhG 1.1", "BhG 1.2", "BhG 1.3", "BhG 1.4", "BhG 1.5"],
        )


class TestUICommentaryFormatting(unittest.TestCase):
    def test_formatter_includes_commentator_name_verse_and_excerpt(self):
        ui = SansRAGUI.__new__(SansRAGUI)
        markdown = ui._format_commentary_matches(
            [
                {
                    "verse_id": "BhG 2.47",
                    "author_key": "baladeva",
                    "author_display_name": "Baladeva",
                    "text": "This is a commentary excerpt about action.",
                    "score": 0.91,
                    "metadata": {"source_dataset": "dataset.xml", "text_variant": "raw"},
                }
            ]
        )

        self.assertIn("BhG 2.47", markdown)
        self.assertIn("Baladeva", markdown)
        self.assertIn("This is a commentary excerpt", markdown)
        self.assertIn("Semantic score: 0.9100", markdown)


if __name__ == "__main__":
    unittest.main()
