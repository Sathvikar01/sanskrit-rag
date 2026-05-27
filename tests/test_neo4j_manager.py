"""Tests for Neo4j manager stats and empty-graph behavior."""
import unittest
from unittest.mock import MagicMock, Mock

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.neo4j_manager import Neo4jManager


class TestNeo4jManagerStats(unittest.TestCase):
    def build_manager(self, run_side_effect):
        manager = Neo4jManager()
        manager._connected = True
        session = Mock()
        session.run.side_effect = run_side_effect
        manager._driver = MagicMock()
        manager._driver.session.return_value.__enter__.return_value = session
        return manager, session

    def test_get_collection_stats_returns_zero_without_missing_label_queries(self):
        manager, session = self.build_manager([[]])

        stats = manager.get_collection_stats(refresh=True)

        self.assertEqual(stats["chunk_count"], 0)
        self.assertEqual(stats["word_count"], 0)
        self.assertEqual(stats["lemma_count"], 0)
        self.assertEqual(session.run.call_count, 1)

    def test_get_collection_stats_counts_only_existing_labels(self):
        chunk_result = Mock()
        chunk_result.single.return_value = {"count": 7}
        word_result = Mock()
        word_result.single.return_value = {"count": 19}
        manager, session = self.build_manager([
            [{"label": "Chunk"}, {"label": "Word"}],
            chunk_result,
            word_result,
        ])

        stats = manager.get_collection_stats(refresh=True)

        self.assertEqual(stats["chunk_count"], 7)
        self.assertEqual(stats["word_count"], 19)
        self.assertEqual(stats["lemma_count"], 0)
        self.assertEqual(session.run.call_count, 3)


if __name__ == "__main__":
    unittest.main()
