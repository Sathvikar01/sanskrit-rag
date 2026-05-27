# Integration Test for SansRAG Pipeline
"""End-to-end integration tests for the complete pipeline."""
import unittest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import json
import os

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import COLLECTION_NAMES, TEST_QUERIES


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete SansRAG pipeline."""
    
    @patch('src.qdrant_manager.QDRANT_AVAILABLE', False)
    @patch('src.embedding_client.requests')
    def test_pipeline_initialization(self, mock_requests):
        from src.main import SansRAGPipeline
        
        pipeline = SansRAGPipeline(verbose=False)
        
        self.assertIsNotNone(pipeline.parser)
        self.assertIsNotNone(pipeline.embedder)
        self.assertIsNotNone(pipeline.qdrant)
        self.assertIsNotNone(pipeline.retriever)
    
    @patch('src.qdrant_manager.QDRANT_AVAILABLE', False)
    def test_search_returns_empty_without_qdrant(self):
        from src.main import SansRAGPipeline
        
        pipeline = SansRAGPipeline(verbose=False)
        results = pipeline.search("dharma", top_k=5)
        
        self.assertIsInstance(results, dict)
        for dtype, result_list in results.items():
            self.assertEqual(len(result_list), 0)
    
    @patch('src.qdrant_manager.QDRANT_AVAILABLE', True)
    @patch('src.qdrant_manager.QdrantClient')
    def test_connect_qdrant_success(self, mock_client):
        from src.main import SansRAGPipeline
        
        pipeline = SansRAGPipeline(verbose=False)
        mock_instance = Mock()
        mock_instance.get_collections.return_value = Mock(collections=[])
        mock_client.return_value = mock_instance
        
        result = pipeline.connect_qdrant(max_retries=1)
        
        self.assertTrue(result)
    
    def test_save_results(self):
        from src.main import SansRAGPipeline
        from src.retriever import HybridSearchResult
        
        pipeline = SansRAGPipeline(verbose=False)
        
        results = {
            "raw": [
                HybridSearchResult(
                    id="test1",
                    text="dharma kṣetra",
                    final_score=0.9,
                    dense_score=0.92,
                    sparse_score=0.88,
                    colbert_score=0.9,
                    bm25_score=0.85,
                    dataset_type="raw",
                    verse_id="BhG 1.1"
                )
            ]
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = pipeline.save_results(results, "test query", output_dir=tmpdir)
            
            self.assertTrue(os.path.exists(filepath))
            
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            self.assertEqual(data["query"], "test query")
            self.assertIn("timestamp", data)
            self.assertIn("regularization", data)
            self.assertIn("results", data)
    
    @patch('src.qdrant_manager.QDRANT_AVAILABLE', False)
    def test_run_test_queries(self):
        from src.main import SansRAGPipeline
        
        pipeline = SansRAGPipeline(verbose=False)
        metrics = pipeline.run_test_queries()
        
        self.assertIn("queries", metrics)
        self.assertEqual(len(metrics["queries"]), len(TEST_QUERIES))
        self.assertIn("total_latency_ms", metrics)
        self.assertIn("avg_latency_ms", metrics)


class TestCLI(unittest.TestCase):
    """Test CLI argument parsing."""
    
    def test_default_args(self):
        from src.main import main
        import argparse
        
        parser = argparse.ArgumentParser()
        parser.add_argument("--ingest", action="store_true")
        parser.add_argument("--test", action="store_true")
        parser.add_argument("--interactive", action="store_true", default=True)
        parser.add_argument("--l1", type=float, default=0.01)
        parser.add_argument("--l2", type=float, default=0.001)
        parser.add_argument("--no-adaptive", action="store_true")
        parser.add_argument("--data-dir", type=str, default=".")
        parser.add_argument("--quiet", action="store_true")
        
        args = parser.parse_args([])
        
        self.assertFalse(args.ingest)
        self.assertFalse(args.test)
        self.assertTrue(args.interactive)
        self.assertEqual(args.l1, 0.01)
        self.assertEqual(args.l2, 0.001)
    
    def test_custom_lambdas(self):
        from src.main import main
        import argparse
        
        parser = argparse.ArgumentParser()
        parser.add_argument("--l1", type=float, default=0.01)
        parser.add_argument("--l2", type=float, default=0.001)
        
        args = parser.parse_args(["--l1", "0.05", "--l2", "0.005"])
        
        self.assertEqual(args.l1, 0.05)
        self.assertEqual(args.l2, 0.005)


class TestColors(unittest.TestCase):
    """Test color output utility."""
    
    def test_color_codes(self):
        from src.main import Colors
        
        self.assertTrue(Colors.HEADER.startswith('\033'))
        self.assertTrue(Colors.END.startswith('\033'))
        self.assertIn('m', Colors.HEADER)


if __name__ == "__main__":
    unittest.main()
