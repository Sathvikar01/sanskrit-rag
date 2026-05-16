"""Tests for SRAG retrieval modules."""


from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid_fusion import reciprocal_rank_fusion, weighted_fusion


class TestBM25Retriever:
    """Test BM25 retrieval."""

    def _create_test_chunks(self):
        """Create test chunks for BM25 testing."""
        from src.preprocessing.chunker import Chunk

        return [
            Chunk(
                chunk_id="chunk_1",
                verse_ref="BhG 1.1",
                chapter_num=1,
                verse_num=1,
                chunk_type="verse",
                text_iast="dharma-kṣetre kuru-kṣetre samavetā yuyutsavaḥ",
                lemmas=["dharma", "kṣetra", "kuru", "samaveta", "yuyutsu"],
            ),
            Chunk(
                chunk_id="chunk_2",
                verse_ref="BhG 2.47",
                chapter_num=2,
                verse_num=47,
                chunk_type="verse",
                text_iast="karmaṇy evādhikāras te mā phaleṣu kadācana",
                lemmas=["karma", "adhikāra", "phala"],
            ),
            Chunk(
                chunk_id="chunk_3",
                verse_ref="BhG 9.22",
                chapter_num=9,
                verse_num=22,
                chunk_type="verse",
                text_iast="ananyāś cintayanto māṃ ye janāḥ paryupāsate",
                lemmas=["ananya", "cintay", "mā", "jana", "paryupās"],
            ),
        ]

    def test_build_index(self):
        retriever = BM25Retriever()
        chunks = self._create_test_chunks()
        retriever.build_index(chunks)
        assert retriever.bm25 is not None
        assert len(retriever.chunk_ids) == 3

    def test_search(self):
        retriever = BM25Retriever()
        chunks = self._create_test_chunks()
        retriever.build_index(chunks)

        results = retriever.search("dharma karma", top_k=2)
        assert len(results) > 0
        assert all("chunk_id" in r for r in results)
        assert all("score" in r for r in results)

    def test_search_empty_query(self):
        retriever = BM25Retriever()
        chunks = self._create_test_chunks()
        retriever.build_index(chunks)

        results = retriever.search("", top_k=5)
        assert len(results) == 0


class TestHybridFusion:
    """Test hybrid fusion methods."""

    def _create_test_results(self):
        """Create test retrieval results."""
        vector_results = [
            {"chunk_id": "chunk_1", "score": 0.9, "rank": 1},
            {"chunk_id": "chunk_2", "score": 0.7, "rank": 2},
            {"chunk_id": "chunk_3", "score": 0.5, "rank": 3},
        ]
        graph_results = [
            {"chunk_id": "chunk_2", "score": 0.8, "rank": 1},
            {"chunk_id": "chunk_1", "score": 0.6, "rank": 2},
            {"chunk_id": "chunk_4", "score": 0.4, "rank": 3},
        ]
        bm25_results = [
            {"chunk_id": "chunk_1", "score": 2.5, "rank": 1},
            {"chunk_id": "chunk_3", "score": 1.8, "rank": 2},
            {"chunk_id": "chunk_2", "score": 1.2, "rank": 3},
        ]
        return vector_results, graph_results, bm25_results

    def test_rrf_fusion(self):
        vector, graph, bm25 = self._create_test_results()
        fused = reciprocal_rank_fusion([vector, graph, bm25])

        assert len(fused) > 0
        assert "rrf_score" in fused[0]
        assert "sources" in fused[0]

        chunk_1 = next(r for r in fused if r["chunk_id"] == "chunk_1")
        assert "vector" in chunk_1["sources"]
        assert "graph" in chunk_1["sources"]
        assert "bm25" in chunk_1["sources"]

    def test_weighted_fusion(self):
        vector, graph, bm25 = self._create_test_results()
        fused = weighted_fusion([vector, graph, bm25])

        assert len(fused) > 0
        assert "weighted_score" in fused[0]

    def test_rrf_ranking(self):
        vector, graph, bm25 = self._create_test_results()
        fused = reciprocal_rank_fusion([vector, graph, bm25])

        chunk_1 = next(r for r in fused if r["chunk_id"] == "chunk_1")
        chunk_2 = next(r for r in fused if r["chunk_id"] == "chunk_2")

        assert chunk_1["rrf_score"] >= chunk_2["rrf_score"]
