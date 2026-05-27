"""Tests for canonical evidence assembly in the answer pipeline."""
import unittest
from unittest.mock import Mock

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.answer_generator import AnswerGenerator
from src.retriever import HybridSearchResult


class DummyMatch:
    def __init__(self, payload):
        self.payload = payload

    def to_dict(self):
        return self.payload


class CanonicalVerseDB:
    def __init__(self, verses):
        self.verses = verses
        self.requested_ids = None

    def get_verses_by_ids(self, verse_ids):
        self.requested_ids = verse_ids
        return [self.verses[verse_id] for verse_id in verse_ids if verse_id in self.verses]


class TestAnswerEvidenceAssembly(unittest.TestCase):
    def build_generator(self, rrf_results, verse_db=None, llm_available=True):
        retriever = Mock()
        retriever._qdrant_available = True
        retriever._neo4j_available = True
        retriever.embedding_client = Mock()
        retriever.cross_db_rrf_search.return_value = rrf_results

        llm = Mock()
        llm.is_available.return_value = llm_available
        llm.generate_answer.return_value = {
            "answer": "Use [Citation 1] for the answer.",
            "citations": [
                {
                    "verse_id": "BhG 2.47",
                    "source": "SQLite",
                    "score": 0.9,
                    "text": "karmany eva adhikaras te",
                }
            ],
        }

        generator = AnswerGenerator(
            gemini_client=llm,
            retriever=retriever,
            qdrant_manager=Mock(),
            neo4j_manager=None,
            verse_db=verse_db,
            top_k=5,
        )
        generator.commentary_manager = Mock()
        generator.commentary_manager.get_best_matches.return_value = [
            DummyMatch(
                {
                    "verse_id": "BhG 2.47",
                    "author_display_name": "Baladeva",
                    "text": "Commentary on action.",
                    "score": 0.88,
                }
            )
        ]
        return generator, retriever, llm

    def test_generate_answer_uses_sqlite_verse_and_commentary_for_llm(self):
        rrf_results = [
            HybridSearchResult(
                id="chunk-a",
                text="retrieved chunk text",
                final_score=0.91,
                dense_score=0.8,
                sparse_score=0.7,
                bm25_score=0.6,
                dataset_type="seg_lemma",
                verse_id="BhG 2.47",
                metadata={"sources": {"qdrant": True, "neo4j": True}},
            )
        ]
        verse_db = CanonicalVerseDB(
            {
                "BhG 2.47": {
                    "verse_id": "BhG 2.47",
                    "chapter": 2,
                    "verse_num": 47,
                    "speaker": "sri-bhagavan uvaca",
                    "lines": ["karmany eva adhikaras te", "ma phalesu kadacana"],
                    "sanskrit_text": "fallback",
                    "word_count": 6,
                }
            }
        )

        generator, retriever, llm = self.build_generator(rrf_results, verse_db)

        result = generator.generate_answer("What is taught about action?")

        retriever.cross_db_rrf_search.assert_called_once()
        self.assertEqual(verse_db.requested_ids, ["BhG 2.47"])

        llm_kwargs = llm.generate_answer.call_args.kwargs
        self.assertEqual(llm_kwargs["retrieved_verses"][0]["source"], "SQLite")
        self.assertIn("karmany eva", llm_kwargs["retrieved_verses"][0]["text"])
        self.assertEqual(llm_kwargs["commentary_matches"][0]["author_display_name"], "Baladeva")
        self.assertTrue(llm_kwargs["retrieval_metadata"]["db_status"]["qdrant"]["contributed"])
        self.assertTrue(llm_kwargs["retrieval_metadata"]["db_status"]["neo4j"]["contributed"])

        self.assertEqual(result.evidence["canonical_verses"][0]["source"], "SQLite")
        self.assertEqual(result.sources["sqlite_verses"], 1)
        self.assertEqual(result.retrieval_stats["unique_verses"], 1)

    def test_generate_answer_returns_clear_no_evidence_without_llm_call(self):
        generator, retriever, llm = self.build_generator([], verse_db=CanonicalVerseDB({}))

        result = generator.generate_answer("Unknown query")

        retriever.cross_db_rrf_search.assert_called_once()
        llm.generate_answer.assert_not_called()
        self.assertFalse(result.evidence["canonical_verses"])
        self.assertIn("neither Qdrant nor Neo4j returned", result.answer)
        self.assertFalse(result.sources["db_status"]["qdrant"]["contributed"])
        self.assertFalse(result.sources["db_status"]["neo4j"]["contributed"])

    def test_generate_answer_handles_non_english_query_when_llm_is_unavailable(self):
        generator, retriever, llm = self.build_generator([], verse_db=CanonicalVerseDB({}), llm_available=False)

        result = generator.generate_answer("dharma yoga")

        retriever.cross_db_rrf_search.assert_called_once()
        llm.generate_answer.assert_not_called()
        self.assertTrue(result.answer)
        self.assertTrue(result.normalized_query.startswith("dharma yoga"))


if __name__ == "__main__":
    unittest.main()
