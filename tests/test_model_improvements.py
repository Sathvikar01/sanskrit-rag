"""Tests for model-improvement evidence utilities."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.answer_generator import AnswerGenerator
from src.commentary_manager import CommentarySearchResult
from src.entity_lexicon import expand_query_with_aliases
from src.evidence_cache import EvidenceCache
from src.evidence_reranker import EvidenceReranker, RerankContext
from src.golden_qa import expected_verse_ids_from_question, retrieval_metrics
from src.query_intent import classify_query_intent
from src.retriever import HybridSearchResult, VerseFilter, parse_verse_references
from src.text_quality import score_text_quality
from src.verse_db import VerseDatabase, ingest_xml_to_sqlite


class TinyVerseDB:
    def __init__(self, verses):
        self.verses = verses

    def get_verses_by_ids(self, verse_ids):
        return [self.verses[verse_id] for verse_id in verse_ids if verse_id in self.verses]


class TestEntityIntentQuality(unittest.TestCase):
    def test_alias_expansion_detects_gita_names(self):
        expanded = expand_query_with_aliases("Why is Krishna called Partha-sarathi?")
        canonical_names = {entity["canonical"] for entity in expanded["entities"]}

        self.assertIn("Krishna", canonical_names)
        self.assertIn("Arjuna", canonical_names)
        self.assertIn("aliases_added", expanded)

    def test_intent_routes_explicit_and_commentary_queries(self):
        explicit = classify_query_intent(
            "Explain BG 1.15",
            verse_filter=parse_verse_references("Explain BG 1.15"),
            entities=[],
        )
        commentary = classify_query_intent("What does the commentary say about dharma?")

        self.assertEqual(explicit.intent, "explicit_verse_lookup")
        self.assertEqual(commentary.intent, "commentary_question")

    def test_text_quality_flags_morphology_noise(self):
        quality = score_text_quality("karma_Case=Nom Gender=Masc Number=Sing VerbForm=Part")

        self.assertLess(quality["quality_score"], 0.7)
        self.assertIn("morphology_only", quality["flags"])


class TestEvidenceReranking(unittest.TestCase):
    def test_explicit_reference_stays_first(self):
        candidates = [
            HybridSearchResult(
                id="bad-high-score",
                text="unrelated action text",
                final_score=0.99,
                verse_id="BhG 3.1",
                metadata={"sources": {"qdrant": True, "neo4j": False}},
            ),
            HybridSearchResult(
                id="explicit-low-score",
                text="yudhisthira ananta-vijaya conchshell evidence",
                final_score=0.01,
                verse_id="BhG 1.16",
                metadata={"sources": {"qdrant": False, "neo4j": True}},
            ),
        ]

        ranked = EvidenceReranker().rerank(
            candidates,
            RerankContext(
                query="What is Yudhisthira's conchshell? BG 1.16",
                verse_filter=VerseFilter(chapter=1, verse_start=16, verse_end=16, verse_ids=["BhG 1.16"]),
                query_intent={"retrieval_profile": {}},
                entities=[],
                commentary_verse_ids=set(),
            ),
            top_k=2,
        )

        self.assertEqual(ranked[0].verse_id, "BhG 1.16")
        self.assertTrue(ranked[0].metadata["reranker"]["explicit_reference"])


class TestAnswerGeneratorImprovements(unittest.TestCase):
    def test_explicit_reference_is_canonical_first_even_if_rrf_returns_other_verse(self):
        retriever = Mock()
        retriever._qdrant_available = True
        retriever._neo4j_available = True
        retriever.embedding_client = Mock()
        retriever.cross_db_rrf_search.return_value = [
            HybridSearchResult(
                id="other",
                text="other verse",
                final_score=0.99,
                verse_id="BhG 3.1",
                metadata={"sources": {"qdrant": True, "neo4j": True}},
            )
        ]

        llm = Mock()
        llm.is_available.return_value = True
        llm.model_name = "test-model"
        llm.generate_answer.return_value = {
            "answer": "Explicit answer [Citation 1]",
            "citations": [{"verse_id": "BhG 1.16", "source": "SQLite", "score": 0.0, "text": "ananta"}],
        }

        verse_db = TinyVerseDB({
            "BhG 1.16": {
                "verse_id": "BhG 1.16",
                "chapter": 1,
                "verse_num": 16,
                "speaker": "",
                "lines": ["anantavijayaṃ rājā kuntī-putro yudhiṣṭhiraḥ"],
                "sanskrit_text": "",
                "word_count": 4,
                "commentaries": [{"commentator": "SQLite Commentator", "text": "Ananta-vijaya belongs to Yudhisthira."}],
            },
            "BhG 3.1": {
                "verse_id": "BhG 3.1",
                "chapter": 3,
                "verse_num": 1,
                "speaker": "",
                "lines": ["unrelated"],
                "sanskrit_text": "",
                "word_count": 1,
                "commentaries": [],
            },
        })

        with tempfile.TemporaryDirectory() as tmpdir:
            generator = AnswerGenerator(llm, retriever, qdrant_manager=Mock(), neo4j_manager=None, verse_db=verse_db)
            generator.cache = EvidenceCache(Path(tmpdir) / "cache.sqlite")
            generator.commentary_manager = None
            result = generator.generate_answer("What is Yudhisthira's conchshell? BG 1.16")
            generator.cache.close()

        llm_kwargs = llm.generate_answer.call_args.kwargs
        self.assertEqual(llm_kwargs["retrieved_verses"][0]["verse_id"], "BhG 1.16")
        self.assertEqual(result.explicit_references, ["BhG 1.16"])
        self.assertEqual(result.query_intent["intent"], "explicit_verse_lookup")
        self.assertGreater(result.confidence, 0.5)
        self.assertEqual(result.commentary_matches[0]["commentary_source"], "SQLite")

    def test_commentary_question_can_add_direct_commentary_verse_candidate(self):
        retriever = Mock()
        retriever._qdrant_available = True
        retriever._neo4j_available = False
        retriever.embedding_client = Mock()
        retriever.cross_db_rrf_search.return_value = []

        llm = Mock()
        llm.is_available.return_value = True
        llm.model_name = "test-model"
        llm.generate_answer.return_value = {
            "answer": "Commentary-backed answer [Citation 1]",
            "citations": [{"verse_id": "BhG 18.66", "source": "SQLite", "score": 0.0, "text": "sarva dharman"}],
        }

        verse_db = TinyVerseDB({
            "BhG 18.66": {
                "verse_id": "BhG 18.66",
                "chapter": 18,
                "verse_num": 66,
                "speaker": "",
                "lines": ["sarva-dharman parityajya"],
                "sanskrit_text": "",
                "word_count": 3,
                "commentaries": [{"commentator": "SQLite Commentator", "text": "Commentary explains surrender."}],
            },
        })

        commentary_manager = Mock()
        commentary_manager.search_commentary.return_value = {
            "baladeva": [
                CommentarySearchResult(
                    id="c1",
                    text="surrender commentary",
                    author="baladeva",
                    verse_id="BhG 18.66",
                    score=0.91,
                    metadata={},
                )
            ]
        }
        commentary_manager.get_best_matches.return_value = []

        with tempfile.TemporaryDirectory() as tmpdir:
            generator = AnswerGenerator(llm, retriever, qdrant_manager=Mock(), neo4j_manager=None, verse_db=verse_db)
            generator.cache = EvidenceCache(Path(tmpdir) / "cache.sqlite")
            generator.commentary_manager = commentary_manager
            result = generator.generate_answer("What does the commentary explain about surrender?")
            generator.cache.close()

        self.assertEqual(result.evidence["canonical_verses"][0]["verse_id"], "BhG 18.66")
        self.assertEqual(result.retrieval_stats["direct_commentary_verse_ids"], ["BhG 18.66"])
        llm.generate_answer.assert_called_once()


class TestSQLiteCommentaryIngestion(unittest.TestCase):
    def test_ingest_saves_multiple_commentary_blocks(self):
        xml = """<TEI xmlns="http://www.tei-c.org/ns/1.0"><text><body><div>
        <p>BhG 1.1</p>
        <lg><l>dharma-kṣetre kuru-kṣetre</l><l>samavetā yuyutsavaḥ</l></lg>
        <p>Śrīdharaḥ - first commentary paragraph</p>
        <p>second commentary paragraph</p>
        <p>Baladevaḥ -- another commentary paragraph</p>
        </div></body></text></TEI>"""

        with tempfile.TemporaryDirectory() as tmpdir:
            xml_path = Path(tmpdir) / "sample.xml"
            db_path = Path(tmpdir) / "verses.db"
            xml_path.write_text(xml, encoding="utf-8")

            db = ingest_xml_to_sqlite(str(xml_path), str(db_path))
            db = VerseDatabase(str(db_path))
            verse = db.get_verse("BhG 1.1")
            db.close()

        self.assertIsNotNone(verse)
        self.assertGreaterEqual(len(verse["commentaries"]), 2)
        self.assertTrue(any("first commentary" in c["text"] for c in verse["commentaries"]))


class TestGoldenQA(unittest.TestCase):
    def test_expected_verse_ids_from_question(self):
        self.assertEqual(
            expected_verse_ids_from_question("What happened? BG 1.16-18"),
            ["BhG 1.16", "BhG 1.17", "BhG 1.18"],
        )

    def test_retrieval_metrics_report_quality_and_coverage(self):
        metrics = retrieval_metrics(
            {
                "evidence": {
                    "canonical_verses": [
                        {"verse_id": "BhG 1.16"},
                        {"verse_id": "BhG 2.47"},
                    ],
                    "commentary_matches": [{"verse_id": "BhG 1.17"}],
                },
                "confidence": 0.8,
            },
            ["BhG 1.16", "BhG 1.17"],
        )

        self.assertTrue(metrics["explicit_verse_hit"])
        self.assertEqual(metrics["expected_coverage"], 0.5)
        self.assertEqual(metrics["first_expected_rank"], 1)
        self.assertGreater(metrics["retrieval_quality"], 0.6)

    def test_range_ingestion_expands_sqlite_verse_ids(self):
        xml = """<TEI xmlns="http://www.tei-c.org/ns/1.0"><text><body><div>
        <p>BhG 1.15-16</p>
        <lg><l>first grouped line</l><l>second grouped line</l></lg>
        <p>Baladeva: grouped commentary</p>
        </div></body></text></TEI>"""

        with tempfile.TemporaryDirectory() as tmpdir:
            xml_path = Path(tmpdir) / "range.xml"
            db_path = Path(tmpdir) / "verses.db"
            xml_path.write_text(xml, encoding="utf-8")

            ingest_xml_to_sqlite(str(xml_path), str(db_path))
            db = VerseDatabase(str(db_path))
            verse_15 = db.get_verse("BhG 1.15")
            verse_16 = db.get_verse("BhG 1.16")
            stats = db.get_stats()
            db.close()

        self.assertEqual(stats["total_verses"], 2)
        self.assertIsNotNone(verse_15)
        self.assertIsNotNone(verse_16)
        self.assertEqual(verse_16["verse_id"], "BhG 1.16")
        self.assertTrue(verse_16["commentaries"])


if __name__ == "__main__":
    unittest.main()
