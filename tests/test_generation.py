"""Tests for SRAG generation modules."""


from src.generation.prompt_templates import (
    format_verse_context,
    build_generation_prompt,
)


class TestPromptTemplates:
    """Test prompt template functions."""

    def _create_test_results(self):
        """Create test reranked results."""
        return [
            {
                "verse_ref": "BhG 2.47",
                "text_iast": "karmaṇy evādhikāras te mā phaleṣu kadācana",
                "text_devanagari": "कर्मण्येवाधिकारस्ते मा फलेषु कदाचन",
                "commentator": "sridhara",
                "chunk_type": "verse",
                "confidence": {"overall_confidence": 0.92},
            },
            {
                "verse_ref": "BhG 2.48",
                "text_iast": "yogasthaḥ kuru karmāṇi",
                "text_devanagari": "योगस्थः कुरु कर्माणि",
                "commentator": None,
                "chunk_type": "verse",
                "confidence": {"overall_confidence": 0.85},
            },
        ]

    def test_format_verse_context_returns_tuple(self):
        results = self._create_test_results()
        verses, commentaries = format_verse_context(results)
        assert isinstance(verses, str)
        assert isinstance(commentaries, str)

    def test_format_verse_context_verses(self):
        results = self._create_test_results()
        verses, commentaries = format_verse_context(results)
        assert "BhG 2.47" in verses
        assert "karmaṇy evādhikāras" in verses

    def test_format_verse_context_commentaries_separated(self):
        results = self._create_test_results()
        verses, commentaries = format_verse_context(results)
        commentary_chunk = {
            "verse_ref": "BhG 2.47",
            "text_iast": "Sridhara commentary on karma yoga",
            "text_devanagari": "",
            "commentator": "sridhara",
            "chunk_type": "commentary",
            "confidence": {"overall_confidence": 0.80},
        }
        verses2, commentaries2 = format_verse_context(results + [commentary_chunk])
        assert "Sridhara commentary" in commentaries2
        assert "Sridhara" not in verses2 or "Sridhara Swamin" not in verses2

    def test_format_verse_context_empty(self):
        verses, commentaries = format_verse_context([])
        assert "No relevant verses found" in verses

    def test_build_generation_prompt(self):
        results = self._create_test_results()
        prompt = build_generation_prompt("What is karma yoga?", results)
        assert "karma yoga" in prompt
        assert "BhG 2.47" in prompt

    def test_build_generation_prompt_includes_commentaries_section(self):
        results = self._create_test_results()
        commentary_chunk = {
            "verse_ref": "BhG 2.47",
            "text_iast": "Sridhara says: karma yoga means...",
            "text_devanagari": "",
            "commentator": "sridhara",
            "chunk_type": "commentary",
            "confidence": {"overall_confidence": 0.80},
        }
        prompt = build_generation_prompt("What is karma yoga?", results + [commentary_chunk])
        assert "Traditional Commentary" in prompt
        assert "Sridhara" in prompt

    def test_build_generation_prompt_no_commentaries(self):
        results = self._create_test_results()
        prompt = build_generation_prompt("What is karma yoga?", results)
        assert "karma yoga" in prompt
        assert "BhG 2.47" in prompt

    def test_empty_results(self):
        prompt = build_generation_prompt("test query", [])
        assert "test query" in prompt

    def test_mixed_chunk_types(self):
        results = [
            {
                "verse_ref": "BhG 2.47",
                "text_iast": "karmaṇy evādhikāras te mā phaleṣu kadācana",
                "text_devanagari": "कर्मण्येवाधिकारस्ते मा फलेषु कदाचन",
                "commentator": None,
                "chunk_type": "verse",
                "confidence": {"overall_confidence": 0.92},
            },
            {
                "verse_ref": "BhG 2.47",
                "text_iast": "Sridhara explains: The meaning of this verse...",
                "text_devanagari": "",
                "commentator": "sridhara",
                "chunk_type": "commentary",
                "confidence": {"overall_confidence": 0.88},
            },
            {
                "verse_ref": "BhG 2.48",
                "text_iast": "yogasthaḥ kuru karmāṇi",
                "text_devanagari": "योगस्थः कुरु कर्माणि",
                "commentator": None,
                "chunk_type": "verse",
                "confidence": {"overall_confidence": 0.85},
            },
        ]
        verses, commentaries = format_verse_context(results)
        assert "BhG 2.47" in verses
        assert "BhG 2.48" in verses
        assert "Sridhara" in commentaries

    def test_prompt_structure_order(self):
        results = self._create_test_results()
        commentary_chunk = {
            "verse_ref": "BhG 2.47",
            "text_iast": "Sridhara commentary text",
            "text_devanagari": "",
            "commentator": "sridhara",
            "chunk_type": "commentary",
            "confidence": {"overall_confidence": 0.80},
        }
        prompt = build_generation_prompt("What is karma yoga?", results + [commentary_chunk])
        verse_pos = prompt.find("## Retrieved Verses")
        comment_pos = prompt.find("## Traditional Commentary")
        question_pos = prompt.find("## User Question")
        assert question_pos < verse_pos < comment_pos
