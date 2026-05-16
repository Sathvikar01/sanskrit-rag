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

    def test_format_verse_context(self):
        results = self._create_test_results()
        context = format_verse_context(results)
        assert "BhG 2.47" in context
        assert "karmaṇy evādhikāras" in context

    def test_build_generation_prompt(self):
        results = self._create_test_results()
        prompt = build_generation_prompt("What is karma yoga?", results)
        assert "karma yoga" in prompt
        assert "BhG 2.47" in prompt

    def test_empty_results(self):
        prompt = build_generation_prompt("test query", [])
        assert "test query" in prompt
