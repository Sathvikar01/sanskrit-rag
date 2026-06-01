"""Tests for dense semantic relevance dataset evaluation helpers."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from run_semantic_relevance_by_dataset import (
    score_ranked_verses,
    strip_explicit_verse_references,
    unique_in_order,
)


def test_strip_explicit_verse_references_removes_bg_ids():
    question = "What is Arjuna bow name? BG 1.29"

    assert strip_explicit_verse_references(question) == "What is Arjuna bow name?"


def test_unique_in_order_preserves_first_occurrence():
    assert unique_in_order(["BhG 1.1", "BhG 1.2", "BhG 1.1"]) == ["BhG 1.1", "BhG 1.2"]


def test_score_ranked_verses_combines_coverage_and_reciprocal_rank():
    metrics = score_ranked_verses(
        ranked_verse_ids=["BhG 1.10", "BhG 1.29", "BhG 1.30"],
        expected_verse_ids=["BhG 1.29", "BhG 1.30"],
        top_k=3,
    )

    assert metrics["expected_coverage"] == 1.0
    assert metrics["first_expected_rank"] == 2
    assert metrics["reciprocal_rank"] == 0.5
    assert metrics["dense_semantic_quality"] == 0.875
