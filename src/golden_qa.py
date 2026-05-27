"""Reusable golden Q&A fixtures and retrieval metrics."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.retriever import parse_verse_references


ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CHAPTER1_QA = ROOT_DIR / "test_qa" / "golden_chapter1_qa.json"


def expected_verse_ids_from_question(question: str) -> List[str]:
    """Infer expected verse IDs from explicit BG references in a question."""
    verse_filter = parse_verse_references(question)
    return verse_filter.verse_ids or []


def with_expected_verse_ids(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Attach expected verse IDs to existing question/reference-answer pairs."""
    enriched = []
    for index, item in enumerate(items, 1):
        question = item.get("question", "")
        enriched.append({
            "question_number": item.get("question_number", index),
            "question": question,
            "reference_answer": item.get("reference_answer", ""),
            "expected_verse_ids": item.get("expected_verse_ids") or expected_verse_ids_from_question(question),
        })
    return enriched


def load_golden_chapter1_qa(
    path: Optional[str | Path] = None,
    default: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Load the 49-question Chapter 1 golden set from JSON or a provided default list."""
    source_path = Path(path) if path else DEFAULT_CHAPTER1_QA
    if source_path.exists():
        payload = json.loads(source_path.read_text(encoding="utf-8"))
        if "results" in payload:
            return with_expected_verse_ids(payload["results"])
        if isinstance(payload, list):
            return with_expected_verse_ids(payload)
    return with_expected_verse_ids(default or [])


def retrieval_metrics(answer_result: Any, expected_verse_ids: List[str]) -> Dict[str, Any]:
    """Compute explicit-reference retrieval metrics for an AnswerResult-like object."""
    data = answer_result.to_dict() if hasattr(answer_result, "to_dict") else dict(answer_result or {})
    evidence = data.get("evidence", {}) or {}
    canonical_ids = [
        verse.get("verse_id")
        for verse in evidence.get("canonical_verses", [])
        if verse.get("verse_id")
    ]
    commentary_ids = [
        match.get("verse_id")
        for match in evidence.get("commentary_matches", [])
        if match.get("verse_id")
    ]
    expected = [verse_id for verse_id in expected_verse_ids if verse_id]
    expected_set = set(expected)
    canonical_set = set(canonical_ids)
    commentary_set = set(commentary_ids)
    matched_canonical = [verse_id for verse_id in expected if verse_id in canonical_set]
    matched_commentary = [verse_id for verse_id in expected if verse_id in commentary_set]

    expected_coverage = len(matched_canonical) / len(expected) if expected else None
    first_rank = None
    for index, verse_id in enumerate(canonical_ids, 1):
        if verse_id in expected_set:
            first_rank = index
            break

    reciprocal_rank = (1.0 / first_rank) if first_rank else 0.0
    commentary_hit = bool(expected_set and expected_set & commentary_set)
    retrieval_quality = None
    if expected:
        retrieval_quality = (
            0.65 * float(expected_coverage or 0.0)
            + 0.25 * reciprocal_rank
            + 0.10 * (1.0 if commentary_hit else 0.0)
        )

    return {
        "expected_verse_ids": expected,
        "canonical_verse_ids": canonical_ids,
        "commentary_verse_ids": commentary_ids,
        "matched_canonical_verse_ids": matched_canonical,
        "matched_commentary_verse_ids": matched_commentary,
        "expected_coverage": round(expected_coverage, 4) if expected_coverage is not None else None,
        "first_expected_rank": first_rank,
        "reciprocal_rank": round(reciprocal_rank, 4),
        "retrieval_quality": round(retrieval_quality, 4) if retrieval_quality is not None else None,
        "explicit_verse_hit": bool(expected_set and expected_set & canonical_set),
        "top_verse_hit": bool(expected and canonical_ids and canonical_ids[0] in expected_set),
        "commentary_hit": commentary_hit,
        "abstained": bool(data.get("abstention_reason")),
        "confidence": data.get("confidence", 0.0),
    }
