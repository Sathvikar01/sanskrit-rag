"""Text normalization and lightweight evidence quality checks."""
from __future__ import annotations

import re
import unicodedata
from typing import Dict, List, Tuple


MOJIBAKE_MARKERS = ("Ã", "Â", "Ä", "Å", "á¹", "à¤", "à¥")
MORPHOLOGY_MARKERS = ("Case=", "Gender=", "Number=", "Person=", "Tense=", "Mood=", "VerbForm=", "Lemma=")


def clean_text(text: str) -> str:
    """Normalize whitespace/control characters without altering Sanskrit content."""
    text = unicodedata.normalize("NFC", text or "")
    text = text.replace("\ufeff", " ").replace("\u200b", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _token_repetition_ratio(text: str) -> float:
    tokens = re.findall(r"[\w\u0900-\u097F]+", text.lower())
    if len(tokens) < 8:
        return 0.0
    unique = len(set(tokens))
    return 1.0 - (unique / max(len(tokens), 1))


def score_text_quality(text: str) -> Dict[str, object]:
    """Score text quality from 0-1 and return reasons for weak evidence."""
    cleaned = clean_text(text)
    reasons: List[str] = []

    if not cleaned:
        return {"text": "", "quality_score": 0.0, "filtered_reason": "empty_text", "flags": ["empty_text"]}

    alpha_count = sum(1 for ch in cleaned if ch.isalpha())
    alpha_ratio = alpha_count / max(len(cleaned), 1)
    morphology_hits = sum(1 for marker in MORPHOLOGY_MARKERS if marker in cleaned)
    mojibake_hits = sum(cleaned.count(marker) for marker in MOJIBAKE_MARKERS)
    repetition_ratio = _token_repetition_ratio(cleaned)

    score = 1.0
    if len(cleaned) < 6:
        score -= 0.35
        reasons.append("too_short")
    if alpha_ratio < 0.35:
        score -= 0.25
        reasons.append("low_alpha_ratio")
    if morphology_hits >= 3 and len(cleaned) < 450:
        score -= 0.35
        reasons.append("morphology_only")
    if mojibake_hits >= 3:
        score -= 0.25
        reasons.append("possible_mojibake")
    if repetition_ratio > 0.58:
        score -= 0.2
        reasons.append("high_repetition")

    score = max(0.0, min(1.0, score))
    filtered_reason = ",".join(reasons) if score < 0.35 else ""
    return {
        "text": cleaned,
        "quality_score": round(score, 4),
        "filtered_reason": filtered_reason,
        "flags": reasons,
    }


def is_usable_text(text: str, min_score: float = 0.35) -> bool:
    """Return True if text is worth using as answer evidence."""
    return float(score_text_quality(text)["quality_score"]) >= min_score


def quality_for_items(items: List[Dict[str, object]], text_key: str = "text") -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    """Attach quality metadata and split usable versus filtered evidence items."""
    usable: List[Dict[str, object]] = []
    filtered: List[Dict[str, object]] = []

    for item in items:
        quality = score_text_quality(str(item.get(text_key, "") or ""))
        updated = {**item, **quality}
        if quality["filtered_reason"]:
            filtered.append(updated)
        else:
            usable.append(updated)

    return usable, filtered
