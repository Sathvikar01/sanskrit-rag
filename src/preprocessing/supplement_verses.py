"""Fetch missing Bhagavad Gita verses from external API."""

import json
import re
from pathlib import Path
from typing import Optional

import requests

from src.preprocessing.iast_devanagari import get_converter
from src.preprocessing.xml_parser import (
    COMMENTATOR_MARKERS,
    VerseData,
    detect_speaker,
)
from src.utils.logger import logger

API_BASE = "https://vedicscriptures.github.io/slok"

MISSING_VERSES: list[tuple[int, int]] = [
    (1, 38),
    (1, 47),
    (2, 35),
    (5, 9),
    (10, 26),
    (12, 6),
    (12, 7),
    (15, 6),
    (15, 9),
    (15, 13),
    (15, 16),
    (16, 2),
    (16, 3),
    (16, 9),
    (16, 13),
    (17, 4),
    (17, 14),
]


def _normalize_iast_line(line: str, ch: int, v: int) -> str:
    """Normalize a single IAST line from the API format to project format."""
    stripped = line.strip()
    stripped = re.sub(rf'\|\|{ch}-{v}\|\|', '', stripped)
    stripped = re.sub(r'\.(\s*)$', r'|\1', stripped)
    stripped = re.sub(r'\.$', '', stripped)
    if stripped and not stripped.endswith('|') and not re.search(r'\|\|\d+\|\|', stripped):
        stripped = stripped.rstrip() + ' |'
    return stripped


def _fetch_verse(ch: int, v: int) -> Optional[dict]:
    """Fetch a single verse from the API."""
    try:
        resp = requests.get(f"{API_BASE}/{ch}/{v}", timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.warning(f"Failed to fetch BhG {ch}.{v}: {e}")
        return None


def _verse_ref(ch: int, v: int) -> str:
    return f"BhG {ch}.{v}"


def get_missing_verses() -> list[VerseData]:
    """Fetch all missing verses from the external API.

    Returns:
        List of VerseData objects for the missing verses.
    """
    converter = get_converter()
    verses: list[VerseData] = []

    for ch, v in MISSING_VERSES:
        ref = _verse_ref(ch, v)
        data = _fetch_verse(ch, v)
        if data is None:
            logger.warning(f"Skipping {ref}: API fetch failed")
            continue

        transliteration = data.get("transliteration", "")
        if not transliteration:
            logger.warning(f"Skipping {ref}: no transliteration in response")
            continue

        raw_lines = transliteration.split("\n")
        iast_lines = []
        for line in raw_lines:
            normalized = _normalize_iast_line(line, ch, v)
            if normalized:
                iast_lines.append(normalized)

        if not iast_lines:
            logger.warning(f"Skipping {ref}: no IAST lines after normalization")
            continue

        speaker = detect_speaker(iast_lines)

        verse = VerseData(
            ref=ref,
            chapter_num=ch,
            verse_num=v,
            verse_lines_iast=iast_lines,
            speaker=speaker,
        )
        verses.append(verse)
        logger.info(f"  Supplemented {ref} (speaker: {speaker or 'unknown'})")

    logger.info(f"Supplemented {len(verses)}/{len(MISSING_VERSES)} missing verses")
    return verses


def save_supplementary_verses(verses: list[VerseData], output_path: str | Path) -> None:
    """Save supplementary verses to a JSON file for reuse."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def _serialize(v: VerseData) -> dict:
        return {
            "ref": v.ref,
            "chapter_num": v.chapter_num,
            "verse_num": v.verse_num,
            "verse_lines_iast": v.verse_lines_iast,
            "speaker": v.speaker,
        }

    data = [_serialize(v) for v in verses]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved {len(verses)} supplementary verses to {output_path}")


def load_supplementary_verses(filepath: str | Path) -> list[VerseData]:
    """Load supplementary verses from saved JSON file."""
    filepath = Path(filepath)
    if not filepath.exists():
        return []

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    verses = []
    for item in data:
        verses.append(
            VerseData(
                ref=item["ref"],
                chapter_num=item["chapter_num"],
                verse_num=item["verse_num"],
                verse_lines_iast=item["verse_lines_iast"],
                speaker=item.get("speaker", ""),
            )
        )
    return verses
