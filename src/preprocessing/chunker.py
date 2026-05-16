"""Chunker for creating verse + commentary chunks from parsed XML data."""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from src.preprocessing.iast_devanagari import get_converter
from src.preprocessing.xml_parser import MorphoData, SegmentationData, VerseData
from src.utils.logger import logger


@dataclass
class Chunk:
    """A retrieval chunk consisting of verse text and/or commentary."""

    chunk_id: str  # e.g., "BhG_1.1_verse", "BhG_1.1_sridhara"
    verse_ref: str  # e.g., "BhG 1.1"
    chapter_num: int
    verse_num: int
    chunk_type: str  # verse, commentary, combined
    commentator: Optional[str] = None  # sridhara, visvanatha, baladeva, or None for verse

    # Text content
    text_iast: str = ""
    text_devanagari: str = ""

    # Verse-specific
    verse_lines_iast: list[str] = field(default_factory=list)
    verse_lines_devanagari: list[str] = field(default_factory=list)
    speaker: str = ""

    # Commentary-specific
    commentary_iast: str = ""
    commentary_devanagari: str = ""

    # Linguistic annotations
    lemmas: list[str] = field(default_factory=list)
    morpho_features: list[str] = field(default_factory=list)
    surface_forms: list[str] = field(default_factory=list)

    # Metadata
    word_count: int = 0


def create_verse_chunk(verse: VerseData, converter=None) -> Chunk:
    """Create a chunk for the original verse text."""
    if converter is None:
        converter = get_converter()

    text_iast = " ".join(verse.verse_lines_iast)
    text_devanagari = converter.iast_to_devanagari(text_iast)

    return Chunk(
        chunk_id=f"{verse.ref.replace(' ', '_')}_verse",
        verse_ref=verse.ref,
        chapter_num=verse.chapter_num,
        verse_num=verse.verse_num,
        chunk_type="verse",
        text_iast=text_iast,
        text_devanagari=text_devanagari,
        verse_lines_iast=verse.verse_lines_iast,
        verse_lines_devanagari=converter.iast_to_devanagari_batch(verse.verse_lines_iast),
        speaker=verse.speaker,
        word_count=len(text_iast.split()),
    )


def create_commentary_chunk(
    verse: VerseData,
    commentator: str,
    lines: list[str],
    converter=None,
) -> Optional[Chunk]:
    """Create a chunk for a specific commentary."""
    if not lines:
        return None

    if converter is None:
        converter = get_converter()

    text_iast = " ".join(lines)
    text_devanagari = converter.iast_to_devanagari(text_iast)

    return Chunk(
        chunk_id=f"{verse.ref.replace(' ', '_')}_{commentator}",
        verse_ref=verse.ref,
        chapter_num=verse.chapter_num,
        verse_num=verse.verse_num,
        chunk_type="commentary",
        commentator=commentator,
        text_iast=text_iast,
        text_devanagari=text_devanagari,
        verse_lines_iast=verse.verse_lines_iast,
        verse_lines_devanagari=converter.iast_to_devanagari_batch(verse.verse_lines_iast),
        speaker=verse.speaker,
        commentary_iast=text_iast,
        commentary_devanagari=text_devanagari,
        word_count=len(text_iast.split()),
    )


def create_combined_chunk(
    verse: VerseData,
    commentator: str,
    commentary_lines: list[str],
    converter=None,
) -> Optional[Chunk]:
    """Create a combined chunk with verse + commentary."""
    if not commentary_lines:
        return None

    if converter is None:
        converter = get_converter()

    verse_text = " ".join(verse.verse_lines_iast)
    commentary_text = " ".join(commentary_lines)
    combined_iast = f"{verse_text} [SEP] {commentary_text}"
    combined_devanagari = converter.iast_to_devanagari(combined_iast)

    return Chunk(
        chunk_id=f"{verse.ref.replace(' ', '_')}_{commentator}_combined",
        verse_ref=verse.ref,
        chapter_num=verse.chapter_num,
        verse_num=verse.verse_num,
        chunk_type="combined",
        commentator=commentator,
        text_iast=combined_iast,
        text_devanagari=combined_devanagari,
        verse_lines_iast=verse.verse_lines_iast,
        verse_lines_devanagari=converter.iast_to_devanagari_batch(verse.verse_lines_iast),
        speaker=verse.speaker,
        commentary_iast=commentary_text,
        commentary_devanagari=converter.iast_to_devanagari(commentary_text),
        word_count=len(combined_iast.split()),
    )


def enrich_chunk_with_linguistics(
    chunk: Chunk,
    morpho: Optional[MorphoData],
    segmentation: Optional[SegmentationData],
) -> Chunk:
    """Add linguistic annotations to a chunk from morpho and segmentation data."""
    if morpho:
        chunk.morpho_features = morpho.verse_lines
    if segmentation:
        for line in segmentation.verse_lines:
            parts = line.split()
            for part in parts:
                if "_" in part:
                    surface, lemma = part.rsplit("_", 1)
                    if surface and lemma:
                        chunk.surface_forms.append(surface.rstrip("-"))
                        chunk.lemmas.append(lemma)
        chunk.lemmas = list(set(chunk.lemmas))
        chunk.surface_forms = list(set(chunk.surface_forms))
    return chunk


def create_all_chunks(
    verses: list[VerseData],
    morpho_data: list[MorphoData],
    segmentation_data: list[SegmentationData],
    chunk_types: list[str] = None,
) -> list[Chunk]:
    """Create all chunks from parsed data.

    Args:
        verses: Parsed verse data from main XML.
        morpho_data: Parsed morphological annotations.
        segmentation_data: Parsed segmentation/lemma data.
        chunk_types: Types of chunks to create. Options: verse, commentary, combined.
                     Default: all three.

    Returns:
        List of Chunk objects.
    """
    if chunk_types is None:
        chunk_types = ["verse", "commentary", "combined"]

    converter = get_converter()

    morpho_map = {m.ref: m for m in morpho_data}
    seg_map = {s.ref: s for s in segmentation_data}

    chunks = []
    commentators = ["sridhara", "visvanatha", "baladeva"]

    for verse in verses:
        morpho = morpho_map.get(verse.ref)
        seg = seg_map.get(verse.ref)

        if "verse" in chunk_types:
            chunk = create_verse_chunk(verse, converter)
            chunk = enrich_chunk_with_linguistics(chunk, morpho, seg)
            chunks.append(chunk)

        for comm in commentators:
            lines = getattr(verse, f"{comm}_lines", [])
            if not lines:
                continue

            if "commentary" in chunk_types:
                chunk = create_commentary_chunk(verse, comm, lines, converter)
                if chunk:
                    chunk = enrich_chunk_with_linguistics(chunk, morpho, seg)
                    chunks.append(chunk)

            if "combined" in chunk_types:
                chunk = create_combined_chunk(verse, comm, lines, converter)
                if chunk:
                    chunk = enrich_chunk_with_linguistics(chunk, morpho, seg)
                    chunks.append(chunk)

    logger.info(f"Created {len(chunks)} chunks from {len(verses)} verses")
    return chunks


def save_chunks(chunks: list[Chunk], output_path: str | Path) -> None:
    """Save chunks to a JSONL file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")

    logger.info(f"Saved {len(chunks)} chunks to {output_path}")


def load_chunks(input_path: str | Path) -> list[Chunk]:
    """Load chunks from a JSONL file."""
    input_path = Path(input_path)
    chunks = []

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            chunks.append(Chunk(**data))

    logger.info(f"Loaded {len(chunks)} chunks from {input_path}")
    return chunks
