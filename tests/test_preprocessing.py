"""Tests for SRAG preprocessing modules."""

import tempfile
from pathlib import Path

import pytest

from src.preprocessing.chunker import Chunk, save_chunks, load_chunks
from src.preprocessing.concept_extractor import ConceptExtractor
from src.preprocessing.iast_devanagari import IASTConverter
from src.preprocessing.morpho_extractor import (
    build_morphological_profile,
    extract_morpho_features_from_line,
    extract_lemmas_from_segmentation_line,
)
from src.preprocessing.xml_parser import parse_verse_ref


class TestIASTConverter:
    """Test IAST to Devanagari conversion."""

    def test_basic_conversion(self):
        converter = IASTConverter()
        result = converter.iast_to_devanagari("dharma")
        assert result == "धर्म"

    def test_complex_conversion(self):
        converter = IASTConverter()
        result = converter.iast_to_devanagari("dharma-kṣetre kuru-kṣetre")
        assert "धर्म" in result
        assert "क्षेत्रे" in result

    def test_roundtrip(self):
        converter = IASTConverter()
        iast = "dhṛtarāṣṭra uvāca"
        deva = converter.iast_to_devanagari(iast)
        back = converter.devanagari_to_iast(deva)
        assert "dhṛtarāṣṭra" in back

    def test_empty_string(self):
        converter = IASTConverter()
        assert converter.iast_to_devanagari("") == ""
        assert converter.iast_to_devanagari(None) is None or converter.iast_to_devanagari(None) == ""

    def test_batch_conversion(self):
        converter = IASTConverter()
        texts = ["dharma", "karma", "yoga"]
        results = converter.iast_to_devanagari_batch(texts)
        assert len(results) == 3
        assert results[0] == "धर्म"


class TestVerseRefParsing:
    """Test verse reference parsing."""

    def test_valid_ref(self):
        chapter, verse = parse_verse_ref("BhG 1.1")
        assert chapter == 1
        assert verse == 1

    def test_complex_ref(self):
        chapter, verse = parse_verse_ref("BhG 18.78")
        assert chapter == 18
        assert verse == 78

    def test_invalid_ref(self):
        with pytest.raises(ValueError):
            parse_verse_ref("invalid")


class TestMorphoExtractor:
    """Test morphological feature extraction."""

    def test_parse_token(self):
        token = "dharma_Case=Loc|Gender=Neut|Number=Sing"
        result = extract_morpho_features_from_line(token)
        assert len(result) == 1
        assert result[0]["lemma"] == "dharma"
        assert result[0]["features"]["Case"] == "Loc"
        assert result[0]["features"]["Gender"] == "Neut"

    def test_parse_line(self):
        line = "dharma_Case=Loc kṣetra_Case=Loc|Gender=Neut|Number=Sing"
        result = extract_morpho_features_from_line(line)
        assert len(result) == 2

    def test_build_profile(self):
        lines = [
            "dharma_Case=Nom|Gender=Masc|Number=Sing",
            "karma_Case=Acc|Gender=Neut|Number=Sing",
        ]
        profile = build_morphological_profile(lines)
        assert profile.total_tokens == 2
        assert profile.noun_count == 2

    def test_segmentation_extraction(self):
        line = "dhṛtarāṣṭraḥ_dhṛtarāṣṭra uvāca_vac"
        result = extract_lemmas_from_segmentation_line(line)
        assert len(result) == 2
        assert result[0]["surface"] == "dhṛtarāṣṭraḥ"
        assert result[0]["lemma"] == "dhṛtarāṣṭra"


class TestConceptExtractor:
    """Test concept extraction."""

    def test_extract_dharma(self):
        extractor = ConceptExtractor()
        found = extractor.extract_from_text("dharma-kṣetre kuru-kṣetre")
        concepts = [f["concept"].name_iast for f in found]
        assert "dharma" in concepts

    def test_extract_multiple(self):
        extractor = ConceptExtractor()
        found = extractor.extract_from_text("karma yoga bhakti")
        concepts = [f["concept"].name_iast for f in found]
        assert "karma" in concepts
        assert "yoga" in concepts
        assert "bhakti" in concepts

    def test_get_concept(self):
        extractor = ConceptExtractor()
        concept = extractor.get_concept_by_name("dharma")
        assert concept is not None
        assert concept.name_english == "duty/righteousness"

    def test_related_concepts(self):
        extractor = ConceptExtractor()
        related = extractor.get_related_concepts("karma")
        names = [c.name_iast for c in related]
        assert "dharma" in names or "moksha" in names

    def test_list_all(self):
        extractor = ConceptExtractor()
        all_concepts = extractor.list_all_concepts()
        assert len(all_concepts) >= 20


class TestChunker:
    """Test chunk creation."""

    def test_chunk_save_load(self):
        chunk = Chunk(
            chunk_id="test_verse",
            verse_ref="BhG 1.1",
            chapter_num=1,
            verse_num=1,
            chunk_type="verse",
            text_iast="test text",
            text_devanagari="टेस्ट",
            word_count=2,
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            temp_path = f.name

        save_chunks([chunk], temp_path)
        loaded = load_chunks(temp_path)

        assert len(loaded) == 1
        assert loaded[0].chunk_id == "test_verse"
        assert loaded[0].text_iast == "test text"

        Path(temp_path).unlink()
