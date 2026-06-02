"""Tests for XML Parser and Semantic Chunking."""
import unittest
import tempfile
import os
from pathlib import Path
from lxml import etree

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.xml_parser import (
    TEIXMLParser,
    SemanticChunker,
    TextChunk,
    parse_sanskrit_text
)


class TestSemanticChunker(unittest.TestCase):
    """Test cases for SemanticChunker class."""
    
    def setUp(self):
        self.chunker = SemanticChunker(chunk_size=100, overlap=20)
    
    def test_count_tokens(self):
        text = "dharma kṣetra kuru kṣetra"
        count = self.chunker.count_tokens(text)
        self.assertGreater(count, 0)
        self.assertIsInstance(count, int)
    
    def test_short_text_no_split(self):
        text = "dharma kṣetra"
        chunks = list(self.chunker.chunk(text))
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].text, text)
    
    def test_long_text_splits(self):
        text = " ".join(["dharma kṣetra kuru saṃjaya"] * 100)
        chunks = list(self.chunker.chunk(text))
        self.assertGreaterEqual(len(chunks), 1)
    
    def test_chunk_id_generation(self):
        text = "test text for id generation"
        id1 = self.chunker._generate_id(text)
        id2 = self.chunker._generate_id(text)
        self.assertEqual(id1, id2)
        self.assertEqual(len(id1), 16)
    
    def test_sentence_split_sanskrit(self):
        text = "dhṛtarāṣṭra uvāca। dharma-kṣetre kuru-kṣetre॥"
        sentences = self.chunker._split_sentences(text)
        self.assertEqual(len(sentences), 2)
    
    def test_metadata_preserved(self):
        text = "dharma kṣetra"
        metadata = {"dataset_type": "raw", "verse_id": "BhG 1.1"}
        chunks = list(self.chunker.chunk(text, metadata))
        self.assertEqual(chunks[0].dataset_type, "raw")
        self.assertEqual(chunks[0].verse_id, "BhG 1.1")


class TestTextChunk(unittest.TestCase):
    """Test cases for TextChunk dataclass."""
    
    def test_to_dict(self):
        chunk = TextChunk(
            id="test123",
            text="dharma kṣetra",
            dataset_type="raw",
            verse_id="BhG 1.1",
            element_type="lg",
            line_number=1,
            metadata={"key": "value"}
        )
        result = chunk.to_dict()
        
        self.assertEqual(result["id"], "test123")
        self.assertEqual(result["text"], "dharma kṣetra")
        self.assertEqual(result["dataset_type"], "raw")
        self.assertEqual(result["verse_id"], "BhG 1.1")
        self.assertEqual(result["metadata"]["key"], "value")


class TestTEIXMLParser(unittest.TestCase):
    """Test cases for TEIXMLParser class."""
    
    def setUp(self):
        self.parser = TEIXMLParser()
        self.sample_xml = """<?xml version='1.0' encoding='utf-8'?>
<TEI xmlns="http://www.tei-c.org/ns/1.0" xml:id="sa_test">
<text xml:lang="sa-Latn">
<body>
<div>
<p>BhG 1.1</p>
<lg>
<l>dharma-kṣetre kuru-kṣetre samavetā yuyutsavaḥ</l>
<l>māmakāḥ pāṇḍavāś caiva kim akurvata saṃjaya</l>
</lg>
<p>This is a prose commentary on the verse.</p>
</div>
</body>
</text>
</TEI>"""
    
    def test_parse_sample_xml(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False, encoding='utf-8') as f:
            f.write(self.sample_xml)
            f.flush()
            temp_path = f.name
        try:
            chunks = self.parser.parse_file(temp_path, "test")
            self.assertGreater(len(chunks), 0)
        finally:
            os.unlink(temp_path)

    def test_extract_verse_id(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False, encoding='utf-8') as f:
            f.write(self.sample_xml)
            f.flush()
            temp_path = f.name
        try:
            chunks = self.parser.parse_file(temp_path, "test")
            verse_ids = [c.verse_id for c in chunks if c.verse_id]
            self.assertTrue(any("BhG" in vid for vid in verse_ids))
        finally:
            os.unlink(temp_path)

    def test_element_type_extraction(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False, encoding='utf-8') as f:
            f.write(self.sample_xml)
            f.flush()
            temp_path = f.name
        try:
            chunks = self.parser.parse_file(temp_path, "test")
            element_types = set(c.element_type for c in chunks)
            self.assertTrue(len(element_types) > 0)
        finally:
            os.unlink(temp_path)

    def test_commentary_marker_normalization_variants(self):
        variants = {
            "Śrīdharaḥ -": "shreedhara",
            "Baladevaḥ_Baladeva": "baladeva",
            "Viśvanātha_Case=Nom|Gender=Masc|Number=Sing": "vishwanatha",
            "śrīdhara__": "shreedhara",
        }

        for label, expected in variants.items():
            element = etree.fromstring(f"<p>{label}</p>")
            self.assertEqual(self.parser._get_commentary_author(element), expected)

    def test_parse_file_uses_div_verse_fallback_for_empty_markers(self):
        sample_xml = """<?xml version='1.0' encoding='utf-8'?>
<TEI xmlns="http://www.tei-c.org/ns/1.0"><text><body>
<div><p>BhG 1.1</p><lg><l>first verse</l></lg></div>
<div><p /><lg><l>range verse text</l></lg></div>
</body></text></TEI>"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False, encoding='utf-8') as f:
            f.write(sample_xml)
            f.flush()
            temp_path = f.name
        try:
            chunks = self.parser.parse_file(
                temp_path,
                "seg_lemma",
                div_verse_ids=[["BhG 1.1"], ["BhG 1.4", "BhG 1.5", "BhG 1.6"]],
            )
            verse_ids = {c.verse_id for c in chunks}
            self.assertTrue({"BhG 1.4", "BhG 1.5", "BhG 1.6"}.issubset(verse_ids))
        finally:
            os.unlink(temp_path)


class TestParseSanskritText(unittest.TestCase):
    """Test cases for parse_sanskrit_text function."""
    
    def test_word_count(self):
        text = "dharma kṣetra kuru"
        features = parse_sanskrit_text(text)
        self.assertEqual(features["word_count"], 3)
    
    def test_sanskrit_detection(self):
        text = "धर्म क्षेत्र"
        features = parse_sanskrit_text(text)
        self.assertTrue(features["has_sanskrit"])
    
    def test_no_sanskrit(self):
        text = "dharma ksetra"
        features = parse_sanskrit_text(text)
        self.assertFalse(features["has_sanskrit"])
    
    def test_whitespace_normalization(self):
        text = "dharma   kṣetra    kuru"
        features = parse_sanskrit_text(text)
        self.assertEqual(features["word_count"], 3)


if __name__ == "__main__":
    unittest.main()
