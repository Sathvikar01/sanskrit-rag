"""XML Parser for TEI-encoded Sanskrit texts with semantic chunking.

Enhanced Features:
- Semantic chunking based on verse boundaries and grammatical structures
- Metadata enrichment (author, chapter, meter, context, cross-references)
- Separate extraction for commentaries (Vishwanatha, Shreedhara, Baladeva)
- Cross-reference detection between verses
"""
import hashlib
import re
import unicodedata
from pathlib import Path
from typing import List, Dict, Any, Optional, Generator, Tuple
from dataclasses import dataclass, field
from lxml import etree
import tiktoken

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import CHUNK_SIZE_TOKENS, CHUNK_OVERLAP_TOKENS, MAX_TEXT_LENGTH


@dataclass
class TextChunk:
    """Represents a semantic chunk of text with enriched metadata."""
    id: str
    text: str
    dataset_type: str
    verse_id: Optional[str] = None
    element_type: Optional[str] = None
    line_number: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "dataset_type": self.dataset_type,
            "verse_id": self.verse_id,
            "element_type": self.element_type,
            "line_number": self.line_number,
            "metadata": self.metadata
        }


@dataclass
class CommentaryChunk:
    """Represents a commentary chunk with author attribution."""
    id: str
    text: str
    author: str
    verse_id: str
    chapter: int
    verse_num: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "author": self.author,
            "verse_id": self.verse_id,
            "chapter": self.chapter,
            "verse_num": self.verse_num,
            "metadata": self.metadata
        }


@dataclass
class VerseMetadata:
    """Rich metadata for a verse."""
    verse_id: str
    chapter: int
    verse_num: int
    text: str
    meter: Optional[str] = None
    authors_referenced: List[str] = field(default_factory=list)
    cross_references: List[str] = field(default_factory=list)
    grammatical_features: Dict[str, List[str]] = field(default_factory=dict)
    key_terms: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verse_id": self.verse_id,
            "chapter": self.chapter,
            "verse_num": self.verse_num,
            "text": self.text,
            "meter": self.meter,
            "authors_referenced": self.authors_referenced,
            "cross_references": self.cross_references,
            "grammatical_features": self.grammatical_features,
            "key_terms": self.key_terms
        }


class SemanticChunker:
    """Semantic text chunker using token counting with verse-aware boundaries."""

    COMMENTARY_AUTHORS = {
        "shreedhara": ["shreedhara", "sridhara", "shridhara", "sridharah"],
        "vishwanatha": ["vishwanatha", "visvanatha", "viswanatha", "visvanathah"],
        "baladeva": ["baladeva", "baladevah"],
    }

    SANSKRIT_STOP_WORDS = {
        "ca", "api", "eva", "hi", "tu", "atha", "tad", "yat", "kim",
        "idam", "etad", "asya", "tasya", "atra", "tatra", "sarva",
        "na", "nanu", "iti", "ha", "khalu", "bhavati", "asti"
    }

    KEY_TERMS = {
        "dharma", "karma", "mokṣa", "yoga", "bhakti", "jñāna",
        "ātman", "brahman", "prakṛti", "puruṣa", "guṇa", "māyā",
        "sattva", "rajas", "tamas", "saṃsāra", "nirvāṇa"
    }

    def __init__(self, chunk_size: int = CHUNK_SIZE_TOKENS, overlap: int = CHUNK_OVERLAP_TOKENS):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.encoder = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        return len(self.encoder.encode(text))

    def chunk(self, text: str, metadata: Dict[str, Any] = None) -> Generator[TextChunk, None, None]:
        if metadata is None:
            metadata = {}

        tokens = self.encoder.encode(text)
        if len(tokens) <= self.chunk_size:
            chunk_id = self._generate_id(text)
            yield TextChunk(
                id=chunk_id,
                text=text,
                dataset_type=metadata.get("dataset_type", "unknown"),
                verse_id=metadata.get("verse_id"),
                element_type=metadata.get("element_type"),
                line_number=metadata.get("line_number", 0),
                metadata=self._enrich_metadata(text, metadata)
            )
            return

        sentences = self._split_sentences(text)
        current_chunk = []
        current_tokens = 0
        chunk_idx = 0

        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)

            if current_tokens + sentence_tokens > self.chunk_size and current_chunk:
                chunk_text = " ".join(current_chunk)
                chunk_id = self._generate_id(chunk_text + str(chunk_idx))
                yield TextChunk(
                    id=chunk_id,
                    text=chunk_text,
                    dataset_type=metadata.get("dataset_type", "unknown"),
                    verse_id=metadata.get("verse_id"),
                    element_type=metadata.get("element_type"),
                    line_number=metadata.get("line_number", 0),
                    metadata=self._enrich_metadata(chunk_text, {**metadata, "chunk_idx": chunk_idx})
                )
                chunk_idx += 1

                overlap_text = current_chunk[-self._get_overlap_sentences(current_chunk):]
                current_chunk = overlap_text
                current_tokens = sum(self.count_tokens(s) for s in current_chunk)

            current_chunk.append(sentence)
            current_tokens += sentence_tokens

        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunk_id = self._generate_id(chunk_text + str(chunk_idx))
            yield TextChunk(
                id=chunk_id,
                text=chunk_text,
                dataset_type=metadata.get("dataset_type", "unknown"),
                verse_id=metadata.get("verse_id"),
                element_type=metadata.get("element_type"),
                line_number=metadata.get("line_number", 0),
                metadata=self._enrich_metadata(chunk_text, {**metadata, "chunk_idx": chunk_idx})
            )

    def _split_sentences(self, text: str) -> List[str]:
        delimiters = r'(?<=[।॥\.\?\!])\s+'
        sentences = re.split(delimiters, text)
        return [s.strip() for s in sentences if s.strip()]

    def _get_overlap_sentences(self, chunk: List[str]) -> int:
        total_tokens = sum(self.count_tokens(s) for s in chunk)
        overlap_tokens = 0
        overlap_count = 0
        for s in reversed(chunk):
            s_tokens = self.count_tokens(s)
            if overlap_tokens + s_tokens > self.overlap:
                break
            overlap_tokens += s_tokens
            overlap_count += 1
        return max(1, overlap_count)

    def _generate_id(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()[:16]

    def _enrich_metadata(self, text: str, base_metadata: Dict[str, Any]) -> Dict[str, Any]:
        enriched = dict(base_metadata)

        key_terms_found = []
        for term in self.KEY_TERMS:
            if term in text.lower():
                key_terms_found.append(term)
        if key_terms_found:
            enriched["key_terms"] = key_terms_found

        iast_words = re.findall(r'[\u0900-\u097F]+|[a-zA-Zāīūṛṝḷḹéóṃḥṅñṭḍṇśṣ]+', text)
        word_count = len(iast_words)
        enriched["word_count"] = word_count
        enriched["has_sanskrit"] = any('\u0900' <= c <= '\u097F' for c in text)

        enriched["estimated_reading_time"] = max(1, word_count // 15)

        return enriched


class TEIXMLParser:
    """Parser for TEI-encoded Sanskrit texts with commentary extraction."""

    TEI_NS = {"tei": "http://www.tei-c.org/ns/1.0"}

    COMMENTARY_MARKERS = {
        "shreedhara": ["shreedhara", "sridhara", "shridhara", "sridharah"],
        "vishwanatha": ["vishwanatha", "visvanatha", "viswanatha", "visvanathah"],
        "baladeva": ["baladeva", "baladevah"],
    }

    def __init__(self, chunker: Optional[SemanticChunker] = None):
        self.chunker = chunker or SemanticChunker()
        self._current_verse_id = None
        self._current_chapter = 0
        self._current_verse_num = 0
        self._current_commentary_author = None

    def parse_file(self, file_path: str, dataset_type: str) -> List[TextChunk]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        tree = etree.parse(str(path), etree.XMLParser(remove_blank_text=True))
        root = tree.getroot()

        chunks = []
        current_verse_id = None
        current_verse_ids = []
        current_chapter = 0
        current_verse_num = 0

        for element in root.iter():
            if self._is_verse_marker(element):
                current_verse_ids = self._extract_verse_ids(element)
                if current_verse_ids:
                    current_verse_id = current_verse_ids[0]
                    current_chapter, current_verse_num = self._parse_verse_id(current_verse_id)
            elif element.tag.endswith("p") or element.tag.endswith("lg"):
                text_elements = self._extract_text_elements(element)
                for text, line_num in text_elements:
                    if text.strip():
                        for verse_id in (current_verse_ids or [current_verse_id]):
                            chapter, verse_num = self._parse_verse_id(verse_id)
                            metadata = {
                                "dataset_type": dataset_type,
                                "verse_id": verse_id,
                                "chapter": chapter or current_chapter,
                                "verse_num": verse_num or current_verse_num,
                                "element_type": element.tag.split("}")[-1] if "}" in element.tag else element.tag,
                                "line_number": line_num
                            }
                            chunks.extend(list(self.chunker.chunk(text, metadata)))

        return chunks

    def parse_with_commentaries(
        self,
        file_path: str,
        dataset_type: str
    ) -> Tuple[List[TextChunk], Dict[str, List[CommentaryChunk]]]:
        """Parse file and extract both main text and commentaries separately.

        Returns:
            Tuple of (main_chunks, commentaries_dict) where commentaries_dict
            has keys: 'vishwanatha', 'shreedhara', 'baladeva'
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        tree = etree.parse(str(path), etree.XMLParser(remove_blank_text=True))
        root = tree.getroot()

        main_chunks = []
        commentaries = {
            "vishwanatha": [],
            "shreedhara": [],
            "baladeva": []
        }

        current_verse_id = None
        current_verse_ids = []
        current_chapter = 0
        current_verse_num = 0
        current_author = None

        for element in root.iter():
            if self._is_verse_marker(element):
                current_verse_ids = self._extract_verse_ids(element)
                if current_verse_ids:
                    current_verse_id = current_verse_ids[0]
                    current_chapter, current_verse_num = self._parse_verse_id(current_verse_id)
                current_author = None
            elif self._is_commentary_marker(element):
                current_author = self._get_commentary_author(element)
            elif element.tag.endswith("p") or element.tag.endswith("lg"):
                text_elements = self._extract_text_elements(element)
                for text, line_num in text_elements:
                    if text.strip():
                        if current_author:
                            chunk_id = self._generate_commentary_id(
                                current_author, current_verse_id, line_num
                            )
                            commentary_chunk = CommentaryChunk(
                                id=chunk_id,
                                text=text,
                                author=current_author,
                                verse_id=current_verse_id,
                                chapter=current_chapter,
                                verse_num=current_verse_num,
                                metadata={
                                    "line_number": line_num,
                                    "source_type": "commentary"
                                }
                            )
                            commentaries[current_author].append(commentary_chunk)
                        else:
                            for verse_id in (current_verse_ids or [current_verse_id]):
                                chapter, verse_num = self._parse_verse_id(verse_id)
                                metadata = {
                                    "dataset_type": dataset_type,
                                    "verse_id": verse_id,
                                    "chapter": chapter or current_chapter,
                                    "verse_num": verse_num or current_verse_num,
                                    "element_type": element.tag.split("}")[-1] if "}" in element.tag else element.tag,
                                    "line_number": line_num
                                }
                                main_chunks.extend(list(self.chunker.chunk(text, metadata)))

        return main_chunks, commentaries

    def _is_verse_marker(self, element) -> bool:
        tag = element.tag.split("}")[-1] if "}" in element.tag else element.tag
        if tag == "p":
            text = element.text or ""
            return bool(re.match(r'^BhG\s*\d+\.\d+', text.strip()))
        return False

    def _extract_verse_info(self, element) -> Optional[Tuple[int, int]]:
        text = element.text or ""
        match = re.match(r'^BhG\s*(\d+)\.(\d+)', text.strip())
        if match:
            return int(match.group(1)), int(match.group(2))
        return None

    def _extract_verse_ids(self, element) -> List[str]:
        text = element.text or ""
        match = re.match(r'^BhG\s*(\d+)\.(\d+)(?:-(\d+))?', text.strip())
        if not match:
            return []
        chapter = int(match.group(1))
        verse_start = int(match.group(2))
        verse_end = int(match.group(3)) if match.group(3) else verse_start
        return [f"BhG {chapter}.{verse_num}" for verse_num in range(verse_start, verse_end + 1)]

    def _parse_verse_id(self, verse_id: Optional[str]) -> Tuple[int, int]:
        match = re.match(r'^BhG\s*(\d+)\.(\d+)', verse_id or "")
        if not match:
            return 0, 0
        return int(match.group(1)), int(match.group(2))

    def _is_commentary_marker(self, element) -> bool:
        tag = element.tag.split("}")[-1] if "}" in element.tag else element.tag
        if tag == "p":
            return self._get_commentary_author(element) is not None
        return False

    def _get_commentary_author(self, element) -> Optional[str]:
        text = self._normalize_commentary_text((element.text or "").strip())
        for author, aliases in self.COMMENTARY_MARKERS.items():
            for alias in aliases:
                if text == alias or text.startswith(f"{alias} "):
                    return author
        return None

    def _normalize_commentary_text(self, text: str) -> str:
        folded = unicodedata.normalize("NFKD", text or "")
        folded = "".join(ch for ch in folded if not unicodedata.combining(ch)).lower()
        folded = folded.replace("|", " ").replace("_", " ").replace("-", " ")
        folded = re.sub(r"[^a-z\s]", " ", folded)
        return re.sub(r"\s+", " ", folded).strip()

    def _extract_text_elements(self, element) -> List[tuple]:
        results = []
        tag = element.tag.split("}")[-1] if "}" in element.tag else element.tag

        if tag == "lg":
            for idx, line in enumerate(element.iterchildren()):
                if line.tag.endswith("l"):
                    text = self._get_element_text(line)
                    if text.strip():
                        results.append((text, idx + 1))
        elif tag == "p":
            text = self._get_element_text(element)
            if text.strip():
                results.append((text, 0))

        return results

    def _get_element_text(self, element) -> str:
        texts = []
        if element.text:
            texts.append(element.text)
        for child in element:
            if child.text:
                texts.append(child.text)
            if child.tail:
                texts.append(child.tail)
        return " ".join(texts).strip()

    def _generate_commentary_id(self, author: str, verse_id: str, line_num: int) -> str:
        verse_str = verse_id.replace(" ", "_").replace(".", "-") if verse_id else "unknown"
        return f"{author}_{verse_str}_L{line_num}_{hashlib.md5(f'{author}{verse_id}{line_num}'.encode()).hexdigest()[:8]}"

    def parse_all_datasets(self, base_dir: str) -> Dict[str, List[TextChunk]]:
        base_path = Path(base_dir)

        datasets = {
            "raw": base_path / "dataset.xml",
            "lemma_morph": base_path / "dataset.lemma-morphosyntax.xml",
            "seg_lemma": base_path / "dataset.segmentation-lemma.xml"
        }

        results = {}
        for dtype, filepath in datasets.items():
            if filepath.exists():
                print(f"Parsing {filepath.name}...")
                results[dtype] = self.parse_file(str(filepath), dtype)
                print(f" Extracted {len(results[dtype])} chunks")
            else:
                print(f"Warning: {filepath} not found")

        return results

    def parse_all_with_commentaries(
        self,
        base_dir: str
    ) -> Tuple[Dict[str, List[TextChunk]], Dict[str, Dict[str, List[CommentaryChunk]]]]:
        """Parse all datasets with separate commentary extraction.

        Returns:
            Tuple of (main_chunks_dict, commentaries_dict) where:
            - main_chunks_dict: {dataset_type: List[TextChunk]}
            - commentaries_dict: {dataset_type: {author: List[CommentaryChunk]}}
        """
        base_path = Path(base_dir)

        datasets = {
            "raw": base_path / "dataset.xml",
            "lemma_morph": base_path / "dataset.lemma-morphosyntax.xml",
            "seg_lemma": base_path / "dataset.segmentation-lemma.xml"
        }

        main_results = {}
        commentary_results = {}

        for dtype, filepath in datasets.items():
            if filepath.exists():
                print(f"Parsing {filepath.name} with commentary extraction...")
                main_chunks, commentaries = self.parse_with_commentaries(
                    str(filepath), dtype
                )
                main_results[dtype] = main_chunks
                commentary_results[dtype] = commentaries
                print(f" Main text: {len(main_chunks)} chunks")
                for author, chunks in commentaries.items():
                    if chunks:
                        print(f" {author}: {len(chunks)} commentary chunks")
            else:
                print(f"Warning: {filepath} not found")

        return main_results, commentary_results

    def extract_verse_metadata(self, verse_id: str, text: str) -> VerseMetadata:
        """Extract rich metadata for a verse."""
        match = re.match(r'BhG\s*(\d+)\.(\d+)', verse_id)
        if match:
            chapter = int(match.group(1))
            verse_num = int(match.group(2))
        else:
            chapter = 0
            verse_num = 0

        meter = self._detect_meter(text)

        key_terms = self._extract_key_terms(text)

        cross_refs = self._extract_cross_references(text)

        gram_features = self._extract_grammatical_features(text)

        return VerseMetadata(
            verse_id=verse_id,
            chapter=chapter,
            verse_num=verse_num,
            text=text,
            meter=meter,
            key_terms=key_terms,
            cross_references=cross_refs,
            grammatical_features=gram_features
        )

    def _detect_meter(self, text: str) -> Optional[str]:
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        if len(lines) == 4:
            return "anushtubh"
        elif len(lines) == 2:
            return "dvipada"
        elif len(lines) == 3:
            return "trishtubh"
        return None

    def _extract_key_terms(self, text: str) -> List[str]:
        found = []
        text_lower = text.lower()
        for term in SemanticChunker.KEY_TERMS:
            if term in text_lower:
                found.append(term)
        return found

    def _extract_cross_references(self, text: str) -> List[str]:
        pattern = r'(?:BhG|BG)\s*\d+\.\d+(?:-\d+)?'
        return list(set(re.findall(pattern, text, re.IGNORECASE)))

    def _extract_grammatical_features(self, text: str) -> Dict[str, List[str]]:
        features = {}

        case_pattern = r'Case=(\w+)'
        cases = re.findall(case_pattern, text)
        if cases:
            features['cases'] = list(set(cases))

        gender_pattern = r'Gender=(\w+)'
        genders = re.findall(gender_pattern, text)
        if genders:
            features['genders'] = list(set(genders))

        number_pattern = r'Number=(\w+)'
        numbers = re.findall(number_pattern, text)
        if numbers:
            features['numbers'] = list(set(numbers))

        return features


def parse_sanskrit_text(text: str) -> Dict[str, Any]:
    """Parse and clean Sanskrit text, extracting linguistic features."""
    text = re.sub(r'\s+', ' ', text).strip()

    words = re.findall(r'[\u0900-\u097F]+|[a-zA-Zāīūṛṝḷḹéóṃḥṅñṭḍṇśṣ]+', text)

    features = {
        "word_count": len(words),
        "char_count": len(text),
        "has_sanskrit": any('\u0900' <= c <= '\u097F' for c in text),
        "words": words[:50],
        "key_concepts": []
    }

    for term in SemanticChunker.KEY_TERMS:
        if term in text.lower():
            features["key_concepts"].append(term)

    return features


if __name__ == "__main__":
    parser = TEIXMLParser()
    base_dir = Path(__file__).parent.parent

    main_chunks, commentaries = parser.parse_all_with_commentaries(str(base_dir))

    total_main = sum(len(c) for c in main_chunks.values())
    print(f"\nTotal main text chunks: {total_main}")

    for dtype, comm_dict in commentaries.items():
        print(f"\n{dtype} commentaries:")
        for author, chunks in comm_dict.items():
            if chunks:
                print(f" {author}: {len(chunks)} chunks")
