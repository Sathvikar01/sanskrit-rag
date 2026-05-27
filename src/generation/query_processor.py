"""Query processor using local IAST conversion and concept extraction."""

import re
from dataclasses import dataclass, field

from src.preprocessing.concept_extractor import ConceptExtractor
from src.preprocessing.iast_devanagari import get_converter
from src.utils.config import Config
from src.utils.logger import logger


@dataclass
class ProcessedQuery:
    """Result of query processing."""

    original_query: str
    query_iast: str
    query_devanagari: str
    concepts: list[str] = field(default_factory=list)
    language_detected: str = ""
    extraction_confidence: float = 0.0


class QueryProcessor:
    """Process user queries locally using IAST converter and concept extractor."""

    def __init__(self, config: Config = None):
        if config is None:
            config = Config()

        self.converter = get_converter()
        self.concept_extractor = ConceptExtractor()

        logger.info("QueryProcessor initialized (local processing)")

    def detect_language(self, query: str) -> str:
        """Detect the language/script of the query."""
        devanagari_pattern = re.compile(r"[\u0900-\u097F]")
        iast_pattern = re.compile(r"[āīūṛṝḷḹṃḥśṣṇṭḍñ]")

        has_devanagari = bool(devanagari_pattern.search(query))
        has_iast = bool(iast_pattern.search(query))

        if has_devanagari:
            return "sanskrit_devanagari"
        elif has_iast:
            return "sanskrit_iast"
        else:
            return "english"

    def process_query(self, query: str) -> ProcessedQuery:
        """Process a user query to extract IAST text and concepts.

        Uses local tools: language detection, IAST/Devanagari conversion,
        and concept extraction. No LLM calls.

        Args:
            query: User's question in any language.

        Returns:
            ProcessedQuery with IAST text, Devanagari, and concepts.
        """
        language = self.detect_language(query)

        if language == "sanskrit_devanagari":
            query_iast = self.converter.devanagari_to_iast(query)
        elif language == "sanskrit_iast":
            query_iast = query
        else:
            query_iast = query

        query_devanagari = self.converter.iast_to_devanagari(query_iast)

        local_concepts = self.concept_extractor.extract_from_text(query_iast)
        concepts = [c["concept"].name_iast for c in local_concepts]

        result = ProcessedQuery(
            original_query=query,
            query_iast=query_iast,
            query_devanagari=query_devanagari,
            concepts=concepts,
            language_detected=language,
            extraction_confidence=0.8 if concepts else 0.3,
        )

        logger.info(
            f"Processed query: lang={language}, concepts={concepts}, "
            f"confidence={result.extraction_confidence:.2f}"
        )

        return result
