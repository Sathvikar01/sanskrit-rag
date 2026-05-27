"""Query processor using MiMo v2.5 for IAST translation and local tools for concepts."""

import json
import re
from dataclasses import dataclass, field

from openai import OpenAI

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


TRANSLATION_PROMPT = """You are a Sanskrit language expert. Your task is IAST TRANSLITERATION, not translation to Hindi.

Query: "{query}"

Rules:
- Convert ALL Sanskrit/Hindi terms in the query to proper IAST transliteration with diacritics
- IAST uses diacritics: ā, ī, ū, ṛ, ṝ, ḷ, ṃ, ḥ, ś, ṣ, ṇ, ṭ, ḍ, ñ, ṅ, ṭh, ḍh
- Keep the query structure in the SAME LANGUAGE - just transliterate Sanskrit words to IAST
- Do NOT translate to Hindi. Only transliterate Sanskrit terms.
- Respond ONLY with the transliterated text, no explanations

Examples:
"What is dharma?" → "What is dharma?"
"Explain karma yoga" → "Explain karma yoga"
"What does Bhagavad Gita say about moksha?" → "What does bhagavad gītā say about moksha?"
"Tell me about the soul" → "Tell me about the ātman"
"How to achieve liberation?" → "How to achieve mokṣa?"
"dharma karmāt sambandhita hai" → "dharma karmāt sambandhita hai"
"bhagavad gita mein kya kaha gaya hai" → "bhagavad gītā mein kyā kaha gayā hai" """


class QueryProcessor:
    """Process user queries using MiMo for IAST translation + local tools for concepts."""

    def __init__(self, config: Config = None):
        if config is None:
            config = Config()

        api_base = config.get("generation.mimo.api_base", "https://token-plan-sgp.xiaomimimo.com/v1")
        model = config.get("generation.mimo.model", "mimo-v2.5")

        self.client = OpenAI(
            api_key=config.mimo_api_key,
            base_url=api_base,
        )
        self.model = model
        self.converter = get_converter()
        self.concept_extractor = ConceptExtractor()

        logger.info(f"QueryProcessor initialized with MiMo ({self.model}) for IAST translation")

    def _translate_to_iast(self, query: str) -> str:
        """Use MiMo to transliterate Sanskrit terms to IAST (not translate to Hindi)."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a Sanskrit transliteration expert. Convert Sanskrit terms to IAST with diacritics. Do NOT translate to Hindi. Keep the original language structure."},
                    {"role": "user", "content": TRANSLATION_PROMPT.format(query=query)},
                ],
                temperature=0.1,
                max_tokens=4096,
            )
            content = response.choices[0].message.content
            if content:
                # Clean up: remove quotes, markdown, extra whitespace
                content = content.strip().strip('"').strip("'").strip()
                content = re.sub(r'^```.*\n?', '', content)
                content = re.sub(r'\n?```$', '', content)
                return content.strip()
            return ""
        except Exception as e:
            logger.error(f"MiMo translation error: {e}")
            return ""

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

        Uses MiMo for IAST translation of English queries, then local tools
        for Devanagari conversion and concept extraction.

        Args:
            query: User's question in any language.

        Returns:
            ProcessedQuery with IAST text, Devanagari, and concepts.
        """
        language = self.detect_language(query)

        if language == "sanskrit_devanagari":
            # Already Devanagari → convert to IAST
            query_iast = self.converter.devanagari_to_iast(query)
        elif language == "sanskrit_iast":
            # Already IAST → use as-is
            query_iast = query
        else:
            # English → translate to IAST via MiMo
            query_iast = self._translate_to_iast(query)
            if not query_iast:
                # Fallback: use original query
                logger.warning("MiMo translation failed, using original query")
                query_iast = query

        query_devanagari = self.converter.iast_to_devanagari(query_iast)

        # Extract concepts from IAST text (better than English)
        local_concepts = self.concept_extractor.extract_from_text(query_iast)
        concepts = [c["concept"].name_iast for c in local_concepts]

        # Also try extracting from original query if no concepts found
        if not concepts:
            local_concepts = self.concept_extractor.extract_from_text(query)
            concepts = [c["concept"].name_iast for c in local_concepts]

        result = ProcessedQuery(
            original_query=query,
            query_iast=query_iast,
            query_devanagari=query_devanagari,
            concepts=concepts,
            language_detected=language,
            extraction_confidence=0.9 if concepts else 0.5,
        )

        logger.info(
            f"Processed query: lang={language}, iast='{query_iast[:60]}', "
            f"concepts={concepts}, confidence={result.extraction_confidence:.2f}"
        )

        return result
