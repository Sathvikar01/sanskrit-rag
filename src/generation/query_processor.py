"""Query processor using MiMo v2.5 for IAST conversion and concept extraction."""

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


QUERY_PROCESSING_PROMPT = """You are a Sanskrit language expert. Your task is to analyze a user query about the Bhagavad Gita and extract structured information.

Given the user query: "{query}"

Please provide the following in JSON format ONLY, no markdown, no explanations:
1. "query_iast": If the query is in English, translate the key Sanskrit terms to IAST transliteration. If the query already contains Sanskrit (IAST or Devanagari), normalize it to IAST.
2. "concepts": A list of philosophical concepts from the Bhagavad Gita relevant to this query. Choose from: dharma, karma, bhakti, jnana, yoga, atman, brahman, ishvara, prakriti, moksha, samsara, gunas, kshetra, ahimsa, tyaga, samkhya, yajna, dhyana, sharira, sukha, duhkha, maya, nishkamakarma, prapatti, sthitaprajna, vishada
3. "language_detected": The language of the original query (e.g., "english", "sanskrit_iast", "sanskrit_devanagari", "hindi", "mixed")
4. "confidence": Your confidence in the extraction (0.0 to 1.0)

Example response:
{{"query_iast": "dharma ki hai bhagavad gītā mein", "concepts": ["dharma", "karma"], "language_detected": "hindi_english_mixed", "confidence": 0.85}}"""


class QueryProcessor:
    """Process user queries using MiMo API."""

    def __init__(self, config: Config = None):
        if config is None:
            config = Config()

        api_base = config.get("generation.mimo.api_base", "https://api.xiaomimimo.com/v1")
        model = config.get("generation.mimo.model", "mimo-v2.5")

        self.client = OpenAI(
            api_key=config.mimo_api_key,
            base_url=api_base,
        )
        self.model = model
        self.converter = get_converter()
        self.concept_extractor = ConceptExtractor()

        logger.info(f"QueryProcessor initialized with MiMo ({self.model})")

    def _call_mimo(self, prompt: str) -> str:
        """Call MiMo API with a prompt."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a Sanskrit language expert. Respond ONLY with valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=1024,
            )
            content = response.choices[0].message.content
            if content:
                logger.debug(f"MiMo response: {content[:200]}")
            else:
                logger.warning("MiMo returned empty content")
            return content or ""
        except Exception as e:
            logger.error(f"MiMo API error: {e}")
            return ""

    def _parse_response(self, response: str) -> dict:
        """Parse MiMo's JSON response."""
        response = response.strip()
        response = re.sub(r"```json\s*", "", response)
        response = re.sub(r"```\s*$", "", response)

        try:
            return json.loads(response)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse MiMo response as JSON: {response[:200]}")
            return {}

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

        Args:
            query: User's question in any language.

        Returns:
            ProcessedQuery with IAST text, Devanagari, and concepts.
        """
        language = self.detect_language(query)

        prompt = QUERY_PROCESSING_PROMPT.format(query=query)
        response = self._call_mimo(prompt)
        parsed = self._parse_response(response)

        query_iast = parsed.get("query_iast", "")
        concepts = parsed.get("concepts", [])
        confidence = parsed.get("confidence", 0.5)

        # Fallback: if API failed or returned empty, use local processing
        if not query_iast:
            logger.warning("MiMo query processing failed, using local fallback")
            return self.process_query_local(query)

        if not concepts:
            local_concepts = self.concept_extractor.extract_from_text(query_iast)
            concepts = [c["concept"].name_iast for c in local_concepts]

        query_devanagari = self.converter.iast_to_devanagari(query_iast)

        result = ProcessedQuery(
            original_query=query,
            query_iast=query_iast,
            query_devanagari=query_devanagari,
            concepts=concepts,
            language_detected=language,
            extraction_confidence=confidence,
        )

        logger.info(
            f"Processed query: lang={language}, concepts={concepts}, "
            f"confidence={confidence:.2f}"
        )

        return result

    def process_query_local(self, query: str) -> ProcessedQuery:
        """Process query locally without API (fallback).

        Args:
            query: User's question.

        Returns:
            ProcessedQuery with basic extraction.
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

        return ProcessedQuery(
            original_query=query,
            query_iast=query_iast,
            query_devanagari=query_devanagari,
            concepts=concepts,
            language_detected=language,
            extraction_confidence=0.3,
        )
