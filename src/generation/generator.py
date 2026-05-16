"""Answer generator using MiMo v2.5 API."""

import json
import re
from dataclasses import dataclass, field
from typing import Optional

from openai import OpenAI

from src.generation.prompt_templates import SYSTEM_PROMPT, build_generation_prompt
from src.reranking.confidence import PipelineConfidence
from src.utils.config import Config
from src.utils.logger import logger


@dataclass
class GeneratedAnswer:
    """Result of answer generation."""

    answer: str
    verses_cited: list[str] = field(default_factory=list)
    concepts_addressed: list[str] = field(default_factory=list)
    generation_confidence: float = 0.0
    pipeline_confidence: dict = field(default_factory=dict)
    model_used: str = ""


class AnswerGenerator:
    """Generate answers using MiMo v2.5 API."""

    def __init__(self, config: Config = None):
        if config is None:
            config = Config()

        provider = config.get("generation.provider", "mimo")

        if provider == "mimo":
            self.api_base = config.get("generation.mimo.api_base", "https://api.xiaomimimo.com/v1")
            self.model = config.get("generation.mimo.model", "MiMo-V2.5")
            self.api_key = config.mimo_api_key
        else:
            raise ValueError(f"Unsupported generation provider: {provider}")

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_base,
        )

        self.max_tokens = config.get("generation.mimo.max_tokens", 2048)
        self.temperature = config.get("generation.mimo.temperature", 0.3)
        self.confidence = PipelineConfidence()

        logger.info(f"AnswerGenerator initialized with {provider} ({self.model})")

    def _extract_verse_citations(self, text: str) -> list[str]:
        """Extract verse citations from generated text.

        Looks for patterns like BhG 1.1, BhG 2.47, etc.
        """
        pattern = r"BhG\s+\d+\.\d+"
        citations = re.findall(pattern, text)
        return list(set(citations))

    def _call_mimo(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> tuple[str, Optional[float]]:
        """Call MiMo API to generate an answer.

        Args:
            system_prompt: System instructions.
            user_prompt: User query with context.

        Returns:
            Tuple of (generated text, confidence score).
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )

            answer = response.choices[0].message.content
            return answer, None

        except Exception as e:
            logger.error(f"MiMo API error: {e}")
            return "", None

    def generate(
        self,
        query: str,
        reranked_results: list[dict],
        concepts: list[str] = None,
    ) -> GeneratedAnswer:
        """Generate an answer to the user's query.

        Args:
            query: User's original question.
            reranked_results: Re-ranked retrieval results.
            concepts: Extracted concept names.

        Returns:
            GeneratedAnswer with the response and metadata.
        """
        if concepts is None:
            concepts = []

        user_prompt = build_generation_prompt(query, reranked_results)

        answer, llm_confidence = self._call_mimo(SYSTEM_PROMPT, user_prompt)

        verses_cited = self._extract_verse_citations(answer)

        generation_confidence = self.confidence.compute_generation_confidence(
            llm_confidence=llm_confidence,
            citation_count=len(verses_cited),
        )

        retrieval_conf = max(
            (r.get("confidence", {}).get("retrieval_confidence", 0) for r in reranked_results),
            default=0.0,
        )
        reranking_conf = max(
            (r.get("confidence", {}).get("reranking_confidence", 0) for r in reranked_results),
            default=0.0,
        )
        pipeline_conf = self.confidence.compute_pipeline_confidence(
            retrieval_conf, reranking_conf, generation_confidence
        )

        result = GeneratedAnswer(
            answer=answer,
            verses_cited=verses_cited,
            concepts_addressed=concepts,
            generation_confidence=generation_confidence,
            pipeline_confidence=pipeline_conf,
            model_used=self.model,
        )

        logger.info(
            f"Generated answer: {len(answer)} chars, "
            f"{len(verses_cited)} citations, "
            f"confidence={pipeline_conf.get('overall_confidence', 0):.2f}"
        )

        return result

    def generate_simple(self, query: str) -> str:
        """Generate a simple answer without retrieval context.

        Args:
            query: User's question.

        Returns:
            Generated text.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": query},
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"MiMo API error: {e}")
            return ""
