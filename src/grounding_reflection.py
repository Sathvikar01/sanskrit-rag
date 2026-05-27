"""Strict Grounding and Self-Correction/Reflection for RAG.

Implements:
1. Strict Grounding Directives - force LLM to only use provided context
2. Self-Correction/Reflection - evaluate generated answer against retrieved chunks
3. Answer verification and hallucination detection
"""
import re
import time
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
import requests

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import NVIDEA_API_KEY


NVIDIA_CHAT_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
NVIDIA_LLM_MODEL = "meta/llama-3.1-8b-instruct"


@dataclass
class VerificationResult:
    """Result of answer verification."""
    is_grounded: bool
    faithfulness_score: float
    hallucinated_claims: List[str] = field(default_factory=list)
    supported_claims: List[str] = field(default_factory=list)
    missing_context: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)


@dataclass
class ReflectionResult:
    """Result of self-correction reflection."""
    original_answer: str
    corrected_answer: str
    issues_found: List[str] = field(default_factory=list)
    corrections_made: List[str] = field(default_factory=list)
    confidence_score: float = 0.0
    needs_revision: bool = False


class StrictGroundingEnforcer:
    """Enforces strict grounding in retrieved context.

    Prompt directives force LLM to:
    - Only answer using provided context
    - State "I do not have enough information" when context is insufficient
    """

    GROUNDING_SYSTEM_PROMPT = """You are a Sanskrit scholar AI assistant with strict grounding requirements.

CRITICAL RULES:
1. ONLY use information from the provided context verses
2. If the answer is NOT in the context, say: "I do not have enough information from the provided verses to answer this question."
3. Do NOT use outside knowledge, general knowledge, or assumptions
4. Cite specific verse IDs for every claim you make
5. If you're unsure whether something is supported by the context, state your uncertainty

Your response must be based EXCLUSIVELY on the provided Sanskrit verses and their translations."""

    GROUNDING_USER_TEMPLATE = """Question: {query}

Provided Context (USE ONLY THIS):
{context}

Remember: Answer ONLY from the provided context. If the information is not there, say "I do not have enough information."

Answer with citations:"""

    VERIFICATION_PROMPT = """Verify if the following answer is properly grounded in the provided context.

Answer to verify:
{answer}

Context provided:
{context}

For each claim in the answer, determine:
1. Is it explicitly stated in the context?
2. Is it a reasonable inference from the context?
3. Is it NOT supported (potential hallucination)?

Output JSON format:
{{
    "is_grounded": true/false,
    "faithfulness_score": 0.0-1.0,
    "hallucinated_claims": ["claim1", "claim2"],
    "supported_claims": ["claim1", "claim2"],
    "missing_context": ["what was needed but not in context"]
}}

JSON output:"""

    def __init__(
        self,
        api_key: str = None,
        model: str = NVIDIA_LLM_MODEL,
        temperature: float = 0.1
    ):
        self.api_key = api_key or NVIDEA_API_KEY
        self.model = model
        self.temperature = temperature
        self.session = requests.Session()

    def generate_grounded_answer(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]]
    ) -> str:
        """Generate a strictly grounded answer."""
        context_text = self._format_context(context_chunks)

        prompt = self.GROUNDING_USER_TEMPLATE.format(
            query=query,
            context=context_text
        )

        return self._call_llm(
            prompt,
            system_prompt=self.GROUNDING_SYSTEM_PROMPT
        )

    def verify_groundedness(
        self,
        answer: str,
        context_chunks: List[Dict[str, Any]]
    ) -> VerificationResult:
        """Verify if answer is properly grounded in context."""
        context_text = self._format_context(context_chunks)

        prompt = self.VERIFICATION_PROMPT.format(
            answer=answer,
            context=context_text
        )

        response = self._call_llm(prompt)

        return self._parse_verification_response(response)

    def _format_context(self, chunks: List[Dict[str, Any]]) -> str:
        """Format context chunks for prompt."""
        formatted = []
        for i, chunk in enumerate(chunks, 1):
            verse_id = chunk.get("verse_id", "Unknown")
            text = chunk.get("text", "")
            formatted.append(f"[{i}] {verse_id}:\n{text}\n")
        return "\n".join(formatted)

    def _call_llm(self, prompt: str, system_prompt: str = None) -> str:
        """Call LLM API."""
        if not self.api_key:
            return ""

        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            data = {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": 1024
            }

            response = self.session.post(
                NVIDIA_CHAT_URL, headers=headers, json=data, timeout=60
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()

        except Exception as e:
            print(f"LLM call failed: {e}")
            return ""

    def _parse_verification_response(self, response: str) -> VerificationResult:
        """Parse verification response from LLM."""
        try:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
                return VerificationResult(
                    is_grounded=data.get("is_grounded", False),
                    faithfulness_score=float(data.get("faithfulness_score", 0.0)),
                    hallucinated_claims=data.get("hallucinated_claims", []),
                    supported_claims=data.get("supported_claims", []),
                    missing_context=data.get("missing_context", [])
                )
        except Exception as e:
            print(f"Failed to parse verification response: {e}")

        return VerificationResult(
            is_grounded=False,
            faithfulness_score=0.0,
            hallucinated_claims=[],
            supported_claims=[],
            missing_context=["Failed to verify"]
        )


class SelfCorrectionReflection:
    """Self-correction and reflection for answer quality.

    Evaluates generated answer against retrieved chunks to verify
    factual alignment before presenting to user.
    """

    REFLECTION_PROMPT = """You are a critical reviewer for Sanskrit scholarly answers.

Review this answer for factual accuracy and alignment with the provided context.

Original Question: {query}

Generated Answer: {answer}

Retrieved Context:
{context}

Evaluate and provide:
1. Issues found (if any)
2. Factual errors or misinterpretations
3. Missing citations
4. Unsupported claims
5. A corrected version of the answer (if needed)

Output JSON:
{{
    "issues_found": ["issue1", "issue2"],
    "factual_errors": ["error1"],
    "missing_citations": ["topic1"],
    "unsupported_claims": ["claim1"],
    "corrected_answer": "the corrected answer text",
    "confidence_score": 0.0-1.0,
    "needs_revision": true/false
}}

JSON output:"""

    def __init__(
        self,
        api_key: str = None,
        model: str = NVIDIA_LLM_MODEL,
        embedding_client: Any = None
    ):
        self.api_key = api_key or NVIDEA_API_KEY
        self.model = model
        self.embedding_client = embedding_client
        self.session = requests.Session()

    def reflect_and_correct(
        self,
        query: str,
        answer: str,
        context_chunks: List[Dict[str, Any]]
    ) -> ReflectionResult:
        """Reflect on answer and provide corrections."""
        context_text = self._format_context(context_chunks)

        prompt = self.REFLECTION_PROMPT.format(
            query=query,
            answer=answer,
            context=context_text
        )

        response = self._call_llm(prompt)

        reflection = self._parse_reflection_response(response)

        confidence = self._compute_semantic_alignment(
            answer, reflection.corrected_answer, context_chunks
        )
        reflection.confidence_score = confidence

        return reflection

    def _format_context(self, chunks: List[Dict[str, Any]]) -> str:
        """Format context chunks."""
        formatted = []
        for i, chunk in enumerate(chunks, 1):
            verse_id = chunk.get("verse_id", "Unknown")
            text = chunk.get("text", "")
            formatted.append(f"[{i}] {verse_id}: {text[:300]}")
        return "\n".join(formatted)

    def _call_llm(self, prompt: str) -> str:
        """Call LLM for reflection."""
        if not self.api_key:
            return "{}"

        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            data = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 1500
            }

            response = self.session.post(
                NVIDIA_CHAT_URL, headers=headers, json=data, timeout=60
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()

        except Exception as e:
            print(f"Reflection LLM call failed: {e}")
            return "{}"

    def _parse_reflection_response(self, response: str) -> ReflectionResult:
        """Parse reflection response."""
        try:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
                return ReflectionResult(
                    original_answer="",
                    corrected_answer=data.get("corrected_answer", ""),
                    issues_found=data.get("issues_found", []),
                    corrections_made=data.get("factual_errors", []),
                    confidence_score=float(data.get("confidence_score", 0.5)),
                    needs_revision=data.get("needs_revision", False)
                )
        except Exception as e:
            print(f"Failed to parse reflection: {e}")

        return ReflectionResult(
            original_answer="",
            corrected_answer="",
            issues_found=["Failed to parse reflection"],
            needs_revision=False
        )

    def _compute_semantic_alignment(
        self,
        original: str,
        corrected: str,
        context: List[Dict[str, Any]]
    ) -> float:
        """Compute semantic alignment score."""
        if not self.embedding_client:
            return 0.5

        try:
            context_text = " ".join([c.get("text", "") for c in context])

            orig_emb = self.embedding_client.embed_query(original)
            corr_emb = self.embedding_client.embed_query(corrected)
            ctx_emb = self.embedding_client.embed_query(context_text)

            orig_ctx_sim = np.dot(orig_emb.dense_vector, ctx_emb.dense_vector) / (
                np.linalg.norm(orig_emb.dense_vector) * np.linalg.norm(ctx_emb.dense_vector)
            )

            corr_ctx_sim = np.dot(corr_emb.dense_vector, ctx_emb.dense_vector) / (
                np.linalg.norm(corr_emb.dense_vector) * np.linalg.norm(ctx_emb.dense_vector)
            )

            return float(max(orig_ctx_sim, corr_ctx_sim))

        except Exception as e:
            print(f"Semantic alignment computation failed: {e}")
            return 0.5


class GroundedAnswerPipeline:
    """Full pipeline for grounded answer generation with verification."""

    def __init__(
        self,
        grounding_enforcer: StrictGroundingEnforcer = None,
        reflection: SelfCorrectionReflection = None,
        embedding_client: Any = None
    ):
        self.grounder = grounding_enforcer or StrictGroundingEnforcer()
        self.reflection = reflection or SelfCorrectionReflection(
            embedding_client=embedding_client
        )
        self.embedding_client = embedding_client

    def generate_verified_answer(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
        max_iterations: int = 2
    ) -> Tuple[str, VerificationResult, ReflectionResult]:
        """Generate and verify a grounded answer.

        Returns:
            Tuple of (final_answer, verification_result, reflection_result)
        """
        current_answer = self.grounder.generate_grounded_answer(query, context_chunks)

        verification = self.grounder.verify_groundedness(current_answer, context_chunks)

        if not verification.is_grounded and max_iterations > 0:
            reflection = self.reflection.reflect_and_correct(
                query, current_answer, context_chunks
            )

            if reflection.needs_revision and reflection.corrected_answer:
                current_answer = reflection.corrected_answer

                verification = self.grounder.verify_groundedness(
                    current_answer, context_chunks
                )
            else:
                verification = self.grounder.verify_groundedness(
                    current_answer, context_chunks
                )
        else:
            reflection = ReflectionResult(
                original_answer=current_answer,
                corrected_answer=current_answer,
                confidence_score=verification.faithfulness_score,
                needs_revision=False
            )

        return current_answer, verification, reflection


class HallucinationDetector:
    """Detect hallucinations in generated answers."""

    HALLUCINATION_INDICATORS = [
        "undoubtedly",
        "it is well known that",
        "scholars agree",
        "tradition states",
        "according to ancient texts",
        "the original meaning is",
        "clearly means",
    ]

    UNCERTAINTY_PHRASES = [
        "I do not have enough information",
        "the provided context does not",
        "I cannot determine",
        "the verses do not specify",
        "there is insufficient information",
    ]

    def detect_hallucination_risk(self, answer: str) -> Dict[str, Any]:
        """Assess hallucination risk in an answer."""
        answer_lower = answer.lower()

        hallucination_count = sum(
            1 for indicator in self.HALLUCINATION_INDICATORS
            if indicator in answer_lower
        )

        uncertainty_count = sum(
            1 for phrase in self.UNCERTAINTY_PHRASES
            if phrase in answer_lower
        )

        citation_pattern = r'\[?\(?(?:Citation|Verse|BG|BhG)\s*\d+\.?\d*'
        citations = re.findall(citation_pattern, answer, re.IGNORECASE)
        citation_count = len(citations)

        risk_score = 0.0
        if hallucination_count > 0:
            risk_score += hallucination_count * 0.2
        if uncertainty_count > 0:
            risk_score -= uncertainty_count * 0.1
        if citation_count > 0:
            risk_score -= min(citation_count * 0.1, 0.3)

        risk_score = max(0.0, min(1.0, risk_score))

        return {
            "hallucination_risk": risk_score,
            "hallucination_indicators_found": hallucination_count,
            "uncertainty_phrases_found": uncertainty_count,
            "citations_count": citation_count,
            "has_proper_citations": citation_count > 0,
            "recommends_review": risk_score > 0.5
        }


if __name__ == "__main__":
    grounder = StrictGroundingEnforcer()
    detector = HallucinationDetector()

    test_context = [
        {"verse_id": "BhG 2.47", "text": "karmaṇy evādhikāras te mā phaleṣu kadācana"},
        {"verse_id": "BhG 2.48", "text": "yoga-sthaḥ kuru karmāṇi saṅgaṁ tyaktvā dhanañjaya"}
    ]

    test_answer = "According to BhG 2.47, Krishna teaches that one has the right to perform their prescribed duty, but should not be attached to the results. Scholars agree this is the essence of karma yoga."

    print("Testing Hallucination Detection:")
    risk = detector.detect_hallucination_risk(test_answer)
    print(f"  Risk: {risk['hallucination_risk']:.2f}")
    print(f"  Indicators found: {risk['hallucination_indicators_found']}")
    print(f"  Citations: {risk['citations_count']}")
    print(f"  Needs review: {risk['recommends_review']}")
