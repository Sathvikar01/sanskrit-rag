"""NVIDIA LLM Client via NVIDIA NIM API for IAST translation and citation-backed answer generation."""
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import requests

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import NVIDEA_API_KEY, GEMINI_TEMPERATURE, GEMINI_MAX_TOKENS

NVIDIA_CHAT_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
NVIDIA_LLM_MODEL = "meta/llama-3.1-8b-instruct"


class NVIDIA_LLM_Client:
    """Client for LLM via NVIDIA NIM API."""

    def __init__(
        self,
        api_key: str = NVIDEA_API_KEY,
        model: str = NVIDIA_LLM_MODEL,
        temperature: float = GEMINI_TEMPERATURE,
        max_tokens: int = GEMINI_MAX_TOKENS
    ):
        self.api_key = api_key
        self.model_name = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._available = bool(self.api_key)
        self._disabled = False
        self._disabled_until = 0
        self._last_check_time = 0
        self._check_interval = 300
        self.session = requests.Session()

    def is_available(self) -> bool:
        if not self._available:
            return False
        if self._disabled:
            if time.time() > self._disabled_until:
                self._disabled = False
                self._disabled_until = 0
            else:
                return False
        return True

    def pre_check_quota(self) -> bool:
        """Quick check if LLM is available by testing a simple call."""
        if not self._available:
            return False
        current_time = time.time()
        if current_time - self._last_check_time < self._check_interval:
            return self.is_available()
        self._last_check_time = current_time
        test_prompt = "Reply with: ok"
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            data = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": test_prompt}],
                "temperature": 0.1,
                "max_tokens": 10
            }
            r = self.session.post(NVIDIA_CHAT_URL, headers=headers, json=data, timeout=10)
            r.raise_for_status()
            return True
        except Exception as e:
            error_msg = str(e).lower()
            if "401" in error_msg or "403" in error_msg or "quota" in error_msg:
                self._disabled = True
                self._disabled_until = time.time() + 300
                return False
            return True

    def _generate(self, prompt: str) -> str:
        if not self.is_available():
            return ""
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            data = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens
            }
            r = self.session.post(NVIDIA_CHAT_URL, headers=headers, json=data, timeout=45)
            r.raise_for_status()
            result = r.json()
            answer = result["choices"][0]["message"]["content"].strip()
            return answer
        except Exception as e:
            error_msg = str(e).lower()
            if "401" in error_msg or "403" in error_msg or "quota" in error_msg:
                print(f"NVIDIA LLM API quota/auth error: {e}")
                self._disabled = True
                self._disabled_until = time.time() + 300
            else:
                print(f"NVIDIA LLM API error: {e}")
            return ""

    def translate_to_iast(self, query: str) -> str:
        """Translate a natural language query to IAST format."""
        prompt = f"""You are a Sanskrit transliteration expert. Convert the following query into IAST (International Alphabet of Sanskrit Transliteration) format.

Rules:
- Preserve the meaning of the query
- Use proper IAST diacritical marks (ā, ī, ū, ṛ, ṝ, ḷ, ḹ, ṃ, ḥ, ṅ, ñ, ṭ, ḍ, ṇ, ś, ṣ)
- If the query is already in IAST, return it as-is
- If the query is in English about Sanskrit concepts, transliterate the Sanskrit terms to IAST and keep English context words
- Return ONLY the IAST text, nothing else

Query: {query}

IAST:"""
        result = self._generate(prompt)
        return result if result else query

    def normalize_with_byt5(self, text: str) -> str:
        """Use LLM to normalize Sanskrit text (ByT5-style normalization)."""
        prompt = f"""You are a Sanskrit text normalization expert. Normalize the following Sanskrit text:

Rules:
- Standardize spelling and diacritical marks to IAST
- Remove extra whitespace and normalize formatting
- Fix any obvious transliteration errors
- Return ONLY the normalized text, nothing else

Text: {text}

Normalized:"""
        result = self._generate(prompt)
        return result if result else text

    def generate_answer(
        self,
        query: str,
        iast_query: str,
        retrieved_verses: List[Dict[str, Any]],
        metadata: List[Dict[str, Any]],
        commentary_matches: Optional[List[Dict[str, Any]]] = None,
        supporting_chunks: Optional[List[Dict[str, Any]]] = None,
        retrieval_metadata: Optional[Dict[str, Any]] = None,
        answer_template: str = "",
        query_intent: Optional[Dict[str, Any]] = None,
        entities: Optional[List[Dict[str, Any]]] = None,
        confidence: float = 0.0,
        answer_mode: str = "current",
    ) -> Dict[str, Any]:
        """Generate a citation-backed answer using canonical verses, commentary, and metadata."""
        verses_context = ""
        for i, verse in enumerate(retrieved_verses):
            verse_id = verse.get("verse_id", "Unknown")
            text = verse.get("text", "")
            source = verse.get("source", "Unknown")
            score = verse.get("score", 0.0)
            verses_context += f"\n[Citation {i+1}] (ID: {verse_id}, Source: {source}, Score: {score:.4f})\n{text}\n"

        commentary_context = ""
        for i, match in enumerate(commentary_matches or []):
            verse_id = match.get("verse_id", "Unknown")
            author = match.get("author_display_name", match.get("author_key", "Unknown commentator"))
            score = float(match.get("score", 0.0) or 0.0)
            text = match.get("text", "")
            commentary_context += (
                f"\n[Commentary {i+1}] (Verse: {verse_id}, Author: {author}, "
                f"Source: {match.get('commentary_source', match.get('metadata', {}).get('source', 'retrieval'))}, "
                f"Score: {score:.4f})\n{text}\n"
            )

        meta_context = ""
        for i, m in enumerate(metadata):
            meta_context += f"\n[Meta {i+1}] {m}\n"

        support_context = ""
        for i, chunk in enumerate((supporting_chunks or [])[:8]):
            sources = chunk.get("sources", {})
            source_names = ", ".join(name for name, enabled in sources.items() if enabled) or "retrieval"
            support_context += (
                f"\n[Support {i+1}] (Verse: {chunk.get('verse_id', 'Unknown')}, "
                f"Sources: {source_names}, Score: {float(chunk.get('score', 0.0) or 0.0):.4f})\n"
                f"{chunk.get('text', '')}\n"
            )

        retrieval_context = retrieval_metadata or {}
        intent_context = query_intent or {}
        entity_context = entities or []

        if answer_mode == "structured_step":
            prompt = f"""You are a Sanskrit scholar AI assistant. Answer using ONLY the evidence below.

User Query: {query}
IAST Query: {iast_query}
Query Intent: {intent_context}
Detected Entities: {entity_context}
Evidence Confidence: {confidence:.4f}

Canonical Original Verses:
{verses_context}

Related Commentary:
{commentary_context}

Additional Metadata:
{meta_context}

Supporting Retrieval Chunks:
{support_context}

Retrieval Metadata:
{retrieval_context}

Answer Template:
{answer_template}

Instructions:
Step 1 - Evidence scan:
- List only the canonical verses or commentary passages that directly answer the query.
- For each selected item, explain in one short phrase why it is relevant.
- If no supplied evidence directly answers the query, write "No directly relevant supplied evidence."

Step 2 - Final answer:
- Answer using only the Step 1 evidence.
- Cite verse IDs using [Citation N] format when referencing a verse.
- Cite commentary as "Author on Verse ID" when commentary influences the interpretation.
- If Step 1 found no directly relevant evidence, abstain clearly instead of using outside memory.
- Be precise, concise, and scholarly.

Answer:"""
        else:
            prompt = f"""You are a Sanskrit scholar AI assistant. Answer the user's question using ONLY the evidence below.

User Query: {query}
IAST Query: {iast_query}
Query Intent: {intent_context}
Detected Entities: {entity_context}
Evidence Confidence: {confidence:.4f}

Canonical Original Verses:
{verses_context}

Related Commentary:
{commentary_context}

Additional Metadata:
{meta_context}

Supporting Retrieval Chunks:
{support_context}

Retrieval Metadata:
{retrieval_context}

Answer Template:
{answer_template}

Instructions:
1. Ground every factual claim in the canonical original verses or related commentary above.
2. Cite specific verse IDs using [Citation N] format when referencing a verse.
3. Cite commentary as "Author on Verse ID" when commentary influences the interpretation.
4. Prefer the canonical original verse text over supporting retrieval chunks if they differ.
5. If the evidence does not contain enough information, say so clearly.
6. Do not use outside memory, training data, or unsupported Bhagavad Gita facts.
7. Include the original Sanskrit text most relevant to the answer when it is present in Canonical Original Verses.
8. Be precise, concise, and scholarly.

Answer:"""

        answer = self._generate(prompt)

        return {
            "answer": answer,
            "query": query,
            "iast_query": iast_query,
            "citations": [
                {
                    "verse_id": v.get("verse_id", ""),
                    "source": v.get("source", ""),
                    "score": v.get("score", 0.0),
                    "text": v.get("text", "")[:200]
                }
                for v in retrieved_verses
            ],
            "num_sources": len(retrieved_verses)
        }

    def generate_answer_simple(
        self,
        query: str,
        context_text: str
    ) -> str:
        """Simple answer generation with context."""
        prompt = f"""You are a Sanskrit scholar AI assistant. Answer the user's question using the provided context.

Query: {query}

Context:
{context_text}

Answer the question based on the context above. Cite sources where possible.

Answer:"""
        return self._generate(prompt)


# Backward compatibility alias
GeminiClient = NVIDIA_LLM_Client


if __name__ == "__main__":
    client = NVIDIA_LLM_Client()
    if client.is_available():
        print("NVIDIA LLM client initialized successfully")
        test = client.translate_to_iast("What does dharma mean")
        print(f"IAST: {test}")
    else:
        print("NVIDIA LLM not available")
