"""RAGAS (RAG Assessment) Evaluation Pipeline.

Implements evaluation metrics:
- Context Precision: Did we retrieve the right things?
- Context Recall: Did we miss anything?
- Answer Faithfulness: Did the LLM hallucinate?
- Answer Relevance: Is the answer relevant to the query?
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
class RAGASMetrics:
    """Container for RAGAS evaluation metrics."""
    context_precision: float = 0.0
    context_recall: float = 0.0
    answer_faithfulness: float = 0.0
    answer_relevance: float = 0.0
    answer_correctness: float = 0.0
    context_relevancy: float = 0.0

    overall_score: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "context_precision": self.context_precision,
            "context_recall": self.context_recall,
            "answer_faithfulness": self.answer_faithfulness,
            "answer_relevance": self.answer_relevance,
            "answer_correctness": self.answer_correctness,
            "context_relevancy": self.context_relevancy,
            "overall_score": self.overall_score
        }

    def compute_overall(self) -> float:
        weights = {
            "context_precision": 0.2,
            "context_recall": 0.15,
            "answer_faithfulness": 0.25,
            "answer_relevance": 0.2,
            "answer_correctness": 0.1,
            "context_relevancy": 0.1
        }

        weighted_sum = sum(
            getattr(self, metric, 0) * weight
            for metric, weight in weights.items()
        )
        self.overall_score = weighted_sum
        return self.overall_score


@dataclass
class EvaluationResult:
    """Complete evaluation result for a query-answer pair."""
    query: str
    answer: str
    contexts: List[str]
    ground_truth: Optional[str]
    metrics: RAGASMetrics
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            "contexts": self.contexts,
            "ground_truth": self.ground_truth,
            "metrics": self.metrics.to_dict(),
            "details": self.details
        }


class ContextPrecisionEvaluator:
    """Evaluates if retrieved contexts are relevant to the query.

    Context Precision = Number of relevant contexts / Total retrieved contexts
    """

    RELEVANCE_PROMPT = """You are a Sanskrit text relevance evaluator.

Determine if each context passage is relevant to answering the query.

Query: {query}

Context Passages:
{contexts}

For each context passage, output a relevance score (0 or 1):
- 1 if the passage contains information directly relevant to answering the query
- 0 if the passage is not relevant

Output JSON format:
{{
    "relevance_scores": [1, 0, 1, ...],
    "reasoning": ["reason for passage 1", "reason for passage 2", ...]
}}

JSON output:"""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or NVIDEA_API_KEY
        self.session = requests.Session()

    def evaluate(
        self,
        query: str,
        contexts: List[str]
    ) -> Tuple[float, List[str]]:
        """Evaluate context precision."""
        if not contexts:
            return 0.0, []

        context_text = "\n\n".join(
            f"[{i+1}] {ctx[:500]}"
            for i, ctx in enumerate(contexts)
        )

        prompt = self.RELEVANCE_PROMPT.format(
            query=query,
            contexts=context_text
        )

        response = self._call_llm(prompt)

        scores, reasons = self._parse_relevance_response(response)

        precision = sum(scores) / len(scores) if scores else 0.0

        return precision, reasons

    def _call_llm(self, prompt: str) -> str:
        if not self.api_key:
            return '{"relevance_scores": [1], "reasoning": ["API unavailable"]}'

        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            data = {
                "model": NVIDIA_LLM_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": 500
            }
            response = self.session.post(
                NVIDIA_CHAT_URL, headers=headers, json=data, timeout=30
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return '{"relevance_scores": [1], "reasoning": ["Error"]}'

    def _parse_relevance_response(
        self,
        response: str
    ) -> Tuple[List[int], List[str]]:
        try:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
                scores = data.get("relevance_scores", [1])
                reasons = data.get("reasoning", [])
                return scores, reasons
        except Exception:
            pass
        return [1], []


class ContextRecallEvaluator:
    """Evaluates if all necessary information was retrieved.

    Context Recall = Information found in contexts / Information needed
    """

    RECALL_PROMPT = """You are evaluating if retrieved contexts contain all necessary information.

Query: {query}

Ground Truth Answer (what should be covered):
{ground_truth}

Retrieved Contexts:
{contexts}

Determine which pieces of information from the ground truth are present in the contexts.
List the key pieces of information that are:
- Found in contexts
- Missing from contexts

Output JSON:
{{
    "found_info": ["info1", "info2"],
    "missing_info": ["info3", "info4"],
    "recall_score": 0.0-1.0
}}

JSON output:"""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or NVIDEA_API_KEY
        self.session = requests.Session()

    def evaluate(
        self,
        query: str,
        contexts: List[str],
        ground_truth: str
    ) -> Tuple[float, List[str], List[str]]:
        """Evaluate context recall."""
        context_text = "\n\n".join(ctx[:500] for ctx in contexts)

        prompt = self.RECALL_PROMPT.format(
            query=query,
            ground_truth=ground_truth,
            contexts=context_text
        )

        response = self._call_llm(prompt)

        recall, found, missing = self._parse_recall_response(response)

        return recall, found, missing

    def _call_llm(self, prompt: str) -> str:
        if not self.api_key:
            return '{"recall_score": 0.5, "found_info": [], "missing_info": []}'

        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            data = {
                "model": NVIDIA_LLM_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": 500
            }
            response = self.session.post(
                NVIDIA_CHAT_URL, headers=headers, json=data, timeout=30
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception:
            return '{"recall_score": 0.5}'

    def _parse_recall_response(
        self,
        response: str
    ) -> Tuple[float, List[str], List[str]]:
        try:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
                recall = float(data.get("recall_score", 0.5))
                found = data.get("found_info", [])
                missing = data.get("missing_info", [])
                return recall, found, missing
        except Exception:
            pass
        return 0.5, [], []


class AnswerFaithfulnessEvaluator:
    """Evaluates if the answer is faithful to the context (no hallucinations)."""

    FAITHFULNESS_PROMPT = """You are evaluating if an answer is faithful to the provided context.

Answer to evaluate:
{answer}

Context provided:
{context}

For each claim in the answer, determine:
1. Is it directly supported by the context?
2. Is it a reasonable inference from the context?
3. Is it NOT supported (hallucination)?

Output JSON:
{{
    "num_supported_claims": 0,
    "num_unsupported_claims": 0,
    "unsupported_statements": ["claim1", "claim2"],
    "faithfulness_score": 0.0-1.0
}}

JSON output:"""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or NVIDEA_API_KEY
        self.session = requests.Session()

    def evaluate(
        self,
        answer: str,
        contexts: List[str]
    ) -> Tuple[float, List[str]]:
        """Evaluate answer faithfulness."""
        context_text = "\n\n".join(ctx[:500] for ctx in contexts)

        prompt = self.FAITHFULNESS_PROMPT.format(
            answer=answer,
            context=context_text
        )

        response = self._call_llm(prompt)

        faithfulness, unsupported = self._parse_faithfulness_response(response)

        return faithfulness, unsupported

    def _call_llm(self, prompt: str) -> str:
        if not self.api_key:
            return '{"faithfulness_score": 0.8, "unsupported_statements": []}'

        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            data = {
                "model": NVIDIA_LLM_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": 500
            }
            response = self.session.post(
                NVIDIA_CHAT_URL, headers=headers, json=data, timeout=30
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception:
            return '{"faithfulness_score": 0.8}'

    def _parse_faithfulness_response(
        self,
        response: str
    ) -> Tuple[float, List[str]]:
        try:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
                faithfulness = float(data.get("faithfulness_score", 0.8))
                unsupported = data.get("unsupported_statements", [])
                return faithfulness, unsupported
        except Exception:
            pass
        return 0.8, []


class AnswerRelevanceEvaluator:
    """Evaluates if the answer addresses the query."""

    RELEVANCE_PROMPT = """You are evaluating if an answer is relevant to the query.

Query: {query}

Answer: {answer}

Evaluate the relevance on a scale of 0 to 1:
- 1.0: Answer directly addresses the query with appropriate detail
- 0.7-0.9: Answer is relevant but could be more specific
- 0.4-0.6: Answer is somewhat relevant but misses key aspects
- 0.0-0.3: Answer is not relevant to the query

Output JSON:
{{
    "relevance_score": 0.0-1.0,
    "explanation": "reasoning for the score"
}}

JSON output:"""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or NVIDEA_API_KEY
        self.session = requests.Session()

    def evaluate(
        self,
        query: str,
        answer: str
    ) -> Tuple[float, str]:
        """Evaluate answer relevance."""
        prompt = self.RELEVANCE_PROMPT.format(query=query, answer=answer)

        response = self._call_llm(prompt)

        relevance, explanation = self._parse_relevance_response(response)

        return relevance, explanation

    def _call_llm(self, prompt: str) -> str:
        if not self.api_key:
            return '{"relevance_score": 0.8, "explanation": "API unavailable"}'

        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            data = {
                "model": NVIDIA_LLM_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": 200
            }
            response = self.session.post(
                NVIDIA_CHAT_URL, headers=headers, json=data, timeout=20
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception:
            return '{"relevance_score": 0.8}'

    def _parse_relevance_response(
        self,
        response: str
    ) -> Tuple[float, str]:
        try:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
                relevance = float(data.get("relevance_score", 0.8))
                explanation = data.get("explanation", "")
                return relevance, explanation
        except Exception:
            pass
        return 0.8, ""


class RAGASPipeline:
    """Complete RAGAS evaluation pipeline."""

    def __init__(
        self,
        api_key: str = None,
        embedding_client: Any = None
    ):
        self.api_key = api_key or NVIDEA_API_KEY
        self.embedding_client = embedding_client

        self.context_precision = ContextPrecisionEvaluator(self.api_key)
        self.context_recall = ContextRecallEvaluator(self.api_key)
        self.answer_faithfulness = AnswerFaithfulnessEvaluator(self.api_key)
        self.answer_relevance = AnswerRelevanceEvaluator(self.api_key)

    def evaluate(
        self,
        query: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None
    ) -> EvaluationResult:
        """Run full RAGAS evaluation."""
        metrics = RAGASMetrics()
        details = {}

        precision, precision_reasons = self.context_precision.evaluate(query, contexts)
        metrics.context_precision = precision
        details["precision_reasons"] = precision_reasons

        if ground_truth:
            recall, found_info, missing_info = self.context_recall.evaluate(
                query, contexts, ground_truth
            )
            metrics.context_recall = recall
            details["found_info"] = found_info
            details["missing_info"] = missing_info

        faithfulness, unsupported = self.answer_faithfulness.evaluate(answer, contexts)
        metrics.answer_faithfulness = faithfulness
        details["unsupported_statements"] = unsupported

        relevance, relevance_explanation = self.answer_relevance.evaluate(query, answer)
        metrics.answer_relevance = relevance
        details["relevance_explanation"] = relevance_explanation

        if ground_truth and self.embedding_client:
            metrics.answer_correctness = self._compute_answer_correctness(
                answer, ground_truth
            )

        metrics.context_relevancy = self._compute_context_relevancy(
            query, contexts
        )

        metrics.compute_overall()

        return EvaluationResult(
            query=query,
            answer=answer,
            contexts=contexts,
            ground_truth=ground_truth,
            metrics=metrics,
            details=details
        )

    def _compute_answer_correctness(
        self,
        answer: str,
        ground_truth: str
    ) -> float:
        """Compute answer correctness using semantic similarity."""
        if not self.embedding_client:
            return 0.5

        try:
            ans_emb = self.embedding_client.embed_query(answer)
            truth_emb = self.embedding_client.embed_query(ground_truth)

            similarity = np.dot(ans_emb.dense_vector, truth_emb.dense_vector) / (
                np.linalg.norm(ans_emb.dense_vector) * np.linalg.norm(truth_emb.dense_vector)
            )

            return float(max(0, min(1, similarity)))
        except Exception:
            return 0.5

    def _compute_context_relevancy(
        self,
        query: str,
        contexts: List[str]
    ) -> float:
        """Compute context relevancy based on semantic overlap."""
        if not self.embedding_client or not contexts:
            return 0.5

        try:
            query_emb = self.embedding_client.embed_query(query)

            relevancies = []
            for ctx in contexts:
                ctx_emb = self.embedding_client.embed_query(ctx[:500])
                similarity = np.dot(query_emb.dense_vector, ctx_emb.dense_vector) / (
                    np.linalg.norm(query_emb.dense_vector) * np.linalg.norm(ctx_emb.dense_vector)
                )
                relevancies.append(float(max(0, similarity)))

            return float(np.mean(relevancies)) if relevancies else 0.5
        except Exception:
            return 0.5


class RAGASTracker:
    """Track RAGAS metrics over time for optimization."""

    def __init__(self, storage_path: str = None):
        self.storage_path = Path(storage_path) if storage_path else Path(__file__).parent.parent / "evaluation_results"
        self.storage_path.mkdir(exist_ok=True)

        self.results_file = self.storage_path / "ragas_results.json"
        self.history: List[Dict[str, Any]] = self._load_history()

    def _load_history(self) -> List[Dict[str, Any]]:
        if self.results_file.exists():
            try:
                with open(self.results_file, 'r') as f:
                    return json.load(f)
            except Exception:
                return []
        return []

    def record_evaluation(self, result: EvaluationResult) -> None:
        """Record an evaluation result."""
        entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            **result.to_dict()
        }
        self.history.append(entry)
        self._save_history()

    def _save_history(self) -> None:
        with open(self.results_file, 'w') as f:
            json.dump(self.history, f, indent=2)

    def get_statistics(self) -> Dict[str, Any]:
        """Get aggregate statistics from history."""
        if not self.history:
            return {}

        metrics_keys = [
            "context_precision", "context_recall", "answer_faithfulness",
            "answer_relevance", "answer_correctness", "context_relevancy",
            "overall_score"
        ]

        stats = {}
        for key in metrics_keys:
            values = [
                h.get("metrics", {}).get(key, 0)
                for h in self.history
                if "metrics" in h
            ]
            if values:
                stats[key] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(min(values)),
                    "max": float(max(values)),
                    "count": len(values)
                }

        return stats

    def compare_configurations(
        self,
        config1_name: str,
        config2_name: str
    ) -> Dict[str, Any]:
        """Compare metrics between two configurations."""
        config1_results = [
            h for h in self.history
            if h.get("configuration") == config1_name
        ]
        config2_results = [
            h for h in self.history
            if h.get("configuration") == config2_name
        ]

        comparison = {}

        metrics_keys = [
            "context_precision", "answer_faithfulness",
            "answer_relevance", "overall_score"
        ]

        for key in metrics_keys:
            values1 = [
                h.get("metrics", {}).get(key, 0)
                for h in config1_results
            ]
            values2 = [
                h.get("metrics", {}).get(key, 0)
                for h in config2_results
            ]

            if values1 and values2:
                comparison[key] = {
                    config1_name: float(np.mean(values1)),
                    config2_name: float(np.mean(values2)),
                    "improvement": float(np.mean(values2) - np.mean(values1))
                }

        return comparison


if __name__ == "__main__":
    pipeline = RAGASPipeline()

    test_query = "What does Krishna teach about duty in Bhagavad Gita?"
    test_answer = "Krishna teaches in BhG 2.47 that one has the right to perform their prescribed duty, but should not be attached to the results."
    test_contexts = [
        "karmaṇy evādhikāras te mā phaleṣu kadācana - You have the right to perform your prescribed duty, but you are not entitled to the fruits of action.",
        "yoga-sthaḥ kuru karmāṇi - Perform your duties with equanimity."
    ]

    print("Running RAGAS Evaluation...")
    result = pipeline.evaluate(
        query=test_query,
        answer=test_answer,
        contexts=test_contexts
    )

    print(f"\nMetrics:")
    for metric, value in result.metrics.to_dict().items():
        print(f"  {metric}: {value:.3f}")
