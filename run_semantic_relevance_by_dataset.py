"""Evaluate dense semantic retrieval quality for each source dataset.

This script is intentionally separate from ``run_qa_comparison.py``:
- it uses dense embeddings directly;
- it evaluates raw, lemma_morph, and seg_lemma source chunks separately;
- by default it strips explicit BG references from questions so the score is
  semantic relevance, not verse-ID parsing.
"""
import argparse
import io
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from config.settings import COLLECTION_NAMES
from src.embedding_client import NVIDIAEmbeddingClient
from src.golden_qa import load_golden_chapter1_qa
from src.xml_parser import TEIXMLParser, TextChunk


DEFAULT_OUTPUT = ROOT_DIR / "results" / "semantic_relevance_by_dataset_chapter1.json"


def configure_utf8_stdio() -> None:
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "buffer"):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dense semantic relevance by source dataset.")
    parser.add_argument("--chapter", type=int, default=1, help="Bhagavad Gita chapter to evaluate.")
    parser.add_argument("--top-k", type=int, default=10, help="Top-k unique verse IDs for scoring.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of questions.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(COLLECTION_NAMES.keys()),
        default=sorted(COLLECTION_NAMES.keys()),
        help="Dataset types to evaluate.",
    )
    parser.add_argument(
        "--keep-verse-refs",
        action="store_true",
        help="Keep explicit BG references in questions. Default strips them for semantic-only scoring.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="JSON output path.",
    )
    return parser.parse_args()


def strip_explicit_verse_references(question: str) -> str:
    """Remove explicit verse IDs so dense retrieval is evaluated semantically."""
    cleaned = re.sub(
        r"\b(?:BG|BhG|Bhagavad\s*Gita)\.?\s*\d+\.\d+(?:-\d+)?\b",
        " ",
        question,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\bchapter\s*\d+\s*(?:verse\s*|v\s*)?\d+(?:-\d+)?\b",
        " ",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .:-")
    return cleaned or question


def unique_in_order(values: Iterable[str]) -> List[str]:
    seen = set()
    ordered = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def score_ranked_verses(
    ranked_verse_ids: List[str],
    expected_verse_ids: List[str],
    top_k: int,
) -> Dict[str, Any]:
    """Score a ranked list of verse IDs against expected verse IDs."""
    expected = [verse_id for verse_id in expected_verse_ids if verse_id]
    expected_set = set(expected)
    ranked_top = ranked_verse_ids[:top_k]
    matched = [verse_id for verse_id in expected if verse_id in set(ranked_top)]
    expected_coverage = len(matched) / len(expected) if expected else None

    first_rank = None
    for index, verse_id in enumerate(ranked_verse_ids, 1):
        if verse_id in expected_set:
            first_rank = index
            break

    reciprocal_rank = (1.0 / first_rank) if first_rank else 0.0
    dense_quality = None
    if expected:
        dense_quality = 0.75 * float(expected_coverage or 0.0) + 0.25 * reciprocal_rank

    return {
        "expected_verse_ids": expected,
        "ranked_verse_ids": ranked_top,
        "matched_verse_ids": matched,
        "expected_coverage": round(expected_coverage, 4) if expected_coverage is not None else None,
        "first_expected_rank": first_rank,
        "reciprocal_rank": round(reciprocal_rank, 4),
        "dense_semantic_quality": round(dense_quality, 4) if dense_quality is not None else None,
        "hit_at_1": bool(expected and ranked_top[:1] and ranked_top[0] in expected_set),
        "hit_at_3": bool(expected_set and set(ranked_top[:3]) & expected_set),
        "hit_at_5": bool(expected_set and set(ranked_top[:5]) & expected_set),
        "hit_at_k": bool(expected_set and set(ranked_top) & expected_set),
    }


def filter_chunks_for_chapter(chunks: List[TextChunk], chapter: int) -> List[TextChunk]:
    prefix = f"BhG {chapter}."
    return [chunk for chunk in chunks if (chunk.verse_id or "").startswith(prefix)]


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def embed_chunks(embedder: NVIDIAEmbeddingClient, chunks: List[TextChunk]) -> np.ndarray:
    embeddings = embedder.get_embeddings_batch(
        [chunk.text for chunk in chunks],
        ids=[chunk.id for chunk in chunks],
        metadata_list=[chunk.to_dict() for chunk in chunks],
        input_type="passage",
    )
    return np.vstack([result.dense_vector for result in embeddings]).astype(np.float32)


def summarize_dataset(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    evaluated = [
        result for result in results
        if result["metrics"].get("dense_semantic_quality") is not None
    ]
    if not evaluated:
        return {
            "evaluated_questions": 0,
            "dense_semantic_quality": 0.0,
            "dense_semantic_quality_percentage": 0.0,
            "expected_coverage": 0.0,
            "mean_reciprocal_rank": 0.0,
            "hit_at_1": 0.0,
            "hit_at_3": 0.0,
            "hit_at_5": 0.0,
            "hit_at_k": 0.0,
        }

    def avg(metric_name: str) -> float:
        values = [
            result["metrics"][metric_name]
            for result in evaluated
            if result["metrics"].get(metric_name) is not None
        ]
        return float(sum(values) / len(values)) if values else 0.0

    def rate(metric_name: str) -> float:
        return sum(1 for result in evaluated if result["metrics"].get(metric_name)) / len(evaluated)

    quality = avg("dense_semantic_quality")
    return {
        "evaluated_questions": len(evaluated),
        "dense_semantic_quality": round(quality, 4),
        "dense_semantic_quality_percentage": round(quality * 100, 2),
        "expected_coverage": round(avg("expected_coverage"), 4),
        "mean_reciprocal_rank": round(avg("reciprocal_rank"), 4),
        "hit_at_1": round(rate("hit_at_1"), 4),
        "hit_at_3": round(rate("hit_at_3"), 4),
        "hit_at_5": round(rate("hit_at_5"), 4),
        "hit_at_k": round(rate("hit_at_k"), 4),
    }


def evaluate_dataset(
    dataset_type: str,
    chunks: List[TextChunk],
    qa_items: List[Dict[str, Any]],
    embedder: NVIDIAEmbeddingClient,
    top_k: int,
    strip_refs: bool,
) -> Dict[str, Any]:
    if not chunks:
        return {
            "dataset_type": dataset_type,
            "candidate_chunks": 0,
            "candidate_verses": 0,
            "summary": summarize_dataset([]),
            "results": [],
        }

    print(f"\nEmbedding {dataset_type}: {len(chunks)} chunks...")
    start = time.time()
    candidate_vectors = l2_normalize(embed_chunks(embedder, chunks))
    elapsed = time.time() - start
    print(f"  Candidate embeddings ready in {elapsed:.1f}s")

    results = []
    for index, qa in enumerate(qa_items, 1):
        original_question = qa["question"]
        semantic_question = (
            strip_explicit_verse_references(original_question) if strip_refs else original_question
        )
        query_embedding = embedder.embed_query(semantic_question)
        if not (query_embedding.metadata or {}).get("dense_available", True):
            ranked_verse_ids = []
            top_chunks = []
        else:
            query_vector = l2_normalize(np.asarray([query_embedding.dense_vector], dtype=np.float32))[0]
            scores = candidate_vectors @ query_vector
            ranked_indices = np.argsort(scores)[::-1]
            ranked_verse_ids = unique_in_order(chunks[i].verse_id for i in ranked_indices)
            top_chunks = [
                {
                    "verse_id": chunks[i].verse_id,
                    "score": round(float(scores[i]), 4),
                    "text_preview": chunks[i].text[:180],
                }
                for i in ranked_indices[:top_k]
            ]

        metrics = score_ranked_verses(
            ranked_verse_ids,
            qa.get("expected_verse_ids", []),
            top_k=top_k,
        )
        results.append({
            "question_number": qa.get("question_number", index),
            "question": original_question,
            "semantic_question": semantic_question,
            "metrics": metrics,
            "top_chunks": top_chunks,
        })

    return {
        "dataset_type": dataset_type,
        "candidate_chunks": len(chunks),
        "candidate_verses": len({chunk.verse_id for chunk in chunks if chunk.verse_id}),
        "summary": summarize_dataset(results),
        "results": results,
    }


def main() -> None:
    configure_utf8_stdio()
    args = parse_args()
    qa_items = load_golden_chapter1_qa()
    if args.limit:
        qa_items = qa_items[:args.limit]

    print("=" * 72)
    print("Dense Semantic Relevance by Source Dataset")
    print("=" * 72)
    print(f"Chapter: {args.chapter}")
    print(f"Questions: {len(qa_items)}")
    print(f"Mode: {'with explicit verse refs' if args.keep_verse_refs else 'semantic-only, verse refs stripped'}")

    parser = TEIXMLParser()
    all_chunks = parser.parse_all_datasets(str(ROOT_DIR))
    embedder = NVIDIAEmbeddingClient()
    output: Dict[str, Any] = {
        "test_name": "Dense semantic relevance by source dataset",
        "chapter": args.chapter,
        "top_k": args.top_k,
        "question_mode": "with_verse_refs" if args.keep_verse_refs else "semantic_only_refs_stripped",
        "embedding_backend": embedder.backend,
        "embedding_model": embedder.model,
        "datasets": {},
    }

    for dataset_type in args.datasets:
        chapter_chunks = filter_chunks_for_chapter(all_chunks.get(dataset_type, []), args.chapter)
        dataset_result = evaluate_dataset(
            dataset_type=dataset_type,
            chunks=chapter_chunks,
            qa_items=qa_items,
            embedder=embedder,
            top_k=args.top_k,
            strip_refs=not args.keep_verse_refs,
        )
        output["datasets"][dataset_type] = dataset_result
        summary = dataset_result["summary"]
        print(
            f"{dataset_type}: quality={summary['dense_semantic_quality_percentage']:.2f}% "
            f"MRR={summary['mean_reciprocal_rank']:.4f} "
            f"hit@{args.top_k}={summary['hit_at_k']:.2%} "
            f"chunks={dataset_result['candidate_chunks']}"
        )

    args.output.parent.mkdir(exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
