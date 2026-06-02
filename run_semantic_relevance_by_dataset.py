"""Evaluate dense semantic retrieval quality for each source dataset.

This script is intentionally separate from ``run_qa_comparison.py``:
- it uses dense embeddings directly;
- it evaluates raw, lemma_morph, and seg_lemma source chunks separately;
- by default it strips explicit BG references from questions so the score is
  semantic relevance, not verse-ID parsing.
"""
import argparse
import hashlib
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
from src.query_normalization import expand_semantic_query
from src.xml_parser import TEIXMLParser, TextChunk


DEFAULT_OUTPUT = ROOT_DIR / "results" / "semantic_relevance_by_dataset_chapter1.json"
QA_FALLBACK_PATHS = [
    ROOT_DIR / "test_qa" / "golden_chapter1_qa.json",
]
DEFAULT_ENSEMBLE_WEIGHTS = {
    "raw": 1.25,
    "seg_lemma": 1.0,
    "lemma_morph": 0.25,
}
DEFAULT_ENSEMBLE_RRF_K = 20


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
        "--candidate-unit",
        choices=["verse", "chunk"],
        default="verse",
        help="Evaluate each dataset as verse-aggregated text or individual chunks.",
    )
    parser.add_argument(
        "--no-query-expansion",
        action="store_true",
        help="Disable Sanskrit/domain query expansion.",
    )
    parser.add_argument(
        "--include-details",
        action="store_true",
        help="Write per-question debug details. Default output is summary-only.",
    )
    parser.add_argument(
        "--no-ensemble",
        action="store_true",
        help="Disable overall RRF ensemble across datasets.",
    )
    parser.add_argument("--ensemble-rrf-k", type=int, default=DEFAULT_ENSEMBLE_RRF_K)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="JSON output path.",
    )
    return parser.parse_args()


def load_eval_questions(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    for path in QA_FALLBACK_PATHS:
        items = load_golden_chapter1_qa(path=path)
        if items:
            return items[:limit] if limit else items
    return []


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
        "expected_coverage": round(expected_coverage, 4) if expected_coverage is not None else None,
        "first_expected_rank": first_rank,
        "reciprocal_rank": round(reciprocal_rank, 4),
        "dense_semantic_quality": round(dense_quality, 4) if dense_quality is not None else None,
        "hit_at_1": bool(expected and ranked_top[:1] and ranked_top[0] in expected_set),
        "hit_at_3": bool(expected_set and set(ranked_top[:3]) & expected_set),
        "hit_at_5": bool(expected_set and set(ranked_top[:5]) & expected_set),
        "hit_at_k": bool(expected_set and set(ranked_top) & expected_set),
    }


def rrf_rank_ensemble(
    dataset_rankings: Dict[str, List[str]],
    weights: Dict[str, float],
    k: int = DEFAULT_ENSEMBLE_RRF_K,
) -> List[str]:
    scores: Dict[str, float] = {}
    for dataset_type, ranked_ids in dataset_rankings.items():
        weight = weights.get(dataset_type, 0.0)
        if weight <= 0:
            continue
        for rank, candidate_id in enumerate(ranked_ids, 1):
            scores[candidate_id] = scores.get(candidate_id, 0.0) + weight / (k + rank)
    return [
        candidate_id
        for candidate_id, _ in sorted(scores.items(), key=lambda item: item[1], reverse=True)
    ]


def filter_chunks_for_chapter(chunks: List[TextChunk], chapter: int) -> List[TextChunk]:
    prefix = f"BhG {chapter}."
    return [chunk for chunk in chunks if (chunk.verse_id or "").startswith(prefix)]


def aggregate_chunks_by_verse(chunks: List[TextChunk], dataset_type: str) -> List[TextChunk]:
    """Collapse source chunks to one candidate per verse for CPU-friendly scoring."""
    grouped: Dict[str, List[TextChunk]] = {}
    for chunk in chunks:
        if chunk.verse_id:
            grouped.setdefault(chunk.verse_id, []).append(chunk)

    aggregated = []
    for verse_id, verse_chunks in sorted(grouped.items(), key=lambda item: item[0]):
        text = "\n".join(unique_in_order(chunk.text for chunk in verse_chunks if chunk.text.strip()))
        chunk_id = hashlib.md5(f"{dataset_type}:{verse_id}:{text}".encode()).hexdigest()[:16]
        first = verse_chunks[0]
        aggregated.append(TextChunk(
            id=chunk_id,
            text=text,
            dataset_type=dataset_type,
            verse_id=verse_id,
            element_type="verse_aggregate",
            line_number=first.line_number,
            metadata={
                "dataset_type": dataset_type,
                "verse_id": verse_id,
                "candidate_unit": "verse",
                "source_chunk_count": len(verse_chunks),
                "chapter": first.metadata.get("chapter"),
                "verse_num": first.metadata.get("verse_num"),
            },
        ))
    return aggregated


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
    expand_queries: bool,
    include_details: bool,
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
    rankings = []
    for index, qa in enumerate(qa_items, 1):
        original_question = qa["question"]
        semantic_question = (
            strip_explicit_verse_references(original_question) if strip_refs else original_question
        )
        semantic_question = expand_semantic_query(semantic_question) if expand_queries else semantic_question
        query_embedding = embedder.embed_query(semantic_question)
        if not (query_embedding.metadata or {}).get("dense_available", True):
            ranked_verse_ids = []
        else:
            query_vector = l2_normalize(np.asarray([query_embedding.dense_vector], dtype=np.float32))[0]
            scores = candidate_vectors @ query_vector
            ranked_indices = np.argsort(scores)[::-1]
            ranked_verse_ids = unique_in_order(chunks[i].verse_id for i in ranked_indices)

        metrics = score_ranked_verses(
            ranked_verse_ids,
            qa.get("expected_verse_ids", []),
            top_k=top_k,
        )
        rankings.append(ranked_verse_ids)
        item = {"metrics": metrics}
        if include_details:
            item.update({
                "question_number": qa.get("question_number", index),
                "question": original_question,
                "semantic_question": semantic_question,
            })
        results.append(item)

    return {
        "dataset_type": dataset_type,
        "candidate_chunks": len(chunks),
        "candidate_verses": len({chunk.verse_id for chunk in chunks if chunk.verse_id}),
        "summary": summarize_dataset(results),
        "results": results if include_details else [],
        "_rankings": rankings,
    }


def evaluate_ensemble(
    qa_items: List[Dict[str, Any]],
    dataset_rankings: Dict[str, List[List[str]]],
    top_k: int,
    weights: Dict[str, float] = DEFAULT_ENSEMBLE_WEIGHTS,
    rrf_k: int = DEFAULT_ENSEMBLE_RRF_K,
    include_details: bool = False,
) -> Dict[str, Any]:
    results = []
    for index, qa in enumerate(qa_items):
        rankings_for_question = {
            dataset_type: rankings[index]
            for dataset_type, rankings in dataset_rankings.items()
            if index < len(rankings)
        }
        ranked = rrf_rank_ensemble(rankings_for_question, weights=weights, k=rrf_k)
        metrics = score_ranked_verses(ranked, qa.get("expected_verse_ids", []), top_k=top_k)
        item = {"metrics": metrics}
        if include_details:
            item["question_number"] = qa.get("question_number", index + 1)
        results.append(item)

    return {
        "summary": summarize_dataset(results),
        "weights": weights,
        "rrf_k": rrf_k,
        "results": results if include_details else [],
    }


def main() -> None:
    configure_utf8_stdio()
    args = parse_args()
    qa_items = load_eval_questions(limit=args.limit)
    if not qa_items:
        raise RuntimeError(
            "No Chapter 1 QA items found. Expected test_qa/golden_chapter1_qa.json "
            "or a prior results/qa_*_chapter1.json file."
        )

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
        "candidate_unit": args.candidate_unit,
        "question_mode": "with_verse_refs" if args.keep_verse_refs else "semantic_only_refs_stripped",
        "query_expansion": not args.no_query_expansion,
        "embedding_backend": embedder.backend,
        "embedding_model": embedder.model,
        "datasets": {},
        "overall": {},
    }
    dataset_rankings: Dict[str, List[List[str]]] = {}

    for dataset_type in args.datasets:
        source_chunks = filter_chunks_for_chapter(all_chunks.get(dataset_type, []), args.chapter)
        chapter_chunks = (
            aggregate_chunks_by_verse(source_chunks, dataset_type)
            if args.candidate_unit == "verse"
            else source_chunks
        )
        dataset_result = evaluate_dataset(
            dataset_type=dataset_type,
            chunks=chapter_chunks,
            qa_items=qa_items,
            embedder=embedder,
            top_k=args.top_k,
            strip_refs=not args.keep_verse_refs,
            expand_queries=not args.no_query_expansion,
            include_details=args.include_details,
        )
        dataset_rankings[dataset_type] = dataset_result.pop("_rankings", [])
        dataset_result["source_chunks"] = len(source_chunks)
        dataset_result["candidate_unit"] = args.candidate_unit
        output["datasets"][dataset_type] = dataset_result
        summary = dataset_result["summary"]
        print(
            f"{dataset_type}: quality={summary['dense_semantic_quality_percentage']:.2f}% "
            f"MRR={summary['mean_reciprocal_rank']:.4f} "
            f"hit@{args.top_k}={summary['hit_at_k']:.2%} "
            f"chunks={dataset_result['candidate_chunks']}"
        )

    if not args.no_ensemble and len(dataset_rankings) > 1:
        ensemble = evaluate_ensemble(
            qa_items=qa_items,
            dataset_rankings=dataset_rankings,
            top_k=args.top_k,
            weights=DEFAULT_ENSEMBLE_WEIGHTS,
            rrf_k=args.ensemble_rrf_k,
            include_details=args.include_details,
        )
        output["overall"] = ensemble
        summary = ensemble["summary"]
        print(
            f"overall: quality={summary['dense_semantic_quality_percentage']:.2f}% "
            f"MRR={summary['mean_reciprocal_rank']:.4f} "
            f"hit@{args.top_k}={summary['hit_at_k']:.2%}"
        )

    args.output.parent.mkdir(exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
