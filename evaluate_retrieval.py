"""Retrieval-only evaluation: measure verse retrieval accuracy with/without verse ID augmentation.

For each dataset question that has a known verse ID, runs retrieval twice:
1. Without verse ID (original question)
2. With verse ID prepended ("BhG X.Y: [question]")

Then compares verse retrieval accuracy (Recall@K, MRR) between the two conditions.
No LLM calls — retrieval -> fusion -> re-ranking only.
"""

import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

import json
import random
import re
import time
from pathlib import Path
from dataclasses import dataclass, field, asdict

from sentence_transformers import SentenceTransformer, util


# ── Bhagavad Gita chapter verse counts (total 700) ──
CHAPTER_VERSES = [46, 72, 43, 42, 29, 47, 30, 28, 34, 42, 55, 20, 35, 27, 20, 24, 28, 78]


def unique_key_to_verse_ref(key: int) -> str:
    cum = 0
    for ch, count in enumerate(CHAPTER_VERSES, 1):
        if key <= cum + count:
            return f"BhG {ch}.{key - cum}"
        cum += count
    return ""


def extract_verse_refs_from_text(text: str) -> set[str]:
    refs = set()
    for match in re.finditer(r'BhG\s+(\d+\.\d+)', text, re.IGNORECASE):
        refs.add(f"BhG {match.group(1)}")
    for match in re.finditer(r'BG\s+(\d+\.\d+)', text, re.IGNORECASE):
        refs.add(f"BhG {match.group(1)}")
    return refs


# ── Dataset loaders with verse ID extraction ──

def load_gita_guidance_qa(path: str) -> list[dict]:
    pairs = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            d = json.loads(line)
            msgs = d.get('messages', [])
            if len(msgs) >= 2:
                user_msg = msgs[0].get('content', '')
                assistant_msg = msgs[1].get('content', '')
                if user_msg and assistant_msg:
                    refs = set()
                    for match in re.finditer(r'Chapter\s+(\d+)[,\s]+Verse\s+(\d+)', assistant_msg, re.IGNORECASE):
                        refs.add(f"BhG {match.group(1)}.{match.group(2)}")
                    refs |= extract_verse_refs_from_text(assistant_msg)
                    if refs:
                        pairs.append({
                            'question': user_msg,
                            'ground_truth': assistant_msg,
                            'source': 'gita_guidance_qa',
                            'verse_refs': sorted(refs),
                        })
    return pairs


def load_hf_gita_qa(path: str) -> list[dict]:
    pairs = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            d = json.loads(line)
            q = d.get('question', '')
            a = d.get('answer', '')
            ch = d.get('chapter_no')
            vs = d.get('verse_no')
            if q and a and ch and vs:
                pairs.append({
                    'question': q,
                    'ground_truth': a,
                    'source': 'hf_gita_qa',
                    'verse_refs': [f"BhG {ch}.{vs}"],
                })
    return pairs


def load_kaggle_gita_qa(path: str) -> list[dict]:
    pairs = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            d = json.loads(line)
            q = d.get('question', '')
            a = d.get('answer', '')
            vs = d.get('verse_source', '')
            if q and a and vs and '.' in vs:
                pairs.append({
                    'question': q,
                    'ground_truth': a,
                    'source': 'kaggle_gita_qa',
                    'verse_refs': [f"BhG {vs}"],
                })
    return pairs


def load_iskcon_vedabase(path: str, max_samples: int) -> list[dict]:
    pairs = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            d = json.loads(line)
            key = d.get('unique_key', 0)
            translation = d.get('translation', '')
            if key and translation:
                ref = unique_key_to_verse_ref(key)
                if ref:
                    pairs.append({
                        'question': "Explain the meaning and context of this Bhagavad Gita verse.",
                        'ground_truth': translation,
                        'source': 'iskcon_vedabase',
                        'verse_refs': [ref],
                    })
    random.Random(42).shuffle(pairs)
    return pairs[:max_samples]


# ── Metrics ──

def compute_verse_retrieval_metrics(expected_refs: list[str], reranked: list[dict]) -> dict:
    expected = set(expected_refs)
    verse_results = [r for r in reranked if r.get("chunk_type") == "verse"]
    metrics = {}
    for k in [1, 3, 5, 10]:
        top_k = verse_results[:k]
        retrieved_refs = {r.get("verse_ref", "") for r in top_k}
        overlap = retrieved_refs & expected
        metrics[f"recall_at_{k}"] = len(overlap) / max(len(expected), 1)
    mrr = 0.0
    for i, r in enumerate(verse_results):
        if r.get("verse_ref", "") in expected:
            mrr = 1.0 / (i + 1)
            break
    metrics["mrr"] = mrr
    metrics["expected_verses"] = sorted(expected)
    metrics["retrieved_verse_refs"] = [r.get("verse_ref", "") for r in verse_results[:10]]
    return metrics


def compute_semantic_metrics(text_a: str, text_b: str, model) -> dict:
    if not text_a or not text_b:
        return {"semantic_similarity": 0.0, "word_overlap": 0.0, "char_length_a": 0, "char_length_b": 0}
    emb_a = model.encode(text_a, convert_to_tensor=True)
    emb_b = model.encode(text_b, convert_to_tensor=True)
    sim = float(util.cos_sim(emb_a, emb_b).item())
    words_a = set(text_a.lower().split())
    words_b = set(text_b.lower().split())
    overlap = len(words_a & words_b) / max(len(words_b), 1)
    return {
        "semantic_similarity": round(sim, 4),
        "word_overlap": round(overlap, 4),
        "char_length_a": len(text_a),
        "char_length_b": len(text_b),
    }


def build_top_verse_text(reranked: list[dict], top_n: int = 3) -> str:
    verses = [r.get("text_iast", "") for r in reranked
              if r.get("chunk_type") == "verse" and r.get("text_iast")]
    return " ".join(verses[:top_n])


# ── Main evaluation ──

def run_evaluation(args):
    print("=" * 70)
    print("SRAG RETRIEVAL-ONLY EVALUATION")
    print("Measures verse retrieval accuracy + semantic relevance")
    print(f"Samples per dataset: {args.samples}")
    print("=" * 70)

    from src.utils.config import Config
    from src.preprocessing.chunker import load_chunks
    from src.reranking.linguistic_reranker import LinguisticReranker
    from src.reranking.adaptive_reranker import detect_query_type
    from src.retrieval.bm25_retriever import BM25Retriever
    from src.retrieval.hybrid_fusion import HybridRetriever
    from src.retrieval.vector_store import VectorStore
    from src.generation.query_processor import QueryProcessor

    config = Config()
    query_processor = QueryProcessor(config)
    vector_store = VectorStore(config)
    bm25_retriever = BM25Retriever()
    hybrid_retriever = HybridRetriever(config)
    reranker = LinguisticReranker(config)

    # Load data
    print("\nLoading chunks...")
    chunks = load_chunks(Path(config.get("data.chunks_file", "data/processed/chunks.jsonl")))
    chunk_map = {c.chunk_id: c for c in chunks}
    print(f"  {len(chunks)} chunks loaded")

    # Build indices
    print("Building indices...")
    faiss_path = Path(config.get("data.faiss_index"))
    metadata_path = Path(config.get("data.faiss_metadata"))
    if faiss_path.exists() and metadata_path.exists():
        vector_store.load(faiss_path, metadata_path)
    else:
        vector_store.build_index(chunks, use_devanagari=True, verse_only=False)
        vector_store.save(faiss_path, metadata_path)
    print(f"  FAISS: {vector_store.index.ntotal} vectors")

    bm25_retriever.build_index(chunks, use_lemmas=True)
    print(f"  BM25: {len(bm25_retriever.chunk_ids)} docs")

    # Sim model
    print("Loading sentence transformer...")
    sim_model = SentenceTransformer('all-MiniLM-L6-v2')

    # Load datasets
    eval_dir = Path("data/evaluation/external")
    datasets = {}

    ds_configs = [
        ("gita_guidance_qa", "gita_guidance_qa.jsonl", load_gita_guidance_qa, None),
        ("hf_gita_qa", "hf_gita_qa.jsonl", load_hf_gita_qa, None),
        ("kaggle_gita_qa", "kaggle_gita_qa.jsonl", load_kaggle_gita_qa, None),
        ("iskcon_vedabase", "iskcon_vedabase.jsonl", load_iskcon_vedabase, args.iskcon_samples),
    ]

    for name, filename, loader, max_s in ds_configs:
        path = eval_dir / filename
        if path.exists():
            kwargs = {} if max_s is None else {"max_samples": max_s}
            data = loader(str(path), **kwargs)
            if data:
                datasets[name] = data
                print(f"Loaded {name}: {len(data)} pairs")

    if not datasets:
        print("No datasets loaded!")
        return

    # Run evaluation per dataset
    all_results = {}
    total_no_id = {"samples": 0, "sum_recall_at_1": 0, "sum_recall_at_3": 0, "sum_recall_at_5": 0,
                   "sum_recall_at_10": 0, "sum_mrr": 0, "sum_semantic": 0, "sum_overlap": 0, "sum_time": 0}
    total_with_id = dict(total_no_id)

    for ds_name, pairs in datasets.items():
        sample = random.Random(42).sample(pairs, min(args.samples, len(pairs)))
        print(f"\n{'=' * 60}")
        print(f"  DATASET: {ds_name} ({len(sample)} samples)")
        print(f"{'=' * 60}")

        ds_results = []

        for i, qa in enumerate(sample):
            question = qa['question']
            ground_truth = qa['ground_truth']
            expected_refs = qa['verse_refs']
            verse_id = expected_refs[0] if expected_refs else ""

            conditions = [("without_id", question)]
            if verse_id:
                conditions.append(("with_id", f"{verse_id}: {question}"))

            row = {
                "question": question[:120],
                "source": ds_name,
                "verse_refs": expected_refs,
            }

            for cond_label, query_text in conditions:
                try:
                    start = time.time()

                    processed = query_processor.process_query_local(query_text)
                    query_type = detect_query_type(processed.query_iast, processed.concepts)

                    top_k = 50
                    vector_results = vector_store.search(processed.query_devanagari, top_k=top_k)
                    bm25_results = bm25_retriever.search(processed.query_iast, top_k=top_k)
                    fused = hybrid_retriever.fuse_results(
                        vector_results, [], bm25_results, top_k=top_k, query_type=query_type,
                    )
                    reranked = reranker.rerank(
                        query_iast=processed.query_iast,
                        concepts=processed.concepts,
                        candidates=fused,
                        all_chunks=chunks,
                        chunk_map=chunk_map,
                    )

                    elapsed = time.time() - start

                    verse_metrics = compute_verse_retrieval_metrics(expected_refs, reranked)
                    top_verse_text = build_top_verse_text(reranked, top_n=3)
                    sem_metrics = compute_semantic_metrics(top_verse_text, ground_truth, sim_model)

                    row[cond_label] = {
                        "time_seconds": round(elapsed, 2),
                        "query_type": query_type,
                        "verse_retrieval": verse_metrics,
                        "semantic": sem_metrics,
                    }

                except Exception as e:
                    row[cond_label] = {"error": str(e)}

            ds_results.append(row)

            # Print per-sample progress
            no_id = row.get("without_id", {})
            with_id = row.get("with_id", {})
            n_recall = no_id.get("verse_retrieval", {}).get("recall_at_1", -1)
            w_recall = with_id.get("verse_retrieval", {}).get("recall_at_1", -1)
            n_sem = no_id.get("semantic", {}).get("semantic_similarity", -1)
            w_sem = with_id.get("semantic", {}).get("semantic_similarity", -1)
            ref_str = ", ".join(expected_refs)
            print(f"  [{i+1}/{len(sample)}] {ref_str} | "
                  f"R@1: {n_recall:.2f}/{w_recall:.2f} | "
                  f"Sem: {n_sem:.3f}/{w_sem:.3f} | "
                  f"Q: {no_id.get('query_type','?')}")

        all_results[ds_name] = ds_results

        # Dataset summary
        valid = [r for r in ds_results if "without_id" in r and "error" not in r["without_id"]]
        if not valid:
            continue

        def avg_metric(results, cond, metric_path):
            vals = []
            for r in results:
                d = r.get(cond, {})
                for key in metric_path:
                    d = d.get(key, {}) if isinstance(d, dict) else {}
                if isinstance(d, (int, float)):
                    vals.append(d)
            return sum(vals) / len(vals) if vals else 0.0

        no_id_avg = {
            "recall_at_1": avg_metric(valid, "without_id", ["verse_retrieval", "recall_at_1"]),
            "recall_at_3": avg_metric(valid, "without_id", ["verse_retrieval", "recall_at_3"]),
            "recall_at_5": avg_metric(valid, "without_id", ["verse_retrieval", "recall_at_5"]),
            "recall_at_10": avg_metric(valid, "without_id", ["verse_retrieval", "recall_at_10"]),
            "mrr": avg_metric(valid, "without_id", ["verse_retrieval", "mrr"]),
            "semantic_similarity": avg_metric(valid, "without_id", ["semantic", "semantic_similarity"]),
            "word_overlap": avg_metric(valid, "without_id", ["semantic", "word_overlap"]),
            "avg_time": avg_metric(valid, "without_id", ["time_seconds"]),
        }

        with_id_valid = [r for r in ds_results if "with_id" in r and "error" not in r["with_id"]]
        with_id_avg = {}
        if with_id_valid:
            with_id_avg = {
                "recall_at_1": avg_metric(with_id_valid, "with_id", ["verse_retrieval", "recall_at_1"]),
                "recall_at_3": avg_metric(with_id_valid, "with_id", ["verse_retrieval", "recall_at_3"]),
                "recall_at_5": avg_metric(with_id_valid, "with_id", ["verse_retrieval", "recall_at_5"]),
                "recall_at_10": avg_metric(with_id_valid, "with_id", ["verse_retrieval", "recall_at_10"]),
                "mrr": avg_metric(with_id_valid, "with_id", ["verse_retrieval", "mrr"]),
                "semantic_similarity": avg_metric(with_id_valid, "with_id", ["semantic", "semantic_similarity"]),
                "word_overlap": avg_metric(with_id_valid, "with_id", ["semantic", "word_overlap"]),
                "avg_time": avg_metric(with_id_valid, "with_id", ["time_seconds"]),
            }

        print(f"\n  --- {ds_name} Summary ---")
        print(f"  {'Metric':<25} {'Without Verse ID':>20} {'With Verse ID':>20}")
        print(f"  {'-'*25} {'-'*20} {'-'*20}")
        for key in ["recall_at_1", "recall_at_3", "recall_at_5", "recall_at_10", "mrr",
                     "semantic_similarity", "word_overlap", "avg_time"]:
            v1 = no_id_avg.get(key, 0)
            v2 = with_id_avg.get(key, 0) if with_id_avg else 0
            suffix = "s" if key == "avg_time" else ""
            print(f"  {key:<25} {v1:>20.4f}{suffix} {v2:>20.4f}{suffix}")

        # Accumulate totals
        for r in valid:
            total_no_id["samples"] += 1
            d = r["without_id"]
            vr = d.get("verse_retrieval", {})
            sm = d.get("semantic", {})
            total_no_id["sum_recall_at_1"] += vr.get("recall_at_1", 0)
            total_no_id["sum_recall_at_3"] += vr.get("recall_at_3", 0)
            total_no_id["sum_recall_at_5"] += vr.get("recall_at_5", 0)
            total_no_id["sum_recall_at_10"] += vr.get("recall_at_10", 0)
            total_no_id["sum_mrr"] += vr.get("mrr", 0)
            total_no_id["sum_semantic"] += sm.get("semantic_similarity", 0)
            total_no_id["sum_overlap"] += sm.get("word_overlap", 0)
            total_no_id["sum_time"] += d.get("time_seconds", 0)

        for r in with_id_valid:
            total_with_id["samples"] += 1
            d = r["with_id"]
            vr = d.get("verse_retrieval", {})
            sm = d.get("semantic", {})
            total_with_id["sum_recall_at_1"] += vr.get("recall_at_1", 0)
            total_with_id["sum_recall_at_3"] += vr.get("recall_at_3", 0)
            total_with_id["sum_recall_at_5"] += vr.get("recall_at_5", 0)
            total_with_id["sum_recall_at_10"] += vr.get("recall_at_10", 0)
            total_with_id["sum_mrr"] += vr.get("mrr", 0)
            total_with_id["sum_semantic"] += sm.get("semantic_similarity", 0)
            total_with_id["sum_overlap"] += sm.get("word_overlap", 0)
            total_with_id["sum_time"] += d.get("time_seconds", 0)

    # Overall summary
    print(f"\n{'=' * 70}")
    print("OVERALL RESULTS")
    print(f"{'=' * 70}")
    print(f"{'Metric':<25} {'Without Verse ID':>20} {'With Verse ID':>20}")
    print(f"{'-' * 25} {'-' * 20} {'-' * 20}")

    key_map = {
        "recall_at_1": "sum_recall_at_1", "recall_at_3": "sum_recall_at_3",
        "recall_at_5": "sum_recall_at_5", "recall_at_10": "sum_recall_at_10",
        "mrr": "sum_mrr", "semantic_similarity": "sum_semantic",
        "word_overlap": "sum_overlap", "avg_time": "sum_time",
    }
    label_map = {
        "recall_at_1": "Recall@1", "recall_at_3": "Recall@3",
        "recall_at_5": "Recall@5", "recall_at_10": "Recall@10",
        "mrr": "MRR", "semantic_similarity": "Semantic Sim.",
        "word_overlap": "Word Overlap", "avg_time": "Avg Time (s)",
    }
    for key in key_map:
        n_val = total_no_id[key_map[key]] / max(total_no_id["samples"], 1)
        w_val = total_with_id[key_map[key]] / max(total_with_id["samples"], 1)
        print(f"  {label_map[key]:<25} {n_val:>20.4f} {w_val:>20.4f}")

    # Save results
    output = {
        "config": {
            "samples_per_dataset": args.samples,
            "iskcon_samples": args.iskcon_samples,
            "model": "all-MiniLM-L6-v2",
            "reranker_top_n": reranker.top_n,
            "note": "No LLM calls. Retrieval -> fusion -> re-ranking only.",
        },
        "overall": {
            "without_verse_id": {
                "total_samples": total_no_id["samples"],
                "avg_recall_at_1": round(total_no_id["sum_recall_at_1"] / max(total_no_id["samples"], 1), 4),
                "avg_recall_at_3": round(total_no_id["sum_recall_at_3"] / max(total_no_id["samples"], 1), 4),
                "avg_recall_at_5": round(total_no_id["sum_recall_at_5"] / max(total_no_id["samples"], 1), 4),
                "avg_recall_at_10": round(total_no_id["sum_recall_at_10"] / max(total_no_id["samples"], 1), 4),
                "avg_mrr": round(total_no_id["sum_mrr"] / max(total_no_id["samples"], 1), 4),
                "avg_semantic_similarity": round(total_no_id["sum_semantic"] / max(total_no_id["samples"], 1), 4),
                "avg_word_overlap": round(total_no_id["sum_overlap"] / max(total_no_id["samples"], 1), 4),
                "avg_time_seconds": round(total_no_id["sum_time"] / max(total_no_id["samples"], 1), 2),
            },
            "with_verse_id": {
                "total_samples": total_with_id["samples"],
                "avg_recall_at_1": round(total_with_id["sum_recall_at_1"] / max(total_with_id["samples"], 1), 4),
                "avg_recall_at_3": round(total_with_id["sum_recall_at_3"] / max(total_with_id["samples"], 1), 4),
                "avg_recall_at_5": round(total_with_id["sum_recall_at_5"] / max(total_with_id["samples"], 1), 4),
                "avg_recall_at_10": round(total_with_id["sum_recall_at_10"] / max(total_with_id["samples"], 1), 4),
                "avg_mrr": round(total_with_id["sum_mrr"] / max(total_with_id["samples"], 1), 4),
                "avg_semantic_similarity": round(total_with_id["sum_semantic"] / max(total_with_id["samples"], 1), 4),
                "avg_word_overlap": round(total_with_id["sum_overlap"] / max(total_with_id["samples"], 1), 4),
                "avg_time_seconds": round(total_with_id["sum_time"] / max(total_with_id["samples"], 1), 2),
            },
        },
        "by_dataset": {},
        "detailed_results": all_results,
    }

    for ds_name, ds_results in all_results.items():
        valid_no = [r for r in ds_results if "without_id" in r and "error" not in r["without_id"]]
        valid_with = [r for r in ds_results if "with_id" in r and "error" not in r["with_id"]]

        def avg_fn(results, cond, path):
            vals = []
            for r in results:
                d = r.get(cond, {})
                for k in path:
                    d = d.get(k, {}) if isinstance(d, dict) else {}
                if isinstance(d, (int, float)):
                    vals.append(d)
            return round(sum(vals) / len(vals), 4) if vals else 0.0

        output["by_dataset"][ds_name] = {
            "total_samples": len(ds_results),
            "without_verse_id": {
                "n": len(valid_no),
                "recall_at_1": avg_fn(valid_no, "without_id", ["verse_retrieval", "recall_at_1"]),
                "recall_at_3": avg_fn(valid_no, "without_id", ["verse_retrieval", "recall_at_3"]),
                "recall_at_5": avg_fn(valid_no, "without_id", ["verse_retrieval", "recall_at_5"]),
                "recall_at_10": avg_fn(valid_no, "without_id", ["verse_retrieval", "recall_at_10"]),
                "mrr": avg_fn(valid_no, "without_id", ["verse_retrieval", "mrr"]),
                "semantic_similarity": avg_fn(valid_no, "without_id", ["semantic", "semantic_similarity"]),
                "word_overlap": avg_fn(valid_no, "without_id", ["semantic", "word_overlap"]),
                "avg_time": avg_fn(valid_no, "without_id", ["time_seconds"]),
            },
            "with_verse_id": {} if not valid_with else {
                "n": len(valid_with),
                "recall_at_1": avg_fn(valid_with, "with_id", ["verse_retrieval", "recall_at_1"]),
                "recall_at_3": avg_fn(valid_with, "with_id", ["verse_retrieval", "recall_at_3"]),
                "recall_at_5": avg_fn(valid_with, "with_id", ["verse_retrieval", "recall_at_5"]),
                "recall_at_10": avg_fn(valid_with, "with_id", ["verse_retrieval", "recall_at_10"]),
                "mrr": avg_fn(valid_with, "with_id", ["verse_retrieval", "mrr"]),
                "semantic_similarity": avg_fn(valid_with, "with_id", ["semantic", "semantic_similarity"]),
                "word_overlap": avg_fn(valid_with, "with_id", ["semantic", "word_overlap"]),
                "avg_time": avg_fn(valid_with, "with_id", ["time_seconds"]),
            },
        }

    output_path = Path("data/evaluation/retrieval_evaluation.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nFull report saved to: {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Retrieval-only evaluation with verse retrieval accuracy metrics")
    parser.add_argument("--samples", type=int, default=30, help="Samples per dataset")
    parser.add_argument("--iskcon-samples", type=int, default=15, help="Max ISKCON samples")
    args = parser.parse_args()

    run_evaluation(args)
