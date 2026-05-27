"""Semantic retrieval evaluation using the full LangGraph pipeline.

Measures how semantically relevant retrieved chunks are to queries,
with and without verse IDs. Uses the pipeline's own BGE-M3 embedding 
model for multilingual (Sanskrit/Devanagari/English) similarity.
"""

import json
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.langchain_components.graph import SRAGGraphPipeline
from src.utils.config import Config
from src.utils.logger import logger

# ── Verse validation ──

_CHAPTER_VERSES = [46, 72, 43, 42, 29, 47, 30, 28, 34, 42, 55, 20, 35, 27, 20, 24, 28, 78]

_SUPPLEMENTED_VERSES = {
    (1,38), (1,47), (2,35), (5,9), (10,26), (12,6), (12,7),
    (15,6), (15,9), (15,13), (15,16), (16,2), (16,3), (16,9),
    (16,13), (17,4), (17,14),
}

def _parse_ref(ref: str):
    parts = ref.replace("BhG ", "").split(".")
    if len(parts) == 2:
        try:
            return int(parts[0]), int(parts[1])
        except ValueError:
            pass
    return 0, 0

def _is_valid_ref(ref: str) -> bool:
    ch, v = _parse_ref(ref)
    if ch < 1 or ch > 18 or v < 1:
        return False
    if v <= _CHAPTER_VERSES[ch - 1]:
        return True
    return (ch, v) in _SUPPLEMENTED_VERSES

def _filter_valid(refs: list[str]) -> list[str]:
    return [r for r in refs if _is_valid_ref(r)]

# ── Dataset loaders ──

def load_gita_guidance_qa(path: str, max_samples: int = 50) -> list[dict]:
    import re
    pairs = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            d = json.loads(line)
            if 'messages' not in d or len(d['messages']) < 2:
                continue
            q = d['messages'][0].get('content', '').strip()
            a = d['messages'][1].get('content', '').strip()
            if not q or not a:
                continue
            refs = re.findall(r'BhG\s+\d+\.\d+', a) or re.findall(r'Chapter\s+(\d+),\s*Verse\s+(\d+)', a)
            if not refs:
                continue
            if isinstance(refs[0], tuple):
                refs = [f"BhG {c}.{v}" for c, v in refs]
            refs = _filter_valid(refs)
            if refs:
                pairs.append({'question': q, 'ground_truth': a, 'source': 'gita_guidance_qa', 'verse_refs': refs})
    return random.Random(42).sample(pairs, min(max_samples, len(pairs)))


def load_hf_gita_qa(path: str, max_samples: int = 50) -> list[dict]:
    pairs = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            d = json.loads(line)
            ch = d.get('chapter_no')
            vs = d.get('verse_no')
            if not ch or not vs:
                continue
            ref = f"BhG {ch}.{vs}"
            if _is_valid_ref(ref):
                pairs.append({
                    'question': d.get('question', ''),
                    'ground_truth': d.get('answer', ''),
                    'source': 'hf_gita_qa',
                    'verse_refs': [ref],
                })
    return random.Random(42).sample(pairs, min(max_samples, len(pairs)))


def load_kaggle_gita_qa(path: str, max_samples: int = 50) -> list[dict]:
    pairs = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            d = json.loads(line)
            vs = d.get('verse_source', '')
            q = d.get('question', '').strip()
            a = d.get('answer', '').strip()
            if not vs or not q or not a:
                continue
            if '.' not in vs:
                continue
            ref = f"BhG {vs}"
            if _is_valid_ref(ref):
                pairs.append({
                    'question': q,
                    'ground_truth': a,
                    'source': 'kaggle_gita_qa',
                    'verse_refs': [ref],
                })
    return random.Random(42).sample(pairs, min(max_samples, len(pairs)))


def load_iskcon_vedabase(path: str, max_samples: int = 25) -> list[dict]:
    pairs = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            d = json.loads(line)
            key = d.get('unique_key', 0)
            translation = d.get('translation', '')
            if not key or not translation:
                continue
            ref = _unique_key_to_verse_ref(key)
            if ref:
                pairs.append({
                    'question': f"Explain {ref}",
                    'ground_truth': translation,
                    'source': 'iskcon_vedabase',
                    'verse_refs': [ref],
                })
    random.Random(42).shuffle(pairs)
    return pairs[:max_samples]


def _unique_key_to_verse_ref(key: int) -> str:
    chapter_verse_counts = [0, 47, 72, 43, 42, 29, 47, 30, 28, 34, 42, 55, 20, 17, 27, 20, 24, 28, 78]
    cumulative = 0
    for ch, count in enumerate(chapter_verse_counts):
        if ch == 0:
            continue
        cumulative += count
        if key < cumulative:
            vs = count - (cumulative - key) + 1
            return f"BhG {ch}.{vs}"
    return None


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
    metrics["retrieved_verse_refs"] = [r.get("verse_ref", "") for r in verse_results[:10]]
    return metrics


def compute_semantic_metrics(query: str, retrieved: list[dict], pipeline) -> dict:
    """Compute query-to-document semantic similarity using the pipeline's embedding model."""
    if not retrieved:
        return {"sim_top1": 0.0, "sim_top3": 0.0, "sim_top5": 0.0, "chunk_types": {}}

    # Get text from retrieved chunks (prefer Devanagari as it's what the vector store uses)
    texts = []
    for r in retrieved[:5]:
        text = r.get("text_devanagari", "") or r.get("text_iast", "")
        texts.append(text)

    if not any(texts):
        return {"sim_top1": 0.0, "sim_top3": 0.0, "sim_top5": 0.0, "chunk_types": {}}

    try:
        query_emb = pipeline.vector_store.encode_texts([query])
        doc_embs = pipeline.vector_store.encode_texts(texts)

        sims = (query_emb @ doc_embs.T).flatten().tolist()
        sims = [max(0.0, min(1.0, s)) for s in sims]

        chunk_types = {}
        for r in retrieved[:10]:
            ct = r.get("chunk_type", "unknown")
            chunk_types[ct] = chunk_types.get(ct, 0) + 1

        return {
            "sim_top1": round(sims[0], 4) if sims else 0.0,
            "sim_top3": round(np.mean(sims[:3]), 4) if len(sims) >= 3 else round(np.mean(sims), 4),
            "sim_top5": round(np.mean(sims), 4),
            "chunk_types": chunk_types,
        }
    except Exception as e:
        return {"sim_top1": 0.0, "sim_top3": 0.0, "sim_top5": 0.0, "error": str(e)}


# ── Main ──

def run_evaluation(args):
    print("=" * 70)
    print("SRAG SEMANTIC RETRIEVAL EVALUATION (LangGraph Pipeline)")
    print(f"Samples per dataset: {args.samples}, ISKCON: {args.iskcon_samples}")
    print("=" * 70)

    config = Config()
    pipeline = SRAGGraphPipeline(config)
    pipeline.preprocess()
    pipeline.build_indices()

    # Load datasets
    eval_dir = Path("data/evaluation/external")
    datasets = {}

    ds_configs = [
        ("gita_guidance_qa", "gita_guidance_qa.jsonl", load_gita_guidance_qa, args.samples),
        ("hf_gita_qa", "hf_gita_qa.jsonl", load_hf_gita_qa, args.samples),
        ("kaggle_gita_qa", "kaggle_gita_qa.jsonl", load_kaggle_gita_qa, args.samples),
        ("iskcon_vedabase", "iskcon_vedabase.jsonl", load_iskcon_vedabase, args.iskcon_samples),
    ]

    for name, filename, loader, max_s in ds_configs:
        path = eval_dir / filename
        if path.exists():
            data = loader(str(path), max_samples=max_s)
            if data:
                datasets[name] = data
                print(f"  Loaded {name}: {len(data)} samples")

    if not datasets:
        print("No datasets loaded!")
        return

    all_results = {}
    totals = {"samples": 0, "sum_recall_1": 0, "sum_recall_3": 0, "sum_mrr": 0,
              "sum_sim_top1": 0, "sum_sim_top3": 0, "sum_sim_top5": 0}
    totals_with_id = dict(totals)

    for ds_name, samples in datasets.items():
        print(f"\n{'=' * 60}")
        print(f"  DATASET: {ds_name} ({len(samples)} samples)")
        print(f"{'=' * 60}")

        ds_results = []

        for i, qa in enumerate(samples):
            question = qa['question']
            expected_refs = qa['verse_refs']
            verse_id = expected_refs[0] if expected_refs else ""

            conds = [("without_id", question)]

            row = {"question": question[:120], "verse_refs": expected_refs}

            for cond_label, query_text in conds:
                try:
                    t0 = time.time()

                    result = pipeline.query(query_text, use_api=False, retrieval_only=True)
                    elapsed = time.time() - t0

                    reranked = result.get("reranked_results", [])
                    verse_ref_detected = result.get("intermediate", {}).get("verse_ref_detected", False)

                    vm = compute_verse_retrieval_metrics(expected_refs, reranked)
                    sm = compute_semantic_metrics(query_text, reranked, pipeline)

                    row[cond_label] = {
                        "time": round(elapsed, 2),
                        "verse_ref_detected": verse_ref_detected,
                        "verse_retrieval": vm,
                        "semantic": sm,
                    }
                except Exception as e:
                    row[cond_label] = {"error": str(e)}

            ds_results.append(row)

            # Print progress
            wi = row.get("without_id", {})
            wv = wi.get("verse_retrieval", {})
            ws = wi.get("semantic", {})
            ri = row.get("with_id", {})
            rv = ri.get("verse_retrieval", {})
            rs = ri.get("semantic", {})
            ref_str = ", ".join(expected_refs)
            print(f"  [{i+1}/{len(samples)}] {ref_str:<16} "
                  f"R@1:{wv.get('recall_at_1',-1):.2f}/{rv.get('recall_at_1',-1):.2f} "
                  f"Sim:{ws.get('sim_top3',-1):.3f}/{rs.get('sim_top3',-1):.3f} "
                  f"Dir:{wi.get('verse_ref_detected','?')}")

        all_results[ds_name] = ds_results

        # Per-dataset summary
        valid = [r for r in ds_results if "without_id" in r and "error" not in r["without_id"]]
        valid_with = [r for r in ds_results if "with_id" in r and "error" not in r["with_id"]]

        def avg(rows, cond, *keys):
            vals = []
            for r in rows:
                d = r.get(cond, {})
                for k in keys:
                    d = d.get(k, {}) if isinstance(d, dict) else {}
                if isinstance(d, (int, float)):
                    vals.append(d)
            return np.mean(vals) if vals else 0.0

        print(f"\n  --- {ds_name} Summary (NoID) ---")
        print(f"  {'Metric':<22} {'NoID':>10}")
        print(f"  {'-'*22} {'-'*10}")
        for metric, keys in [
            ("Recall@1", ["verse_retrieval", "recall_at_1"]),
            ("Recall@3", ["verse_retrieval", "recall_at_3"]),
            ("MRR", ["verse_retrieval", "mrr"]),
            ("Sim@top1", ["semantic", "sim_top1"]),
            ("Sim@top3", ["semantic", "sim_top3"]),
            ("Sim@top5", ["semantic", "sim_top5"]),
        ]:
            v = avg(valid, "without_id", *keys)
            print(f"  {metric:<22} {v*100:>9.2f}%")

        # Accumulate totals
        for r in valid:
            totals["samples"] += 1
            d = r["without_id"]
            totals["sum_recall_1"] += d.get("verse_retrieval", {}).get("recall_at_1", 0)
            totals["sum_recall_3"] += d.get("verse_retrieval", {}).get("recall_at_3", 0)
            totals["sum_mrr"] += d.get("verse_retrieval", {}).get("mrr", 0)
            totals["sum_sim_top1"] += d.get("semantic", {}).get("sim_top1", 0)
            totals["sum_sim_top3"] += d.get("semantic", {}).get("sim_top3", 0)
            totals["sum_sim_top5"] += d.get("semantic", {}).get("sim_top5", 0)

        for r in valid_with:
            totals_with_id["samples"] += 1
            d = r["with_id"]
            totals_with_id["sum_recall_1"] += d.get("verse_retrieval", {}).get("recall_at_1", 0)
            totals_with_id["sum_recall_3"] += d.get("verse_retrieval", {}).get("recall_at_3", 0)
            totals_with_id["sum_mrr"] += d.get("verse_retrieval", {}).get("mrr", 0)
            totals_with_id["sum_sim_top1"] += d.get("semantic", {}).get("sim_top1", 0)
            totals_with_id["sum_sim_top3"] += d.get("semantic", {}).get("sim_top3", 0)
            totals_with_id["sum_sim_top5"] += d.get("semantic", {}).get("sim_top5", 0)

    # Overall
    n = max(totals["samples"], 1)
    print(f"\n{'=' * 70}")
    print("OVERALL SEMANTIC RETRIEVAL RESULTS (NoID)")
    print(f"{'=' * 70}")
    print(f"  {'Metric':<22} {'NoID':>10}")
    print(f"  {'-'*22} {'-'*10}")
    print(f"  {'Recall@1':<22} {totals['sum_recall_1']/n*100:>9.2f}%")
    print(f"  {'Recall@3':<22} {totals['sum_recall_3']/n*100:>9.2f}%")
    print(f"  {'MRR':<22} {totals['sum_mrr']/n*100:>9.2f}%")
    print(f"  {'Sim@top1':<22} {totals['sum_sim_top1']/n*100:>9.2f}%")
    print(f"  {'Sim@top3':<22} {totals['sum_sim_top3']/n*100:>9.2f}%")
    print(f"  {'Sim@top5':<22} {totals['sum_sim_top5']/n*100:>9.2f}%")
    print(f"  N = {totals['samples']}")

    # Save
    output_path = "data/evaluation/semantic_evaluation.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({"config": {"samples": args.samples, "iskcon_samples": args.iskcon_samples},
                    "results": all_results}, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=30, help="Samples per dataset")
    parser.add_argument("--iskcon-samples", type=int, default=15, help="ISKCON samples")
    args = parser.parse_args()
    run_evaluation(args)
