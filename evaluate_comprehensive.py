"""Comprehensive SRAG evaluation across all datasets with normalization comparison."""

import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

import json, time, re, random
from pathlib import Path
from sentence_transformers import SentenceTransformer, util


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
                    pairs.append({'question': user_msg, 'ground_truth': assistant_msg, 'source': 'gita_guidance_qa'})
    return pairs


def load_edwin_arnold_qa(path: str) -> list[dict]:
    pairs = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            d = json.loads(line)
            q = d.get('question', '')
            a = d.get('answer', '')
            if q and a:
                pairs.append({'question': q, 'ground_truth': a, 'source': 'edwin_arnold_qa'})
    return pairs


def load_iskcon_vedabase(path: str, max_samples: int = 100) -> list[dict]:
    pairs = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            d = json.loads(line)
            key = d.get('unique_key', '')
            translation = d.get('translation', '')
            if key and translation:
                pairs.append({
                    'question': f"Explain the meaning of Bhagavad Gita verse {key}.",
                    'ground_truth': translation,
                    'source': 'iskcon_vedabase',
                    'verse_key': key,
                })
    random.shuffle(pairs)
    return pairs[:max_samples]


def load_hf_gita_qa(path: str) -> list[dict]:
    pairs = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            d = json.loads(line)
            q = d.get('question', '')
            a = d.get('answer', '')
            if q and a:
                pairs.append({'question': q, 'ground_truth': a, 'source': 'hf_gita_qa'})
    return pairs


def load_kaggle_gita_qa(path: str) -> list[dict]:
    pairs = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            d = json.loads(line)
            q = d.get('question', '')
            a = d.get('answer', '')
            if q and a:
                pairs.append({'question': q, 'ground_truth': a, 'source': 'kaggle_gita_qa'})
    return pairs


def extract_verse_refs(text: str) -> set[str]:
    refs = set()
    for match in re.finditer(r'BhG\s+(\d+\.\d+)', text, re.IGNORECASE):
        refs.add(match.group(1))
    for match in re.finditer(r'BG\s+(\d+\.\d+)', text, re.IGNORECASE):
        refs.add(match.group(1))
    return refs


def compute_metrics(srag_answer: str, ground_truth: str, model) -> dict:
    emb1 = model.encode(srag_answer, convert_to_tensor=True)
    emb2 = model.encode(ground_truth, convert_to_tensor=True)
    semantic_sim = float(util.cos_sim(emb1, emb2).item())

    srag_words = set(srag_answer.lower().split())
    gt_words = set(ground_truth.lower().split())
    word_overlap = len(srag_words & gt_words) / max(len(gt_words), 1)

    srag_verses = extract_verse_refs(srag_answer)
    gt_verses = extract_verse_refs(ground_truth)
    verse_recall = len(srag_verses & gt_verses) / max(len(gt_verses), 1) if gt_verses else None

    return {
        'semantic_similarity': round(semantic_sim, 4),
        'word_overlap': round(word_overlap, 4),
        'verse_recall': round(verse_recall, 4) if verse_recall is not None else None,
        'srag_length': len(srag_answer),
        'gt_length': len(ground_truth),
    }


def run_evaluation(
    pipeline,
    sim_model,
    qa_pairs: list[dict],
    sample_size: int = 30,
    normalize_method: str = "none",
    label: str = "",
) -> list[dict]:
    """Run evaluation on a set of QA pairs."""
    actual_size = min(sample_size, len(qa_pairs))
    random.seed(42)
    sample = random.sample(qa_pairs, actual_size)

    print(f"\n{'='*60}")
    print(f"EVALUATION: {label}")
    print(f"Samples: {actual_size}, Normalize: {normalize_method}")
    print(f"{'='*60}")

    results = []
    total_time = 0

    # Set normalization method on pipeline
    if hasattr(pipeline, 'reranker'):
        pipeline.reranker.normalize_method = normalize_method

    for i, qa in enumerate(sample):
        question = qa['question']
        ground_truth = qa['ground_truth']

        print(f"  [{i+1}/{actual_size}] {question[:70]}...")

        try:
            start = time.time()
            result = pipeline.query(question, use_api=True)
            elapsed = time.time() - start
            total_time += elapsed

            srag_answer = result.get('answer', '')
            metrics = compute_metrics(srag_answer, ground_truth, sim_model)

            entry = {
                'question': question[:200],
                'ground_truth': ground_truth[:500],
                'srag_answer': srag_answer[:500],
                'source': qa['source'],
                'query_type': result.get('query_type', ''),
                'time_seconds': round(elapsed, 2),
                **metrics,
            }
            results.append(entry)

            print(f"    Semantic: {metrics['semantic_similarity']:.3f} | "
                  f"Type: {entry['query_type']} | Time: {elapsed:.1f}s")
        except Exception as e:
            print(f"    ERROR: {e}")
            results.append({'question': question[:200], 'source': qa['source'], 'error': str(e)})

    # Summary
    successful = [r for r in results if 'error' not in r]
    if successful:
        avg_sim = sum(r['semantic_similarity'] for r in successful) / len(successful)
        avg_time = total_time / len(successful)
        high = sum(1 for r in successful if r['semantic_similarity'] >= 0.5)
        med = sum(1 for r in successful if 0.3 <= r['semantic_similarity'] < 0.5)
        low = sum(1 for r in successful if r['semantic_similarity'] < 0.3)
        print(f"\n  SUMMARY ({label}):")
        print(f"    Evaluated: {len(successful)}, Errors: {len(results) - len(successful)}")
        print(f"    Avg Semantic Similarity: {avg_sim:.4f}")
        print(f"    High (>=0.5): {high}, Medium (0.3-0.5): {med}, Low (<0.3): {low}")
        print(f"    Avg Response Time: {avg_time:.1f}s")

    return results


def generate_report(all_results: dict, output_path: Path):
    """Generate comprehensive comparison report."""
    report = {
        "summary": {},
        "by_dataset": {},
        "by_normalization": {},
        "detailed_results": all_results,
    }

    for label, results in all_results.items():
        successful = [r for r in results if 'error' not in r]
        if not successful:
            continue

        avg_sim = sum(r['semantic_similarity'] for r in successful) / len(successful)
        avg_overlap = sum(r['word_overlap'] for r in successful) / len(successful)
        avg_time = sum(r.get('time_seconds', 0) for r in successful) / len(successful)
        high = sum(1 for r in successful if r['semantic_similarity'] >= 0.5)
        med = sum(1 for r in successful if 0.3 <= r['semantic_similarity'] < 0.5)
        low = sum(1 for r in successful if r['semantic_similarity'] < 0.3)

        report["by_dataset"][label] = {
            "count": len(successful),
            "errors": len(results) - len(successful),
            "avg_semantic_similarity": round(avg_sim, 4),
            "avg_word_overlap": round(avg_overlap, 4),
            "avg_response_time": round(avg_time, 2),
            "high_similarity": high,
            "medium_similarity": med,
            "low_similarity": low,
        }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\nReport saved to: {output_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--iskcon-samples", type=int, default=15)
    parser.add_argument("--normalize", default="none", choices=["none", "minmax", "l2", "zscore"])
    parser.add_argument("--output", default="data/evaluation/srag_comprehensive_report.json")
    args = parser.parse_args()

    from src.utils.config import Config
    config = Config()

    from src.langchain_components.graph import SRAGGraphPipeline
    pipeline = SRAGGraphPipeline(config)

    pipeline.preprocess()
    pipeline.build_indices()
    try:
        if hasattr(pipeline, '_get_graph_retriever'):
            pipeline._get_graph_retriever()
    except Exception:
        pass

    sim_model = SentenceTransformer('all-MiniLM-L6-v2')
    eval_dir = Path("data/evaluation/external")

    # Load all datasets
    datasets = {}

    guidance_path = eval_dir / "gita_guidance_qa.jsonl"
    if guidance_path.exists():
        datasets["gita_guidance_qa"] = load_gita_guidance_qa(str(guidance_path))
        print(f"Loaded gita_guidance_qa: {len(datasets['gita_guidance_qa'])} pairs")

    arnold_path = eval_dir / "edwin_arnold_qa.jsonl"
    if arnold_path.exists():
        datasets["edwin_arnold_qa"] = load_edwin_arnold_qa(str(arnold_path))
        print(f"Loaded edwin_arnold_qa: {len(datasets['edwin_arnold_qa'])} pairs")

    hf_path = eval_dir / "hf_gita_qa.jsonl"
    if hf_path.exists():
        datasets["hf_gita_qa"] = load_hf_gita_qa(str(hf_path))
        print(f"Loaded hf_gita_qa: {len(datasets['hf_gita_qa'])} pairs")

    kaggle_path = eval_dir / "kaggle_gita_qa.jsonl"
    if kaggle_path.exists():
        datasets["kaggle_gita_qa"] = load_kaggle_gita_qa(str(kaggle_path))
        print(f"Loaded kaggle_gita_qa: {len(datasets['kaggle_gita_qa'])} pairs")

    iskcon_path = eval_dir / "iskcon_vedabase.jsonl"
    if iskcon_path.exists():
        datasets["iskcon_vedabase"] = load_iskcon_vedabase(str(iskcon_path), max_samples=args.iskcon_samples)
        print(f"Loaded iskcon_vedabase: {len(datasets['iskcon_vedabase'])} pairs")

    # Run evaluations
    all_results = {}

    for name, pairs in datasets.items():
        label = f"{name} (normalize={args.normalize})"
        results = run_evaluation(
            pipeline, sim_model, pairs,
            sample_size=args.samples,
            normalize_method=args.normalize,
            label=label,
        )
        all_results[label] = results

    # Generate report
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    generate_report(all_results, output_path)

    # Print comparison table
    print("\n" + "=" * 70)
    print("COMPARISON TABLE")
    print("=" * 70)
    print(f"{'Dataset':<35} {'Samples':>8} {'Avg Sim':>10} {'High':>6} {'Med':>6} {'Low':>6}")
    print("-" * 70)
    for label, results in all_results.items():
        successful = [r for r in results if 'error' not in r]
        if not successful:
            continue
        avg_sim = sum(r['semantic_similarity'] for r in successful) / len(successful)
        high = sum(1 for r in successful if r['semantic_similarity'] >= 0.5)
        med = sum(1 for r in successful if 0.3 <= r['semantic_similarity'] < 0.5)
        low = sum(1 for r in successful if r['semantic_similarity'] < 0.3)
        short_label = label[:35]
        print(f"{short_label:<35} {len(successful):>8} {avg_sim:>10.4f} {high:>6} {med:>6} {low:>6}")

    pipeline.close()


if __name__ == "__main__":
    main()
