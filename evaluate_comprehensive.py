"""Comprehensive SRAG evaluation: compare output semantic similarity to ground-truth datasets (Gita Guidance QA, Edwin Arnold QA, ISKCON VedaBase)."""

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
            sanskrit = d.get('sanskrit_data', '')
            translation = d.get('translation', '')
            if key and translation:
                pairs.append({
                    'question': f"Explain the meaning of Bhagavad Gita verse {key}.",
                    'ground_truth': translation,
                    'source': 'iskcon_vedabase',
                    'verse_key': key,
                    'sanskrit': sanskrit,
                })
    random.shuffle(pairs)
    return pairs[:max_samples]


def extract_verse_refs(text: str) -> set[str]:
    refs = set()
    for match in re.finditer(r'BhG\s+(\d+\.\d+)', text, re.IGNORECASE):
        refs.add(match.group(1))
    for match in re.finditer(r'BG\s+(\d+\.\d+)', text, re.IGNORECASE):
        refs.add(match.group(1))
    for match in re.finditer(r'Chapter\s+(\d+)[,\s]+Verse\s+(\d+)', text, re.IGNORECASE):
        refs.add(f"{match.group(1)}.{match.group(2)}")
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
        'srag_verses': sorted(srag_verses),
        'gt_verses': sorted(gt_verses),
        'srag_length': len(srag_answer),
        'gt_length': len(ground_truth),
    }


def run_comprehensive_evaluation(sample_size: int = 50, use_langgraph: bool = True, iskcon_samples: int = 30):
    print("=" * 70)
    print("SRAG COMPREHENSIVE EVALUATION — All 3 Datasets")
    print(f"QA sample size: {sample_size}, Pipeline: {'LangGraph' if use_langgraph else 'Standard'}, ISKCON samples: {iskcon_samples}")
    print("=" * 70)

    from src.utils.config import Config
    config = Config()

    if use_langgraph:
        from src.langchain_components.graph import SRAGGraphPipeline
        pipeline = SRAGGraphPipeline(config)
    else:
        from main import SRAGPipeline
        pipeline = SRAGPipeline(config)

    pipeline.preprocess()
    pipeline.build_indices()
    try:
        if hasattr(pipeline, '_get_graph_retriever'):
            pipeline._get_graph_retriever()
    except Exception:
        pass

    eval_dir = Path("data/evaluation/external")
    sim_model = SentenceTransformer('all-MiniLM-L6-v2')

    all_pairs = []
    guidance_path = eval_dir / "gita_guidance_qa.jsonl"
    arnold_path = eval_dir / "edwin_arnold_qa.jsonl"
    iskcon_path = eval_dir / "iskcon_vedabase.jsonl"

    if guidance_path.exists():
        all_pairs.extend(load_gita_guidance_qa(str(guidance_path)))
    if arnold_path.exists():
        all_pairs.extend(load_edwin_arnold_qa(str(arnold_path)))

    iskcon_pairs = []
    if iskcon_path.exists():
        iskcon_pairs = load_iskcon_vedabase(str(iskcon_path), max_samples=iskcon_samples)
        print(f"Loaded {len(iskcon_pairs)} ISKCON VedaBase samples for comparison")

    if not all_pairs:
        print("No QA datasets found.")
        return

    print(f"\nLoaded {len(all_pairs)} QA pairs:")
    sources = {}
    for p in all_pairs:
        sources[p['source']] = sources.get(p['source'], 0) + 1
    for src, count in sources.items():
        print(f"  - {src}: {count}")

    actual_size = min(sample_size, len(all_pairs))
    random.seed(42)
    sample = random.sample(all_pairs, actual_size)

    print(f"\nRunning evaluation on {actual_size} QA samples + {len(iskcon_pairs)} ISKCON samples...")
    results = []
    total_time = 0

    for i, qa in enumerate(sample):
        question = qa['question']
        ground_truth = qa['ground_truth']

        print(f"\n[{i+1}/{actual_size} QA] {question[:80]}...")
        try:
            start = time.time()
            result = pipeline.query(question, use_api=True)
            elapsed = time.time() - start
            total_time += elapsed

            srag_answer = result.get('answer', '')
            srag_concepts = result.get('concepts_extracted', [])
            srag_verses = result.get('verses_cited', [])
            confidence = result.get('pipeline_confidence', {})
            query_type = result.get('query_type', '')

            metrics = compute_metrics(srag_answer, ground_truth, sim_model)

            entry = {
                'question': question[:200],
                'ground_truth': ground_truth[:500],
                'srag_answer': srag_answer[:500],
                'source': qa['source'],
                'concepts': srag_concepts,
                'verses_cited': srag_verses,
                'confidence': confidence,
                'query_type': query_type,
                'time_seconds': round(elapsed, 2),
                **metrics,
            }
            results.append(entry)
            print(f"  Semantic: {metrics['semantic_similarity']:.3f} | Type: {query_type} | Time: {elapsed:.1f}s")
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({'question': question[:200], 'source': qa['source'], 'error': str(e)})

    # ISKCON evaluation
    iskcon_results = []
    for i, qa in enumerate(iskcon_pairs):
        question = qa['question']
        ground_truth = qa['ground_truth']

        print(f"\n[{i+1}/{len(iskcon_pairs)} ISKCON] Verse {qa.get('verse_key', '')}...")
        try:
            start = time.time()
            result = pipeline.query(question, use_api=True)
            elapsed = time.time() - start

            srag_answer = result.get('answer', '')
            query_type = result.get('query_type', '')
            confidence = result.get('pipeline_confidence', {})

            metrics = compute_metrics(srag_answer, ground_truth, sim_model)

            entry = {
                'question': question[:200],
                'ground_truth': ground_truth[:300],
                'srag_answer': srag_answer[:300],
                'source': 'iskcon_vedabase',
                'verse_key': qa.get('verse_key', ''),
                'concepts': result.get('concepts_extracted', []),
                'verses_cited': result.get('verses_cited', []),
                'confidence': confidence,
                'query_type': query_type,
                'time_seconds': round(elapsed, 2),
                **metrics,
            }
            iskcon_results.append(entry)
            print(f"  Semantic: {metrics['semantic_similarity']:.3f} | Type: {query_type} | Time: {elapsed:.1f}s")
        except Exception as e:
            print(f"  ERROR: {e}")
            iskcon_results.append({'question': question[:200], 'source': 'iskcon_vedabase', 'error': str(e)})

    # consolidate
    all_results = results + iskcon_results

    # generate report
    successful = [r for r in all_results if 'error' not in r]
    if successful:
        avg_semantic = sum(r['semantic_similarity'] for r in successful) / len(successful)
        avg_word_overlap = sum(r['word_overlap'] for r in successful) / len(successful)

        high_sim = sum(1 for r in successful if r['semantic_similarity'] >= 0.5)
        med_sim = sum(1 for r in successful if 0.3 <= r['semantic_similarity'] < 0.5)
        low_sim = sum(1 for r in successful if r['semantic_similarity'] < 0.3)

        by_source = {}
        for r in successful:
            src = r['source']
            by_source.setdefault(src, []).append(r)

        by_query_type = {}
        for r in successful:
            qt = r.get('query_type', 'unknown')
            by_query_type.setdefault(qt, []).append(r)

        report = {
            'summary': {
                'total_evaluated': len(successful),
                'total_errors': len(all_results) - len(successful),
                'pipeline_type': 'langgraph' if use_langgraph else 'standard',
                'avg_semantic_similarity': round(avg_semantic, 4),
                'avg_word_overlap': round(avg_word_overlap, 4),
                'high_similarity_count': high_sim,
                'medium_similarity_count': med_sim,
                'low_similarity_count': low_sim,
            },
            'by_source': {},
            'by_query_type': {},
            'detailed_results': all_results,
        }

        for src, src_results in by_source.items():
            report['by_source'][src] = {
                'count': len(src_results),
                'avg_semantic_similarity': round(sum(r['semantic_similarity'] for r in src_results) / len(src_results), 4),
                'avg_word_overlap': round(sum(r['word_overlap'] for r in src_results) / len(src_results), 4),
            }

        for qt, qt_results in by_query_type.items():
            report['by_query_type'][qt] = {
                'count': len(qt_results),
                'avg_semantic_similarity': round(sum(r['semantic_similarity'] for r in qt_results) / len(qt_results), 4),
                'avg_word_overlap': round(sum(r['word_overlap'] for r in qt_results) / len(qt_results), 4),
            }

        output_path = Path("data/evaluation/srag_comprehensive_report.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print("\n" + "=" * 70)
        print("COMPREHENSIVE EVALUATION RESULTS")
        print("=" * 70)
        print(f"\nPipeline: {'LangGraph' if use_langgraph else 'Standard'}")
        print(f"Total evaluated: {len(successful)}")
        print(f"Errors: {len(all_results) - len(successful)}")
        print(f"\n--- Semantic Similarity ---")
        print(f"  Average: {avg_semantic:.4f}")
        print(f"  High (>=0.5): {high_sim} ({100*high_sim/len(successful):.0f}%)")
        print(f"  Medium (0.3-0.5): {med_sim} ({100*med_sim/len(successful):.0f}%)")
        print(f"  Low (<0.3): {low_sim} ({100*low_sim/len(successful):.0f}%)")
        print(f"\n--- By Dataset Source ---")
        for src, stats in report['by_source'].items():
            print(f"  {src}: {stats['count']} samples, semantic={stats['avg_semantic_similarity']:.4f}")
        print(f"\n--- By Query Type ---")
        for qt, stats in report['by_query_type'].items():
            print(f"  {qt}: {stats['count']} samples, semantic={stats['avg_semantic_similarity']:.4f}")
        print(f"\nFull report saved to: {output_path}")
    else:
        print("\nNo successful results to report.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--iskcon-samples", type=int, default=30)
    parser.add_argument("--langgraph", action="store_true", default=True)
    args = parser.parse_args()
    run_comprehensive_evaluation(sample_size=args.samples, use_langgraph=args.langgraph, iskcon_samples=args.iskcon_samples)
