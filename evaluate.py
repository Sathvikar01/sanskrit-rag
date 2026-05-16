"""SRAG Evaluation Pipeline - Compare with ground truth datasets."""

import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

import json
import time
import re
import random
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
                    pairs.append({
                        'question': user_msg,
                        'ground_truth': assistant_msg,
                        'source': 'gita_guidance_qa',
                    })
    return pairs


def load_edwin_arnold_qa(path: str) -> list[dict]:
    pairs = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            d = json.loads(line)
            q = d.get('question', '')
            a = d.get('answer', '')
            if q and a:
                pairs.append({
                    'question': q,
                    'ground_truth': a,
                    'source': 'edwin_arnold_qa',
                })
    return pairs


def extract_verse_refs(text: str) -> set[str]:
    patterns = [
        r'BhG\s+(\d+\.\d+)',
        r'BG\s+(\d+\.\d+)',
        r'Chapter\s+(\d+)[,\s]+Verse\s+(\d+)',
    ]
    refs = set()
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            if len(match.groups()) == 1:
                refs.add(match.group(1))
            elif len(match.groups()) == 2:
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


def run_evaluation(sample_size: int = 100, use_langgraph: bool = False):
    print("=" * 70)
    print("SRAG EVALUATION PIPELINE")
    print(f"Sample size: {sample_size}, Pipeline: {'LangGraph' if use_langgraph else 'Standard'}")
    print("=" * 70)

    eval_dir = Path("data/evaluation/external")
    qa_pairs = []

    guidance_path = eval_dir / "gita_guidance_qa.jsonl"
    arnold_path = eval_dir / "edwin_arnold_qa.jsonl"

    if guidance_path.exists():
        qa_pairs.extend(load_gita_guidance_qa(str(guidance_path)))
    if arnold_path.exists():
        qa_pairs.extend(load_edwin_arnold_qa(str(arnold_path)))

    if not qa_pairs:
        print("No QA datasets found.")
        return

    print(f"\nLoaded {len(qa_pairs)} QA pairs:")
    sources = {}
    for p in qa_pairs:
        sources[p['source']] = sources.get(p['source'], 0) + 1
    for src, count in sources.items():
        print(f"  - {src}: {count}")

    print("\nLoading sentence transformer for semantic comparison...")
    sim_model = SentenceTransformer('all-MiniLM-L6-v2')

    print("Initializing SRAG pipeline...")
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
    except Exception as e:
        print(f"Graph connection failed: {e}")

    actual_size = min(sample_size, len(qa_pairs))
    random.seed(42)
    sample = random.sample(qa_pairs, actual_size)

    print(f"\nRunning evaluation on {actual_size} samples...")
    results = []
    total_time = 0

    for i, qa in enumerate(sample):
        question = qa['question']
        ground_truth = qa['ground_truth']

        print(f"\n[{i+1}/{actual_size}] {question[:80]}...")

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
                'question': question,
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

            print(f"  Semantic: {metrics['semantic_similarity']:.3f} | "
                  f"Word overlap: {metrics['word_overlap']:.3f} | "
                  f"Type: {query_type} | Time: {elapsed:.1f}s")
            if metrics['verse_recall'] is not None:
                print(f"  Verse recall: {metrics['verse_recall']:.3f}")

        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                'question': question,
                'source': qa['source'],
                'error': str(e),
            })

    successful = [r for r in results if 'error' not in r]
    if successful:
        avg_semantic = sum(r['semantic_similarity'] for r in successful) / len(successful)
        avg_word_overlap = sum(r['word_overlap'] for r in successful) / len(successful)
        avg_time = total_time / len(successful)

        verse_results = [r for r in successful if r.get('verse_recall') is not None]
        avg_verse_recall = (sum(r['verse_recall'] for r in verse_results) / len(verse_results)) if verse_results else 0

        high_sim = sum(1 for r in successful if r['semantic_similarity'] >= 0.5)
        med_sim = sum(1 for r in successful if 0.3 <= r['semantic_similarity'] < 0.5)
        low_sim = sum(1 for r in successful if r['semantic_similarity'] < 0.3)

        by_source = {}
        for r in successful:
            src = r['source']
            if src not in by_source:
                by_source[src] = []
            by_source[src].append(r)

        by_query_type = {}
        for r in successful:
            qt = r.get('query_type', 'unknown')
            if qt not in by_query_type:
                by_query_type[qt] = []
            by_query_type[qt].append(r)

        report = {
            'summary': {
                'total_evaluated': len(successful),
                'total_errors': len(results) - len(successful),
                'pipeline_type': 'langgraph' if use_langgraph else 'standard',
                'avg_semantic_similarity': round(avg_semantic, 4),
                'avg_word_overlap': round(avg_word_overlap, 4),
                'avg_verse_recall': round(avg_verse_recall, 4),
                'avg_response_time_seconds': round(avg_time, 2),
                'high_similarity_count': high_sim,
                'medium_similarity_count': med_sim,
                'low_similarity_count': low_sim,
            },
            'by_source': {},
            'by_query_type': {},
            'detailed_results': results,
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

        output_path = Path("data/evaluation/srag_evaluation_report.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print("\n" + "=" * 70)
        print("EVALUATION RESULTS")
        print("=" * 70)
        print(f"\nPipeline: {'LangGraph' if use_langgraph else 'Standard'}")
        print(f"Total evaluated: {len(successful)}")
        print(f"Errors: {len(results) - len(successful)}")
        print(f"\n--- Semantic Similarity ---")
        print(f"  Average: {avg_semantic:.4f}")
        print(f"  High (>=0.5): {high_sim} ({100*high_sim/len(successful):.0f}%)")
        print(f"  Medium (0.3-0.5): {med_sim} ({100*med_sim/len(successful):.0f}%)")
        print(f"  Low (<0.3): {low_sim} ({100*low_sim/len(successful):.0f}%)")
        print(f"\n--- Word Overlap ---")
        print(f"  Average: {avg_word_overlap:.4f}")
        print(f"\n--- Verse Citation Recall ---")
        print(f"  Average: {avg_verse_recall:.4f}")
        print(f"\n--- Response Time ---")
        print(f"  Average: {avg_time:.1f}s")
        print(f"\n--- By Dataset Source ---")
        for src, stats in report['by_source'].items():
            print(f"  {src}: {stats['count']} samples, "
                  f"semantic={stats['avg_semantic_similarity']:.4f}")
        print(f"\n--- By Query Type ---")
        for qt, stats in report['by_query_type'].items():
            print(f"  {qt}: {stats['count']} samples, "
                  f"semantic={stats['avg_semantic_similarity']:.4f}")

        print(f"\nFull report saved to: {output_path}")
    else:
        print("\nNo successful results to report.")


if __name__ == "__main__":
    from src.utils.config import Config

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--langgraph", action="store_true")
    args = parser.parse_args()

    run_evaluation(sample_size=args.samples, use_langgraph=args.langgraph)
