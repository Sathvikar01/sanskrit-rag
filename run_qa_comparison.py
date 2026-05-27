"""Run Q&A comparison test against the SansRAG pipeline.

Takes the 49 reference Q&A pairs from the Bhagavad Gita Chapter 1 PDF,
runs each question through the pipeline, and compares semantic relevance.
"""
import sys
import io
import argparse
import json
import time
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from config.settings import L1_REG_LAMBDA, L2_REG_LAMBDA, RRF_TOP_K
from src.embedding_client import NVIDIAEmbeddingClient
from src.gemini_client import GeminiClient
from src.qdrant_manager import QdrantManager
from src.neo4j_manager import Neo4jManager
from src.retriever import RegularizedRetriever
from src.answer_generator import AnswerGenerator, AnswerResult
from src.golden_qa import load_golden_chapter1_qa, retrieval_metrics
from src.verse_db import EXPECTED_BHAGAVAD_GITA_VERSE_COUNT, VerseDatabase, ingest_xml_to_sqlite

REFERENCE_QA = [
    {"question": "Why Lord Kṛṣṇa name is Yashoda nandana? BG 1.15", "reference_answer": "because He awarded His childhood pastimes to Yaśodā at Vṛndāvana"},
    {"question": "Why Lord Kṛṣṇa name is Pārtha-sārathi? BG 1.15", "reference_answer": "because He worked as charioteer of His friend Arjuna"},
    {"question": "What is the conchshell name of Yudhiṣṭhira? BG 1.16-18", "reference_answer": "Ananta-vijaya"},
    {"question": "What is Nakula and Sahadeva conchshell name? BG 1.16-18", "reference_answer": "Sughoṣa and Maṇipuṣpaka"},
    {"question": "how did the different conchshells sound was like? BG 1.19", "reference_answer": "Vibrating both in the sky and on the earth"},
    {"question": "How did sons of Dhrtarastra felt after listening to conchshells sound of Pandavas side? BG 1.19", "reference_answer": "hearts of the sons of Dhṛtarāṣṭra were shattered by the sounds vibrated by the Pāṇḍavas' party"},
    {"question": "One who takes shelter of the Supreme Lord has nothing to fear, even in the midst of the greatest calamity. BG 1.19", "reference_answer": "TRUE"},
    {"question": "What was Arjuna chariot bearing the flag marked with? BG 1.20", "reference_answer": "Hanumān"},
    {"question": "Who cooperated with Lord Rāma in the battle between Rāma and Rāvaṇa? BG 1.20", "reference_answer": "Hanuman"},
    {"question": "Who is known as the goddess of fortune? BG 1.20", "reference_answer": "Sītā"},
    {"question": "What did Arjuna told Lord Kṛṣṇa in the begining of battlefield? BG 1.21-22", "reference_answer": "O infallible one, please draw my chariot between the two armies so that I may see those present here, who desire to fight"},
    {"question": "What forced Arjuna to come onto the battleﬁeld? BG 1.21-22", "reference_answer": "the obstinacy of Duryodhana, who was never agreeable to any peaceful negotiation"},
    {"question": "What does Arjuna Think about Dhṛtarāṣṭra and his sons? BG 1.23", "reference_answer": "the evil-minded"},
    {"question": "Why Arjuna is referred to as Guḍākeśa? BG 1.24", "reference_answer": "one who conquers sleep is called guḍākeśa"},
    {"question": "Whom Arjuna could see in the battle field? BG 1.26", "reference_answer": "the armies of both parties, his fathers, grandfathers, teachers, maternal uncles, brothers, sons, grandsons, friends, and also his fathers-in-law and well-wishers."},
    {"question": "What happened to Arjuna when he saw all these different grades of friends and relatives? BG 1.27", "reference_answer": "he became overwhelmed with compassion"},
    {"question": "What was Arjuna`s reaction after seeing friends and relatives in the battle field? BG 1.28", "reference_answer": "Arjuna's bodily limbs quivering and his mouth drying up, but he was also crying out of compassion"},
    {"question": "What is Arjuna bow name? BG 1.29", "reference_answer": "Gāṇḍīva"},
    {"question": "Why Arjuna`s body trembling, his hair is standing on end, his bow Gāṇḍīva is slipping from his hand, and his skin is burning? BG 1.29", "reference_answer": "All these are due to a material conception of life."},
    {"question": "what is the significance of name keshi for Kṛṣṇa? BG 1.30", "reference_answer": "He is the killer of the Keśī demon"},
    {"question": "Excessive attachment for material things puts a man in such fearfulness and loss of mental equilibrium take place in persons who are too affected by material conditions. BG 1.30", "reference_answer": "TRUE"},
    {"question": "What is The words nimittāni viparītāni meaning? BG 1.30", "reference_answer": "When a man sees only frustration in his expectations, he thinks, \"Why am I here?"},
    {"question": "Where one's real self-interest lies in ? BG 1.30", "reference_answer": "One's real self-interest lies in Viṣṇu, or Kṛṣṇa"},
    {"question": "What is ksatriya`s occupation? BG 1.31", "reference_answer": "Fighting in the battlefield for kingdom"},
    {"question": "Arjuna indicates that Krsna should understand what will satisfy Arjuna`s sense. BG 1.32-35", "reference_answer": "TRUE"},
    {"question": "Lord can excuse a person on his own account. But he excuse no one who has done harm to his devotee. BG 1.32-35", "reference_answer": "TRUE"},
    {"question": "According to vedic injunctions how many kind of aggressors are there? BG 1.36", "reference_answer": "C. 6"},
    {"question": "Arjuna`s character was saintly by nature. BG 1.36", "reference_answer": "TRUE"},
    {"question": "Whom Arjuna is referred as husband of the goddess of fortune? BG. 1.36", "reference_answer": "Krsna"},
    {"question": "A ksatriya is not suppose to refuse to battle or gamble when he is so invited by rival party. BG 1.37-38", "reference_answer": "TRUE"},
    {"question": "Who is responsible to purifying processes in the family? BG. 1.39", "reference_answer": "The elder members"},
    {"question": "Who would require protection from elder member of the family according to varnasrama? BG 1.40", "reference_answer": "Both Children and women"},
    {"question": "According to Chanakya pandita, women are not generally not very intelligent. BG 1.40", "reference_answer": "TRUE"},
    {"question": "According to fruitive activity there is need to offer periodical food and water to the forefathers of the family. BG 1.41", "reference_answer": "TRUE"},
    {"question": "What kind of food can deliver one from all kinds of sinful reactions? BG 1.41", "reference_answer": "Eating the remnents of food offerd to Vishnu"},
    {"question": "How sanatana dharma or varnasrama dharma designed as? BG 1.42", "reference_answer": "To enable the human being to attain his ultimate salvation"},
    {"question": "What kind of leaders are called blind according to sanatana dharma? BG 1.42", "reference_answer": "People forget the aim of life as vishnu. Such people are called blind"},
    {"question": "What kind of system in Varnasrama intution? BG 1.43", "reference_answer": "by which before death one has to under go the process of atonement for his sinful activities."},
    {"question": "what is prayaschitta? BG 1.43", "reference_answer": "one who is always engaged in sinful activities must utilize the process of atonement called prayaschitta."},
    {"question": "One without doing prayascitta will be transferred to hellish planet to under go miserable life as the result of sinful activity. BG 1.43", "reference_answer": "TRUE"},
    {"question": "Arjuna being saintly devotee of lord is always conscious of moral principle. BG 1.44", "reference_answer": "TRUE"},
    {"question": "According to ksatriya fighting principles an unarmed and unwilling foe should not be attached.", "reference_answer": "TRUE"},
    {"question": "Why Arjuna sat on the chariot keeping aside his bow and arrow? BG 1.46", "reference_answer": "Because his mind was over whelmed with grief"},
    {"question": "Why Arjuna was distressed? BG 1.46", "reference_answer": "Because Arjuna is not intend to kill his own kinsmen in the battlefield."},
    {"question": "What is royal happiness according to Arjuna? BG 1.44", "reference_answer": "Commitig sinful activities like killing own family members."},
    {"question": "Why Lord Krsna name is also known as Vasudeva? BG. 1.15", "reference_answer": "Because he appeard as the son of Vasudeva"},
    {"question": "What did Krsna said to Arjuna in front of Bhisma, Drona and all the other chieftens of Kuru? BG 1.25", "reference_answer": "Just behold Partha, all the kurus assembled here."},
    {"question": "Why Arjuna is also known as Partha? BG 1.25", "reference_answer": "The word Partha meaning the son of Prtha or Kunti."},
    {"question": "What did Sanjaya said to Dhrtarastra at the end of Bhagavad gita Chapter-1? BG 1.46", "reference_answer": "After speaking to Lord Krsna on the battle field Arjuna sat on the Chariot keeping aside his bow and arrow."},
]

REFERENCE_QA = load_golden_chapter1_qa(default=REFERENCE_QA)


class DisabledGeminiClient:
    """No-op LLM client for retrieval-only evaluation runs."""

    model_name = "retrieval-only"

    def is_available(self) -> bool:
        return False

    def translate_to_iast(self, text: str) -> str:
        return text

    def normalize_with_byt5(self, text: str) -> str:
        return text


def parse_args():
    parser = argparse.ArgumentParser(description="Run SansRAG Chapter 1 QA evaluation.")
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Evaluate labeled retrieval quality without LLM answer generation or answer similarity grading.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of questions evaluated.",
    )
    parser.add_argument(
        "--answer-mode",
        choices=["current", "structured_step"],
        default="current",
        help="Answer generation mode for non-retrieval-only runs.",
    )
    return parser.parse_args()


def compute_semantic_similarity(text1: str, text2: str, embedder: NVIDIAEmbeddingClient) -> float:
    """Compute cosine similarity between two texts using embeddings."""
    try:
        emb1 = embedder.embed_query(text1)
        emb2 = embedder.embed_query(text2)

        import numpy as np
        v1 = np.array(emb1.dense_vector)
        v2 = np.array(emb2.dense_vector)

        dot = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot / (norm1 * norm2))
    except Exception as e:
        print(f"  Embedding error: {e}")
        return 0.0


def keyword_overlap(reference: str, generated: str) -> float:
    """Compute keyword overlap ratio."""
    ref_words = set(reference.lower().split())
    gen_words = set(generated.lower().split())

    if not ref_words:
        return 0.0

    overlap = ref_words & gen_words
    return len(overlap) / len(ref_words)


def grade_match(similarity: float, keyword_score: float, is_true_false: bool, generated_answer: str = "") -> dict:
    """Grade the match quality."""
    combined = (similarity * 0.6) + (keyword_score * 0.4)

    if is_true_false:
        if "true" in generated_answer.lower() or "correct" in generated_answer.lower():
            grade = "EXACT"
            correctness = 1.0
        elif "false" in generated_answer.lower() or "incorrect" in generated_answer.lower():
            grade = "WRONG"
            correctness = 0.0
        else:
            grade = "PARTIAL" if combined > 0.4 else "WRONG"
            correctness = combined if combined > 0.4 else 0.0
    else:
        if combined >= 0.7:
            grade = "EXACT"
            correctness = combined
        elif combined >= 0.4:
            grade = "PARTIAL"
            correctness = combined
        elif combined >= 0.2:
            grade = "RELATED"
            correctness = combined
        else:
            grade = "UNRELATED"
            correctness = combined

    return {"grade": grade, "correctness": round(correctness, 4), "combined_score": round(combined, 4)}


def main():
    args = parse_args()
    qa_items = REFERENCE_QA[:args.limit] if args.limit else REFERENCE_QA

    print("=" * 70)
    print("SansRAG Q&A Comparison Test - Bhagavad Gita Chapter 1")
    if args.retrieval_only:
        print("Mode: Retrieval-only quality evaluation")
    else:
        print(f"Answer Mode: {args.answer_mode}")
    print("=" * 70)

    embedder = NVIDIAEmbeddingClient()
    gemini = DisabledGeminiClient() if args.retrieval_only else GeminiClient()
    qdrant = QdrantManager()
    neo4j = Neo4jManager()
    verse_db = VerseDatabase()
    verse_db.connect()
    verse_stats = verse_db.get_stats()
    if verse_stats["total_verses"] == 0 or verse_stats["total_verses"] < EXPECTED_BHAGAVAD_GITA_VERSE_COUNT:
        xml_path = ROOT_DIR / "dataset.xml"
        if xml_path.exists():
            print("Refreshing SQLite verse DB with range-expanded canonical verses...")
            verse_db.close()
            ingest_xml_to_sqlite(str(xml_path), verse_db.db_path)
            verse_db = VerseDatabase(verse_db.db_path)
            verse_db.connect()

    qdrant_ok = qdrant.connect()
    neo4j_ok = neo4j.connect()
    print(f"Qdrant: {'Connected' if qdrant_ok else 'Not connected'}")
    print(f"Neo4j:  {'Connected' if neo4j_ok else 'Not connected'}")

    # Pre-check Gemini quota to avoid 45s timeout per question
    if args.retrieval_only:
        print("Gemini: Disabled for retrieval-only mode")
    elif gemini.is_available():
        print("Checking Gemini quota...", end=" ")
        gemini.translate_to_iast("test")
        if gemini.is_available():
            print("Available")
        else:
            print("Quota exhausted, disabling Gemini")
    else:
        print("Gemini: Not available")

    retriever = RegularizedRetriever(
        embedding_client=embedder,
        qdrant_manager=qdrant if qdrant_ok else None,
        neo4j_manager=neo4j if neo4j_ok else None,
        l1_lambda=L1_REG_LAMBDA,
        l2_lambda=L2_REG_LAMBDA,
        adaptive=True
    )

    answer_gen = AnswerGenerator(
        gemini_client=gemini,
        retriever=retriever,
        qdrant_manager=qdrant if qdrant_ok else None,
        neo4j_manager=neo4j if neo4j_ok else None,
        verse_db=verse_db,
        top_k=RRF_TOP_K,
        answer_mode=args.answer_mode,
    )

    results = []
    total_start = time.time()

    for i, qa in enumerate(qa_items, 1):
        question = qa["question"]
        ref_answer = qa["reference_answer"]
        expected_verse_ids = qa.get("expected_verse_ids", [])
        is_tf = ref_answer.strip().upper() == "TRUE"

        print(f"\n[{i}/{len(qa_items)}] Q: {question[:60]}...")

        try:
            answer_result: AnswerResult = answer_gen.generate_answer(question, answer_mode=args.answer_mode)
            generated_answer = answer_result.answer
            latency = answer_result.latency_ms
            citations = answer_result.citations
            retrieval_info = retrieval_metrics(answer_result, expected_verse_ids)
        except Exception as e:
            generated_answer = f"Error: {e}"
            latency = 0
            citations = []
            retrieval_info = retrieval_metrics({}, expected_verse_ids)

        if args.retrieval_only:
            similarity = 0.0
            keyword_score = 0.0
            match_info = {"grade": "NOT_GRADED", "correctness": 0.0, "combined_score": 0.0}
            generated_answer = ""
        else:
            similarity = compute_semantic_similarity(ref_answer, generated_answer, embedder)
            keyword_score = keyword_overlap(ref_answer, generated_answer)
            match_info = grade_match(similarity, keyword_score, is_tf, generated_answer)

        result = {
            "question_number": i,
            "question": question,
            "reference_answer": ref_answer,
            "generated_answer": generated_answer,
            "is_true_false": is_tf,
            "semantic_similarity": round(similarity, 4),
            "keyword_overlap": round(keyword_score, 4),
            "grade": match_info["grade"],
            "correctness": match_info["correctness"],
            "combined_score": match_info["combined_score"],
            "latency_ms": round(latency, 2),
            "num_citations": len(citations),
            "retrieval_metrics": retrieval_info,
            "citations": [{"verse_id": c.get("verse_id", ""), "score": c.get("score", 0.0)} for c in citations[:3]]
        }
        results.append(result)

        print(f"  Grade: {match_info['grade']} | Score: {match_info['combined_score']:.4f} | "
              f"Similarity: {similarity:.4f} | Keywords: {keyword_score:.4f} | "
              f"Retrieval Quality: {retrieval_info.get('retrieval_quality')} | "
              f"Latency: {latency:.0f}ms")

    total_time = time.time() - total_start

    grades = {}
    for r in results:
        g = r["grade"]
        grades[g] = grades.get(g, 0) + 1

    avg_correctness = sum(r["correctness"] for r in results) / len(results)
    avg_similarity = sum(r["semantic_similarity"] for r in results) / len(results)
    avg_keyword = sum(r["keyword_overlap"] for r in results) / len(results)
    avg_latency = sum(r["latency_ms"] for r in results) / len(results)
    evaluated_results = [r for r in results if r["retrieval_metrics"].get("retrieval_quality") is not None]
    explicit_hit_rate = (
        sum(1 for r in evaluated_results if r["retrieval_metrics"]["explicit_verse_hit"]) / len(evaluated_results)
        if evaluated_results else 0.0
    )
    top_verse_hit_rate = (
        sum(1 for r in evaluated_results if r["retrieval_metrics"]["top_verse_hit"]) / len(evaluated_results)
        if evaluated_results else 0.0
    )
    commentary_hit_rate = (
        sum(1 for r in evaluated_results if r["retrieval_metrics"]["commentary_hit"]) / len(evaluated_results)
        if evaluated_results else 0.0
    )
    retrieval_scores = [
        r["retrieval_metrics"]["retrieval_quality"]
        for r in results
        if r["retrieval_metrics"].get("retrieval_quality") is not None
    ]
    coverage_scores = [
        r["retrieval_metrics"]["expected_coverage"]
        for r in results
        if r["retrieval_metrics"].get("expected_coverage") is not None
    ]
    reciprocal_ranks = [
        r["retrieval_metrics"]["reciprocal_rank"]
        for r in results
        if r["retrieval_metrics"].get("retrieval_quality") is not None
    ]
    semantic_retrieval_quality = sum(retrieval_scores) / len(retrieval_scores) if retrieval_scores else 0.0
    average_expected_coverage = sum(coverage_scores) / len(coverage_scores) if coverage_scores else 0.0
    mean_reciprocal_rank = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0

    summary = {
        "test_name": "Bhagavad Gita Chapter 1 Q&A Comparison",
        "mode": "retrieval_only" if args.retrieval_only else "qa_comparison",
        "answer_mode": "retrieval_only" if args.retrieval_only else args.answer_mode,
        "total_questions": len(qa_items),
        "total_time_seconds": round(total_time, 2),
        "grade_distribution": grades,
        "average_correctness": round(avg_correctness, 4),
        "average_semantic_similarity": round(avg_similarity, 4),
        "average_keyword_overlap": round(avg_keyword, 4),
        "average_latency_ms": round(avg_latency, 2),
        "explicit_verse_hit_rate": round(explicit_hit_rate, 4),
        "top_verse_hit_rate": round(top_verse_hit_rate, 4),
        "commentary_hit_rate": round(commentary_hit_rate, 4),
        "semantic_retrieval_quality": round(semantic_retrieval_quality, 4),
        "semantic_retrieval_quality_percentage": round(semantic_retrieval_quality * 100, 2),
        "average_expected_coverage": round(average_expected_coverage, 4),
        "mean_reciprocal_rank": round(mean_reciprocal_rank, 4),
        "retrieval_evaluated_questions": len(retrieval_scores),
        "accuracy_percentage": round((grades.get("EXACT", 0) / len(qa_items)) * 100, 2),
        "partial_or_better_percentage": round(
            ((grades.get("EXACT", 0) + grades.get("PARTIAL", 0)) / len(qa_items)) * 100, 2
        ),
        "related_or_better_percentage": round(
            ((grades.get("EXACT", 0) + grades.get("PARTIAL", 0) + grades.get("RELATED", 0)) / len(qa_items)) * 100, 2
        )
    }

    output = {
        "summary": summary,
        "results": results
    }

    if args.retrieval_only:
        output_name = "qa_retrieval_quality_chapter1.json"
    elif args.answer_mode == "structured_step":
        output_name = "qa_comparison_chapter1_structured_step.json"
    else:
        output_name = "qa_comparison_chapter1.json"
    output_path = ROOT_DIR / "results" / output_name
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total Questions: {len(qa_items)}")
    print(f"Total Time: {total_time:.2f}s")
    print(f"\nGrade Distribution:")
    for g in ["EXACT", "PARTIAL", "RELATED", "UNRELATED", "WRONG", "NOT_GRADED"]:
        if g in grades:
            print(f"  {g}: {grades[g]} ({grades[g]/len(qa_items)*100:.1f}%)")
    print(f"\nAverage Correctness: {avg_correctness:.4f}")
    print(f"Average Semantic Similarity: {avg_similarity:.4f}")
    print(f"Average Keyword Overlap: {avg_keyword:.4f}")
    print(f"Average Latency: {avg_latency:.2f}ms")
    print(f"Explicit Verse Hit Rate: {explicit_hit_rate:.2%}")
    print(f"Top Verse Hit Rate: {top_verse_hit_rate:.2%}")
    print(f"Commentary Hit Rate: {commentary_hit_rate:.2%}")
    print(f"Semantic Retrieval Quality: {semantic_retrieval_quality:.2%}")
    print(f"Expected Coverage: {average_expected_coverage:.2%}")
    print(f"MRR: {mean_reciprocal_rank:.4f}")
    print(f"\nAccuracy (EXACT): {summary['accuracy_percentage']:.2f}%")
    print(f"Partial or Better: {summary['partial_or_better_percentage']:.2f}%")
    print(f"Related or Better: {summary['related_or_better_percentage']:.2f}%")
    print(f"\nResults saved to: {output_path}")

    qdrant.disconnect()
    neo4j.disconnect()
    verse_db.close()


if __name__ == "__main__":
    main()
