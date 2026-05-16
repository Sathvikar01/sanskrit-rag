"""Benchmark and ablation study framework for SRAG."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.evaluation.metrics import compute_all_metrics
from src.utils.config import Config
from src.utils.logger import logger


@dataclass
class TestQuery:
    """A test query with ground truth."""

    query_id: str
    query: str
    relevant_verse_refs: list[str] = field(default_factory=list)
    relevant_chunk_ids: list[str] = field(default_factory=list)
    concepts: list[str] = field(default_factory=list)
    difficulty: str = "medium"


@dataclass
class BenchmarkResult:
    """Result of a benchmark run."""

    system_name: str
    metrics: dict[str, float]
    per_query_results: list[dict] = field(default_factory=list)
    config: dict = field(default_factory=dict)


def load_test_queries(filepath: str | Path) -> list[TestQuery]:
    """Load test queries from JSON file."""
    filepath = Path(filepath)
    if not filepath.exists():
        logger.warning(f"Test queries file not found: {filepath}")
        return []

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    queries = []
    for item in data:
        queries.append(
            TestQuery(
                query_id=item.get("query_id", ""),
                query=item.get("query", ""),
                relevant_verse_refs=item.get("relevant_verse_refs", []),
                relevant_chunk_ids=item.get("relevant_chunk_ids", []),
                concepts=item.get("concepts", []),
                difficulty=item.get("difficulty", "medium"),
            )
        )

    logger.info(f"Loaded {len(queries)} test queries")
    return queries


def save_test_queries(queries: list[TestQuery], filepath: str | Path):
    """Save test queries to JSON file."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    data = []
    for q in queries:
        data.append(
            {
                "query_id": q.query_id,
                "query": q.query,
                "relevant_verse_refs": q.relevant_verse_refs,
                "relevant_chunk_ids": q.relevant_chunk_ids,
                "concepts": q.concepts,
                "difficulty": q.difficulty,
            }
        )

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved {len(queries)} test queries to {filepath}")


def verse_ref_to_chunk_ids(ref: str, chunk_types: list[str] = None) -> list[str]:
    """Convert a verse reference to possible chunk IDs.

    Args:
        ref: Verse reference like "BhG 1.1".
        chunk_types: Types of chunks to include.

    Returns:
        List of possible chunk IDs.
    """
    if chunk_types is None:
        chunk_types = ["verse", "sridhara", "visvanatha", "baladeva"]

    ref_id = ref.replace(" ", "_")
    return [f"{ref_id}_{ct}" for ct in chunk_types]


class Benchmark:
    """Benchmark runner for SRAG system."""

    def __init__(self, config: Config = None):
        if config is None:
            config = Config()

        self.config = config
        self.results_dir = Path(config.get("data.evaluation_dir", "data/evaluation"))
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def run_benchmark(
        self,
        pipeline,
        test_queries: list[TestQuery],
        system_name: str = "srag_full",
        top_k: int = 50,
    ) -> BenchmarkResult:
        """Run benchmark evaluation.

        Args:
            pipeline: SRAGPipeline instance.
            test_queries: List of test queries with ground truth.
            system_name: Name of the system being evaluated.
            top_k: Number of retrieval candidates.

        Returns:
            BenchmarkResult with metrics.
        """
        logger.info(f"Running benchmark: {system_name} ({len(test_queries)} queries)")

        ranked_lists = []
        relevant_sets = []
        per_query = []

        for tq in test_queries:
            try:
                processed = pipeline.query_processor.process_query_local(tq.query)

                candidates = pipeline.retrieve(
                    processed.query_iast,
                    processed.query_devanagari,
                    processed.concepts,
                    top_k=top_k,
                )

                reranked = pipeline.reranker.rerank(
                    query_iast=processed.query_iast,
                    concepts=processed.concepts,
                    candidates=candidates,
                    all_chunks=pipeline.chunks,
                    chunk_map=pipeline.chunk_map,
                )

                ranked_ids = [r["chunk_id"] for r in reranked]
                relevant_ids = set()
                for ref in tq.relevant_verse_refs:
                    relevant_ids.update(verse_ref_to_chunk_ids(ref))

                ranked_lists.append(ranked_ids)
                relevant_sets.append(relevant_ids)

                per_query.append(
                    {
                        "query_id": tq.query_id,
                        "query": tq.query,
                        "ranked_ids": ranked_ids[:10],
                        "relevant_ids": list(relevant_ids),
                        "hits": len(set(ranked_ids[:10]).intersection(relevant_ids)),
                    }
                )

            except Exception as e:
                logger.error(f"Error processing query {tq.query_id}: {e}")
                ranked_lists.append([])
                relevant_sets.append(set())

        metrics = compute_all_metrics(ranked_lists, relevant_sets)

        result = BenchmarkResult(
            system_name=system_name,
            metrics=metrics,
            per_query_results=per_query,
        )

        logger.info(f"Benchmark complete: {metrics}")
        return result

    def run_ablation(
        self,
        pipeline,
        test_queries: list[TestQuery],
    ) -> list[BenchmarkResult]:
        """Run ablation study comparing different system configurations.

        Args:
            pipeline: SRAGPipeline instance.
            test_queries: List of test queries.

        Returns:
            List of BenchmarkResult for each configuration.
        """
        systems = [
            ("vector_only", {"use_vector": True, "use_graph": False, "use_bm25": False}),
            ("vector_bm25", {"use_vector": True, "use_graph": False, "use_bm25": True}),
            ("vector_graph", {"use_vector": True, "use_graph": True, "use_bm25": False}),
            ("full_hybrid", {"use_vector": True, "use_graph": True, "use_bm25": True}),
        ]

        results = []
        for system_name, config in systems:
            logger.info(f"Ablation: Running {system_name}")
            result = self.run_benchmark(pipeline, test_queries, system_name)
            result.config = config
            results.append(result)

        return results

    def save_results(self, results: list[BenchmarkResult], filename: str = "benchmark_results.json"):
        """Save benchmark results to JSON."""
        filepath = self.results_dir / filename

        data = []
        for r in results:
            data.append(
                {
                    "system_name": r.system_name,
                    "metrics": r.metrics,
                    "config": r.config,
                    "num_queries": len(r.per_query_results),
                }
            )

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved benchmark results to {filepath}")

    def print_results(self, results: list[BenchmarkResult]):
        """Print benchmark results in a formatted table."""
        print("\n" + "=" * 80)
        print("BENCHMARK RESULTS")
        print("=" * 80)

        metric_names = ["mrr", "ndcg@5", "ndcg@10", "recall@5", "recall@10"]

        header = f"{'System':<20}"
        for metric in metric_names:
            header += f"{metric:>12}"
        print(header)
        print("-" * 80)

        for result in results:
            row = f"{result.system_name:<20}"
            for metric in metric_names:
                value = result.metrics.get(metric, 0.0)
                row += f"{value:>12.4f}"
            print(row)

        print("=" * 80)


# Sample test queries for Bhagavad Gita
SAMPLE_TEST_QUERIES = [
    TestQuery(
        query_id="q001",
        query="What is the concept of dharma in the Bhagavad Gita?",
        relevant_verse_refs=["BhG 1.1", "BhG 2.31", "BhG 3.35", "BhG 4.7", "BhG 18.47"],
        concepts=["dharma"],
        difficulty="easy",
    ),
    TestQuery(
        query_id="q002",
        query="What does Krishna say about karma yoga?",
        relevant_verse_refs=["BhG 3.3", "BhG 3.19", "BhG 4.20", "BhG 2.47"],
        concepts=["karma", "yoga"],
        difficulty="easy",
    ),
    TestQuery(
        query_id="q003",
        query="Explain the nature of the self (atman) as described in the Gita.",
        relevant_verse_refs=["BhG 2.17", "BhG 2.19", "BhG 2.20", "BhG 2.22", "BhG 13.31"],
        concepts=["atman"],
        difficulty="medium",
    ),
    TestQuery(
        query_id="q004",
        query="What is bhakti yoga according to the Bhagavad Gita?",
        relevant_verse_refs=["BhG 9.22", "BhG 12.6", "BhG 12.7", "BhG 18.65", "BhG 18.66"],
        concepts=["bhakti", "yoga"],
        difficulty="easy",
    ),
    TestQuery(
        query_id="q005",
        query="What is the meaning of sthitaprajna in the Gita?",
        relevant_verse_refs=["BhG 2.54", "BhG 2.55", "BhG 2.56", "BhG 2.57", "BhG 2.58"],
        concepts=["sthitaprajna", "jnana"],
        difficulty="medium",
    ),
    TestQuery(
        query_id="q006",
        query="Describe the three gunas and their effects.",
        relevant_verse_refs=["BhG 14.5", "BhG 14.6", "BhG 14.7", "BhG 14.8", "BhG 14.9", "BhG 14.10"],
        concepts=["gunas", "prakriti"],
        difficulty="medium",
    ),
    TestQuery(
        query_id="q007",
        query="What is the significance of surrender (prapatti) in the Gita?",
        relevant_verse_refs=["BhG 18.62", "BhG 18.65", "BhG 18.66", "BhG 9.22"],
        concepts=["prapatti", "bhakti"],
        difficulty="hard",
    ),
    TestQuery(
        query_id="q008",
        query="How does the Gita describe the cycle of samsara?",
        relevant_verse_refs=["BhG 2.22", "BhG 8.16", "BhG 9.21", "BhG 15.7"],
        concepts=["samsara", "moksha"],
        difficulty="medium",
    ),
    TestQuery(
        query_id="q009",
        query="What is the teaching about desireless action (nishkama karma)?",
        relevant_verse_refs=["BhG 2.47", "BhG 3.19", "BhG 4.20"],
        concepts=["nishkamakarma", "karma"],
        difficulty="easy",
    ),
    TestQuery(
        query_id="q010",
        query="Explain the concept of maya in the Bhagavad Gita.",
        relevant_verse_refs=["BhG 7.14", "BhG 7.15", "BhG 7.25", "BhG 18.61"],
        concepts=["maya", "prakriti"],
        difficulty="hard",
    ),
]


def create_sample_test_queries(filepath: str | Path):
    """Create sample test queries file."""
    save_test_queries(SAMPLE_TEST_QUERIES, filepath)
