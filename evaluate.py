"""Evaluation script for SRAG system."""

import argparse
import sys
from pathlib import Path

from src.evaluation.benchmark import (
    Benchmark,
    create_sample_test_queries,
    load_test_queries,
)
from src.utils.config import Config
from src.utils.logger import logger


def main():
    """Run SRAG evaluation."""
    parser = argparse.ArgumentParser(description="SRAG Evaluation")

    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--mode",
        choices=["benchmark", "ablation", "create-sample"],
        default="benchmark",
        help="Evaluation mode",
    )
    parser.add_argument(
        "--test-queries",
        type=str,
        help="Path to test queries JSON file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_results.json",
        help="Output filename for results",
    )

    args = parser.parse_args()
    config = Config(args.config)

    if args.mode == "create-sample":
        output_path = args.test_queries or "data/evaluation/test_queries.json"
        create_sample_test_queries(output_path)
        print(f"Created sample test queries at {output_path}")
        return

    from main import SRAGPipeline

    test_queries_path = args.test_queries or config.get(
        "evaluation.test_queries_file", "data/evaluation/test_queries.json"
    )
    test_queries = load_test_queries(test_queries_path)

    if not test_queries:
        print("No test queries found. Run with --mode create-sample first.")
        sys.exit(1)

    pipeline = SRAGPipeline(config)

    try:
        pipeline.preprocess()
        pipeline.build_indices()

        try:
            pipeline._get_graph_retriever()
        except Exception as e:
            logger.warning(f"Graph connection failed: {e}")

        benchmark = Benchmark(config)

        if args.mode == "benchmark":
            results = [benchmark.run_benchmark(pipeline, test_queries)]
        elif args.mode == "ablation":
            results = benchmark.run_ablation(pipeline, test_queries)

        benchmark.print_results(results)
        benchmark.save_results(results, args.output)

    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
