"""Evaluation metrics for SRAG retrieval system."""

import math



def mean_reciprocal_rank(ranked_lists: list[list[str]], relevant_sets: list[set[str]]) -> float:
    """Compute Mean Reciprocal Rank (MRR).

    Args:
        ranked_lists: List of ranked result lists (chunk_ids).
        relevant_sets: List of sets of relevant chunk_ids.

    Returns:
        MRR score.
    """
    if not ranked_lists:
        return 0.0

    rr_sum = 0.0
    for ranked, relevant in zip(ranked_lists, relevant_sets):
        for rank, item in enumerate(ranked, 1):
            if item in relevant:
                rr_sum += 1.0 / rank
                break

    return rr_sum / len(ranked_lists)


def ndcg_at_k(
    ranked_lists: list[list[str]],
    relevant_sets: list[set[str]],
    k: int = 10,
) -> float:
    """Compute Normalized Discounted Cumulative Gain (NDCG@k).

    Args:
        ranked_lists: List of ranked result lists.
        relevant_sets: List of sets of relevant chunk_ids.
        k: Cutoff rank.

    Returns:
        NDCG@k score.
    """
    if not ranked_lists:
        return 0.0

    total_ndcg = 0.0
    for ranked, relevant in zip(ranked_lists, relevant_sets):
        dcg = 0.0
        for i, item in enumerate(ranked[:k]):
            if item in relevant:
                dcg += 1.0 / math.log2(i + 2)

        ideal_dcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant), k)))

        if ideal_dcg > 0:
            total_ndcg += dcg / ideal_dcg

    return total_ndcg / len(ranked_lists)


def recall_at_k(
    ranked_lists: list[list[str]],
    relevant_sets: list[set[str]],
    k: int = 10,
) -> float:
    """Compute Recall@k.

    Args:
        ranked_lists: List of ranked result lists.
        relevant_sets: List of sets of relevant chunk_ids.
        k: Cutoff rank.

    Returns:
        Recall@k score.
    """
    if not ranked_lists:
        return 0.0

    total_recall = 0.0
    for ranked, relevant in zip(ranked_lists, relevant_sets):
        if not relevant:
            continue

        retrieved_at_k = set(ranked[:k])
        hits = len(retrieved_at_k.intersection(relevant))
        total_recall += hits / len(relevant)

    return total_recall / len(ranked_lists)


def precision_at_k(
    ranked_lists: list[list[str]],
    relevant_sets: list[set[str]],
    k: int = 10,
) -> float:
    """Compute Precision@k.

    Args:
        ranked_lists: List of ranked result lists.
        relevant_sets: List of sets of relevant chunk_ids.
        k: Cutoff rank.

    Returns:
        Precision@k score.
    """
    if not ranked_lists:
        return 0.0

    total_precision = 0.0
    for ranked, relevant in zip(ranked_lists, relevant_sets):
        retrieved_at_k = set(ranked[:k])
        if retrieved_at_k:
            hits = len(retrieved_at_k.intersection(relevant))
            total_precision += hits / len(retrieved_at_k)

    return total_precision / len(ranked_lists)


def mean_average_precision(
    ranked_lists: list[list[str]],
    relevant_sets: list[set[str]],
) -> float:
    """Compute Mean Average Precision (MAP).

    Args:
        ranked_lists: List of ranked result lists.
        relevant_sets: List of sets of relevant chunk_ids.

    Returns:
        MAP score.
    """
    if not ranked_lists:
        return 0.0

    total_ap = 0.0
    for ranked, relevant in zip(ranked_lists, relevant_sets):
        if not relevant:
            continue

        hits = 0
        sum_precision = 0.0
        for rank, item in enumerate(ranked, 1):
            if item in relevant:
                hits += 1
                sum_precision += hits / rank

        if hits > 0:
            total_ap += sum_precision / len(relevant)

    return total_ap / len(ranked_lists)


def compute_all_metrics(
    ranked_lists: list[list[str]],
    relevant_sets: list[set[str]],
    ks: list[int] = None,
) -> dict[str, float]:
    """Compute all evaluation metrics.

    Args:
        ranked_lists: List of ranked result lists.
        relevant_sets: List of sets of relevant chunk_ids.
        ks: List of k values for cutoff metrics.

    Returns:
        Dictionary of metric names to scores.
    """
    if ks is None:
        ks = [5, 10]

    metrics = {
        "mrr": mean_reciprocal_rank(ranked_lists, relevant_sets),
        "map": mean_average_precision(ranked_lists, relevant_sets),
    }

    for k in ks:
        metrics[f"ndcg@{k}"] = ndcg_at_k(ranked_lists, relevant_sets, k)
        metrics[f"recall@{k}"] = recall_at_k(ranked_lists, relevant_sets, k)
        metrics[f"precision@{k}"] = precision_at_k(ranked_lists, relevant_sets, k)

    return metrics
