# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
import statistics
from typing import Any

from llama_stack_api import AggregationFunctionType, ScoringResultRow


def aggregate_accuracy(scoring_results: list[ScoringResultRow]) -> dict[str, Any]:
    """Compute accuracy metrics from scoring results.

    Args:
        scoring_results: list of scoring result rows with score values

    Returns:
        Dictionary with accuracy, num_correct, and num_total
    """
    num_correct = sum(result["score"] for result in scoring_results)
    avg_score = num_correct / len(scoring_results)

    return {
        "accuracy": avg_score,
        "num_correct": num_correct,
        "num_total": len(scoring_results),
    }


def aggregate_average(scoring_results: list[ScoringResultRow]) -> dict[str, Any]:
    """Compute the arithmetic average of non-null scores.

    Args:
        scoring_results: list of scoring result rows with score values

    Returns:
        Dictionary with average score
    """
    return {
        "average": sum(result["score"] for result in scoring_results if result["score"] is not None)
        / len([_ for _ in scoring_results if _["score"] is not None]),
    }


def aggregate_weighted_average(scoring_results: list[ScoringResultRow]) -> dict[str, Any]:
    """Compute the weighted average of non-null scores.

    Args:
        scoring_results: list of scoring result rows with score and weight values

    Returns:
        Dictionary with weighted_average score
    """
    return {
        "weighted_average": sum(
            result["score"] * result["weight"]
            for result in scoring_results
            if result["score"] is not None and result["weight"] is not None
        )
        / sum(result["weight"] for result in scoring_results if result["weight"] is not None),
    }


def aggregate_categorical_count(
    scoring_results: list[ScoringResultRow],
) -> dict[str, Any]:
    """Count occurrences of each unique score category.

    Args:
        scoring_results: list of scoring result rows with score values

    Returns:
        Dictionary with categorical_count mapping each category to its count
    """
    scores = [str(r["score"]) for r in scoring_results]
    unique_scores = sorted(set(scores))
    return {"categorical_count": {s: scores.count(s) for s in unique_scores}}


def aggregate_median(scoring_results: list[ScoringResultRow]) -> dict[str, Any]:
    """Compute the median of non-null scores.

    Args:
        scoring_results: list of scoring result rows with score values

    Returns:
        Dictionary with median score value
    """
    scores = [r["score"] for r in scoring_results if r["score"] is not None]
    median = statistics.median(scores) if scores else None
    return {"median": median}


# TODO: decide whether we want to make aggregation functions as a registerable resource
AGGREGATION_FUNCTIONS = {
    AggregationFunctionType.accuracy: aggregate_accuracy,
    AggregationFunctionType.average: aggregate_average,
    AggregationFunctionType.weighted_average: aggregate_weighted_average,
    AggregationFunctionType.categorical_count: aggregate_categorical_count,
    AggregationFunctionType.median: aggregate_median,
}


def aggregate_metrics(
    scoring_results: list[ScoringResultRow], metrics: list[AggregationFunctionType]
) -> dict[str, Any]:
    """Aggregate scoring results using the specified metric functions.

    Args:
        scoring_results: list of scoring result rows
        metrics: list of aggregation function types to apply

    Returns:
        Dictionary mapping each metric to its aggregated result
    """
    agg_results = {}
    for metric in metrics:
        if metric not in AGGREGATION_FUNCTIONS:
            raise ValueError(f"Aggregation function {metric} not found")
        agg_fn = AGGREGATION_FUNCTIONS[metric]
        agg_results[metric] = agg_fn(scoring_results)
    return agg_results
