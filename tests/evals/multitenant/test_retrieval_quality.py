# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Retrieval quality evaluation under tenant isolation gating.

Measures whether applying metadata filters for tenant isolation degrades
retrieval quality for the authorized tenant. Key metrics:

  - **Recall@k**: fraction of known-relevant authorized docs retrieved.
  - **Precision@k**: fraction of retrieved docs that are relevant AND authorized.
  - **Mean Reciprocal Rank (MRR)**: average reciprocal rank of the first relevant hit.

Expected results:
  - Gated retrieval maintains the same recall on authorized documents as an
    ideal per-tenant index (no quality degradation from sharing infrastructure).
  - Precision improves under gating because unauthorized irrelevant results
    are filtered out.

Run::

    uv run pytest tests/evals/multitenant/test_retrieval_quality.py -v
"""

import numpy as np
import pytest

from .conftest import (
    QUERY_EMBEDDINGS,
    TOPICS,
    matches_filters,
)

TOP_K = 5
SCORE_THRESHOLD = 0.0


def _recall_at_k(retrieved_chunks, relevant_doc_ids: set) -> float:
    """Fraction of relevant documents that appear in the retrieved set."""
    if not relevant_doc_ids:
        return 0.0
    retrieved_ids = {c.metadata.get("document_id") for c in retrieved_chunks}
    hits = retrieved_ids & relevant_doc_ids
    return len(hits) / len(relevant_doc_ids)


def _precision_at_k(retrieved_chunks, relevant_doc_ids: set) -> float:
    """Fraction of retrieved documents that are relevant."""
    if not retrieved_chunks:
        return 0.0
    relevant_count = sum(1 for c in retrieved_chunks if c.metadata.get("document_id") in relevant_doc_ids)
    return relevant_count / len(retrieved_chunks)


def _mrr(retrieved_chunks, relevant_doc_ids: set) -> float:
    """Reciprocal rank of the first relevant document in results."""
    for rank, c in enumerate(retrieved_chunks, start=1):
        if c.metadata.get("document_id") in relevant_doc_ids:
            return 1.0 / rank
    return 0.0


class TestRetrievalQualityUnderGating:
    """Verify that tenant-scoped gating does not degrade retrieval quality."""

    @pytest.mark.parametrize("topic", TOPICS)
    async def test_gated_recall_matches_per_tenant_index(self, shared_vector_index, tenant_a_vector_index, topic):
        """Recall@k on authorized docs is the same whether we use a shared gated
        index or a physically separate per-tenant index."""
        query_emb = QUERY_EMBEDDINGS[topic]
        # The relevant doc for this topic from tenant-a
        relevant_ids = {f"tenant-a-{topic}"}

        # Per-tenant index (ground truth)
        per_tenant_result = await tenant_a_vector_index.query_vector(
            embedding=query_emb, k=TOP_K, score_threshold=SCORE_THRESHOLD
        )
        per_tenant_recall = _recall_at_k(per_tenant_result.chunks, relevant_ids)

        # Shared index with gating
        shared_result = await shared_vector_index.query_vector(
            embedding=query_emb, k=TOP_K, score_threshold=SCORE_THRESHOLD
        )
        tenant_filter = {"type": "eq", "key": "tenant_id", "value": "tenant-a"}
        gated_chunks = [c for c in shared_result.chunks if matches_filters(c.metadata, tenant_filter)]
        gated_recall = _recall_at_k(gated_chunks, relevant_ids)

        assert gated_recall >= per_tenant_recall, (
            f"Topic '{topic}': gated recall ({gated_recall:.2f}) should be >= "
            f"per-tenant recall ({per_tenant_recall:.2f})"
        )

    async def test_aggregate_retrieval_metrics(self, shared_vector_index, tenant_a_vector_index):
        """Aggregate recall, precision, and MRR across all topics for both configurations."""
        tenant_filter = {"type": "eq", "key": "tenant_id", "value": "tenant-a"}

        metrics = {
            "ungated": {"recall": [], "precision": [], "mrr": []},
            "chunk_gated": {"recall": [], "precision": [], "mrr": []},
            "per_tenant": {"recall": [], "precision": [], "mrr": []},
        }

        for topic in TOPICS:
            query_emb = QUERY_EMBEDDINGS[topic]
            relevant_ids = {f"tenant-a-{topic}"}

            # Ungated (shared index, no filter)
            result = await shared_vector_index.query_vector(
                embedding=query_emb, k=TOP_K, score_threshold=SCORE_THRESHOLD
            )
            metrics["ungated"]["recall"].append(_recall_at_k(result.chunks, relevant_ids))
            metrics["ungated"]["precision"].append(_precision_at_k(result.chunks, relevant_ids))
            metrics["ungated"]["mrr"].append(_mrr(result.chunks, relevant_ids))

            # Chunk-level gated
            gated = [c for c in result.chunks if matches_filters(c.metadata, tenant_filter)]
            metrics["chunk_gated"]["recall"].append(_recall_at_k(gated, relevant_ids))
            metrics["chunk_gated"]["precision"].append(_precision_at_k(gated, relevant_ids))
            metrics["chunk_gated"]["mrr"].append(_mrr(gated, relevant_ids))

            # Per-tenant index
            pt_result = await tenant_a_vector_index.query_vector(
                embedding=query_emb, k=TOP_K, score_threshold=SCORE_THRESHOLD
            )
            metrics["per_tenant"]["recall"].append(_recall_at_k(pt_result.chunks, relevant_ids))
            metrics["per_tenant"]["precision"].append(_precision_at_k(pt_result.chunks, relevant_ids))
            metrics["per_tenant"]["mrr"].append(_mrr(pt_result.chunks, relevant_ids))

        print("\n" + "=" * 70)
        print("RETRIEVAL QUALITY METRICS (querying as tenant-a)")
        print("=" * 70)
        print(f"{'Configuration':<20} {'Recall@5':>10} {'Precision@5':>12} {'MRR':>8}")
        print("-" * 70)
        for name, m in metrics.items():
            avg_recall = np.mean(m["recall"])
            avg_precision = np.mean(m["precision"])
            avg_mrr = np.mean(m["mrr"])
            print(f"{name:<20} {avg_recall:>10.4f} {avg_precision:>12.4f} {avg_mrr:>8.4f}")
            print(f"[METRIC] {name}_recall_at_5 = {avg_recall:.4f}")
            print(f"[METRIC] {name}_precision_at_5 = {avg_precision:.4f}")
            print(f"[METRIC] {name}_mrr = {avg_mrr:.4f}")
        print("=" * 70)

        # Gated recall must be >= ungated recall on authorized docs
        # (ungated may include unauthorized docs that dilute precision but
        # recall on authorized docs should not decrease)
        gated_recall = np.mean(metrics["chunk_gated"]["recall"])
        per_tenant_recall = np.mean(metrics["per_tenant"]["recall"])
        assert gated_recall >= per_tenant_recall - 0.01, (
            f"Gated recall ({gated_recall:.4f}) should not be worse than per-tenant recall ({per_tenant_recall:.4f})"
        )

        # Gated precision must be higher than ungated (fewer irrelevant results)
        gated_precision = np.mean(metrics["chunk_gated"]["precision"])
        ungated_precision = np.mean(metrics["ungated"]["precision"])
        assert gated_precision >= ungated_precision, (
            f"Gated precision ({gated_precision:.4f}) should be >= ungated precision ({ungated_precision:.4f})"
        )


class TestRetrievalQualityPerTopic:
    """Per-topic breakdown for detailed paper tables."""

    async def test_per_topic_metrics_table(self, shared_vector_index, tenant_a_vector_index):
        """Detailed per-topic metrics for both gated and per-tenant configurations."""
        tenant_filter = {"type": "eq", "key": "tenant_id", "value": "tenant-a"}

        print("\n" + "=" * 80)
        print("PER-TOPIC RETRIEVAL QUALITY (querying as tenant-a)")
        print("=" * 80)
        print(f"{'Topic':<15} {'Config':<15} {'Recall@5':>10} {'Precision@5':>12} {'MRR':>8}")
        print("-" * 80)

        for topic in TOPICS:
            query_emb = QUERY_EMBEDDINGS[topic]
            relevant_ids = {f"tenant-a-{topic}"}

            shared_result = await shared_vector_index.query_vector(
                embedding=query_emb, k=TOP_K, score_threshold=SCORE_THRESHOLD
            )
            gated = [c for c in shared_result.chunks if matches_filters(c.metadata, tenant_filter)]
            pt_result = await tenant_a_vector_index.query_vector(
                embedding=query_emb, k=TOP_K, score_threshold=SCORE_THRESHOLD
            )

            for name, chunks in [
                ("ungated", shared_result.chunks),
                ("chunk_gated", gated),
                ("per_tenant", pt_result.chunks),
            ]:
                r = _recall_at_k(chunks, relevant_ids)
                p = _precision_at_k(chunks, relevant_ids)
                m = _mrr(chunks, relevant_ids)
                print(f"{topic:<15} {name:<15} {r:>10.4f} {p:>12.4f} {m:>8.4f}")

        print("=" * 80)
