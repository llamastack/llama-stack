# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""QReCC benchmark — conversational QA with scoped corpus per conversation."""

from __future__ import annotations

import json
import logging  # allow-direct-logging
import random
from collections import defaultdict

import huggingface_hub
from lib.ingest import ingest_corpus
from lib.metrics import answer_metrics
from lib.query import rag_conversation
from lib.utils import Checkpoint

from benchmarks.base import BenchmarkRunner

logger = logging.getLogger("rag-benchmarks")

# Number of distractor passages to add per conversation
DISTRACTORS_PER_CONV = 500
# Max conversations to evaluate (full dataset is 14K)
DEFAULT_MAX_CONVERSATIONS = 100


class QReCCBenchmark(BenchmarkRunner):
    """QReCC conversational QA benchmark with scoped corpus."""

    name = "qrecc"

    def __init__(self, max_conversations: int | None = None, use_rewritten: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.max_conversations = max_conversations or DEFAULT_MAX_CONVERSATIONS
        self.use_rewritten = use_rewritten
        self.conversations: list[dict] = []
        self.all_passages: dict[str, str] = {}

    def download(self) -> None:
        cache_dir = str(self.data_dir / "qrecc")
        self._test_path = huggingface_hub.hf_hub_download(
            "svakulenk0/qrecc",
            "qrecc-test.json",
            repo_type="dataset",
            cache_dir=cache_dir,
        )
        logger.info("Downloaded QReCC dataset")

    def load_data(self) -> None:
        with open(self._test_path) as f:
            test_data = json.load(f)

        # Group turns by conversation_id
        conv_turns: dict[str, list] = defaultdict(list)
        for item in test_data:
            conv_id = str(item["Conversation_no"])
            conv_turns[conv_id].append(item)

        # Sort turns within each conversation by turn number
        for conv_id in conv_turns:
            conv_turns[conv_id].sort(key=lambda x: x["Turn_no"])

        # Build conversation list
        conv_ids = sorted(conv_turns.keys(), key=int)[: self.max_conversations]
        for conv_id in conv_ids:
            turns = conv_turns[conv_id]
            conv = {
                "conv_id": conv_id,
                "turns": [],
            }
            for turn in turns:
                query_key = "Rewrite" if self.use_rewritten else "Question"
                conv["turns"].append(
                    {
                        "turn_no": turn["Turn_no"],
                        "query": turn[query_key],
                        "answer": turn.get("Answer", ""),
                        "context": turn.get("Context", []) or [],
                    }
                )
            self.conversations.append(conv)

        logger.info(
            f"Loaded QReCC: {len(self.conversations)} conversations, "
            f"{sum(len(c['turns']) for c in self.conversations)} total turns, "
            f"{len(self.all_passages)} unique passages in pool"
        )

    def ingest(self) -> None:
        # Ingestion is per-conversation, handled in evaluate()
        pass

    def evaluate(self) -> dict:
        all_predictions = {}
        all_ground_truths = {}
        conversations_processed = 0

        ckpt = Checkpoint(str(self.output_dir / "eval_checkpoint.json"))
        completed_convs = set(ckpt.get("completed_conversations", []))

        for conv in self.conversations:
            conv_id = conv["conv_id"]
            if conv_id in completed_convs:
                logger.info(f"Skipping already-evaluated conversation {conv_id}")
                continue

            # Build scoped corpus: gold passages + distractors
            scoped_corpus = {}
            for turn in conv["turns"]:
                for passage in turn["gold_passages"]:
                    pid = f"conv{conv_id}_gold_{len(scoped_corpus)}"
                    scoped_corpus[pid] = {"title": "", "text": passage}

            # Add random distractor passages
            pool_ids = list(self.all_passages.keys())
            n_distractors = min(DISTRACTORS_PER_CONV, len(pool_ids))
            distractor_ids = random.sample(pool_ids, n_distractors)
            for pid in distractor_ids:
                did = f"conv{conv_id}_dist_{pid}"
                scoped_corpus[did] = {"title": "", "text": self.all_passages[pid]}

            if not scoped_corpus:
                logger.warning(f"Conversation {conv_id} has no passages, skipping")
                continue

            # Ingest scoped corpus
            checkpoint_path = str(self.output_dir / f"conv_{conv_id}_checkpoint.json")
            try:
                vs_id, mapping = ingest_corpus(
                    client=self.client,
                    corpus=scoped_corpus,
                    vector_store_name=f"qrecc-conv-{conv_id}",
                    checkpoint_path=checkpoint_path,
                    resume=self.resume,
                )
            except Exception as e:
                logger.error(f"Ingestion failed for conversation {conv_id}: {e}")
                continue

            # Run conversation turns
            turns_input = [{"query_id": f"{conv_id}_{t['turn_no']}", "query": t["query"]} for t in conv["turns"]]

            try:
                results = rag_conversation(
                    client=self.client,
                    model=self.model,
                    turns=turns_input,
                    vector_store_ids=[vs_id],
                    mapping=mapping,
                    max_num_results=10,
                    search_mode=self.search_mode,
                )
            except Exception as e:
                logger.error(f"Query failed for conversation {conv_id}: {e}")
                continue

            # Collect predictions and ground truths
            for turn, result in zip(conv["turns"], results, strict=False):
                qid = f"{conv_id}_{turn['turn_no']}"
                all_predictions[qid] = result["answer"]
                all_ground_truths[qid] = turn["answer"]

            conversations_processed += 1
            completed_convs.add(conv_id)
            ckpt.set("completed_conversations", list(completed_convs))

            # Clean up vector store to avoid accumulation
            try:
                self.client.vector_stores.delete(vs_id)
            except Exception:
                pass

            if conversations_processed % 10 == 0:
                logger.info(f"Processed {conversations_processed}/{len(self.conversations)} conversations")

        # Compute metrics
        metrics = answer_metrics(all_predictions, all_ground_truths)
        metrics["dataset"] = "qrecc"
        metrics["mode"] = "rewritten" if self.use_rewritten else "original"
        metrics["num_conversations"] = conversations_processed
        metrics["search_mode"] = self.search_mode or "default"

        # Save per-query results
        per_query_path = self.output_dir / "per_query_results.json"
        per_query_path.write_text(
            json.dumps(
                {
                    qid: {"prediction": all_predictions[qid], "ground_truth": all_ground_truths[qid]}
                    for qid in all_predictions
                },
                indent=2,
            )
        )

        return metrics
