# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Doc2Dial benchmark — document-grounded dialogue evaluation."""

from __future__ import annotations

import json
import logging  # allow-direct-logging

from lib.ingest import ingest_corpus
from lib.metrics import answer_metrics
from lib.query import rag_conversation
from lib.utils import Checkpoint, IDMapping

from benchmarks.base import BenchmarkRunner

logger = logging.getLogger("rag-benchmarks")

DEFAULT_MAX_CONVERSATIONS = 200


class Doc2DialBenchmark(BenchmarkRunner):
    """Doc2Dial document-grounded dialogue benchmark."""

    name = "doc2dial"

    def __init__(self, max_conversations: int | None = None, **kwargs):
        super().__init__(**kwargs)
        self.max_conversations = max_conversations or DEFAULT_MAX_CONVERSATIONS
        self.corpus: dict[str, dict] = {}
        self.conversations: list[dict] = []
        self.vector_store_id: str | None = None
        self.mapping: IDMapping | None = None

    def download(self) -> None:
        data_dir = self.data_dir / "doc2dial"
        doc_path = data_dir / "doc2dial_doc.json"
        dial_path = data_dir / "doc2dial_dial_validation.json"
        if not doc_path.exists() or not dial_path.exists():
            raise FileNotFoundError(
                f"Doc2Dial data not found at {data_dir}. "
                "Download doc2dial_v1.0.1.zip from https://doc2dial.github.io/data.html "
                "and unzip it into the data directory."
            )
        with open(doc_path) as f:
            self._doc_data = json.load(f)
        with open(dial_path) as f:
            self._dial_data = json.load(f)
        logger.info("Loaded Doc2Dial data from local files")

    def load_data(self) -> None:
        # Extract documents across all domains
        for _domain, docs in self._doc_data["doc_data"].items():
            for doc_key, doc in docs.items():
                doc_id = doc.get("doc_id", doc_key)
                self.corpus[doc_id] = {
                    "title": doc.get("title", ""),
                    "text": doc.get("doc_text", ""),
                }

        # Extract dialogues from validation split
        all_dialogues = []
        for _domain, doc_dials in self._dial_data["dial_data"].items():
            for _doc_key, dial_list in doc_dials.items():
                for dial in dial_list:
                    all_dialogues.append(dial)

        # Build conversation list with user-agent turn pairs
        for dial in all_dialogues[: self.max_conversations]:
            dial_id = dial["dial_id"]
            raw_turns = dial.get("turns", [])

            # Pair user queries with agent responses
            turns = []
            pending_query = None
            for turn in raw_turns:
                if turn["role"] == "user":
                    pending_query = turn["utterance"]
                elif turn["role"] == "agent" and pending_query:
                    turns.append(
                        {
                            "query": pending_query,
                            "answer": turn["utterance"],
                            "doc_id": dial.get("doc_id", ""),
                        }
                    )
                    pending_query = None

            if turns:
                self.conversations.append({"dial_id": dial_id, "turns": turns})

        logger.info(
            f"Loaded Doc2Dial: {len(self.corpus)} documents, "
            f"{len(self.conversations)} conversations, "
            f"{sum(len(c['turns']) for c in self.conversations)} total turns"
        )

    def ingest(self) -> None:
        if not self.corpus:
            logger.warning("No documents to ingest")
            return

        checkpoint_path = str(self.output_dir / "checkpoint.json")
        self.vector_store_id, self.mapping = ingest_corpus(
            client=self.client,
            corpus=self.corpus,
            vector_store_name="doc2dial",
            checkpoint_path=checkpoint_path,
            resume=self.resume,
        )

    def evaluate(self) -> dict:
        if not self.vector_store_id:
            return {"error": "No vector store available"}

        all_predictions = {}
        all_ground_truths = {}
        conversations_processed = 0

        ckpt = Checkpoint(str(self.output_dir / "eval_checkpoint.json"))
        completed_dials = set(ckpt.get("completed_dialogues", []))

        for conv in self.conversations:
            dial_id = conv["dial_id"]
            if dial_id in completed_dials:
                continue

            turns_input = [
                {"query_id": f"{dial_id}_{i}", "query": t["query"]}
                for i, t in enumerate(conv["turns"])
                if t["query"]  # skip empty queries
            ]

            if not turns_input:
                continue

            try:
                results = rag_conversation(
                    client=self.client,
                    model=self.model,
                    turns=turns_input,
                    vector_store_ids=[self.vector_store_id],
                    mapping=self.mapping,
                    max_num_results=10,
                    search_mode=self.search_mode,
                )
            except Exception as e:
                logger.error(f"Conversation {dial_id} failed: {e}")
                continue

            valid_turns = [t for t in conv["turns"] if t["query"]]
            for i, (turn, result) in enumerate(zip(valid_turns, results, strict=False)):
                qid = f"{dial_id}_{i}"
                all_predictions[qid] = result["answer"]
                all_ground_truths[qid] = turn["answer"]

            conversations_processed += 1
            completed_dials.add(dial_id)
            ckpt.set("completed_dialogues", list(completed_dials))

            if conversations_processed % 20 == 0:
                logger.info(f"Processed {conversations_processed}/{len(self.conversations)} dialogues")

        # Compute metrics
        metrics = answer_metrics(all_predictions, all_ground_truths)
        metrics["dataset"] = "doc2dial"
        metrics["num_conversations"] = conversations_processed
        metrics["num_documents"] = len(self.corpus)
        metrics["search_mode"] = self.search_mode or "default"

        # Save per-query results
        per_query_path = self.output_dir / "per_query_results.json"
        per_query_path.write_text(
            json.dumps(
                {
                    qid: {"prediction": all_predictions.get(qid, ""), "ground_truth": all_ground_truths.get(qid, "")}
                    for qid in all_predictions
                },
                indent=2,
            )
        )

        return metrics
