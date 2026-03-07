# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Self-improving RAG agent using Llama Stack APIs.

An RL-style optimization loop where an outer "optimizer agent" iteratively
improves the system prompt of an inner RAG agent — using Llama Stack's own
APIs (Prompts, Responses, Conversations, Chat Completions) as tools.
"""

import inspect
import json
import sqlite3
from typing import Annotated, get_args, get_origin

from llama_stack_client import LlamaStackClient

# ---------------------------------------------------------------------------
# Tool schema generation — derives OpenAI function tool definitions from
# Python function signatures using type hints and Annotated descriptions.
# ---------------------------------------------------------------------------

PYTHON_TYPE_TO_JSON = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}

_tool_registry: dict[str, callable] = {}


def tool(fn):
    """Decorator: register a function as a tool the optimizer agent can call."""
    _tool_registry[fn.__name__] = fn
    return fn


def fn_to_tool_schema(fn) -> dict:
    """Derive an OpenAI function tool schema from a Python function."""
    sig = inspect.signature(fn)
    hints = fn.__annotations__
    properties = {}
    required = []

    for name, param in sig.parameters.items():
        hint = hints.get(name, str)

        # Extract description from Annotated[type, "description"]
        description = None
        if get_origin(hint) is Annotated:
            args = get_args(hint)
            hint = args[0]
            if len(args) > 1 and isinstance(args[1], str):
                description = args[1]

        # Handle Optional (T | None)
        is_optional = False
        if get_origin(hint) is type(int | None):  # UnionType
            inner_types = [t for t in get_args(hint) if t is not type(None)]
            if inner_types:
                hint = inner_types[0]
                is_optional = True

        prop = {"type": PYTHON_TYPE_TO_JSON.get(hint, "string")}
        if description:
            prop["description"] = description
        properties[name] = prop

        if param.default is inspect.Parameter.empty and not is_optional:
            required.append(name)

    return {
        "type": "function",
        "name": fn.__name__,
        "description": (fn.__doc__ or "").strip(),
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required,
        },
    }


# ---------------------------------------------------------------------------
# RAGAgent — the inner agent whose system prompt is being optimized.
# Uses Responses API with file_search over a vector store.
# ---------------------------------------------------------------------------


class RAGAgent:
    def __init__(self, client: LlamaStackClient, model: str, vector_store_id: str):
        self.client = client
        self.model = model
        self.vector_store_id = vector_store_id

    @classmethod
    def from_files(
        cls,
        client: LlamaStackClient,
        model: str,
        name: str,
        file_paths: list[str],
        embedding_model: str = "all-MiniLM-L6-v2",
        embedding_dimension: int = 384,
    ) -> "RAGAgent":
        """Create a RAGAgent with a new vector store populated from local files."""
        import time

        vector_store = client.vector_stores.create(
            name=name,
            embedding_model=embedding_model,
            embedding_dimension=embedding_dimension,
        )

        for path in file_paths:
            file = client.files.create(file=open(path, "rb"), purpose="assistants")
            attach = client.vector_stores.files.create(
                vector_store_id=vector_store.id, file_id=file.id,
            )
            while attach.status == "in_progress":
                time.sleep(0.5)
                attach = client.vector_stores.files.retrieve(
                    vector_store_id=vector_store.id, file_id=file.id,
                )

        return cls(client, model, vector_store.id)

    def query(self, question: str, system_prompt: str) -> str:
        """Run a RAG query: search the vector store and generate an answer."""
        response = self.client.responses.create(
            model=self.model,
            input=question,
            instructions=system_prompt,
            tools=[{"type": "file_search", "vector_store_ids": [self.vector_store_id]}],
            stream=False,
        )
        return response.output_text


# ---------------------------------------------------------------------------
# ScoreLedger — SQLite-backed record of (prompt_version, score) pairs.
# ---------------------------------------------------------------------------


class ScoreLedger:
    def __init__(self, db_path: str = "prompt_optimizer.db"):
        self.db = sqlite3.connect(db_path)
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS results (
                prompt_id TEXT,
                version INTEGER,
                score REAL,
                reasoning TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

    def log(self, prompt_id: str, version: int, score: float, reasoning: str) -> str:
        self.db.execute(
            "INSERT INTO results (prompt_id, version, score, reasoning) VALUES (?, ?, ?, ?)",
            (prompt_id, version, score, reasoning),
        )
        self.db.commit()
        return f"Logged: version={version}, score={score}"

    def history(self, prompt_id: str) -> list[dict]:
        rows = self.db.execute(
            "SELECT version, score, reasoning, timestamp FROM results WHERE prompt_id = ? ORDER BY version",
            (prompt_id,),
        ).fetchall()
        return [{"version": r[0], "score": r[1], "reasoning": r[2], "timestamp": r[3]} for r in rows]


# ---------------------------------------------------------------------------
# OptimizerAgent — the outer RL agent that iteratively improves the RAG
# agent's system prompt. Uses Responses API with function tools that call
# Llama Stack APIs and the score ledger.
# ---------------------------------------------------------------------------


class OptimizerAgent:
    def __init__(
        self,
        client: LlamaStackClient,
        model: str,
        judge_model: str,
        rag_agent: RAGAgent,
        ledger: ScoreLedger,
        prompt_id: str,
        test_cases: list[dict],
    ):
        self.client = client
        self.model = model
        self.judge_model = judge_model
        self.rag_agent = rag_agent
        self.ledger = ledger
        self.prompt_id = prompt_id
        self.test_cases = test_cases
        self.conversation_id = client.conversations.create().id
        self._register_tools()

    def _register_tools(self):
        """Register function tools that the optimizer agent can call.
        Schemas are auto-derived from type hints and docstrings."""

        @tool
        def update_prompt(
            new_prompt: Annotated[str, "The improved system prompt text"],
            current_version: Annotated[int, "Current version number (for optimistic locking)"],
        ) -> dict:
            """Create a new version of the system prompt. Returns the new version number."""
            result = self.client.prompts.update(
                self.prompt_id,
                prompt=new_prompt,
                version=current_version,
            )
            return {"prompt_id": result.prompt_id, "version": result.version}

        @tool
        def get_prompt(
            version: Annotated[int | None, "Specific version to fetch"] = None,
        ) -> dict:
            """Fetch the current system prompt text and version."""
            result = self.client.prompts.retrieve(self.prompt_id, version=version)
            return {"prompt": result.prompt, "version": result.version}

        @tool
        def run_rag_test(
            question: Annotated[str, "The test question to ask the RAG agent"],
            system_prompt: Annotated[str, "The system prompt to use"],
        ) -> str:
            """Run the inner RAG agent on a test question. Returns the agent's answer."""
            return self.rag_agent.query(question, system_prompt)

        @tool
        def judge_answer(
            question: Annotated[str, "The original question"],
            expected: Annotated[str, "The expected answer"],
            actual: Annotated[str, "The RAG agent's actual answer"],
        ) -> dict:
            """Score a RAG answer using LLM-as-judge. Returns {score, reasoning}."""
            response = self.client.chat.completions.create(
                model=self.judge_model,
                messages=[{
                    "role": "user",
                    "content": (
                        f"Score the following answer on a scale of 0.0 to 1.0.\n\n"
                        f"Question: {question}\n"
                        f"Expected answer: {expected}\n"
                        f"Actual answer: {actual}\n\n"
                        f'Respond with JSON: {{"score": <float>, "reasoning": "<brief explanation>"}}'
                    ),
                }],
            )
            return json.loads(response.choices[0].message.content)

        @tool
        def log_result(
            version: Annotated[int, "The prompt version that was tested"],
            score: Annotated[float, "The average score across test cases"],
            reasoning: Annotated[str, "Summary of why this score was achieved"],
        ) -> str:
            """Log a prompt version's score to the results ledger."""
            return self.ledger.log(self.prompt_id, version, score, reasoning)

        @tool
        def get_history() -> list[dict]:
            """Get the full optimization history: all prompt versions and their scores."""
            return self.ledger.history(self.prompt_id)

    def _execute_tool_call(self, name: str, arguments: str) -> str:
        args = json.loads(arguments)
        result = _tool_registry[name](**args)
        return json.dumps(result) if not isinstance(result, str) else result

    def run(self, max_iterations: int = 5):
        """Run the optimization loop for max_iterations rounds."""
        test_cases_str = json.dumps(self.test_cases, indent=2)

        for iteration in range(max_iterations):
            print(f"\n{'='*60}")
            print(f"Iteration {iteration + 1}/{max_iterations}")
            print(f"{'='*60}")

            prompt = (
                f"You are optimizing a RAG agent's system prompt.\n\n"
                f"Test cases the RAG agent must answer well:\n{test_cases_str}\n\n"
                f"Your workflow:\n"
                f"1. Call get_history to see past scores\n"
                f"2. Call get_prompt to read the current prompt\n"
                f"3. Propose an improved prompt and call update_prompt\n"
                f"4. For each test case, call run_rag_test, then judge_answer\n"
                f"5. Average the scores and call log_result\n\n"
                f"Focus on: using retrieved context, being concise, citing sources."
            )

            inputs = [{"role": "user", "content": prompt}]

            # Agentic loop: keep going until the model stops calling tools
            while True:
                response = self.client.responses.create(
                    model=self.model,
                    input=inputs,
                    tools=[fn_to_tool_schema(fn) for fn in _tool_registry.values()],
                    conversation=self.conversation_id,
                    stream=False,
                )

                function_calls = [o for o in response.output if o.type == "function_call"]
                if not function_calls:
                    print(f"Optimizer: {response.output_text}")
                    break

                # Execute tool calls and feed results back
                inputs = []
                for fc in function_calls:
                    print(f"  Tool: {fc.name}({fc.arguments[:80]}...)")
                    result = self._execute_tool_call(fc.name, fc.arguments)
                    inputs.append(fc)
                    inputs.append({
                        "type": "function_call_output",
                        "call_id": fc.call_id,
                        "output": result,
                    })

    def best_prompt(self) -> dict:
        """Return the highest-scoring prompt from the ledger."""
        history = self.ledger.history(self.prompt_id)
        best = max(history, key=lambda h: h["score"])
        prompt = self.client.prompts.retrieve(self.prompt_id, version=best["version"])
        return {"version": best["version"], "score": best["score"], "prompt": prompt.prompt}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    client = LlamaStackClient(base_url="http://localhost:8321")

    MODEL = "ollama/llama3.1:8b"

    TEST_CASES = [
        {"question": "What is the maximum context length of Llama 3.1?", "expected": "128K tokens"},
        {"question": "What languages does Llama 3.1 support?", "expected": "English, German, French, Italian, Portuguese, Hindi, Spanish, Thai"},
    ]

    # Create the inner RAG agent with a vector store from local files
    rag_agent = RAGAgent.from_files(
        client, model=MODEL, name="llama-docs", file_paths=["llama3_model_card.txt"],
    )

    # Create the initial system prompt
    initial = client.prompts.create(
        prompt="You are a helpful assistant. Answer questions based on the provided context.",
    )

    # Create the score ledger
    ledger = ScoreLedger()

    # Create and run the optimizer
    optimizer = OptimizerAgent(
        client=client,
        model=MODEL,
        judge_model="ollama/gpt-oss:20b",
        rag_agent=rag_agent,
        ledger=ledger,
        prompt_id=initial.prompt_id,
        test_cases=TEST_CASES,
    )
    optimizer.run(max_iterations=5)

    # Show results
    print(f"\n{'='*60}")
    print("Optimization complete!")
    print(f"{'='*60}")
    for h in ledger.history(initial.prompt_id):
        print(f"  v{h['version']}: score={h['score']:.2f} — {h['reasoning']}")

    best = optimizer.best_prompt()
    print(f"\nBest prompt (v{best['version']}, score={best['score']:.2f}):")
    print(f"  {best['prompt']}")
