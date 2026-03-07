---
slug: building-agentic-flows-with-conversations-and-responses
title: "Building a Self-Improving Agent with Llama Stack"
authors:
  - name: Llama Stack Team
    title: Core Team
    url: https://github.com/llamastack
    image_url: https://llamastack.github.io/img/llama-stack-logo.png
tags: [agents, responses-api, conversations, prompts, tutorial]
date: 2026-03-01
---

What if your AI agent could improve itself? In this post, we build an RL-style optimization loop where an outer "optimizer agent" iteratively improves the system prompt of an inner RAG agent — using Llama Stack's own APIs as tools. The optimizer proposes prompt changes, tests them against a benchmark, scores the results with an LLM judge, and learns from its history.

This is self-referential: **Llama Stack improving Llama Stack agents**.

<!--truncate-->

## What We're Building

The system has two agents:

- **RAGAgent** (inner): A question-answering agent that searches a vector store and generates answers. Its system prompt is the thing being optimized.
- **OptimizerAgent** (outer): An RL-style agent that iteratively proposes better system prompts, evaluates them, and tracks results.

The optimizer uses the Responses API with function tools that call back into Llama Stack. The Conversations API ties it together — the optimizer's reasoning history persists across iterations, so it can learn from what it tried before.

```
┌─────────────────────────────────────────────────────┐
│  OptimizerAgent (Responses API + function tools)    │
│                                                     │
│  for each iteration:                                │
│    1. get_history()        → read past scores       │
│    2. get_prompt()         → read current prompt    │
│    3. update_prompt()      → write improved prompt  │
│    4. run_rag_test()       → test the RAG agent     │
│    5. judge_answer()       → score with LLM judge   │
│    6. log_result()         → record the score       │
│                                                     │
│  Conversation tracks reasoning across iterations    │
├─────────────────────────────────────────────────────┤
│  RAGAgent (Responses API + file_search)             │
│  Vector Store ◄── file_search                       │
└─────────────────────────────────────────────────────┘
```

## Prerequisites

- A running Llama Stack server with Ollama: `llama stack run ollama`
- Python SDK: `pip install llama-stack-client`
- Two models via Ollama: `llama3.1:8b` for the RAG agent and `gpt-oss:20b` as the judge

## The Inner Agent: RAGAgent

The RAG agent is simple — it takes a question and a system prompt, searches a vector store using `file_search`, and returns an answer. The Responses API handles the retrieval and generation in a single call:

```python
class RAGAgent:
    def __init__(self, client: LlamaStackClient, model: str, vector_store_id: str):
        self.client = client
        self.model = model
        self.vector_store_id = vector_store_id

    @classmethod
    def from_files(cls, client, model, name, file_paths):
        """Create a RAGAgent with a new vector store populated from local files.
        Also accepts embedding_model and embedding_dimension kwargs."""
        vector_store = client.vector_stores.create(name=name)
        for path in file_paths:
            file = client.files.create(file=open(path, "rb"), purpose="assistants")
            client.vector_stores.files.create(
                vector_store_id=vector_store.id, file_id=file.id
            )
            # ... poll until indexing completes ...
        return cls(client, model, vector_store.id)

    def query(self, question: str, system_prompt: str) -> str:
        response = self.client.responses.create(
            model=self.model,
            input=question,
            instructions=system_prompt,
            tools=[{"type": "file_search", "vector_store_ids": [self.vector_store_id]}],
            stream=False,
        )
        return response.output_text
```

The `from_files` classmethod handles vector store creation, file upload, and indexing. The `instructions` parameter sets the system prompt, and `file_search` tells the Responses API to search the vector store for relevant context before generating. The system prompt is what the optimizer will iterate on — how the agent uses retrieved context, whether it cites sources, how concise it is.

## The Outer Agent: OptimizerAgent

The optimizer registers function tools that the LLM can call. Each tool is a thin wrapper around a Llama Stack API call or a ledger operation. Here are the key ones:

### Prompt management tools (Prompts API)

```python
class OptimizerAgent:
    ...

    def _register_tools(self):

        @tool
        def update_prompt(
            new_prompt: Annotated[str, "The improved system prompt text"],
            current_version: Annotated[
                int, "Current version number (for optimistic locking)"
            ],
        ) -> dict:
            """Create a new version of the system prompt."""
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
```

The Prompts API handles versioning automatically — each `update` increments the version. The `current_version` parameter provides optimistic locking.

### Testing and judging tools

```python
class OptimizerAgent:
    ...

    def _register_tools(self):
        ...

        @tool
        def run_rag_test(
            question: Annotated[str, "The test question to ask the RAG agent"],
            system_prompt: Annotated[str, "The system prompt to use"],
        ) -> str:
            """Run the inner RAG agent on a test question."""
            return self.rag_agent.query(question, system_prompt)

        @tool
        def judge_answer(
            question: Annotated[str, "The original question"],
            expected: Annotated[str, "The expected answer"],
            actual: Annotated[str, "The RAG agent's actual answer"],
        ) -> dict:
            """Score a RAG answer using LLM-as-judge."""
            response = self.client.chat.completions.create(
                model=self.judge_model,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            f"Score the following answer on a scale of 0.0 to 1.0.\n\n"
                            f"Question: {question}\nExpected: {expected}\nActual: {actual}\n\n"
                            f'Respond with JSON: {{"score": <float>, "reasoning": "<explanation>"}}'
                        ),
                    }
                ],
            )
            return json.loads(response.choices[0].message.content)
```

The `run_rag_test` tool calls back into the Responses API — the optimizer agent is using Llama Stack to test Llama Stack. The `judge_answer` tool uses Chat Completions to score the output.

### Score ledger tools

The optimizer also has `log_result` and `get_history` tools backed by a `ScoreLedger` — a simple SQLite table that maps `(prompt_id, version)` to scores. The Prompts API stores prompt text and versions; the ledger just tracks scores. See the [full implementation](./prompt_optimizer_sketch.py) for details.

## The Optimization Loop

This is the core of the optimizer. Each iteration, the agent uses the Responses API with its function tools. The Conversations API preserves reasoning across iterations — the agent remembers what it tried and why:

```python
class OptimizerAgent:
    ...

    def run(self, max_iterations: int = 5):
        for iteration in range(max_iterations):
            inputs = [{"role": "user", "content": optimization_prompt}]

            # Agentic loop: keep going until the model stops calling tools
            while True:
                response = self.client.responses.create(
                    model=self.model,
                    input=inputs,
                    tools=tool_schemas,
                    conversation=self.conversation_id,
                    stream=False,
                )

                function_calls = [
                    o for o in response.output if o.type == "function_call"
                ]
                if not function_calls:
                    break  # Model is done — no more tool calls

                # Execute tool calls and feed results back
                inputs = []
                for fc in function_calls:
                    result = self._execute_tool_call(fc.name, fc.arguments)
                    inputs.append(fc)
                    inputs.append(
                        {
                            "type": "function_call_output",
                            "call_id": fc.call_id,
                            "output": result,
                        }
                    )
```

The pattern is straightforward:
1. Call `responses.create()` with function tools and a conversation ID
2. If the response contains `function_call` outputs, execute them client-side
3. Feed the results back as `function_call_output` items
4. Repeat until the model responds with text instead of tool calls
5. The `conversation` parameter means all turns are persisted — the next iteration sees the full history

## Running It

First, make sure you have a Llama Stack server running with Ollama:

```bash
ollama pull llama3.1:8b
ollama pull gpt-oss:20b
uv run --with llama-stack llama stack run ollama
```

Then set up the vector store with some documents for the RAG agent to search over, create the initial prompt, and run the optimizer:

```python
from llama_stack_client import LlamaStackClient

client = LlamaStackClient(base_url="http://localhost:8321")
MODEL = "ollama/llama3.1:8b"

# Create the inner RAG agent with a vector store from local files
rag_agent = RAGAgent.from_files(
    client,
    model=MODEL,
    name="llama-docs",
    file_paths=["llama3_model_card.txt"],
)

# Verify the RAG agent works
answer = rag_agent.query(
    "What is the max context length of Llama 3.1?",
    system_prompt="Answer based on the provided context.",
)
print(f"RAG agent says: {answer}")

# Create the initial system prompt via Prompts API
initial = client.prompts.create(
    prompt="You are a helpful assistant. Answer questions based on the provided context.",
)

# Run the optimizer
optimizer = OptimizerAgent(
    client=client,
    model=MODEL,
    judge_model="ollama/gpt-oss:20b",
    rag_agent=rag_agent,
    ledger=ScoreLedger(),
    prompt_id=initial.prompt_id,
    test_cases=[
        {
            "question": "What is the max context length of Llama 3.1?",
            "expected": "128K tokens",
        },
        {
            "question": "What languages does Llama 3.1 support?",
            "expected": "English, German, French, Italian, Portuguese, Hindi, Spanish, Thai",
        },
    ],
)
optimizer.run(max_iterations=5)

# Show the best prompt
best = optimizer.best_prompt()
print(f"Best prompt (v{best['version']}, score={best['score']:.2f}):")
print(f"  {best['prompt']}")
```

The full implementation with tool schema generation and all supporting code is available at [prompt_optimizer_sketch.py](./prompt_optimizer_sketch.py).

## How It Works Under the Hood

Server-side tools like `file_search` and `web_search` are executed automatically by the Responses API — the server handles the retrieval loop. Client-side `function` tools return tool calls for you to execute, which is what our `while True` loop handles. The `conversation` parameter persists all turns server-side, so the optimizer sees its full history without manual state management. And the Prompts API auto-increments versions on each update, with optimistic locking via `current_version` to prevent silent overwrites.

## What's Next

This pattern — using Llama Stack APIs as tools for an agent built on Llama Stack — generalizes beyond prompt optimization:

- **MCP tools** for connecting to external services (databases, APIs, code execution sandboxes)
- **Web search** alongside file_search for agents that combine local knowledge with live web results
- **Multiple RAG agents** with different vector stores, optimized in parallel

To learn more:
- [Responses API documentation](/docs/building_applications/agent)
- [Conversations API documentation](/docs/building_applications/agent)
- [Prompts API documentation](/docs/providers)
- [Vector Stores documentation](/docs/building_applications/rag)
- [Join our Discord](https://discord.gg/llama-stack)
