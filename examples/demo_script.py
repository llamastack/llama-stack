# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack_client import Agent, AgentEventLogger, RAGDocument, LlamaStackClient

# vector_db_id = "my_demo_vector_db"
vector_db_name = "my_demo_vector_db"

# Initialize client with telemetry headers
# All API calls will automatically generate traces sent to Jaeger
client = LlamaStackClient(
    base_url="http://localhost:8321",
    default_headers={
        "X-Telemetry-Service": "llama-stack-rag-demo",
        "X-Telemetry-Version": "1.0.0",
    }
)

print("=" * 80)
print("🔭 Telemetry enabled: Traces will be sent to Jaeger")
print("   View traces at: http://localhost:16686")
print("   Service name: llama-stack-rag-demo")
print("=" * 80)
print()

models = client.models.list()

# Select the first LLM from vLLM provider and first embedding model
model_id = next(m for m in models if m.model_type == "llm" and m.provider_id == "vllm").identifier
embedding_model_id = (
    em := next(m for m in models if m.model_type == "embedding")
).identifier
embedding_dimension = em.metadata["embedding_dimension"]

# ✅ FIXED: Use vector_stores.create instead of vector_dbs.register
vector_store = client.vector_stores.create(
    name=vector_db_name,
    extra_body={
        "embedding_model": embedding_model_id,
    },
)
vector_db_id = vector_store.id


# vector_db = client.vector_dbs.register(
#     vector_db_id=vector_db_id,
#     embedding_model=embedding_model_id,
#     embedding_dimension=embedding_dimension,
#     provider_id="faiss",
# )
# vector_db_id = vector_db.identifier
source = "https://www.paulgraham.com/greatwork.html"
print("rag_tool> Ingesting document:", source)
document = RAGDocument(
    document_id="document_1",
    content=source,
    mime_type="text/html",
    metadata={},
)
client.tool_runtime.rag_tool.insert(
    documents=[document],
    vector_db_id=vector_db_id,
    chunk_size_in_tokens=100,
)
agent = Agent(
    client,
    model=model_id,
    instructions="You are a helpful assistant",
    tools=[
        {
            "name": "builtin::rag/knowledge_search",
            "args": {"vector_db_ids": [vector_db_id]},
        }
    ],
)

prompt = "How do you do great work?"
print("prompt>", prompt)

use_stream = True
response = agent.create_turn(
    messages=[{"role": "user", "content": prompt}],
    session_id=agent.create_session("rag_session"),
    stream=use_stream,
)

# Only call `AgentEventLogger().log(response)` for streaming responses.
if use_stream:
    for log in AgentEventLogger().log(response):
        log.print()
else:
    print(response)

print()
print("=" * 80)
print("✅ Demo completed!")
print("🔭 View telemetry traces in Jaeger UI: http://localhost:16686")
print("   - Service: llama-stack-rag-demo")
print("   - Look for traces showing RAG operations, inference calls, and tool execution")
print("=" * 80)