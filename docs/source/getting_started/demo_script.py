# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack_client import Agent, AgentEventLogger, RAGDocument, LlamaStackClient

vector_db_id = "my_demo_vector_db"
client = LlamaStackClient(base_url="http://localhost:8321")

models = client.models.list()

# Select the first LLM and first embedding models
model_id = next(m for m in models if m.model_type == "llm").identifier
embedding_model_id = (
    em := next(m for m in models if m.model_type == "embedding")
).identifier
embedding_dimension = em.metadata["embedding_dimension"]

vector_db = client.vector_dbs.register(
    vector_db_id=vector_db_id,
    embedding_model=embedding_model_id,
    embedding_dimension=embedding_dimension,
    provider_id="faiss",
)
vector_db_id = vector_db.identifier
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
