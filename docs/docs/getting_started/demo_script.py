# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Demo script showing RAG with both Responses API and Chat Completions API.

This example demonstrates two approaches to RAG with Llama Stack:
1. Responses API - Automatic agentic tool calling with file search
2. Chat Completions API - Manual retrieval with explicit control

Run this script after starting a Llama Stack server:
    llama stack run starter
"""

import io

import requests
from openai import OpenAI

# Initialize OpenAI client pointing to Llama Stack server
client = OpenAI(base_url="http://localhost:8321/v1/", api_key="none")

# Shared setup: Create vector store and upload document
print("=" * 80)
print("SETUP: Creating vector store and uploading document")
print("=" * 80)

url = "https://www.paulgraham.com/greatwork.html"
print(f"Fetching document from: {url}")

vs = client.vector_stores.create()
print(f"Vector store created: {vs.id}")

response = requests.get(url)
pseudo_file = io.BytesIO(str(response.content).encode("utf-8"))
uploaded_file = client.files.create(
    file=(url, pseudo_file, "text/html"), purpose="assistants"
)
client.vector_stores.files.create(vector_store_id=vs.id, file_id=uploaded_file.id)
print(f"File uploaded and added to vector store: {uploaded_file.id}")

query = "How do you do great work?"

# ============================================================================
# APPROACH 1: Responses API (Recommended for most use cases)
# ============================================================================
print("\n" + "=" * 80)
print("APPROACH 1: Responses API (Automatic Tool Calling)")
print("=" * 80)
print(f"Query: {query}\n")

resp = client.responses.create(
    model="ollama/llama3.2:3b",  # feel free to change this to any other model
    input=query,
    tools=[{"type": "file_search", "vector_store_ids": [vs.id]}],
    include=["file_search_call.results"],
)

print("Response (Responses API):")
print("-" * 80)
print(resp.output[-1].content[-1].text)
print("-" * 80)

# ============================================================================
# APPROACH 2: Chat Completions API
# ============================================================================
print("\n" + "=" * 80)
print("APPROACH 2: Chat Completions API (Manual Retrieval)")
print("=" * 80)
print(f"Query: {query}\n")

# Step 1: Search vector store explicitly
print("Searching vector store...")
search_results = client.vector_stores.search(
    vector_store_id=vs.id, query=query, max_num_results=3, rewrite_query=False
)

# Step 2: Extract context from search results
context_chunks = []
for result in search_results.data:
    # result.content is a list of Content objects, extract the text from each
    if hasattr(result, "content") and result.content:
        for content_item in result.content:
            if hasattr(content_item, "text") and content_item.text:
                context_chunks.append(content_item.text)

context = "\n\n".join(context_chunks)
print(f"Found {len(context_chunks)} relevant chunks\n")

# Step 3: Use Chat Completions with retrieved context
print("Generating response with chat completions...")
completion = client.chat.completions.create(
    model="ollama/llama3.2:3b",  # Feel free to change this to any other model
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant. Use the provided context to answer the user's question.",
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {query}\n\nPlease provide a comprehensive answer based on the context above.",
        },
    ],
    temperature=0.7,
)

print("Response (Chat Completions API):")
print("-" * 80)
print(completion.choices[0].message.content)
print("-" * 80)

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(
    """
Both approaches successfully performed RAG:

1. Responses API:
   - Automatic tool calling (model decides when to search)
   - Simpler code, less control
   - Best for: Conversational agents, automatic workflows

2. Chat Completions API:
   - Manual retrieval (you control the search)
   - More code, more control
   - Best for: Custom RAG patterns, batch processing, specialized workflows
"""
)
