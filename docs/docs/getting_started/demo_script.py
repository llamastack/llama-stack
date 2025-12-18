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
import os

import requests
from openai import OpenAI

# Initialize OpenAI client pointing to Llama Stack server
client = OpenAI(base_url="http://localhost:8321/v1/", api_key="none")

print("RAG demonstration\n")

url = "https://www.paulgraham.com/greatwork.html"
print(f"Fetching document from: {url}")

vs = client.vector_stores.create()

response = requests.get(url)
pseudo_file = io.BytesIO(str(response.content).encode("utf-8"))
uploaded_file = client.files.create(
    file=(url, pseudo_file, "text/html"), purpose="assistants"
)
client.vector_stores.files.create(vector_store_id=vs.id, file_id=uploaded_file.id)
print(f"File uploaded and added to vector store: {uploaded_file.id}")

query = "How do you do great work?"
print(f"Query: {query}\n")

print(
    """
RAG using Responses API:
   - Automatic tool calling (model decides when to search)
   - Simpler code, less control
   - Best for: Conversational agents, automatic workflows

"""
)

print("Reply via Responses API:\n")
resp = client.responses.create(
    model=os.getenv("INFERENCE_MODEL", "ollama/llama3.2:3b"),
    input=query,
    tools=[{"type": "file_search", "vector_store_ids": [vs.id]}],
    include=["file_search_call.results"],
)

print("-" * 80)
print(resp.output[-1].content[-1].text)
print("-" * 80)

print(
    """

RAG using Chat Completions API:
   - Manual retrieval (you control the search)
   - More code, more control
   - Best for: Custom RAG patterns, batch processing, specialized workflows
"""
)

print("Searching vector store...")
search_results = client.vector_stores.search(
    vector_store_id=vs.id, query=query, max_num_results=3, rewrite_query=False
)

# Extract context from search results
context_chunks = []
for result in search_results.data:
    # result.content is a list of Content objects, extract the text from each
    for content_item in result.content:
        context_chunks.append(content_item.text)

context = "\n\n".join(context_chunks)
print(f"Found {len(context_chunks)} relevant chunks\n")

print("Reply via Chat Completions API:\n")
completion = client.chat.completions.create(
    model=os.getenv("INFERENCE_MODEL", "ollama/llama3.2:3b"),
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

print("-" * 80)
print(completion.choices[0].message.content)
print("-" * 80)
