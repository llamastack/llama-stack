# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


import io, requests
from openai import OpenAI

url="https://www.paulgraham.com/greatwork.html"
client = OpenAI(base_url="http://localhost:8321/v1/", api_key="none")

vs = client.vector_stores.create()
response = requests.get(url)
pseudo_file = io.BytesIO(str(response.content).encode('utf-8'))
uploaded_file = client.files.create(file=(url, pseudo_file, "text/html"), purpose="assistants")
client.vector_stores.files.create(vector_store_id=vs.id, file_id=uploaded_file.id)

resp = client.responses.create(
    model="openai/gpt-4o",
    input="How do you do great work? Use the existing knowledge_search tool.",
    tools=[{"type": "file_search", "vector_store_ids": [vs.id]}],
    include=["file_search_call.results"],
)

print(resp)
