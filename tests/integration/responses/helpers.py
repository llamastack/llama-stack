# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import time


def new_vector_store(openai_client, name, embedding_model, embedding_dimension):
    """Create a new vector store, cleaning up any existing one with the same name."""
    # Ensure we don't reuse an existing vector store
    vector_stores = openai_client.vector_stores.list()
    for vector_store in vector_stores:
        if vector_store.name == name:
            openai_client.vector_stores.delete(vector_store_id=vector_store.id)

    # Create a new vector store

    vector_store = openai_client.vector_stores.create(
        name=name,
        extra_body={"embedding_model": embedding_model, "embedding_dimension": embedding_dimension},
    )
    return vector_store


def upload_file(openai_client, name, file_path):
    """Upload a file, cleaning up any existing file with the same name."""
    # Ensure we don't reuse an existing file
    files = openai_client.files.list()
    for file in files:
        if file.filename == name:
            openai_client.files.delete(file_id=file.id)

    # Upload a text file with our document content
    return openai_client.files.create(file=open(file_path, "rb"), purpose="assistants")


def wait_for_file_attachment(compat_client, vector_store_id, file_id):
    """Wait for a file to be attached to a vector store."""
    file_attach_response = compat_client.vector_stores.files.retrieve(
        vector_store_id=vector_store_id,
        file_id=file_id,
    )

    while file_attach_response.status == "in_progress":
        time.sleep(0.1)
        file_attach_response = compat_client.vector_stores.files.retrieve(
            vector_store_id=vector_store_id,
            file_id=file_id,
        )

    assert file_attach_response.status == "completed", f"Expected file to be attached, got {file_attach_response}"
    assert not file_attach_response.last_error
    return file_attach_response


def setup_mcp_tools(tools, mcp_server_info):
    """Replace placeholder MCP server URLs with actual server info."""
    # Create a deep copy to avoid modifying the original test case
    import copy

    tools_copy = copy.deepcopy(tools)

    for tool in tools_copy:
        if tool["type"] == "mcp" and tool["server_url"] == "<FILLED_BY_TEST_RUNNER>":
            tool["server_url"] = mcp_server_info["server_url"]
    return tools_copy
