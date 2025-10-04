# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
import os

import pytest


@pytest.fixture
def sample_search_query():
    return "What are the latest developments in quantum computing?"


@pytest.fixture
def sample_wolfram_alpha_query():
    return "What is the square root of 16?"


def test_web_search_tool(llama_stack_client, sample_search_query):
    """Test the web search tool functionality."""
    if "TAVILY_SEARCH_API_KEY" not in os.environ:
        pytest.skip("TAVILY_SEARCH_API_KEY not set, skipping test")

    tools = llama_stack_client.tool_runtime.list_tools()
    assert any(tool.name == "web_search" for tool in tools)

    response = llama_stack_client.tool_runtime.invoke_tool(
        tool_name="web_search", kwargs={"query": sample_search_query}
    )
    # Verify the response
    assert response.content is not None
    assert len(response.content) > 0
    assert isinstance(response.content, str)

    content = json.loads(response.content)
    assert "query" in content
    assert "top_k" in content
    assert len(content["top_k"]) > 0

    first = content["top_k"][0]
    assert "title" in first
    assert "url" in first


def test_wolfram_alpha_tool(llama_stack_client, sample_wolfram_alpha_query):
    """Test the wolfram alpha tool functionality."""
    if "WOLFRAM_ALPHA_API_KEY" not in os.environ:
        pytest.skip("WOLFRAM_ALPHA_API_KEY not set, skipping test")

    tools = llama_stack_client.tool_runtime.list_tools()
    assert any(tool.name == "wolfram_alpha" for tool in tools)
    response = llama_stack_client.tool_runtime.invoke_tool(
        tool_name="wolfram_alpha", kwargs={"query": sample_wolfram_alpha_query}
    )

    assert response.content is not None
    assert len(response.content) > 0
    assert isinstance(response.content, str)

    content = json.loads(response.content)
    result = content["queryresult"]
    assert "success" in result
    assert result["success"]
    assert "pods" in result
    assert len(result["pods"]) > 0
