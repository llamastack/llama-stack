# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
from llama_stack_api.models.llama.llama3.tool_utils import ToolUtils


class TestMaybeExtractCustomToolCall:
    def test_valid_single_tool_call(self):
        input_string = '[get_weather(location="San Francisco", units="celsius")]'
        result = ToolUtils.maybe_extract_custom_tool_call(input_string)

        assert result is not None
        assert len(result) == 2
        assert result[0] == "get_weather"
        assert result[1] == {"location": "San Francisco", "units": "celsius"}

    def test_valid_multiple_tool_calls(self):
        input_string = '[search(query="python programming"), get_time(timezone="UTC")]'
        result = ToolUtils.maybe_extract_custom_tool_call(input_string)

        # Note: maybe_extract_custom_tool_call currently only returns the first tool call
        assert result is not None
        assert len(result) == 2
        assert result[0] == "search"
        assert result[1] == {"query": "python programming"}

    def test_different_value_types(self):
        input_string = '[analyze_data(count=42, enabled=True, ratio=3.14, name="test", options=None)]'
        result = ToolUtils.maybe_extract_custom_tool_call(input_string)

        assert result is not None
        assert len(result) == 2
        assert result[0] == "analyze_data"
        assert result[1] == {"count": 42, "enabled": True, "ratio": 3.14, "name": "test", "options": None}

    def test_nested_structures(self):
        input_string = '[complex_function(filters={"min": 10, "max": 100}, tags=["important", "urgent"])]'
        result = ToolUtils.maybe_extract_custom_tool_call(input_string)

        # This test checks that nested structures are handled
        assert result is not None
        assert len(result) == 2
        assert result[0] == "complex_function"
        assert "filters" in result[1]
        assert sorted(result[1]["filters"].items()) == sorted({"min": 10, "max": 100}.items())

        assert "tags" in result[1]
        assert result[1]["tags"] == ["important", "urgent"]

    def test_hyphenated_function_name(self):
        input_string = '[weather-forecast(city="London")]'
        result = ToolUtils.maybe_extract_custom_tool_call(input_string)

        assert result is not None
        assert len(result) == 2
        assert result[0] == "weather-forecast"  # Function name remains hyphenated
        assert result[1] == {"city": "London"}

    def test_empty_input(self):
        input_string = "[]"
        result = ToolUtils.maybe_extract_custom_tool_call(input_string)

        assert result is None

    def test_invalid_format(self):
        invalid_inputs = [
            'get_weather(location="San Francisco")',  # Missing outer brackets
            '{get_weather(location="San Francisco")}',  # Wrong outer brackets
            '[get_weather(location="San Francisco"]',  # Unmatched brackets
            '[get_weather{location="San Francisco"}]',  # Wrong inner brackets
            "just some text",  # Not a tool call format at all
        ]

        for input_string in invalid_inputs:
            result = ToolUtils.maybe_extract_custom_tool_call(input_string)
            assert result is None

    def test_quotes_handling(self):
        input_string = '[search(query="Text with \\"quotes\\" inside")]'
        result = ToolUtils.maybe_extract_custom_tool_call(input_string)

        # This test checks that escaped quotes are handled correctly
        assert result is not None

    def test_single_quotes_in_arguments(self):
        input_string = "[add-note(name='demonote', content='demonstrating Llama Stack and MCP integration')]"
        result = ToolUtils.maybe_extract_custom_tool_call(input_string)

        assert result is not None
        assert len(result) == 2
        assert result[0] == "add-note"  # Function name remains hyphenated
        assert result[1] == {"name": "demonote", "content": "demonstrating Llama Stack and MCP integration"}

    def test_json_format(self):
        input_string = '{"type": "function", "name": "search_web", "parameters": {"query": "AI research"}}'
        result = ToolUtils.maybe_extract_custom_tool_call(input_string)

        assert result is not None
        assert len(result) == 2
        assert result[0] == "search_web"
        assert result[1] == {"query": "AI research"}

    def test_python_list_format(self):
        input_string = "[calculate(x=10, y=20)]"
        result = ToolUtils.maybe_extract_custom_tool_call(input_string)

        assert result is not None
        assert len(result) == 2
        assert result[0] == "calculate"
        assert result[1] == {"x": 10, "y": 20}

    def test_complex_nested_structures(self):
        input_string = '[advanced_query(config={"filters": {"categories": ["books", "electronics"], "price_range": {"min": 10, "max": 500}}, "sort": {"field": "relevance", "order": "desc"}})]'
        result = ToolUtils.maybe_extract_custom_tool_call(input_string)

        assert result is not None
        assert len(result) == 2
        assert result[0] == "advanced_query"

        # Verify the overall structure
        assert "config" in result[1]
        assert isinstance(result[1]["config"], dict)

        # Verify the first level of nesting
        config = result[1]["config"]
        assert "filters" in config
        assert "sort" in config

        # Verify the second level of nesting (filters)
        filters = config["filters"]
        assert "categories" in filters
        assert "price_range" in filters

        # Verify the list within the dict
        assert filters["categories"] == ["books", "electronics"]

        # Verify the nested dict within another dict
        assert filters["price_range"]["min"] == 10
        assert filters["price_range"]["max"] == 500

        # Verify the sort dictionary
        assert config["sort"]["field"] == "relevance"
        assert config["sort"]["order"] == "desc"
