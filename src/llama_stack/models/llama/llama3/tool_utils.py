# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
import re

from llama_stack.log import get_logger
from llama_stack.models.llama.datatypes import BuiltinTool, ToolCall, ToolPromptFormat

from ..datatypes import RecursiveType

logger = get_logger(name=__name__, category="models::llama")

BUILTIN_TOOL_PATTERN = r'\b(?P<tool_name>\w+)\.call\(query="(?P<query>[^"]*)"\)'
CUSTOM_TOOL_CALL_PATTERN = re.compile(r"<function=(?P<function_name>[^}]+)>(?P<args>{.*?})")


def is_json(s):
    try:
        parsed = json.loads(s)
        # Return True for valid objects and not for ints, strings, etc
        return isinstance(parsed, dict)
    except json.JSONDecodeError:
        return False
    return True


def parse_llama_tool_call_format(input_string):
    """
    Parse tool calls in the format:
    [func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)]

    Returns a list of (function_name, arguments_dict) tuples or None if parsing fails.
    """
    # Strip outer brackets and whitespace
    input_string = input_string.strip()
    if not (input_string.startswith("[") and input_string.endswith("]")):
        return None

    content = input_string[1:-1].strip()
    if not content:
        return None

    result = []

    # State variables for parsing
    pos = 0
    length = len(content)

    while pos < length:
        # Find function name
        name_end = content.find("(", pos)
        if name_end == -1:
            break

        func_name = content[pos:name_end].strip()

        # Find closing parenthesis for this function call
        paren_level = 1
        args_start = name_end + 1
        args_end = args_start

        while args_end < length and paren_level > 0:
            if content[args_end] == "(":
                paren_level += 1
            elif content[args_end] == ")":
                paren_level -= 1
            args_end += 1

        if paren_level != 0:
            # Unmatched parentheses
            return None

        # Parse arguments
        args_str = content[args_start : args_end - 1].strip()
        args_dict = {}

        if args_str:
            # Split by commas, but respect nested structures
            parts = []
            part_start = 0
            in_quotes = False
            quote_char = None
            nested_level = 0

            for i, char in enumerate(args_str):
                if char in ('"', "'") and (i == 0 or args_str[i - 1] != "\\"):
                    if not in_quotes:
                        in_quotes = True
                        quote_char = char
                    elif char == quote_char:
                        in_quotes = False
                        quote_char = None
                elif not in_quotes:
                    if char in ("{", "["):
                        nested_level += 1
                    elif char in ("}", "]"):
                        nested_level -= 1
                    elif char == "," and nested_level == 0:
                        parts.append(args_str[part_start:i].strip())
                        part_start = i + 1

            parts.append(args_str[part_start:].strip())

            # Process each key=value pair
            for part in parts:
                if "=" in part:
                    key, value = part.split("=", 1)
                    key = key.strip()
                    value = value.strip()

                    # Try to convert value to appropriate Python type
                    if (value.startswith('"') and value.endswith('"')) or (
                        value.startswith("'") and value.endswith("'")
                    ):
                        # String
                        value = value[1:-1]
                    elif value.lower() == "true":
                        value = True
                    elif value.lower() == "false":
                        value = False
                    elif value.lower() == "none":
                        value = None
                    elif value.startswith("{") and value.endswith("}"):
                        # This is a nested dictionary
                        try:
                            # Try to parse as JSON
                            value = json.loads(value.replace("'", '"'))
                        except json.JSONDecodeError:
                            # Keep as string if parsing fails
                            pass
                    elif value.startswith("[") and value.endswith("]"):
                        # This is a nested list
                        try:
                            # Try to parse as JSON
                            value = json.loads(value.replace("'", '"'))
                        except json.JSONDecodeError:
                            # Keep as string if parsing fails
                            pass
                    else:
                        # Try to convert to number
                        try:
                            if "." in value:
                                value = float(value)
                            else:
                                value = int(value)
                        except ValueError:
                            # Keep as string if not a valid number
                            pass

                    args_dict[key] = value

        result.append((func_name, args_dict))

        # Move to the next function call
        pos = args_end

        # Skip the comma between function calls if present
        if pos < length and content[pos] == ",":
            pos += 1

    return result if result else None


class ToolUtils:
    @staticmethod
    def is_builtin_tool_call(message_body: str) -> bool:
        match = re.search(ToolUtils.BUILTIN_TOOL_PATTERN, message_body)
        return match is not None

    @staticmethod
    def maybe_extract_builtin_tool_call(message_body: str) -> tuple[str, str] | None:
        # Find the first match in the text
        match = re.search(BUILTIN_TOOL_PATTERN, message_body)

        # Check if a match is found and return it
        if match:
            tool_name = match.group("tool_name")
            query = match.group("query")
            return tool_name, query
        else:
            return None

    @staticmethod
    def maybe_extract_custom_tool_call(message_body: str) -> tuple[str, str] | None:
        # NOTE: Custom function too calls are still experimental
        # Sometimes, response is of the form
        # {"type": "function", "name": "function_name", "parameters": {...}
        # and some times
        # <function=function_name>(parameters)</function>

        # Find the first match in the text
        match = re.search(CUSTOM_TOOL_CALL_PATTERN, message_body)
        if match:
            tool_name = match.group("function_name")
            query = match.group("args")
            try:
                return tool_name, json.loads(query.replace("'", '"'))
            except Exception as e:
                print("Exception while parsing json query for custom tool call", query, e)
                return None
        elif is_json(message_body):
            response = json.loads(message_body)
            if ("type" in response and response["type"] == "function") or (
                "name" in response and "parameters" in response
            ):
                function_name = response["name"]
                args = response["parameters"]
                return function_name, args
            else:
                return None
        elif function_calls := parse_llama_tool_call_format(message_body):
            # FIXME: Enable multiple tool calls
            return function_calls[0]
        else:
            logger.debug(f"Did not parse tool call from message body: {message_body}")
            return None

    @staticmethod
    def encode_tool_call(t: ToolCall, tool_prompt_format: ToolPromptFormat) -> str:
        args = json.loads(t.arguments)
        if t.tool_name == BuiltinTool.brave_search:
            q = args["query"]
            return f'brave_search.call(query="{q}")'
        elif t.tool_name == BuiltinTool.wolfram_alpha:
            q = args["query"]
            return f'wolfram_alpha.call(query="{q}")'
        elif t.tool_name == BuiltinTool.photogen:
            q = args["query"]
            return f'photogen.call(query="{q}")'
        elif t.tool_name == BuiltinTool.code_interpreter:
            return args["code"]
        else:
            fname = t.tool_name

            if tool_prompt_format == ToolPromptFormat.json:
                return json.dumps(
                    {
                        "type": "function",
                        "name": fname,
                        "parameters": args,
                    }
                )
            elif tool_prompt_format == ToolPromptFormat.function_tag:
                return f"<function={fname}>{t.arguments}</function>"

            elif tool_prompt_format == ToolPromptFormat.python_list:

                def format_value(value: RecursiveType) -> str:
                    if isinstance(value, str):
                        return f'"{value}"'
                    elif isinstance(value, int | float | bool) or value is None:
                        return str(value)
                    elif isinstance(value, list):
                        return f"[{', '.join(format_value(v) for v in value)}]"
                    elif isinstance(value, dict):
                        return f"{{{', '.join(f'{k}={format_value(v)}' for k, v in value.items())}}}"
                    else:
                        raise ValueError(f"Unsupported type: {type(value)}")

                args_str = ", ".join(f"{k}={format_value(v)}" for k, v in args.items())
                return f"[{fname}({args_str})]"
            else:
                raise ValueError(f"Unsupported tool prompt format: {tool_prompt_format}")
