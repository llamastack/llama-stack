# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
import textwrap

from llama_stack.models.llama.datatypes import (
    RawMessage,
    StopReason,
    ToolCall,
    ToolPromptFormat,
)

from ..prompt_format import (
    TextCompletionContent,
    UseCase,
    llama3_1_builtin_code_interpreter_dialog,
)


def user_tool_call():
    content = textwrap.dedent(
        """
        Questions: Can you retrieve the details for the user with the ID 7890, who has black as their special request?
        Here is a list of functions in JSON format that you can invoke:
        [
            {
                "name": "get_user_info",
                "description": "Retrieve details for a specific user by their unique identifier. Note that the provided function is in Python 3 syntax.",
                "parameters": {
                    "type": "dict",
                    "required": [
                        "user_id"
                    ],
                    "properties": {
                        "user_id": {
                        "type": "integer",
                        "description": "The unique identifier of the user. It is used to fetch the specific user details from the database."
                    },
                    "special": {
                        "type": "string",
                        "description": "Any special information or parameters that need to be considered while fetching user details.",
                        "default": "none"
                        }
                    }
                }
            }
        ]

        Should you decide to return the function call(s),Put it in the format of [func1(params_name=params_value, params_name2=params_value2...), func2(params)]

        NO other text MUST be included.
        """
    )
    return content.strip()


def system_tool_call():
    content = textwrap.dedent(
        """
        You are an expert in composing functions. You are given a question and a set of possible functions.
        Based on the question, you will need to make one or more function/tool calls to achieve the purpose.
        If none of the function can be used, point it out. If the given question lacks the parameters required by the function,
        also point it out. You should only return the function call in tools call sections.

        If you decide to invoke any of the function(s), you MUST put it in the format of [func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)]
        You SHOULD NOT include any other text in the response.

        Here is a list of functions in JSON format that you can invoke.

        [
            {
                "name": "get_weather",
                "description": "Get weather info for places",
                "parameters": {
                    "type": "dict",
                    "required": [
                        "city"
                    ],
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "The name of the city to get the weather for"
                        },
                        "metric": {
                            "type": "string",
                            "description": "The metric for weather. Options are: celsius, fahrenheit",
                            "default": "celsius"
                        }
                    }
                }
            }
        ]
        """
    )
    return content.strip()


def usecases():
    return [
        UseCase(
            title="User and assistant conversation",
            description="Here is a regular multi-turn user assistant conversation and how its formatted.",
            dialogs=[
                [
                    RawMessage(role="system", content="You are a helpful assistant"),
                    RawMessage(role="user", content="Who are you?"),
                ]
            ],
            notes="This format is unchanged from Llama3.1",
        ),
        UseCase(
            title="Zero shot function calling",
            description=textwrap.dedent(
                """
                For Llama3.2 1B and 3B instruct models, we are introducing a new format for zero shot function calling.
                This new format is designed to be more flexible and powerful than the previous format.
                All available functions can be provided in the system message. A key difference is in the format of how the assistant responds with function calls.
                It is pythonic in the form of `[func1(params_name=params_value, params_name2=params_value2...), func2(params)]` instead of the `json` or `<function>` tag that were defined in Llama3.1.
                Here is an example for the same,
                """
            ),
            dialogs=[
                # Zero shot tool calls as system message
                [
                    RawMessage(role="system", content=system_tool_call()),
                    RawMessage(role="user", content="What is the weather in SF and Seattle?"),
                ],
            ],
            notes=textwrap.dedent(
                """
                - The output supports multiple tool calls natively
                - JSON format for defining the functions in the system prompt is similar to Llama3.1
                """
            ),
        ),
        UseCase(
            title="Zero shot function calling with user message",
            description=textwrap.dedent(
                """
                While the default is to provide all function calls in a system message, in Llama3.2 text models you can also provide information for all the available tools in a user message.
                """
            ),
            dialogs=[
                # Zero shot tool call as user message
                [
                    RawMessage(role="user", content=user_tool_call()),
                ],
            ],
            notes=textwrap.dedent(
                """
                - The tool call format for the model is the same whether your function calls are provided in the system or user message.
                - While builtin tool calls end with a <|eom_id|>, notice the <|eot_id|> for zero shot tool calls.
                """
            ),
        ),
        UseCase(
            title="Code Interpreter",
            description=textwrap.dedent(
                """
                Code Interpreter continues to work in 3.2 text models similar to Llama 3.1 model family.
                Here is an example,
                """
            ),
            dialogs=[llama3_1_builtin_code_interpreter_dialog()],
            notes=textwrap.dedent(
                """
                - Note `Environment: ipython` in the system prompt.
                - Note that the response starts with `<|python_tag|>` and ends with `<|eom_id|>`
                """
            ),
        ),
        UseCase(
            title="Zero shot function calling E2E format",
            description=textwrap.dedent(
                """
                Here is an example of the e2e cycle of tool calls with the model in a muti-step way.
                """
            ),
            dialogs=[
                [
                    RawMessage(role="system", content=system_tool_call()),
                    RawMessage(role="user", content="What is the weather in SF?"),
                    RawMessage(
                        role="assistant",
                        content="",
                        stop_reason=StopReason.end_of_turn,
                        tool_calls=[
                            ToolCall(
                                call_id="cc",
                                tool_name="get_weather",
                                arguments={
                                    "city": "San Francisco",
                                    "metric": "celsius",
                                },
                            )
                        ],
                    ),
                    RawMessage(
                        role="tool",
                        content=json.dumps("25 C"),
                    ),
                ],
            ],
            notes=textwrap.dedent(
                """
                - The output of the function call is provided back to the model as a tool response ( in json format ).
                - Notice `<|start_header_id|>ipython<|end_header_id|>` as the header message preceding the tool response.
                - The model finally summarizes the information from the tool response and returns the result to the user.
                """
            ),
            tool_prompt_format=ToolPromptFormat.python_list,
        ),
        UseCase(
            title="Prompt format for base models",
            description=textwrap.dedent(
                """
                For base models (Llama3.2-1B and Llama3.2-3B), the prompt format for a simple completion is as follows
                """
            ),
            dialogs=[
                TextCompletionContent(content="The color of the sky is blue but sometimes it can also be"),
            ],
            notes="Same as Llama3.1",
        ),
    ]
