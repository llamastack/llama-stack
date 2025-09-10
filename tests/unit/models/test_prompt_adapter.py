# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from llama_stack.apis.inference import (
    ChatCompletionRequest,
    CompletionMessage,
    StopReason,
    SystemMessage,
    SystemMessageBehavior,
    ToolCall,
    ToolConfig,
    UserMessage,
)
from llama_stack.models.llama.datatypes import (
    BuiltinTool,
    ToolDefinition,
    ToolParamDefinition,
    ToolPromptFormat,
)
from llama_stack.providers.utils.inference.prompt_adapter import (
    chat_completion_request_to_messages,
    chat_completion_request_to_prompt,
    interleaved_content_as_str,
)

MODEL = "Llama3.1-8B-Instruct"
MODEL3_2 = "Llama3.2-3B-Instruct"


async def test_system_default():
    content = "Hello !"
    request = ChatCompletionRequest(
        model=MODEL,
        messages=[
            UserMessage(content=content),
        ],
    )
    messages = chat_completion_request_to_messages(request, MODEL)
    assert len(messages) == 2
    assert messages[-1].content == content
    assert "Cutting Knowledge Date: December 2023" in interleaved_content_as_str(messages[0].content)


async def test_system_builtin_only():
    content = "Hello !"
    request = ChatCompletionRequest(
        model=MODEL,
        messages=[
            UserMessage(content=content),
        ],
        tools=[
            ToolDefinition(tool_name=BuiltinTool.code_interpreter),
            ToolDefinition(tool_name=BuiltinTool.brave_search),
        ],
    )
    messages = chat_completion_request_to_messages(request, MODEL)
    assert len(messages) == 2
    assert messages[-1].content == content
    assert "Cutting Knowledge Date: December 2023" in interleaved_content_as_str(messages[0].content)
    assert "Tools: brave_search" in interleaved_content_as_str(messages[0].content)


async def test_system_custom_only():
    content = "Hello !"
    request = ChatCompletionRequest(
        model=MODEL,
        messages=[
            UserMessage(content=content),
        ],
        tools=[
            ToolDefinition(
                tool_name="custom1",
                description="custom1 tool",
                parameters={
                    "param1": ToolParamDefinition(
                        param_type="str",
                        description="param1 description",
                        required=True,
                    ),
                },
            )
        ],
        tool_config=ToolConfig(tool_prompt_format=ToolPromptFormat.json),
    )
    messages = chat_completion_request_to_messages(request, MODEL)
    assert len(messages) == 3
    assert "Environment: ipython" in interleaved_content_as_str(messages[0].content)

    assert "Return function calls in JSON format" in interleaved_content_as_str(messages[1].content)
    assert messages[-1].content == content


async def test_system_custom_and_builtin():
    content = "Hello !"
    request = ChatCompletionRequest(
        model=MODEL,
        messages=[
            UserMessage(content=content),
        ],
        tools=[
            ToolDefinition(tool_name=BuiltinTool.code_interpreter),
            ToolDefinition(tool_name=BuiltinTool.brave_search),
            ToolDefinition(
                tool_name="custom1",
                description="custom1 tool",
                parameters={
                    "param1": ToolParamDefinition(
                        param_type="str",
                        description="param1 description",
                        required=True,
                    ),
                },
            ),
        ],
    )
    messages = chat_completion_request_to_messages(request, MODEL)
    assert len(messages) == 3

    assert "Environment: ipython" in interleaved_content_as_str(messages[0].content)
    assert "Tools: brave_search" in interleaved_content_as_str(messages[0].content)

    assert "Return function calls in JSON format" in interleaved_content_as_str(messages[1].content)
    assert messages[-1].content == content


async def test_completion_message_encoding():
    request = ChatCompletionRequest(
        model=MODEL3_2,
        messages=[
            UserMessage(content="hello"),
            CompletionMessage(
                content="",
                stop_reason=StopReason.end_of_turn,
                tool_calls=[
                    ToolCall(
                        tool_name="custom1",
                        arguments={"param1": "value1"},
                        call_id="123",
                    )
                ],
            ),
        ],
        tools=[
            ToolDefinition(
                tool_name="custom1",
                description="custom1 tool",
                parameters={
                    "param1": ToolParamDefinition(
                        param_type="str",
                        description="param1 description",
                        required=True,
                    ),
                },
            ),
        ],
        tool_config=ToolConfig(tool_prompt_format=ToolPromptFormat.python_list),
    )
    prompt = await chat_completion_request_to_prompt(request, request.model)
    assert '[custom1(param1="value1")]' in prompt

    request.model = MODEL
    request.tool_config = ToolConfig(tool_prompt_format=ToolPromptFormat.json)
    prompt = await chat_completion_request_to_prompt(request, request.model)
    assert '{"type": "function", "name": "custom1", "parameters": {"param1": "value1"}}' in prompt


async def test_user_provided_system_message():
    content = "Hello !"
    system_prompt = "You are a pirate"
    request = ChatCompletionRequest(
        model=MODEL,
        messages=[
            SystemMessage(content=system_prompt),
            UserMessage(content=content),
        ],
        tools=[
            ToolDefinition(tool_name=BuiltinTool.code_interpreter),
        ],
    )
    messages = chat_completion_request_to_messages(request, MODEL)
    assert len(messages) == 2
    assert interleaved_content_as_str(messages[0].content).endswith(system_prompt)

    assert messages[-1].content == content


async def test_replace_system_message_behavior_builtin_tools():
    content = "Hello !"
    system_prompt = "You are a pirate"
    request = ChatCompletionRequest(
        model=MODEL,
        messages=[
            SystemMessage(content=system_prompt),
            UserMessage(content=content),
        ],
        tools=[
            ToolDefinition(tool_name=BuiltinTool.code_interpreter),
        ],
        tool_config=ToolConfig(
            tool_choice="auto",
            tool_prompt_format=ToolPromptFormat.python_list,
            system_message_behavior=SystemMessageBehavior.replace,
        ),
    )
    messages = chat_completion_request_to_messages(request, MODEL3_2)
    assert len(messages) == 2
    assert interleaved_content_as_str(messages[0].content).endswith(system_prompt)
    assert "Environment: ipython" in interleaved_content_as_str(messages[0].content)
    assert messages[-1].content == content


async def test_replace_system_message_behavior_custom_tools():
    content = "Hello !"
    system_prompt = "You are a pirate"
    request = ChatCompletionRequest(
        model=MODEL,
        messages=[
            SystemMessage(content=system_prompt),
            UserMessage(content=content),
        ],
        tools=[
            ToolDefinition(tool_name=BuiltinTool.code_interpreter),
            ToolDefinition(
                tool_name="custom1",
                description="custom1 tool",
                parameters={
                    "param1": ToolParamDefinition(
                        param_type="str",
                        description="param1 description",
                        required=True,
                    ),
                },
            ),
        ],
        tool_config=ToolConfig(
            tool_choice="auto",
            tool_prompt_format=ToolPromptFormat.python_list,
            system_message_behavior=SystemMessageBehavior.replace,
        ),
    )
    messages = chat_completion_request_to_messages(request, MODEL3_2)

    assert len(messages) == 2
    assert interleaved_content_as_str(messages[0].content).endswith(system_prompt)
    assert "Environment: ipython" in interleaved_content_as_str(messages[0].content)
    assert messages[-1].content == content


async def test_replace_system_message_behavior_custom_tools_with_template():
    content = "Hello !"
    system_prompt = "You are a pirate {{ function_description }}"
    request = ChatCompletionRequest(
        model=MODEL,
        messages=[
            SystemMessage(content=system_prompt),
            UserMessage(content=content),
        ],
        tools=[
            ToolDefinition(tool_name=BuiltinTool.code_interpreter),
            ToolDefinition(
                tool_name="custom1",
                description="custom1 tool",
                parameters={
                    "param1": ToolParamDefinition(
                        param_type="str",
                        description="param1 description",
                        required=True,
                    ),
                },
            ),
        ],
        tool_config=ToolConfig(
            tool_choice="auto",
            tool_prompt_format=ToolPromptFormat.python_list,
            system_message_behavior=SystemMessageBehavior.replace,
        ),
    )
    messages = chat_completion_request_to_messages(request, MODEL3_2)

    assert len(messages) == 2
    assert "Environment: ipython" in interleaved_content_as_str(messages[0].content)
    assert "You are a pirate" in interleaved_content_as_str(messages[0].content)
    # function description is present in the system prompt
    assert '"name": "custom1"' in interleaved_content_as_str(messages[0].content)
    assert messages[-1].content == content
