# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from time import sleep

import pytest
from pydantic import BaseModel

from llama_stack.models.llama.sku_list import resolve_model

from ..test_cases.test_case import TestCase

PROVIDER_LOGPROBS_TOP_K = {"remote::together", "remote::fireworks", "remote::vllm"}


def skip_if_model_doesnt_support_completion(client_with_models, model_id):
    models = {m.identifier: m for m in client_with_models.models.list()}
    models.update({m.provider_resource_id: m for m in client_with_models.models.list()})
    provider_id = models[model_id].provider_id
    providers = {p.provider_id: p for p in client_with_models.providers.list()}
    provider = providers[provider_id]
    if (
        provider.provider_type
        in (
            "remote::openai",
            "remote::anthropic",
            "remote::gemini",
            "remote::vertexai",
            "remote::groq",
            "remote::sambanova",
            "remote::azure",
        )
        or "openai-compat" in provider.provider_type
    ):
        pytest.skip(f"Model {model_id} hosted by {provider.provider_type} doesn't support completion")


def skip_if_model_doesnt_support_json_schema_structured_output(client_with_models, model_id):
    models = {m.identifier: m for m in client_with_models.models.list()}
    models.update({m.provider_resource_id: m for m in client_with_models.models.list()})
    provider_id = models[model_id].provider_id
    providers = {p.provider_id: p for p in client_with_models.providers.list()}
    provider = providers[provider_id]
    if provider.provider_type in ("remote::sambanova", "remote::azure", "remote::watsonx"):
        pytest.skip(
            f"Model {model_id} hosted by {provider.provider_type} doesn't support json_schema structured output"
        )


def get_llama_model(client_with_models, model_id):
    models = {}
    for m in client_with_models.models.list():
        models[m.identifier] = m
        models[m.provider_resource_id] = m

    assert model_id in models, f"Model {model_id} not found"

    model = models[model_id]
    ids = (model.identifier, model.provider_resource_id)
    for mid in ids:
        if resolve_model(mid):
            return mid

    return model.metadata.get("llama_model", None)


@pytest.mark.parametrize(
    "test_case",
    [
        "inference:completion:sanity",
    ],
)
def test_text_completion_non_streaming(client_with_models, text_model_id, test_case):
    skip_if_model_doesnt_support_completion(client_with_models, text_model_id)
    tc = TestCase(test_case)

    response = client_with_models.inference.completion(
        content=tc["content"],
        stream=False,
        model_id=text_model_id,
        sampling_params={
            "max_tokens": 50,
        },
    )
    assert len(response.content) > 10
    # assert "blue" in response.content.lower().strip()


@pytest.mark.parametrize(
    "test_case",
    [
        "inference:completion:sanity",
    ],
)
def test_text_completion_streaming(client_with_models, text_model_id, test_case):
    skip_if_model_doesnt_support_completion(client_with_models, text_model_id)
    tc = TestCase(test_case)

    response = client_with_models.inference.completion(
        content=tc["content"],
        stream=True,
        model_id=text_model_id,
        sampling_params={
            "max_tokens": 50,
        },
    )
    streamed_content = [chunk.delta for chunk in response]
    content_str = "".join(streamed_content).lower().strip()
    # assert "blue" in content_str
    assert len(content_str) > 10


@pytest.mark.parametrize(
    "test_case",
    [
        "inference:completion:stop_sequence",
    ],
)
def test_text_completion_stop_sequence(client_with_models, text_model_id, inference_provider_type, test_case):
    skip_if_model_doesnt_support_completion(client_with_models, text_model_id)
    # This is only supported/tested for remote vLLM: https://github.com/meta-llama/llama-stack/issues/1771
    if inference_provider_type != "remote::vllm":
        pytest.xfail(f"{inference_provider_type} doesn't support 'stop' parameter yet")
    tc = TestCase(test_case)

    response = client_with_models.inference.completion(
        content=tc["content"],
        stream=True,
        model_id=text_model_id,
        sampling_params={
            "max_tokens": 50,
            "stop": ["1963"],
        },
    )
    streamed_content = [chunk.delta for chunk in response]
    content_str = "".join(streamed_content).lower().strip()
    assert "1963" not in content_str


@pytest.mark.parametrize(
    "test_case",
    [
        "inference:completion:log_probs",
    ],
)
def test_text_completion_log_probs_non_streaming(client_with_models, text_model_id, inference_provider_type, test_case):
    skip_if_model_doesnt_support_completion(client_with_models, text_model_id)
    if inference_provider_type not in PROVIDER_LOGPROBS_TOP_K:
        pytest.xfail(f"{inference_provider_type} doesn't support log probs yet")

    tc = TestCase(test_case)

    response = client_with_models.inference.completion(
        content=tc["content"],
        stream=False,
        model_id=text_model_id,
        sampling_params={
            "max_tokens": 5,
        },
        logprobs={
            "top_k": 1,
        },
    )
    assert response.logprobs, "Logprobs should not be empty"
    assert 1 <= len(response.logprobs) <= 5  # each token has 1 logprob and here max_tokens=5
    assert all(len(logprob.logprobs_by_token) == 1 for logprob in response.logprobs)


@pytest.mark.parametrize(
    "test_case",
    [
        "inference:completion:log_probs",
    ],
)
def test_text_completion_log_probs_streaming(client_with_models, text_model_id, inference_provider_type, test_case):
    skip_if_model_doesnt_support_completion(client_with_models, text_model_id)
    if inference_provider_type not in PROVIDER_LOGPROBS_TOP_K:
        pytest.xfail(f"{inference_provider_type} doesn't support log probs yet")

    tc = TestCase(test_case)

    response = client_with_models.inference.completion(
        content=tc["content"],
        stream=True,
        model_id=text_model_id,
        sampling_params={
            "max_tokens": 5,
        },
        logprobs={
            "top_k": 1,
        },
    )
    streamed_content = list(response)
    for chunk in streamed_content:
        if chunk.delta:  # if there's a token, we expect logprobs
            assert chunk.logprobs, "Logprobs should not be empty"
            assert all(len(logprob.logprobs_by_token) == 1 for logprob in chunk.logprobs)
        else:  # no token, no logprobs
            assert not chunk.logprobs, "Logprobs should be empty"


@pytest.mark.parametrize(
    "test_case",
    [
        "inference:completion:structured_output",
    ],
)
def test_text_completion_structured_output(client_with_models, text_model_id, test_case):
    skip_if_model_doesnt_support_completion(client_with_models, text_model_id)
    skip_if_model_doesnt_support_json_schema_structured_output(client_with_models, text_model_id)

    class AnswerFormat(BaseModel):
        name: str
        year_born: str
        year_retired: str

    tc = TestCase(test_case)

    user_input = tc["user_input"]
    response = client_with_models.inference.completion(
        model_id=text_model_id,
        content=user_input,
        stream=False,
        sampling_params={
            "max_tokens": 50,
        },
        response_format={
            "type": "json_schema",
            "json_schema": AnswerFormat.model_json_schema(),
        },
    )
    answer = AnswerFormat.model_validate_json(response.content)
    expected = tc["expected"]
    assert answer.name == expected["name"]
    assert answer.year_born == expected["year_born"]
    assert answer.year_retired == expected["year_retired"]


@pytest.mark.parametrize(
    "test_case",
    [
        "inference:chat_completion:non_streaming_01",
        "inference:chat_completion:non_streaming_02",
    ],
)
def test_text_chat_completion_non_streaming(client_with_models, text_model_id, test_case):
    tc = TestCase(test_case)
    question = tc["question"]
    expected = tc["expected"]

    response = client_with_models.inference.chat_completion(
        model_id=text_model_id,
        messages=[
            {
                "role": "user",
                "content": question,
            }
        ],
        stream=False,
    )
    message_content = response.completion_message.content.lower().strip()
    assert len(message_content) > 0
    assert expected.lower() in message_content


@pytest.mark.parametrize(
    "test_case",
    [
        "inference:chat_completion:streaming_01",
        "inference:chat_completion:streaming_02",
    ],
)
def test_text_chat_completion_streaming(client_with_models, text_model_id, test_case):
    tc = TestCase(test_case)
    question = tc["question"]
    expected = tc["expected"]

    response = client_with_models.inference.chat_completion(
        model_id=text_model_id,
        messages=[{"role": "user", "content": question}],
        stream=True,
        timeout=120,  # Increase timeout to 2 minutes for large conversation history
    )
    streamed_content = [str(chunk.event.delta.text.lower().strip()) for chunk in response]
    assert len(streamed_content) > 0
    assert expected.lower() in "".join(streamed_content)


@pytest.mark.parametrize(
    "test_case",
    [
        "inference:chat_completion:tool_calling",
    ],
)
def test_text_chat_completion_with_tool_calling_and_non_streaming(client_with_models, text_model_id, test_case):
    tc = TestCase(test_case)

    response = client_with_models.inference.chat_completion(
        model_id=text_model_id,
        messages=tc["messages"],
        tools=tc["tools"],
        tool_choice="auto",
        stream=False,
    )
    # some models can return content for the response in addition to the tool call
    assert response.completion_message.role == "assistant"

    assert len(response.completion_message.tool_calls) == 1
    assert response.completion_message.tool_calls[0].tool_name == tc["tools"][0]["tool_name"]
    assert response.completion_message.tool_calls[0].arguments == tc["expected"]


# Will extract streamed text and separate it from tool invocation content
# The returned tool inovcation content will be a string so it's easy to comapare with expected value
# e.g. "[get_weather, {'location': 'San Francisco, CA'}]"
def extract_tool_invocation_content(response):
    tool_invocation_content: str = ""
    for chunk in response:
        delta = chunk.event.delta
        if delta.type == "tool_call" and delta.parse_status == "succeeded":
            call = delta.tool_call
            tool_invocation_content += f"[{call.tool_name}, {call.arguments}]"
    return tool_invocation_content


@pytest.mark.parametrize(
    "test_case",
    [
        "inference:chat_completion:tool_calling",
    ],
)
def test_text_chat_completion_with_tool_calling_and_streaming(client_with_models, text_model_id, test_case):
    tc = TestCase(test_case)

    response = client_with_models.inference.chat_completion(
        model_id=text_model_id,
        messages=tc["messages"],
        tools=tc["tools"],
        tool_choice="auto",
        stream=True,
    )
    tool_invocation_content = extract_tool_invocation_content(response)
    expected_tool_name = tc["tools"][0]["tool_name"]
    expected_argument = tc["expected"]
    assert tool_invocation_content == f"[{expected_tool_name}, {expected_argument}]"


@pytest.mark.parametrize(
    "test_case",
    [
        "inference:chat_completion:tool_calling",
    ],
)
def test_text_chat_completion_with_tool_choice_required(client_with_models, text_model_id, test_case):
    tc = TestCase(test_case)

    response = client_with_models.inference.chat_completion(
        model_id=text_model_id,
        messages=tc["messages"],
        tools=tc["tools"],
        tool_config={
            "tool_choice": "required",
        },
        stream=True,
    )
    tool_invocation_content = extract_tool_invocation_content(response)
    expected_tool_name = tc["tools"][0]["tool_name"]
    expected_argument = tc["expected"]
    assert tool_invocation_content == f"[{expected_tool_name}, {expected_argument}]"


@pytest.mark.parametrize(
    "test_case",
    [
        "inference:chat_completion:tool_calling",
    ],
)
def test_text_chat_completion_with_tool_choice_none(client_with_models, text_model_id, test_case):
    tc = TestCase(test_case)

    response = client_with_models.inference.chat_completion(
        model_id=text_model_id,
        messages=tc["messages"],
        tools=tc["tools"],
        tool_config={"tool_choice": "none"},
        stream=True,
    )
    tool_invocation_content = extract_tool_invocation_content(response)
    assert tool_invocation_content == ""


@pytest.mark.parametrize(
    "test_case",
    [
        "inference:chat_completion:structured_output",
    ],
)
def test_text_chat_completion_structured_output(client_with_models, text_model_id, test_case):
    skip_if_model_doesnt_support_json_schema_structured_output(client_with_models, text_model_id)

    class NBAStats(BaseModel):
        year_for_draft: int
        num_seasons_in_nba: int

    class AnswerFormat(BaseModel):
        first_name: str
        last_name: str
        year_of_birth: int
        nba_stats: NBAStats

    tc = TestCase(test_case)

    response = client_with_models.inference.chat_completion(
        model_id=text_model_id,
        messages=tc["messages"],
        response_format={
            "type": "json_schema",
            "json_schema": AnswerFormat.model_json_schema(),
        },
        stream=False,
    )
    answer = AnswerFormat.model_validate_json(response.completion_message.content)
    expected = tc["expected"]
    assert answer.first_name == expected["first_name"]
    assert answer.last_name == expected["last_name"]
    assert answer.year_of_birth == expected["year_of_birth"]
    assert answer.nba_stats.num_seasons_in_nba == expected["num_seasons_in_nba"]
    assert answer.nba_stats.year_for_draft == expected["year_for_draft"]


@pytest.mark.parametrize("streaming", [True, False])
@pytest.mark.parametrize(
    "test_case",
    [
        "inference:chat_completion:tool_calling_tools_absent",
    ],
)
def test_text_chat_completion_tool_calling_tools_not_in_request(
    client_with_models, text_model_id, test_case, streaming
):
    tc = TestCase(test_case)

    # TODO: more dynamic lookup on tool_prompt_format for model family
    tool_prompt_format = "json" if "3.1" in text_model_id else "python_list"
    request = {
        "model_id": text_model_id,
        "messages": tc["messages"],
        "tools": tc["tools"],
        "tool_choice": "auto",
        "tool_prompt_format": tool_prompt_format,
        "stream": streaming,
    }

    response = client_with_models.inference.chat_completion(**request)

    if streaming:
        for chunk in response:
            delta = chunk.event.delta
            if delta.type == "tool_call" and delta.parse_status == "succeeded":
                assert delta.tool_call.tool_name == "get_object_namespace_list"
            if delta.type == "tool_call" and delta.parse_status == "failed":
                # expect raw message that failed to parse in tool_call
                assert isinstance(delta.tool_call, str)
                assert len(delta.tool_call) > 0
    else:
        for tc in response.completion_message.tool_calls:
            assert tc.tool_name == "get_object_namespace_list"


@pytest.mark.parametrize(
    "test_case",
    [
        # Tests if the model can handle simple messages like "Hi" or
        # a message unrelated to one of the tool calls
        "inference:chat_completion:text_then_tool",
        # Tests if the model can do full tool call with responses correctly
        "inference:chat_completion:tool_then_answer",
        # Tests if model can generate multiple params and
        # read outputs correctly
        "inference:chat_completion:array_parameter",
    ],
)
def test_text_chat_completion_with_multi_turn_tool_calling(client_with_models, text_model_id, test_case):
    """This test tests the model's tool calling loop in various scenarios"""
    if "llama-4" not in text_model_id.lower() and "llama4" not in text_model_id.lower():
        pytest.xfail("Not tested for non-llama4 models yet")

    tc = TestCase(test_case)
    messages = []

    # keep going until either
    # 1. we have messages to test in multi-turn
    # 2. no messages bust last message is tool response
    while len(tc["messages"]) > 0 or (len(messages) > 0 and messages[-1]["role"] == "tool"):
        # do not take new messages if last message is tool response
        if len(messages) == 0 or messages[-1]["role"] != "tool":
            new_messages = tc["messages"].pop(0)
            messages += new_messages

        # pprint(messages)
        response = client_with_models.inference.chat_completion(
            model_id=text_model_id,
            messages=messages,
            tools=tc["tools"],
            stream=False,
            sampling_params={
                "strategy": {
                    "type": "top_p",
                    "top_p": 0.9,
                    "temperature": 0.6,
                }
            },
        )
        op_msg = response.completion_message
        messages.append(op_msg.model_dump())
        # print(op_msg)

        assert op_msg.role == "assistant"
        expected = tc["expected"].pop(0)
        assert len(op_msg.tool_calls) == expected["num_tool_calls"]

        if expected["num_tool_calls"] > 0:
            assert op_msg.tool_calls[0].tool_name == expected["tool_name"]
            assert op_msg.tool_calls[0].arguments == expected["tool_arguments"]

            tool_response = tc["tool_responses"].pop(0)
            messages.append(
                # Tool Response Message
                {
                    "role": "tool",
                    "call_id": op_msg.tool_calls[0].call_id,
                    "content": tool_response["response"],
                }
            )
        else:
            actual_answer = op_msg.content.lower()
            # pprint(actual_answer)
            assert expected["answer"] in actual_answer

        # sleep to avoid rate limit
        sleep(1)
