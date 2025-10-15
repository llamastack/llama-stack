# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any
from uuid import uuid4

import pytest
from llama_stack_client import AgentEventLogger
from llama_stack_client.lib.agents.agent import Agent
from llama_stack_client.lib.agents.turn_events import StepCompleted
from llama_stack_client.types.shared_params.agent_config import AgentConfig, ToolConfig

from llama_stack.apis.agents.agents import (
    AgentConfig as Server__AgentConfig,
)
from llama_stack.apis.agents.agents import (
    ToolChoice,
)


def text_message(content: str, *, role: str = "user") -> dict[str, Any]:
    return {
        "type": "message",
        "role": role,
        "content": [{"type": "input_text", "text": content}],
    }


def build_agent(client: Any, config: dict[str, Any], **overrides: Any) -> Agent:
    merged = {**config, **overrides}
    return Agent(
        client=client,
        model=merged["model"],
        instructions=merged["instructions"],
        tools=merged.get("tools"),
    )


def collect_turn(
    agent: Agent,
    session_id: str,
    messages: list[dict[str, Any]],
    *,
    extra_headers: dict[str, Any] | None = None,
):
    chunks = list(agent.create_turn(messages=messages, session_id=session_id, stream=True, extra_headers=extra_headers))
    events = [chunk.event for chunk in chunks]
    final_response = next((chunk.response for chunk in reversed(chunks) if chunk.response), None)
    if final_response is None:
        raise AssertionError("Turn did not yield a final response")
    return chunks, events, final_response


def get_boiling_point(liquid_name: str, celcius: bool = True) -> int:
    """
    Returns the boiling point of a liquid in Celcius or Fahrenheit.

    :param liquid_name: The name of the liquid
    :param celcius: Whether to return the boiling point in Celcius
    :return: The boiling point of the liquid in Celcius or Fahrenheit
    """
    if liquid_name.lower() == "polyjuice":
        if celcius:
            return -100
        else:
            return -212
    else:
        return -1


def get_boiling_point_with_metadata(liquid_name: str, celcius: bool = True) -> dict[str, Any]:
    """
    Returns the boiling point of a liquid in Celcius or Fahrenheit

    :param liquid_name: The name of the liquid
    :param celcius: Whether to return the boiling point in Celcius
    :return: The boiling point of the liquid in Celcius or Fahrenheit
    """
    if liquid_name.lower() == "polyjuice":
        if celcius:
            temp = -100
        else:
            temp = -212
    else:
        temp = -1
    return {"content": temp, "metadata": {"source": "https://www.google.com"}}


@pytest.fixture(scope="session")
def agent_config(llama_stack_client, text_model_id):
    available_shields = [shield.identifier for shield in llama_stack_client.shields.list()]
    available_shields = available_shields[:1]
    agent_config = dict(
        model=text_model_id,
        instructions="You are a helpful assistant",
        sampling_params={
            "strategy": {
                "type": "top_p",
                "temperature": 0.0001,
                "top_p": 0.9,
            },
            "max_tokens": 512,
        },
        tools=[],
        input_shields=available_shields,
        output_shields=available_shields,
        enable_session_persistence=False,
    )
    return agent_config


@pytest.fixture(scope="session")
def agent_config_without_safety(text_model_id):
    agent_config = dict(
        model=text_model_id,
        instructions="You are a helpful assistant",
        sampling_params={
            "strategy": {
                "type": "top_p",
                "temperature": 0.0001,
                "top_p": 0.9,
            },
            "max_tokens": 512,
        },
        tools=[],
        enable_session_persistence=False,
    )
    return agent_config


def test_agent_simple(llama_stack_client, agent_config):
    agent = build_agent(llama_stack_client, agent_config)
    session_id = agent.create_session(f"test-session-{uuid4()}")

    chunks, events, _ = collect_turn(
        agent,
        session_id,
        messages=[text_message("Give me a sentence that contains the word: hello")],
    )

    logs = [str(log) for log in AgentEventLogger().log(chunks) if log is not None]
    logs_str = "".join(logs)

    assert "hello" in logs_str.lower()

    if len(agent_config["input_shields"]) > 0:
        pytest.skip("Shield support not available in new Agent implementation")


def test_tool_config(agent_config):
    common_params = dict(
        model="meta-llama/Llama-3.2-3B-Instruct",
        instructions="You are a helpful assistant",
        sampling_params={
            "strategy": {
                "type": "top_p",
                "temperature": 1.0,
                "top_p": 0.9,
            },
            "max_tokens": 512,
        },
        toolgroups=[],
        enable_session_persistence=False,
    )
    agent_config = AgentConfig(
        **common_params,
    )
    Server__AgentConfig(**common_params)

    agent_config = AgentConfig(
        **common_params,
        tool_choice="auto",
    )
    server_config = Server__AgentConfig(**agent_config)
    assert server_config.tool_config.tool_choice == ToolChoice.auto

    agent_config = AgentConfig(
        **common_params,
        tool_choice="auto",
        tool_config=ToolConfig(
            tool_choice="auto",
        ),
    )
    server_config = Server__AgentConfig(**agent_config)
    assert server_config.tool_config.tool_choice == ToolChoice.auto

    agent_config = AgentConfig(
        **common_params,
        tool_config=ToolConfig(
            tool_choice="required",
        ),
    )
    server_config = Server__AgentConfig(**agent_config)
    assert server_config.tool_config.tool_choice == ToolChoice.required

    agent_config = AgentConfig(
        **common_params,
        tool_choice="required",
        tool_config=ToolConfig(
            tool_choice="auto",
        ),
    )
    with pytest.raises(ValueError, match="tool_choice is deprecated"):
        Server__AgentConfig(**agent_config)


def test_builtin_tool_web_search(llama_stack_client, agent_config):
    agent_config = {
        **agent_config,
        "instructions": "You are a helpful assistant that can use web search to answer questions.",
        "tools": [
            {"type": "web_search"},
        ],
    }
    agent = build_agent(llama_stack_client, agent_config)
    session_id = agent.create_session(f"test-session-{uuid4()}")

    _, events, _ = collect_turn(
        agent,
        session_id,
        messages=[text_message("Who are the latest board members to join Meta's board of directors?")],
    )

    found_tool_execution = False
    for event in events:
        if isinstance(event, StepCompleted) and event.step_type == "tool_execution":
            assert event.result.tool_calls[0].tool_name == "brave_search"
            found_tool_execution = True
            break
    assert found_tool_execution


@pytest.mark.skip(reason="Code interpreter is currently disabled in the Stack")
def test_builtin_tool_code_execution(llama_stack_client, agent_config):
    agent_config = {
        **agent_config,
        "tools": [
            "builtin::code_interpreter",
        ],
    }
    agent = build_agent(llama_stack_client, agent_config)
    session_id = agent.create_session(f"test-session-{uuid4()}")

    chunks, _, _ = collect_turn(
        agent,
        session_id,
        messages=[
            text_message("Write code and execute it to find the answer for: What is the 100th prime number?"),
        ],
    )
    logs = [str(log) for log in AgentEventLogger().log(chunks) if log is not None]
    logs_str = "".join(logs)

    assert "541" in logs_str
    assert "Tool:code_interpreter Response" in logs_str


# This test must be run in an environment where `bwrap` is available. If you are running against a
# server, this means the _server_ must have `bwrap` available. If you are using library client, then
# you must have `bwrap` available in test's environment.
def test_custom_tool(llama_stack_client, agent_config):
    client_tool = get_boiling_point
    agent_config = {
        **agent_config,
        "tools": [client_tool],
    }

    agent = build_agent(llama_stack_client, agent_config)
    session_id = agent.create_session(f"test-session-{uuid4()}")

    chunks, _, _ = collect_turn(
        agent,
        session_id,
        messages=[text_message("What is the boiling point of the liquid polyjuice in celsius?")],
    )

    logs = [str(log) for log in AgentEventLogger().log(chunks) if log is not None]
    logs_str = "".join(logs)
    assert "-100" in logs_str
    assert "get_boiling_point" in logs_str


def test_custom_tool_infinite_loop(llama_stack_client, agent_config):
    client_tool = get_boiling_point
    agent_config = {
        **agent_config,
        "instructions": "You are a helpful assistant Always respond with tool calls no matter what. ",
        "tools": [client_tool],
        "max_infer_iters": 5,
    }

    agent = build_agent(llama_stack_client, agent_config)
    session_id = agent.create_session(f"test-session-{uuid4()}")

    _, events, _ = collect_turn(
        agent,
        session_id,
        messages=[text_message("Get the boiling point of polyjuice with a tool call.")],
    )

    num_tool_calls = sum(
        1 for event in events if isinstance(event, StepCompleted) and event.step_type == "tool_execution"
    )
    assert num_tool_calls <= 5


def test_tool_choice_required(llama_stack_client, agent_config):
    tool_execution_steps = run_agent_with_tool_choice(llama_stack_client, agent_config, "required")
    assert len(tool_execution_steps) > 0


@pytest.mark.xfail(reason="Agent tool choice configuration not yet supported")
def test_tool_choice_none(llama_stack_client, agent_config):
    tool_execution_steps = run_agent_with_tool_choice(llama_stack_client, agent_config, "none")
    assert len(tool_execution_steps) == 0


def test_tool_choice_get_boiling_point(llama_stack_client, agent_config):
    if "llama" not in agent_config["model"].lower():
        pytest.xfail("NotImplemented for non-llama models")

    tool_execution_steps = run_agent_with_tool_choice(llama_stack_client, agent_config, "get_boiling_point")
    assert (
        len(tool_execution_steps) >= 1 and tool_execution_steps[0].result.tool_calls[0].tool_name == "get_boiling_point"
    )


def run_agent_with_tool_choice(client, agent_config, tool_choice):
    client_tool = get_boiling_point

    test_agent_config = {
        **agent_config,
        "tool_config": {"tool_choice": tool_choice},
        "tools": [client_tool],
        "max_infer_iters": 2,
    }

    agent = build_agent(client, test_agent_config)
    session_id = agent.create_session(f"test-session-{uuid4()}")

    _, events, _ = collect_turn(
        agent,
        session_id,
        messages=[text_message("What is the boiling point of the liquid polyjuice in celsius?")],
    )

    return [event for event in events if isinstance(event, StepCompleted) and event.step_type == "tool_execution"]


@pytest.mark.parametrize(
    "client_tools",
    [(get_boiling_point, False), (get_boiling_point_with_metadata, True)],
)
def test_create_turn_response(llama_stack_client, agent_config, client_tools):
    client_tool, expects_metadata = client_tools
    agent_config = {
        **agent_config,
        "input_shields": [],
        "output_shields": [],
        "tools": [client_tool],
    }

    agent = build_agent(llama_stack_client, agent_config)
    session_id = agent.create_session(f"test-session-{uuid4()}")

    input_prompt = f"Call {client_tools[0].__name__} tool and answer What is the boiling point of polyjuice?"
    _, events, final_response = collect_turn(
        agent,
        session_id,
        messages=[text_message(input_prompt)],
    )

    tool_events = [
        event for event in events if isinstance(event, StepCompleted) and event.step_type == "tool_execution"
    ]
    assert len(tool_events) >= 1
    tool_exec = tool_events[0]
    assert tool_exec.result.tool_calls[0].tool_name.startswith("get_boiling_point")
    if expects_metadata:
        assert tool_exec.result.tool_responses[0]["metadata"]["source"] == "https://www.google.com"

    inference_events = [
        event for event in events if isinstance(event, StepCompleted) and event.step_type == "inference"
    ]
    assert len(inference_events) >= 2
    assert "polyjuice" in final_response.output_text.lower()


def test_multi_tool_calls(llama_stack_client, agent_config):
    if "gpt" not in agent_config["model"] and "llama-4" not in agent_config["model"].lower():
        pytest.xfail("Only tested on GPT and Llama 4 models")

    agent_config = {
        **agent_config,
        "tools": [get_boiling_point],
    }

    agent = build_agent(llama_stack_client, agent_config)
    session_id = agent.create_session(f"test-session-{uuid4()}")

    _, events, final_response = collect_turn(
        agent,
        session_id,
        messages=[
            text_message(
                "Call get_boiling_point twice to answer: What is the boiling point of polyjuice in both celsius and fahrenheit?.\nUse the tool responses to answer the question."
            )
        ],
    )

    tool_exec_events = [
        event for event in events if isinstance(event, StepCompleted) and event.step_type == "tool_execution"
    ]
    assert len(tool_exec_events) >= 1
    tool_exec = tool_exec_events[0]
    assert len(tool_exec.result.tool_calls) == 2
    assert tool_exec.result.tool_calls[0].tool_name.startswith("get_boiling_point")
    assert tool_exec.result.tool_calls[1].tool_name.startswith("get_boiling_point")

    output = final_response.output_text.lower()
    assert "-100" in output and "-212" in output
