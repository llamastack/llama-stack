# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import pytest

from llama_stack.apis.agents import AgentConfig, Turn
from llama_stack.apis.inference import SamplingParams, UserMessage
from llama_stack.providers.datatypes import Api
from llama_stack.providers.utils.kvstore import kvstore_impl
from llama_stack.providers.utils.kvstore.config import SqliteKVStoreConfig


@pytest.fixture
def sample_messages():
    return [
        UserMessage(content="What's the weather like today?"),
    ]


def pick_inference_model(inference_model):
    return inference_model


def create_agent_session(agents_impl, agent_config):
    return agents_impl.create_agent_session(agent_config)


@pytest.fixture
def common_params(inference_model):
    inference_model = pick_inference_model(inference_model)

    return dict(
        model=inference_model,
        instructions="You are a helpful assistant.",
        enable_session_persistence=True,
        sampling_params=SamplingParams(temperature=0.7, top_p=0.95),
        input_shields=[],
        output_shields=[],
        tools=[],
        max_infer_iters=5,
    )


@pytest.mark.skip(reason="This test needs to be migrated to api / client-sdk world")
async def test_delete_agents_and_sessions(self, agents_stack, common_params):
    agents_impl = agents_stack.impls[Api.agents]
    agent_id, session_id = await create_agent_session(
        agents_impl,
        AgentConfig(
            **{
                **common_params,
                "input_shields": [],
                "output_shields": [],
            }
        ),
    )

    run_config = agents_stack.run_config
    provider_config = run_config.providers["agents"][0].config
    persistence_store = await kvstore_impl(SqliteKVStoreConfig(**provider_config["persistence_store"]))

    await agents_impl.delete_agents_session(agent_id, session_id)
    session_response = await persistence_store.get(f"session:{agent_id}:{session_id}")

    await agents_impl.delete_agents(agent_id)
    agent_response = await persistence_store.get(f"agent:{agent_id}")

    assert session_response is None
    assert agent_response is None


@pytest.mark.skip(reason="This test needs to be migrated to api / client-sdk world")
async def test_get_agent_turns_and_steps(self, agents_stack, sample_messages, common_params):
    agents_impl = agents_stack.impls[Api.agents]

    agent_id, session_id = await create_agent_session(
        agents_impl,
        AgentConfig(
            **{
                **common_params,
                "input_shields": [],
                "output_shields": [],
            }
        ),
    )

    # Create and execute a turn
    turn_request = dict(
        agent_id=agent_id,
        session_id=session_id,
        messages=sample_messages,
        stream=True,
    )

    turn_response = [chunk async for chunk in await agents_impl.create_agent_turn(**turn_request)]

    final_event = turn_response[-1].event.payload
    turn_id = final_event.turn.turn_id

    provider_config = agents_stack.run_config.providers["agents"][0].config
    persistence_store = await kvstore_impl(SqliteKVStoreConfig(**provider_config["persistence_store"]))
    turn = await persistence_store.get(f"session:{agent_id}:{session_id}:{turn_id}")
    response = await agents_impl.get_agents_turn(agent_id, session_id, turn_id)

    assert isinstance(response, Turn)
    assert response == final_event.turn
    assert turn == final_event.turn.model_dump_json()

    steps = final_event.turn.steps
    step_id = steps[0].step_id
    step_response = await agents_impl.get_agents_step(agent_id, session_id, turn_id, step_id)

    assert step_response.step == steps[0]
