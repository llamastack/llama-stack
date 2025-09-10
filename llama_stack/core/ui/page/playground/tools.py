# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import enum
import json
import uuid

import streamlit as st
from llama_stack_client import Agent
from llama_stack_client.lib.agents.react.agent import ReActAgent
from llama_stack_client.lib.agents.react.tool_parser import ReActOutput

from llama_stack.core.ui.modules.api import llama_stack_api


class AgentType(enum.Enum):
    REGULAR = "Regular"
    REACT = "ReAct"


def tool_chat_page():
    st.title("🛠 Tools")

    client = llama_stack_api.client
    models = client.models.list()
    model_list = [model.identifier for model in models if model.api_model_type == "llm"]

    tool_groups = client.toolgroups.list()
    tool_groups_list = [tool_group.identifier for tool_group in tool_groups]
    mcp_tools_list = [tool for tool in tool_groups_list if tool.startswith("mcp::")]
    builtin_tools_list = [tool for tool in tool_groups_list if not tool.startswith("mcp::")]
    selected_vector_dbs = []

    def reset_agent():
        st.session_state.clear()
        st.cache_resource.clear()

    with st.sidebar:
        st.title("Configuration")
        st.subheader("Model")
        model = st.selectbox(label="Model", options=model_list, on_change=reset_agent, label_visibility="collapsed")

        st.subheader("Available ToolGroups")

        toolgroup_selection = st.pills(
            label="Built-in tools",
            options=builtin_tools_list,
            selection_mode="multi",
            on_change=reset_agent,
            format_func=lambda tool: "".join(tool.split("::")[1:]),
            help="List of built-in tools from your llama stack server.",
        )

        if "builtin::rag" in toolgroup_selection:
            vector_dbs = llama_stack_api.client.vector_dbs.list() or []
            if not vector_dbs:
                st.info("No vector databases available for selection.")
            vector_dbs = [vector_db.identifier for vector_db in vector_dbs]
            selected_vector_dbs = st.multiselect(
                label="Select Document Collections to use in RAG queries",
                options=vector_dbs,
                on_change=reset_agent,
            )

        mcp_selection = st.pills(
            label="MCP Servers",
            options=mcp_tools_list,
            selection_mode="multi",
            on_change=reset_agent,
            format_func=lambda tool: "".join(tool.split("::")[1:]),
            help="List of MCP servers registered to your llama stack server.",
        )

        toolgroup_selection.extend(mcp_selection)

        grouped_tools = {}
        total_tools = 0

        for toolgroup_id in toolgroup_selection:
            tools = client.tools.list(toolgroup_id=toolgroup_id)
            grouped_tools[toolgroup_id] = [tool.identifier for tool in tools]
            total_tools += len(tools)

        st.markdown(f"Active Tools: 🛠 {total_tools}")

        for group_id, tools in grouped_tools.items():
            with st.expander(f"🔧 Tools from `{group_id}`"):
                for idx, tool in enumerate(tools, start=1):
                    st.markdown(f"{idx}. `{tool.split(':')[-1]}`")

        st.subheader("Agent Configurations")
        st.subheader("Agent Type")
        agent_type = st.radio(
            label="Select Agent Type",
            options=["Regular", "ReAct"],
            on_change=reset_agent,
        )

        if agent_type == "ReAct":
            agent_type = AgentType.REACT
        else:
            agent_type = AgentType.REGULAR

        max_tokens = st.slider(
            "Max Tokens",
            min_value=0,
            max_value=4096,
            value=512,
            step=64,
            help="The maximum number of tokens to generate",
            on_change=reset_agent,
        )

    for i, tool_name in enumerate(toolgroup_selection):
        if tool_name == "builtin::rag":
            tool_dict = dict(
                name="builtin::rag",
                args={
                    "vector_db_ids": list(selected_vector_dbs),
                },
            )
            toolgroup_selection[i] = tool_dict

    @st.cache_resource
    def create_agent():
        if "agent_type" in st.session_state and st.session_state.agent_type == AgentType.REACT:
            return ReActAgent(
                client=client,
                model=model,
                tools=toolgroup_selection,
                response_format={
                    "type": "json_schema",
                    "json_schema": ReActOutput.model_json_schema(),
                },
                sampling_params={"strategy": {"type": "greedy"}, "max_tokens": max_tokens},
            )
        else:
            return Agent(
                client,
                model=model,
                instructions="You are a helpful assistant. When you use a tool always respond with a summary of the result.",
                tools=toolgroup_selection,
                sampling_params={"strategy": {"type": "greedy"}, "max_tokens": max_tokens},
            )

    st.session_state.agent_type = agent_type

    agent = create_agent()

    if "agent_session_id" not in st.session_state:
        st.session_state["agent_session_id"] = agent.create_session(session_name=f"tool_demo_{uuid.uuid4()}")

    session_id = st.session_state["agent_session_id"]

    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input(placeholder=""):
        with st.chat_message("user"):
            st.markdown(prompt)

        st.session_state.messages.append({"role": "user", "content": prompt})

        turn_response = agent.create_turn(
            session_id=session_id,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )

        def response_generator(turn_response):
            if st.session_state.get("agent_type") == AgentType.REACT:
                return _handle_react_response(turn_response)
            else:
                return _handle_regular_response(turn_response)

        def _handle_react_response(turn_response):
            current_step_content = ""
            final_answer = None
            tool_results = []

            for response in turn_response:
                if not hasattr(response.event, "payload"):
                    yield (
                        "\n\n🚨 :red[_Llama Stack server Error:_]\n"
                        "The response received is missing an expected `payload` attribute.\n"
                        "This could indicate a malformed response or an internal issue within the server.\n\n"
                        f"Error details: {response}"
                    )
                    return

                payload = response.event.payload

                if payload.event_type == "step_progress" and hasattr(payload.delta, "text"):
                    current_step_content += payload.delta.text
                    continue

                if payload.event_type == "step_complete":
                    step_details = payload.step_details

                    if step_details.step_type == "inference":
                        yield from _process_inference_step(current_step_content, tool_results, final_answer)
                        current_step_content = ""
                    elif step_details.step_type == "tool_execution":
                        tool_results = _process_tool_execution(step_details, tool_results)
                        current_step_content = ""
                    else:
                        current_step_content = ""

            if not final_answer and tool_results:
                yield from _format_tool_results_summary(tool_results)

        def _process_inference_step(current_step_content, tool_results, final_answer):
            try:
                react_output_data = json.loads(current_step_content)
                thought = react_output_data.get("thought")
                action = react_output_data.get("action")
                answer = react_output_data.get("answer")

                if answer and answer != "null" and answer is not None:
                    final_answer = answer

                if thought:
                    with st.expander("🤔 Thinking...", expanded=False):
                        st.markdown(f":grey[__{thought}__]")

                if action and isinstance(action, dict):
                    tool_name = action.get("tool_name")
                    tool_params = action.get("tool_params")
                    with st.expander(f'🛠 Action: Using tool "{tool_name}"', expanded=False):
                        st.json(tool_params)

                if answer and answer != "null" and answer is not None:
                    yield f"\n\n✅ **Final Answer:**\n{answer}"

            except json.JSONDecodeError:
                yield f"\n\nFailed to parse ReAct step content:\n```json\n{current_step_content}\n```"
            except Exception as e:
                yield f"\n\nFailed to process ReAct step: {e}\n```json\n{current_step_content}\n```"

            return final_answer

        def _process_tool_execution(step_details, tool_results):
            try:
                if hasattr(step_details, "tool_responses") and step_details.tool_responses:
                    for tool_response in step_details.tool_responses:
                        tool_name = tool_response.tool_name
                        content = tool_response.content
                        tool_results.append((tool_name, content))
                        with st.expander(f'⚙️ Observation (Result from "{tool_name}")', expanded=False):
                            try:
                                parsed_content = json.loads(content)
                                st.json(parsed_content)
                            except json.JSONDecodeError:
                                st.code(content, language=None)
                else:
                    with st.expander("⚙️ Observation", expanded=False):
                        st.markdown(":grey[_Tool execution step completed, but no response data found._]")
            except Exception as e:
                with st.expander("⚙️ Error in Tool Execution", expanded=False):
                    st.markdown(f":red[_Error processing tool execution: {str(e)}_]")

            return tool_results

        def _format_tool_results_summary(tool_results):
            yield "\n\n**Here's what I found:**\n"
            for tool_name, content in tool_results:
                try:
                    parsed_content = json.loads(content)

                    if tool_name == "web_search" and "top_k" in parsed_content:
                        yield from _format_web_search_results(parsed_content)
                    elif "results" in parsed_content and isinstance(parsed_content["results"], list):
                        yield from _format_results_list(parsed_content["results"])
                    elif isinstance(parsed_content, dict) and len(parsed_content) > 0:
                        yield from _format_dict_results(parsed_content)
                    elif isinstance(parsed_content, list) and len(parsed_content) > 0:
                        yield from _format_list_results(parsed_content)
                except json.JSONDecodeError:
                    yield f"\n**{tool_name}** was used but returned complex data. Check the observation for details.\n"
                except (TypeError, AttributeError, KeyError, IndexError) as e:
                    print(f"Error processing {tool_name} result: {type(e).__name__}: {e}")

        def _format_web_search_results(parsed_content):
            for i, result in enumerate(parsed_content["top_k"], 1):
                if i <= 3:
                    title = result.get("title", "Untitled")
                    url = result.get("url", "")
                    content_text = result.get("content", "").strip()
                    yield f"\n- **{title}**\n  {content_text}\n  [Source]({url})\n"

        def _format_results_list(results):
            for i, result in enumerate(results, 1):
                if i <= 3:
                    if isinstance(result, dict):
                        name = result.get("name", result.get("title", "Result " + str(i)))
                        description = result.get("description", result.get("content", result.get("summary", "")))
                        yield f"\n- **{name}**\n  {description}\n"
                    else:
                        yield f"\n- {result}\n"

        def _format_dict_results(parsed_content):
            yield "\n```\n"
            for key, value in list(parsed_content.items())[:5]:
                if isinstance(value, str) and len(value) < 100:
                    yield f"{key}: {value}\n"
                else:
                    yield f"{key}: [Complex data]\n"
            yield "```\n"

        def _format_list_results(parsed_content):
            yield "\n"
            for _, item in enumerate(parsed_content[:3], 1):
                if isinstance(item, str):
                    yield f"- {item}\n"
                elif isinstance(item, dict) and "text" in item:
                    yield f"- {item['text']}\n"
                elif isinstance(item, dict) and len(item) > 0:
                    first_value = next(iter(item.values()))
                    if isinstance(first_value, str) and len(first_value) < 100:
                        yield f"- {first_value}\n"

        def _handle_regular_response(turn_response):
            for response in turn_response:
                if hasattr(response.event, "payload"):
                    print(response.event.payload)
                    if response.event.payload.event_type == "step_progress":
                        if hasattr(response.event.payload.delta, "text"):
                            yield response.event.payload.delta.text
                    if response.event.payload.event_type == "step_complete":
                        if response.event.payload.step_details.step_type == "tool_execution":
                            if response.event.payload.step_details.tool_calls:
                                tool_name = str(response.event.payload.step_details.tool_calls[0].tool_name)
                                yield f'\n\n🛠 :grey[_Using "{tool_name}" tool:_]\n\n'
                            else:
                                yield "No tool_calls present in step_details"
                else:
                    yield f"Error occurred in the Llama Stack Cluster: {response}"

        with st.chat_message("assistant"):
            response_content = st.write_stream(response_generator(turn_response))

        st.session_state.messages.append({"role": "assistant", "content": response_content})


tool_chat_page()
