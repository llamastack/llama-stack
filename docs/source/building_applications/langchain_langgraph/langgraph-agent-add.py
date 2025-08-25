# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, ToolMessage
from langchain.agents import tool
from langchain_openai import ChatOpenAI
from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages

# --- Tool ---
@tool
def add_numbers(x: int, y: int) -> int:
    """Add two integers together."""
    return x + y

tools = [add_numbers]

# --- LLM that supports function-calling ---
llm = ChatOpenAI(
    model="ollama/llama3.2:3b-instruct-fp16",
    openai_api_key="none",
    openai_api_base="http://localhost:8321/v1/openai/v1"
).bind_tools(tools)

# --- Node that runs the agent ---
def agent_node(state):
    messages = state["messages"]
    if "scratchpad" in state:
        messages += format_to_openai_tool_messages(state["scratchpad"])
    response = llm.invoke(messages)
    print(f"LLM returned Chat Completion object: {response.response_metadata}")
    return {
        "messages": messages + [response],
        "intermediate_step": response,
    }

# --- Node that executes tool call ---
def tool_node(state):
    tool_call = state["intermediate_step"].tool_calls[0]
    result = add_numbers.invoke(tool_call["args"])
    return {
        "messages": state["messages"] + [
            ToolMessage(tool_call_id=tool_call["id"], content=str(result))
        ]
    }

# --- Build LangGraph ---
graph = StateGraph(dict)
graph.add_node("agent", agent_node)
graph.add_node("tool", tool_node)

graph.set_entry_point("agent")
graph.add_edge("agent", "tool")
graph.add_edge("tool", END)

compiled_graph = graph.compile()

# --- Run it ---
initial_state = {
    "messages": [HumanMessage(content="What is 16 plus 9?")]
}

final_state = compiled_graph.invoke(initial_state)

# --- Output ---
for msg in final_state["messages"]:
    print(f"{msg.type.upper()}: {msg.content}")
