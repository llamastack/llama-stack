# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import streamlit as st

from llama_stack.core.ui.modules.api import llama_stack_api

# Sidebar configurations
with st.sidebar:
    st.header("Configuration")
    available_models = llama_stack_api.client.models.list()
    available_models = [model.identifier for model in available_models if model.model_type == "llm"]
    selected_model = st.selectbox(
        "Choose a model",
        available_models,
        index=0,
    )

    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.1,
        help="Controls the randomness of the response. Higher values make the output more creative and unexpected, lower values make it more conservative and predictable",
    )

    top_p = st.slider(
        "Top P",
        min_value=0.0,
        max_value=1.0,
        value=0.95,
        step=0.1,
    )

    max_tokens = st.slider(
        "Max Tokens",
        min_value=0,
        max_value=4096,
        value=512,
        step=1,
        help="The maximum number of tokens to generate",
    )

    repetition_penalty = st.slider(
        "Repetition Penalty",
        min_value=1.0,
        max_value=2.0,
        value=1.0,
        step=0.1,
        help="Controls the likelihood for generating the same word or phrase multiple times in the same sentence or paragraph. 1 implies no penalty, 2 will strongly discourage model to repeat words or phrases.",
    )

    stream = st.checkbox("Stream", value=True)
    system_prompt = st.text_area(
        "System Prompt",
        value="You are a helpful AI assistant.",
        help="Initial instructions given to the AI to set its behavior and context",
    )

    # Add clear chat button to sidebar
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# Main chat interface
st.title("🦙 Chat")


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Example: What is Llama Stack?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        if temperature > 0.0:
            strategy = {
                "type": "top_p",
                "temperature": temperature,
                "top_p": top_p,
            }
        else:
            strategy = {"type": "greedy"}

        response = llama_stack_api.client.inference.chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            model_id=selected_model,
            stream=stream,
            sampling_params={
                "strategy": strategy,
                "max_tokens": max_tokens,
                "repetition_penalty": repetition_penalty,
            },
        )

        if stream:
            for chunk in response:
                if chunk.event.event_type == "progress":
                    full_response += chunk.event.delta.text
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        else:
            full_response = response.completion_message.content
            message_placeholder.markdown(full_response)

        st.session_state.messages.append({"role": "assistant", "content": full_response})
