# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import uuid

import streamlit as st
from llama_stack_client import Agent, AgentEventLogger, RAGDocument

from llama_stack.apis.common.content_types import ToolCallDelta
from llama_stack.core.ui.modules.api import llama_stack_api
from llama_stack.core.ui.modules.utils import data_url_from_file


def rag_chat_page():
    st.title("🦙 RAG")

    def reset_agent_and_chat():
        st.session_state.clear()
        st.cache_resource.clear()

    def should_disable_input():
        return "displayed_messages" in st.session_state and len(st.session_state.displayed_messages) > 0

    def log_message(message):
        with st.chat_message(message["role"]):
            if "tool_output" in message and message["tool_output"]:
                with st.expander(label="Tool Output", expanded=False, icon="🛠"):
                    st.write(message["tool_output"])
            st.markdown(message["content"])

    with st.sidebar:
        # File/Directory Upload Section
        st.subheader("Upload Documents", divider=True)
        uploaded_files = st.file_uploader(
            "Upload file(s) or directory",
            accept_multiple_files=True,
            type=["txt", "pdf", "doc", "docx"],  # Add more file types as needed
        )
        # Process uploaded files
        if uploaded_files:
            st.success(f"Successfully uploaded {len(uploaded_files)} files")
            # Add memory bank name input field
            vector_db_name = st.text_input(
                "Document Collection Name",
                value="rag_vector_db",
                help="Enter a unique identifier for this document collection",
            )
            if st.button("Create Document Collection"):
                documents = [
                    RAGDocument(
                        document_id=uploaded_file.name,
                        content=data_url_from_file(uploaded_file),
                    )
                    for i, uploaded_file in enumerate(uploaded_files)
                ]

                providers = llama_stack_api.client.providers.list()
                vector_io_provider = None

                for x in providers:
                    if x.api == "vector_io":
                        vector_io_provider = x.provider_id

                llama_stack_api.client.vector_dbs.register(
                    vector_db_id=vector_db_name,  # Use the user-provided name
                    embedding_dimension=384,
                    embedding_model="all-MiniLM-L6-v2",
                    provider_id=vector_io_provider,
                )

                # insert documents using the custom vector db name
                llama_stack_api.client.tool_runtime.rag_tool.insert(
                    vector_db_id=vector_db_name,  # Use the user-provided name
                    documents=documents,
                    chunk_size_in_tokens=512,
                )
                st.success("Vector database created successfully!")

        st.subheader("RAG Parameters", divider=True)

        rag_mode = st.radio(
            "RAG mode",
            ["Direct", "Agent-based"],
            captions=[
                "RAG is performed by directly retrieving the information and augmenting the user query",
                "RAG is performed by an agent activating a dedicated knowledge search tool.",
            ],
            on_change=reset_agent_and_chat,
            disabled=should_disable_input(),
        )

        # select memory banks
        vector_dbs = llama_stack_api.client.vector_dbs.list()
        vector_dbs = [vector_db.identifier for vector_db in vector_dbs]
        selected_vector_dbs = st.multiselect(
            label="Select Document Collections to use in RAG queries",
            options=vector_dbs,
            on_change=reset_agent_and_chat,
            disabled=should_disable_input(),
        )

        st.subheader("Inference Parameters", divider=True)
        available_models = llama_stack_api.client.models.list()
        available_models = [model.identifier for model in available_models if model.model_type == "llm"]
        selected_model = st.selectbox(
            label="Choose a model",
            options=available_models,
            index=0,
            on_change=reset_agent_and_chat,
            disabled=should_disable_input(),
        )
        system_prompt = st.text_area(
            "System Prompt",
            value="You are a helpful assistant. ",
            help="Initial instructions given to the AI to set its behavior and context",
            on_change=reset_agent_and_chat,
            disabled=should_disable_input(),
        )
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.1,
            help="Controls the randomness of the response. Higher values make the output more creative and unexpected, lower values make it more conservative and predictable",
            on_change=reset_agent_and_chat,
            disabled=should_disable_input(),
        )

        top_p = st.slider(
            "Top P",
            min_value=0.0,
            max_value=1.0,
            value=0.95,
            step=0.1,
            on_change=reset_agent_and_chat,
            disabled=should_disable_input(),
        )

        # Add clear chat button to sidebar
        if st.button("Clear Chat", use_container_width=True):
            reset_agent_and_chat()
            st.rerun()

    # Chat Interface
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "displayed_messages" not in st.session_state:
        st.session_state.displayed_messages = []

    # Display chat history
    for message in st.session_state.displayed_messages:
        log_message(message)

    if temperature > 0.0:
        strategy = {
            "type": "top_p",
            "temperature": temperature,
            "top_p": top_p,
        }
    else:
        strategy = {"type": "greedy"}

    @st.cache_resource
    def create_agent():
        return Agent(
            llama_stack_api.client,
            model=selected_model,
            instructions=system_prompt,
            sampling_params={
                "strategy": strategy,
            },
            tools=[
                dict(
                    name="builtin::rag/knowledge_search",
                    args={
                        "vector_db_ids": list(selected_vector_dbs),
                    },
                )
            ],
        )

    if rag_mode == "Agent-based":
        agent = create_agent()
        if "agent_session_id" not in st.session_state:
            st.session_state["agent_session_id"] = agent.create_session(session_name=f"rag_demo_{uuid.uuid4()}")

        session_id = st.session_state["agent_session_id"]

    def agent_process_prompt(prompt):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Send the prompt to the agent
        response = agent.create_turn(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            session_id=session_id,
        )

        # Display assistant response
        with st.chat_message("assistant"):
            retrieval_message_placeholder = st.expander(label="Tool Output", expanded=False, icon="🛠")
            message_placeholder = st.empty()
            full_response = ""
            retrieval_response = ""
            for log in AgentEventLogger().log(response):
                log.print()
                if log.role == "tool_execution":
                    retrieval_response += log.content.replace("====", "").strip()
                    retrieval_message_placeholder.write(retrieval_response)
                else:
                    full_response += log.content
                    message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)

            st.session_state.messages.append({"role": "assistant", "content": full_response})
            st.session_state.displayed_messages.append(
                {"role": "assistant", "content": full_response, "tool_output": retrieval_response}
            )

    def direct_process_prompt(prompt):
        # Add the system prompt in the beginning of the conversation
        if len(st.session_state.messages) == 0:
            st.session_state.messages.append({"role": "system", "content": system_prompt})

        # Query the vector DB
        rag_response = llama_stack_api.client.tool_runtime.rag_tool.query(
            content=prompt, vector_db_ids=list(selected_vector_dbs)
        )
        prompt_context = rag_response.content

        with st.chat_message("assistant"):
            with st.expander(label="Retrieval Output", expanded=False):
                st.write(prompt_context)

            retrieval_message_placeholder = st.empty()
            message_placeholder = st.empty()
            full_response = ""
            retrieval_response = ""

            # Construct the extended prompt
            extended_prompt = f"Please answer the following query using the context below.\n\nCONTEXT:\n{prompt_context}\n\nQUERY:\n{prompt}"

            # Run inference directly
            st.session_state.messages.append({"role": "user", "content": extended_prompt})
            response = llama_stack_api.client.inference.chat_completion(
                messages=st.session_state.messages,
                model_id=selected_model,
                sampling_params={
                    "strategy": strategy,
                },
                stream=True,
            )

            # Display assistant response
            for chunk in response:
                response_delta = chunk.event.delta
                if isinstance(response_delta, ToolCallDelta):
                    retrieval_response += response_delta.tool_call.replace("====", "").strip()
                    retrieval_message_placeholder.info(retrieval_response)
                else:
                    full_response += chunk.event.delta.text
                    message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)

        response_dict = {"role": "assistant", "content": full_response, "stop_reason": "end_of_message"}
        st.session_state.messages.append(response_dict)
        st.session_state.displayed_messages.append(response_dict)

    # Chat input
    if prompt := st.chat_input("Ask a question about your documents"):
        # Add user message to chat history
        st.session_state.displayed_messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # store the prompt to process it after page refresh
        st.session_state.prompt = prompt

        # force page refresh to disable the settings widgets
        st.rerun()

    if "prompt" in st.session_state and st.session_state.prompt is not None:
        if rag_mode == "Agent-based":
            agent_process_prompt(st.session_state.prompt)
        else:  # rag_mode == "Direct"
            direct_process_prompt(st.session_state.prompt)
        st.session_state.prompt = None


rag_chat_page()
