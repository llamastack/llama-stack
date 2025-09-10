# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json

import pandas as pd
import streamlit as st

from llama_stack.core.ui.modules.api import llama_stack_api


def select_benchmark_1():
    # Select Benchmarks
    st.subheader("1. Choose An Eval Task")
    benchmarks = llama_stack_api.client.benchmarks.list()
    benchmarks = {et.identifier: et for et in benchmarks}
    benchmarks_names = list(benchmarks.keys())
    selected_benchmark = st.selectbox(
        "Choose an eval task.",
        options=benchmarks_names,
        help="Choose an eval task. Each eval task is parameterized by a dataset, and list of scoring functions.",
    )
    with st.expander("View Eval Task"):
        st.json(benchmarks[selected_benchmark], expanded=True)

    st.session_state["selected_benchmark"] = selected_benchmark
    st.session_state["benchmarks"] = benchmarks
    if st.button("Confirm", key="confirm_1"):
        st.session_state["selected_benchmark_1_next"] = True


def define_eval_candidate_2():
    if not st.session_state.get("selected_benchmark_1_next", None):
        return

    st.subheader("2. Define Eval Candidate")
    st.info(
        """
        Define the configurations for the evaluation candidate model or agent used for generation.
        Select "model" if you want to run generation with inference API, or "agent" if you want to run generation with agent API through specifying AgentConfig.
        """
    )
    with st.expander("Define Eval Candidate", expanded=True):
        # Define Eval Candidate
        candidate_type = st.radio("Candidate Type", ["model", "agent"])

        available_models = llama_stack_api.client.models.list()
        available_models = [model.identifier for model in available_models]
        selected_model = st.selectbox(
            "Choose a model",
            available_models,
            index=0,
        )

        # Sampling Parameters
        st.markdown("##### Sampling Parameters")
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
        if candidate_type == "model":
            if temperature > 0.0:
                strategy = {
                    "type": "top_p",
                    "temperature": temperature,
                    "top_p": top_p,
                }
            else:
                strategy = {"type": "greedy"}

            eval_candidate = {
                "type": "model",
                "model": selected_model,
                "sampling_params": {
                    "strategy": strategy,
                    "max_tokens": max_tokens,
                    "repetition_penalty": repetition_penalty,
                },
            }
        elif candidate_type == "agent":
            system_prompt = st.text_area(
                "System Prompt",
                value="You are a helpful AI assistant.",
                help="Initial instructions given to the AI to set its behavior and context",
            )
            tools_json = st.text_area(
                "Tools Configuration (JSON)",
                value=json.dumps(
                    [
                        {
                            "type": "brave_search",
                            "engine": "brave",
                            "api_key": "ENTER_BRAVE_API_KEY_HERE",
                        }
                    ]
                ),
                help="Enter tool configurations in JSON format. Each tool should have a name, description, and parameters.",
                height=200,
            )
            try:
                tools = json.loads(tools_json)
            except json.JSONDecodeError:
                st.error("Invalid JSON format for tools configuration")
                tools = []
            eval_candidate = {
                "type": "agent",
                "config": {
                    "model": selected_model,
                    "instructions": system_prompt,
                    "tools": tools,
                    "tool_choice": "auto",
                    "tool_prompt_format": "json",
                    "input_shields": [],
                    "output_shields": [],
                    "enable_session_persistence": False,
                },
            }
        st.session_state["eval_candidate"] = eval_candidate

    if st.button("Confirm", key="confirm_2"):
        st.session_state["selected_eval_candidate_2_next"] = True


def run_evaluation_3():
    if not st.session_state.get("selected_eval_candidate_2_next", None):
        return

    st.subheader("3. Run Evaluation")
    # Add info box to explain configurations being used
    st.info(
        """
        Review the configurations that will be used for this evaluation run, make any necessary changes, and then click the "Run Evaluation" button.
        """
    )
    selected_benchmark = st.session_state["selected_benchmark"]
    benchmarks = st.session_state["benchmarks"]
    eval_candidate = st.session_state["eval_candidate"]

    dataset_id = benchmarks[selected_benchmark].dataset_id
    rows = llama_stack_api.client.datasets.iterrows(
        dataset_id=dataset_id,
    )
    total_rows = len(rows.data)
    # Add number of examples control
    num_rows = st.number_input(
        "Number of Examples to Evaluate",
        min_value=1,
        max_value=total_rows,
        value=5,
        help="Number of examples from the dataset to evaluate. ",
    )

    benchmark_config = {
        "type": "benchmark",
        "eval_candidate": eval_candidate,
        "scoring_params": {},
    }

    with st.expander("View Evaluation Task", expanded=True):
        st.json(benchmarks[selected_benchmark], expanded=True)
    with st.expander("View Evaluation Task Configuration", expanded=True):
        st.json(benchmark_config, expanded=True)

    # Add run button and handle evaluation
    if st.button("Run Evaluation"):
        progress_text = "Running evaluation..."
        progress_bar = st.progress(0, text=progress_text)
        rows = rows.data
        if num_rows < total_rows:
            rows = rows[:num_rows]

        # Create separate containers for progress text and results
        progress_text_container = st.empty()
        results_container = st.empty()
        output_res = {}
        for i, r in enumerate(rows):
            # Update progress
            progress = i / len(rows)
            progress_bar.progress(progress, text=progress_text)
            # Run evaluation for current row
            eval_res = llama_stack_api.client.eval.evaluate_rows(
                benchmark_id=selected_benchmark,
                input_rows=[r],
                scoring_functions=benchmarks[selected_benchmark].scoring_functions,
                benchmark_config=benchmark_config,
            )

            for k in r.keys():
                if k not in output_res:
                    output_res[k] = []
                output_res[k].append(r[k])

            for k in eval_res.generations[0].keys():
                if k not in output_res:
                    output_res[k] = []
                output_res[k].append(eval_res.generations[0][k])

            for scoring_fn in benchmarks[selected_benchmark].scoring_functions:
                if scoring_fn not in output_res:
                    output_res[scoring_fn] = []
                output_res[scoring_fn].append(eval_res.scores[scoring_fn].score_rows[0])

            progress_text_container.write(f"Expand to see current processed result ({i + 1} / {len(rows)})")
            results_container.json(eval_res, expanded=2)

        progress_bar.progress(1.0, text="Evaluation complete!")
        # Display results in dataframe
        if output_res:
            output_df = pd.DataFrame(output_res)
            st.subheader("Evaluation Results")
            st.dataframe(output_df)


def native_evaluation_page():
    st.set_page_config(page_title="Evaluations (Generation + Scoring)", page_icon="🦙")
    st.title("📊 Evaluations (Generation + Scoring)")

    select_benchmark_1()
    define_eval_candidate_2()
    run_evaluation_3()


native_evaluation_page()
