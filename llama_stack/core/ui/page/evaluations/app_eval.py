# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json

import pandas as pd
import streamlit as st

from llama_stack.core.ui.modules.api import llama_stack_api
from llama_stack.core.ui.modules.utils import process_dataset


def application_evaluation_page():
    st.set_page_config(page_title="Evaluations (Scoring)", page_icon="🦙")
    st.title("📊 Evaluations (Scoring)")

    # File uploader
    uploaded_file = st.file_uploader("Upload Dataset", type=["csv", "xlsx", "xls"])

    if uploaded_file is None:
        st.error("No file uploaded")
        return

    # Process uploaded file
    df = process_dataset(uploaded_file)
    if df is None:
        st.error("Error processing file")
        return

    # Display dataset information
    st.success("Dataset loaded successfully!")

    # Display dataframe preview
    st.subheader("Dataset Preview")
    st.dataframe(df)

    # Select Scoring Functions to Run Evaluation On
    st.subheader("Select Scoring Functions")
    scoring_functions = llama_stack_api.client.scoring_functions.list()
    scoring_functions = {sf.identifier: sf for sf in scoring_functions}
    scoring_functions_names = list(scoring_functions.keys())
    selected_scoring_functions = st.multiselect(
        "Choose one or more scoring functions",
        options=scoring_functions_names,
        help="Choose one or more scoring functions.",
    )

    available_models = llama_stack_api.client.models.list()
    available_models = [m.identifier for m in available_models]

    scoring_params = {}
    if selected_scoring_functions:
        st.write("Selected:")
        for scoring_fn_id in selected_scoring_functions:
            scoring_fn = scoring_functions[scoring_fn_id]
            st.write(f"- **{scoring_fn_id}**: {scoring_fn.description}")
            new_params = None
            if scoring_fn.params:
                new_params = {}
                for param_name, param_value in scoring_fn.params.to_dict().items():
                    if param_name == "type":
                        new_params[param_name] = param_value
                        continue

                    if param_name == "judge_model":
                        value = st.selectbox(
                            f"Select **{param_name}** for {scoring_fn_id}",
                            options=available_models,
                            index=0,
                            key=f"{scoring_fn_id}_{param_name}",
                        )
                        new_params[param_name] = value
                    else:
                        value = st.text_area(
                            f"Enter value for **{param_name}** in {scoring_fn_id} in valid JSON format",
                            value=json.dumps(param_value, indent=2),
                            height=80,
                        )
                        try:
                            new_params[param_name] = json.loads(value)
                        except json.JSONDecodeError:
                            st.error(f"Invalid JSON for **{param_name}** in {scoring_fn_id}")

                st.json(new_params)
            scoring_params[scoring_fn_id] = new_params

        # Add run evaluation button & slider
        total_rows = len(df)
        num_rows = st.slider("Number of rows to evaluate", 1, total_rows, total_rows)

        if st.button("Run Evaluation"):
            progress_text = "Running evaluation..."
            progress_bar = st.progress(0, text=progress_text)
            rows = df.to_dict(orient="records")
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
                score_res = llama_stack_api.run_scoring(
                    r,
                    scoring_function_ids=selected_scoring_functions,
                    scoring_params=scoring_params,
                )

                for k in r.keys():
                    if k not in output_res:
                        output_res[k] = []
                    output_res[k].append(r[k])

                for fn_id in selected_scoring_functions:
                    if fn_id not in output_res:
                        output_res[fn_id] = []
                    output_res[fn_id].append(score_res.results[fn_id].score_rows[0])

                # Display current row results using separate containers
                progress_text_container.write(f"Expand to see current processed result ({i + 1} / {len(rows)})")
                results_container.json(
                    score_res.to_json(),
                    expanded=2,
                )

            progress_bar.progress(1.0, text="Evaluation complete!")

            # Display results in dataframe
            if output_res:
                output_df = pd.DataFrame(output_res)
                st.subheader("Evaluation Results")
                st.dataframe(output_df)


application_evaluation_page()
