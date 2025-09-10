# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import streamlit as st

from llama_stack.core.ui.modules.api import llama_stack_api


def benchmarks():
    # Benchmarks Section
    st.header("Benchmarks")

    benchmarks_info = {d.identifier: d.to_dict() for d in llama_stack_api.client.benchmarks.list()}

    if len(benchmarks_info) > 0:
        selected_benchmark = st.selectbox("Select an eval task", list(benchmarks_info.keys()), key="benchmark_inspect")
        st.json(benchmarks_info[selected_benchmark], expanded=True)
