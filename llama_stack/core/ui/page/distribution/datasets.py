# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import streamlit as st

from llama_stack.core.ui.modules.api import llama_stack_api


def datasets():
    st.header("Datasets")

    datasets_info = {d.identifier: d.to_dict() for d in llama_stack_api.client.datasets.list()}
    if len(datasets_info) > 0:
        selected_dataset = st.selectbox("Select a dataset", list(datasets_info.keys()))
        st.json(datasets_info[selected_dataset], expanded=True)
