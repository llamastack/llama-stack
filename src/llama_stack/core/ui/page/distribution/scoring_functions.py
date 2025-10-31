# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import streamlit as st

from llama_stack.core.ui.modules.api import llama_stack_api


def scoring_functions():
    st.header("Scoring Functions")

    scoring_functions_info = {s.identifier: s.to_dict() for s in llama_stack_api.client.scoring_functions.list()}

    selected_scoring_function = st.selectbox("Select a scoring function", list(scoring_functions_info.keys()))
    st.json(scoring_functions_info[selected_scoring_function], expanded=True)
