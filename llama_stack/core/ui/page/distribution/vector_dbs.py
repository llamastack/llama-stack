# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import streamlit as st

from llama_stack.core.ui.modules.api import llama_stack_api


def vector_dbs():
    st.header("Vector Databases")
    vector_dbs_info = {v.identifier: v.to_dict() for v in llama_stack_api.client.vector_dbs.list()}

    if len(vector_dbs_info) > 0:
        selected_vector_db = st.selectbox("Select a vector database", list(vector_dbs_info.keys()))
        st.json(vector_dbs_info[selected_vector_db])
    else:
        st.info("No vector databases found")
