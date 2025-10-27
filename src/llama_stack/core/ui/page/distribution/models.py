# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import streamlit as st

from llama_stack.core.ui.modules.api import llama_stack_api


def models():
    # Models Section
    st.header("Models")
    models_info = {m.identifier: m.to_dict() for m in llama_stack_api.client.models.list()}

    selected_model = st.selectbox("Select a model", list(models_info.keys()))
    st.json(models_info[selected_model])
