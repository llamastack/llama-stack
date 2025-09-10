# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import streamlit as st

from llama_stack.core.ui.modules.api import llama_stack_api


def shields():
    # Shields Section
    st.header("Shields")

    shields_info = {s.identifier: s.to_dict() for s in llama_stack_api.client.shields.list()}

    selected_shield = st.selectbox("Select a shield", list(shields_info.keys()))
    st.json(shields_info[selected_shield])
