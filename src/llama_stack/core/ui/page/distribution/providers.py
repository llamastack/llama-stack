# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import streamlit as st

from llama_stack.core.ui.modules.api import llama_stack_api


def providers():
    st.header("🔍 API Providers")
    apis_providers_lst = llama_stack_api.client.providers.list()
    api_to_providers = {}
    for api_provider in apis_providers_lst:
        if api_provider.api in api_to_providers:
            api_to_providers[api_provider.api].append(api_provider)
        else:
            api_to_providers[api_provider.api] = [api_provider]

    for api in api_to_providers.keys():
        st.markdown(f"###### {api}")
        st.dataframe([x.to_dict() for x in api_to_providers[api]], width=500)


providers()
