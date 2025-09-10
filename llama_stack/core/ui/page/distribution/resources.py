# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from streamlit_option_menu import option_menu

from llama_stack.core.ui.page.distribution.datasets import datasets
from llama_stack.core.ui.page.distribution.eval_tasks import benchmarks
from llama_stack.core.ui.page.distribution.models import models
from llama_stack.core.ui.page.distribution.scoring_functions import scoring_functions
from llama_stack.core.ui.page.distribution.shields import shields
from llama_stack.core.ui.page.distribution.vector_dbs import vector_dbs


def resources_page():
    options = [
        "Models",
        "Vector Databases",
        "Shields",
        "Scoring Functions",
        "Datasets",
        "Benchmarks",
    ]
    icons = ["magic", "memory", "shield", "file-bar-graph", "database", "list-task"]
    selected_resource = option_menu(
        None,
        options,
        icons=icons,
        orientation="horizontal",
        styles={
            "nav-link": {
                "font-size": "12px",
            },
        },
    )
    if selected_resource == "Benchmarks":
        benchmarks()
    elif selected_resource == "Vector Databases":
        vector_dbs()
    elif selected_resource == "Datasets":
        datasets()
    elif selected_resource == "Models":
        models()
    elif selected_resource == "Scoring Functions":
        scoring_functions()
    elif selected_resource == "Shields":
        shields()


resources_page()
