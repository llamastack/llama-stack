# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import json
from datetime import datetime
from pathlib import Path

import requests
from docutils import nodes

# Read version from pyproject.toml
with Path(__file__).parent.parent.parent.joinpath("pyproject.toml").open("rb") as f:
    pypi_url = "https://pypi.org/pypi/llama-stack/json"
    headers = {
        'User-Agent': 'pip/23.0.1 (python 3.11)',  # Mimic pip's user agent
        'Accept': 'application/json'
    }
    version_tag = json.loads(requests.get(pypi_url, headers=headers).text)["info"]["version"]
    print(f"{version_tag=}")

    # generate the full link including text and url here
    llama_stack_version_url = (
        f"https://github.com/meta-llama/llama-stack/releases/tag/v{version_tag}"
    )
    llama_stack_version_link = f"<a href='{llama_stack_version_url}'>release notes</a>"

project = "llama-stack"
copyright = f"{datetime.now().year}, Meta"
author = "Meta"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_rtd_theme",
    "sphinx_rtd_dark_mode",
    "sphinx_tabs.tabs",
    "sphinxcontrib.redoc",
    "sphinxcontrib.mermaid",
    "sphinxcontrib.video",
    "sphinx_reredirects"
]

redirects = {
    "providers/post_training/index": "../../advanced_apis/post_training/index.html",
    "providers/eval/index": "../../advanced_apis/eval/index.html",
    "providers/scoring/index": "../../advanced_apis/scoring/index.html",
    "playground/index": "../../building_applications/playground/index.html",
    "openai/index": "../../providers/index.html#openai-api-compatibility",
    "introduction/index": "../concepts/index.html#llama-stack-architecture"
}

myst_enable_extensions = ["colon_fence"]

html_theme = "sphinx_rtd_theme"
html_use_relative_paths = True
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

myst_enable_extensions = [
    "amsmath",
    "attrs_inline",
    "attrs_block",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    # "linkify",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
]

myst_substitutions = {
    "docker_hub": "https://hub.docker.com/repository/docker/llamastack",
    "llama_stack_version": version_tag,
    "llama_stack_version_link": llama_stack_version_link,
}

suppress_warnings = ["myst.header"]

# Copy button settings
copybutton_prompt_text = "$ "  # for bash prompts
copybutton_prompt_is_regexp = True
copybutton_remove_prompts = True
copybutton_line_continuation_character = "\\"

# Source suffix
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = "alabaster"
html_theme_options = {
    "canonical_url": "https://github.com/meta-llama/llama-stack",
    "collapse_navigation": False,
    # "style_nav_header_background": "#c3c9d4",
    'display_version': True,
    'version_selector': True,
}

default_dark_mode = False

html_static_path = ["../_static"]
# html_logo = "../_static/llama-stack-logo.png"
# html_style = "../_static/css/my_theme.css"


def setup(app):
    app.add_css_file("css/my_theme.css")
    app.add_js_file("js/detect_theme.js")
    app.add_js_file("js/horizontal_nav.js")
    app.add_js_file("js/keyboard_shortcuts.js")

    def dockerhub_role(name, rawtext, text, lineno, inliner, options={}, content=[]):
        url = f"https://hub.docker.com/r/llamastack/{text}"
        node = nodes.reference(rawtext, text, refuri=url, **options)
        return [node], []

    def repopath_role(name, rawtext, text, lineno, inliner, options={}, content=[]):
        parts = text.split("::")
        if len(parts) == 2:
            link_text = parts[0]
            url_path = parts[1]
        else:
            link_text = text
            url_path = text

        url = f"https://github.com/meta-llama/llama-stack/tree/main/{url_path}"
        node = nodes.reference(rawtext, link_text, refuri=url, **options)
        return [node], []

    app.add_role("dockerhub", dockerhub_role)
    app.add_role("repopath", repopath_role)
