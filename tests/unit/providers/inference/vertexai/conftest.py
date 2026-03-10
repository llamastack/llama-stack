# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import sys
from types import ModuleType, SimpleNamespace
from typing import Any, cast


def _install_google_shims() -> None:
    """Install Google SDK shim modules for tests."""
    google_module = sys.modules.get("google")
    if google_module is None:
        google_module = ModuleType("google")
        google_module.__path__ = []  # type: ignore[attr-defined]
        sys.modules["google"] = google_module

    genai_module = ModuleType("google.genai")

    class MockClient:
        def __init__(self, **kwargs):
            """Initialize shim attributes from kwargs."""
            self.kwargs = kwargs
            self.models = SimpleNamespace(list=lambda: [])
            self.aio = SimpleNamespace(models=SimpleNamespace())

    cast(Any, genai_module).Client = MockClient
    sys.modules["google.genai"] = genai_module
    cast(Any, google_module).genai = genai_module

    genai_types_module = ModuleType("google.genai.types")

    class FunctionCallingConfig:
        def __init__(self, **kwargs):
            """Initialize shim attributes from kwargs."""
            self.__dict__.update(kwargs)

    class ToolConfig:
        def __init__(self, **kwargs):
            """Initialize shim attributes from kwargs."""
            self.__dict__.update(kwargs)

    class Tool:
        def __init__(self, **kwargs):
            """Initialize shim attributes from kwargs."""
            self.__dict__.update(kwargs)

    class ThinkingConfig:
        def __init__(self, **kwargs):
            """Initialize shim attributes from kwargs."""
            self.__dict__.update(kwargs)

    class GenerateContentConfig:
        def __init__(self, **kwargs):
            """Initialize shim attributes from kwargs."""
            if "thinking_config" in kwargs and isinstance(kwargs["thinking_config"], dict):
                kwargs["thinking_config"] = ThinkingConfig(**kwargs["thinking_config"])
            self.__dict__.update(kwargs)

    class ListModelsConfig:
        def __init__(self, **kwargs):
            """Initialize shim attributes from kwargs."""
            self.__dict__.update(kwargs)

    class HttpOptions:
        def __init__(self, **kwargs):
            """Initialize shim attributes from kwargs."""
            self.__dict__.update(kwargs)

    class EmbedContentConfig:
        def __init__(self, **kwargs):
            """Initialize shim attributes from kwargs."""
            self.__dict__.update(kwargs)

    cast(Any, genai_types_module).FunctionCallingConfig = FunctionCallingConfig
    cast(Any, genai_types_module).ToolConfig = ToolConfig
    cast(Any, genai_types_module).Tool = Tool
    cast(Any, genai_types_module).GenerateContentConfig = GenerateContentConfig
    cast(Any, genai_types_module).ThinkingConfig = ThinkingConfig
    cast(Any, genai_types_module).ListModelsConfig = ListModelsConfig
    cast(Any, genai_types_module).HttpOptions = HttpOptions
    cast(Any, genai_types_module).EmbedContentConfig = EmbedContentConfig

    sys.modules["google.genai.types"] = genai_types_module
    cast(Any, genai_module).types = genai_types_module

    oauth2_module = ModuleType("google.oauth2")
    sys.modules["google.oauth2"] = oauth2_module
    cast(Any, google_module).oauth2 = oauth2_module

    credentials_module = ModuleType("google.oauth2.credentials")

    class Credentials:
        def __init__(self, token: str):
            """Initialize shim attributes from kwargs."""
            self.token = token

    cast(Any, credentials_module).Credentials = Credentials
    sys.modules["google.oauth2.credentials"] = credentials_module
    cast(Any, oauth2_module).credentials = credentials_module


_install_google_shims()
