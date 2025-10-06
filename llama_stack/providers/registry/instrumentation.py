# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Registry for instrumentation providers (non-API providers).

This registry is string-based to avoid importing provider modules at import
 time (prevents circular imports). Classes are instantiated lazily by the
 StackRunConfig validator using `instantiate_class_type`.

Please implement your instrumentation provider as a subclass of `InstrumentationProvider` and register it in this registry.

Example:

```
from llama_stack.core.instrumentation import InstrumentationProvider

class MyInstrumentationProvider(InstrumentationProvider):
    fastapi_middleware(self, app: FastAPI) -> None:
        pass
```
"""

from typing import NamedTuple


class InstrumentationEntry(NamedTuple):
    provider_class: str  # fully-qualified class path
    config_class: str  # fully-qualified class path
    description: str


instrumentation_registry: dict[str, InstrumentationEntry] = {
    "otel": InstrumentationEntry(
        provider_class="llama_stack.providers.inline.instrumentation.otel.otel.OTelInstrumentationProvider",
        config_class="llama_stack.providers.inline.instrumentation.otel.config.OTelConfig",
        description="OpenTelemetry instrumentation",
    ),
}
