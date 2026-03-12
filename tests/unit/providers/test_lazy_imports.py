# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Tests to verify that provider modules do not eagerly load heavy dependencies.

Each test uses subprocess isolation to check that importing a module
does not pull in heavy libraries (torch, numpy, etc.) until they are
explicitly needed.
"""

import subprocess
import sys


def _check_module_import_isolation(module_path: str, forbidden_modules: list[str]) -> dict:
    """
    Run a subprocess to import a module and check which forbidden modules are loaded.

    Returns a dict with 'loaded' (list of unexpectedly loaded modules) and 'success' (bool).
    """
    check_script = f"""
import sys

# Record modules before import
before = set(sys.modules.keys())

# Import the target module
{module_path}

# Check which forbidden modules were loaded
after = set(sys.modules.keys())
new_modules = after - before

forbidden = {forbidden_modules!r}
loaded = [m for m in forbidden if any(m == mod or mod.startswith(m + '.') for mod in new_modules)]

# Output result
import json
print(json.dumps({{"loaded": loaded, "new_count": len(new_modules)}}))
"""

    result = subprocess.run(
        [sys.executable, "-c", check_script],
        capture_output=True,
        text=True,
        timeout=60,
    )

    if result.returncode != 0:
        return {"loaded": [], "error": result.stderr, "success": False}

    import json

    output = json.loads(result.stdout.strip())
    output["success"] = True
    return output


class TestBraintrustLazyImports:
    """Test that braintrust scoring provider doesn't load autoevals/pyarrow at import time."""

    def test_braintrust_import_no_autoevals(self):
        """Verify braintrust module import doesn't load autoevals or pyarrow."""
        result = _check_module_import_isolation(
            "from llama_stack.providers.inline.scoring.braintrust import braintrust",
            ["autoevals", "pyarrow"],
        )

        assert result.get("success"), f"Import failed: {result.get('error', 'unknown error')}"
        assert not result["loaded"], (
            f"Heavy modules loaded unexpectedly during braintrust import: {result['loaded']}. "
            "These should be lazily loaded only when scoring is performed."
        )


def _check_no_forbidden_imports(module_path: str, forbidden: list[str]) -> tuple[bool, str]:
    """Import a module in a subprocess and check that forbidden modules are not loaded."""
    code = f"""
import sys
import importlib
importlib.import_module("{module_path}")
loaded = [m for m in {forbidden!r} if m in sys.modules]
if loaded:
    print("FORBIDDEN:" + ",".join(loaded))
else:
    print("OK")
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30)
    output = result.stdout.strip()
    if output.startswith("FORBIDDEN:"):
        return False, output.split(":", 1)[1]
    return True, ""


class TestEmbeddingMixinLazyImports:
    """Verify embedding_mixin.py does not eagerly import torch."""

    def test_no_torch_on_import(self):
        ok, loaded = _check_no_forbidden_imports(
            "llama_stack.providers.utils.inference.embedding_mixin",
            ["torch"],
        )
        assert ok, f"embedding_mixin.py eagerly loaded: {loaded}"
