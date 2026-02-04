# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Tests to verify that provider modules do not eagerly load heavy dependencies.

These tests ensure that importing provider modules does not trigger loading of
heavy dependencies like torch, numpy, pyarrow, or transformers until those
dependencies are actually needed. This is important for reducing startup memory
consumption.
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


class TestPromptGuardLazyImports:
    """Test that prompt_guard safety provider doesn't load torch/transformers at import time."""

    def test_prompt_guard_import_no_torch(self):
        """Verify prompt_guard module import doesn't load torch or transformers."""
        result = _check_module_import_isolation(
            "from llama_stack.providers.inline.safety.prompt_guard import prompt_guard",
            ["torch", "transformers"],
        )

        assert result.get("success"), f"Import failed: {result.get('error', 'unknown error')}"
        assert not result["loaded"], (
            f"Heavy modules loaded unexpectedly during prompt_guard import: {result['loaded']}. "
            "These should be lazily loaded only when initialize() is called."
        )


class TestGeneratorsLazyImports:
    """Test that meta_reference generators don't load torch at import time.

    Note: The generators module has transitive dependencies on inference.py
    which imports chat_format.py, and that imports torch. The lazy loading
    in generators.py itself is effective, but complete isolation requires
    refactoring the entire meta_reference inference provider hierarchy.
    """

    def test_generators_import_no_lmformatenforcer(self):
        """Verify generators module import doesn't load lmformatenforcer.

        Note: torch may still be loaded through transitive dependencies on
        llama_stack.models which eagerly import torch. This test verifies
        that at least lmformatenforcer (used only in get_logits_processor)
        is not loaded at import time.
        """
        result = _check_module_import_isolation(
            "from llama_stack.providers.inline.inference.meta_reference import generators",
            ["lmformatenforcer"],
        )

        # Skip if import failed (e.g., torch not installed)
        if not result.get("success"):
            import pytest

            pytest.skip(f"Import failed (likely missing torch): {result.get('error', '')[:200]}")

        assert not result["loaded"], (
            f"Heavy modules loaded unexpectedly during generators import: {result['loaded']}. "
            "lmformatenforcer should be lazily loaded only when get_logits_processor is called."
        )


class TestFaissLazyImports:
    """Test that faiss vector_io provider doesn't load faiss/numpy at import time."""

    def test_faiss_import_no_heavy_deps(self):
        """Verify faiss module import doesn't load faiss or numpy."""
        result = _check_module_import_isolation(
            "from llama_stack.providers.inline.vector_io.faiss import faiss",
            ["faiss", "numpy"],
        )

        assert result.get("success"), f"Import failed: {result.get('error', 'unknown error')}"
        assert not result["loaded"], (
            f"Heavy modules loaded unexpectedly during faiss import: {result['loaded']}. "
            "These should be lazily loaded only when FaissIndex is created."
        )


class TestSqliteVecLazyImports:
    """Test that sqlite_vec vector_io provider doesn't load numpy at import time."""

    def test_sqlite_vec_import_no_numpy(self):
        """Verify sqlite_vec module import doesn't load numpy or sqlite_vec."""
        result = _check_module_import_isolation(
            "from llama_stack.providers.inline.vector_io.sqlite_vec import sqlite_vec",
            ["numpy", "sqlite_vec"],
        )

        assert result.get("success"), f"Import failed: {result.get('error', 'unknown error')}"
        assert not result["loaded"], (
            f"Heavy modules loaded unexpectedly during sqlite_vec import: {result['loaded']}. "
            "These should be lazily loaded only when SQLiteVecIndex is created."
        )


class TestVectorStoreLazyImports:
    """Test that vector_store utility doesn't load numpy at import time."""

    def test_vector_store_import_no_numpy(self):
        """Verify vector_store module import doesn't load numpy."""
        result = _check_module_import_isolation(
            "from llama_stack.providers.utils.memory import vector_store",
            ["numpy"],
        )

        assert result.get("success"), f"Import failed: {result.get('error', 'unknown error')}"
        assert not result["loaded"], (
            f"Heavy modules loaded unexpectedly during vector_store import: {result['loaded']}. "
            "These should be lazily loaded only when vector operations are performed."
        )


class TestEmbeddingMixinLazyImports:
    """Test that embedding_mixin doesn't load torch at import time."""

    def test_embedding_mixin_import_no_torch(self):
        """Verify embedding_mixin module import doesn't load torch."""
        result = _check_module_import_isolation(
            "from llama_stack.providers.utils.inference import embedding_mixin",
            ["torch", "sentence_transformers"],
        )

        assert result.get("success"), f"Import failed: {result.get('error', 'unknown error')}"
        assert not result["loaded"], (
            f"Heavy modules loaded unexpectedly during embedding_mixin import: {result['loaded']}. "
            "These should be lazily loaded only when _load_sentence_transformer_model() is called."
        )
