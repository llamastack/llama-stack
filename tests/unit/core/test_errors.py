# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


import httpx

from llama_stack.core.exceptions.mapping import translate_exception_to_http
from llama_stack_api.common.errors import (
    ClientListCommand,
    LlamaStackError,
    ModelNotFoundError,
    ResourceNotFoundError,
)


class TestClientListCommand:
    def test_basic_command(self):
        cmd = ClientListCommand("files.list")
        assert str(cmd) == "Use 'client.files.list()'."

    def test_with_argument(self):
        cmd = ClientListCommand("connectors.list_tools", "my-connector")
        assert str(cmd) == "Use 'client.connectors.list_tools(\"my-connector\")'."

    def test_with_multiple_arguments(self):
        cmd = ClientListCommand("widgets.find", ["arg1", "arg2"])
        assert str(cmd) == 'Use \'client.widgets.find("arg1", "arg2")\'.'

    def test_with_resource_plural(self):
        cmd = ClientListCommand("batches.list", resource_name_plural="batches")
        assert str(cmd) == "Use 'client.batches.list()' to list available batches."


class TestTranslateExceptionToHttp:
    """Tests for translate_exception_to_http which walks the MRO to find
    the first mapped exception type in EXCEPTION_MAP."""

    # ── Direct matches for each mapped type ──────────────────────────

    def test_value_error(self):
        exc = ValueError("bad input")
        result = translate_exception_to_http(exc)
        assert result is not None
        assert result.status_code == httpx.codes.BAD_REQUEST
        assert "bad input" in result.detail

    def test_permission_error(self):
        exc = PermissionError("access denied")
        result = translate_exception_to_http(exc)
        assert result is not None
        assert result.status_code == httpx.codes.FORBIDDEN
        assert "access denied" in result.detail

    def test_connection_error(self):
        exc = ConnectionError("connection refused")
        result = translate_exception_to_http(exc)
        assert result is not None
        assert result.status_code == httpx.codes.BAD_GATEWAY

    def test_timeout_error(self):
        exc = TimeoutError("timed out")
        result = translate_exception_to_http(exc)
        assert result is not None
        assert result.status_code == httpx.codes.GATEWAY_TIMEOUT
        assert "timed out" in result.detail

    def test_asyncio_timeout_error(self):
        exc = TimeoutError("async timed out")
        result = translate_exception_to_http(exc)
        assert result is not None
        assert result.status_code == httpx.codes.GATEWAY_TIMEOUT

    def test_not_implemented_error(self):
        exc = NotImplementedError("not supported")
        result = translate_exception_to_http(exc)
        assert result is not None
        assert result.status_code == httpx.codes.NOT_IMPLEMENTED
        assert "not supported" in result.detail

    # ── Subclass matching via MRO ────────────────────────────────────

    def test_subclass_one_level_deep(self):
        """A direct subclass of a mapped type should match via MRO."""

        class CustomValueError(ValueError):
            pass

        exc = CustomValueError("custom")
        result = translate_exception_to_http(exc)
        assert result is not None
        assert result.status_code == httpx.codes.BAD_REQUEST
        assert "custom" in result.detail

    def test_subclass_two_levels_deep(self):
        """ResourceNotFoundError(ValueError, LlamaStackError) should
        match ValueError via MRO since LlamaStackError is not mapped."""
        exc = ResourceNotFoundError("abc", "Widget")
        result = translate_exception_to_http(exc)
        assert result is not None
        assert result.status_code == httpx.codes.BAD_REQUEST

    def test_subclass_three_levels_deep(self):
        """ModelNotFoundError -> ResourceNotFoundError -> ValueError
        should still resolve to ValueError in the map."""
        exc = ModelNotFoundError("llama-3")
        result = translate_exception_to_http(exc)
        assert result is not None
        assert result.status_code == httpx.codes.BAD_REQUEST

    # ── Multiple mapped parents: MRO order determines winner ─────────

    def test_multiple_mapped_parents_first_wins(self):
        """When a class inherits from two mapped types, MRO puts the
        first parent before the second, so it should win."""

        class PermThenValueError(PermissionError, ValueError):
            pass

        exc = PermThenValueError("denied")
        result = translate_exception_to_http(exc)
        assert result is not None
        # PermissionError comes first in MRO → 403 FORBIDDEN
        assert result.status_code == httpx.codes.FORBIDDEN

    def test_multiple_mapped_parents_reversed_order(self):
        """Reversing the parent order flips which mapped type wins."""

        class ValueThenPermError(ValueError, PermissionError):
            pass

        exc = ValueThenPermError("invalid")
        result = translate_exception_to_http(exc)
        assert result is not None
        # ValueError comes first in MRO → 400 BAD_REQUEST
        assert result.status_code == httpx.codes.BAD_REQUEST

    # ── Unmapped types before mapped type in the MRO ─────────────────

    def test_unmapped_ancestors_before_mapped(self):
        """KeyError -> LookupError are not in the map. For
        class Err(KeyError, ValueError), MRO is:
        Err -> KeyError -> LookupError -> ValueError -> Exception
        so it should skip unmapped types and match ValueError."""

        class LookupAndValueError(KeyError, ValueError):
            pass

        exc = LookupAndValueError("missing")
        result = translate_exception_to_http(exc)
        assert result is not None
        assert result.status_code == httpx.codes.BAD_REQUEST

    # ── No match returns None ────────────────────────────────────────

    def test_unmapped_runtime_error(self):
        exc = RuntimeError("boom")
        assert translate_exception_to_http(exc) is None

    def test_unmapped_key_error(self):
        exc = KeyError("missing")
        assert translate_exception_to_http(exc) is None

    def test_bare_exception(self):
        exc = Exception("generic")
        assert translate_exception_to_http(exc) is None

    def test_llama_stack_error_base_not_in_map(self):
        """LlamaStackError inherits from Exception which is NOT in the
        map. The base class itself should not match since it's handled
        separately by translate_exception via its status_code attr."""

        class BareStackError(LlamaStackError):
            status_code = httpx.codes.IM_A_TEAPOT

        exc = BareStackError("teapot")
        assert translate_exception_to_http(exc) is None
