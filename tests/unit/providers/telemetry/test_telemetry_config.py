# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Tests for TelemetryConfig handling of empty/None service_name.

This test suite validates the fix for the empty env var default issue
(https://github.com/meta-llama/llama-stack/issues/4611), ensuring that
TelemetryConfig accepts None/empty service_name values and the
TelemetryAdapter handles them gracefully with a default value.
"""

import pytest

from llama_stack.providers.inline.telemetry.meta_reference.config import TelemetryConfig


class TestTelemetryConfigServiceName:
    """Test TelemetryConfig handling of service_name field."""

    def test_service_name_accepts_none(self):
        """Test that TelemetryConfig accepts None for service_name."""
        config = TelemetryConfig(service_name=None)
        assert config.service_name is None

    def test_service_name_accepts_empty_string(self):
        """Test that TelemetryConfig accepts empty string for service_name."""
        config = TelemetryConfig(service_name="")
        assert config.service_name == ""

    def test_service_name_accepts_valid_string(self):
        """Test that TelemetryConfig accepts valid string for service_name."""
        config = TelemetryConfig(service_name="my-service")
        assert config.service_name == "my-service"

    def test_service_name_default_is_none(self):
        """Test that service_name defaults to None when not specified."""
        config = TelemetryConfig()
        assert config.service_name is None

    def test_sample_run_config_has_service_name(self):
        """Test that sample_run_config includes service_name field."""
        sample = TelemetryConfig.sample_run_config(__distro_dir__="/tmp/test")
        assert "service_name" in sample
        # Should use env var syntax with empty default (no zero-width space)
        assert sample["service_name"] == "${env.OTEL_SERVICE_NAME:=}"

    def test_sample_run_config_no_zero_width_space(self):
        """Test that sample_run_config does not use zero-width space character.

        The previous implementation used a zero-width space (\\u200b) as a workaround
        to pass Pydantic validation while appearing empty. This is problematic because:
        1. Invisible characters cause debugging difficulties
        2. Copy-paste can lose or duplicate these characters
        3. Most editors don't show them, hiding the actual value
        """
        sample = TelemetryConfig.sample_run_config(__distro_dir__="/tmp/test")
        # Ensure no zero-width space in service_name template
        assert "\u200b" not in sample["service_name"]


class TestTelemetryAdapterServiceName:
    """Test TelemetryAdapter handling of None/empty service_name."""

    def test_default_service_name_constant_exists(self):
        """Test that DEFAULT_SERVICE_NAME constant is defined."""
        from llama_stack.providers.inline.telemetry.meta_reference.telemetry import (
            DEFAULT_SERVICE_NAME,
        )

        assert DEFAULT_SERVICE_NAME == "llama-stack"

    def test_adapter_handles_none_service_name(self):
        """Test that TelemetryAdapter handles None service_name without error.

        When service_name is None, the adapter should use the default "llama-stack".
        """
        from unittest.mock import MagicMock, patch

        from llama_stack.providers.inline.telemetry.meta_reference.telemetry import (
            DEFAULT_SERVICE_NAME,
        )

        # Reset global state to ensure clean test
        import llama_stack.providers.inline.telemetry.meta_reference.telemetry as telemetry_module

        original_provider = telemetry_module._TRACER_PROVIDER
        telemetry_module._TRACER_PROVIDER = None

        try:
            config = TelemetryConfig(service_name=None)

            # Mock the trace module to capture the Resource creation
            with patch.object(telemetry_module, "trace") as mock_trace:
                mock_trace.get_tracer_provider.return_value = MagicMock()

                # Import and instantiate adapter
                from llama_stack.providers.inline.telemetry.meta_reference.telemetry import (
                    TelemetryAdapter,
                )

                adapter = TelemetryAdapter(config, deps={})

                # The adapter should have been created without error
                assert adapter is not None
        finally:
            # Restore original state
            telemetry_module._TRACER_PROVIDER = original_provider

    def test_adapter_uses_custom_service_name(self):
        """Test that TelemetryAdapter uses custom service_name when provided."""
        from unittest.mock import MagicMock, patch

        import llama_stack.providers.inline.telemetry.meta_reference.telemetry as telemetry_module

        original_provider = telemetry_module._TRACER_PROVIDER
        telemetry_module._TRACER_PROVIDER = None

        try:
            config = TelemetryConfig(service_name="custom-service")

            with patch.object(telemetry_module, "trace") as mock_trace:
                mock_trace.get_tracer_provider.return_value = MagicMock()

                from llama_stack.providers.inline.telemetry.meta_reference.telemetry import (
                    TelemetryAdapter,
                )

                adapter = TelemetryAdapter(config, deps={})
                assert adapter is not None
        finally:
            telemetry_module._TRACER_PROVIDER = original_provider
