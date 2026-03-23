# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Tests for the OTEL streaming tool-call crash workaround.

opentelemetry-instrumentation-openai-v2's ChoiceBuffer crashes with TypeError
when a provider sends ``arguments=None`` in a tool-call streaming delta.
Our workaround: _patch_otel_choice_buffer() monkey-patches ChoiceBuffer to
normalize None → "" before appending.

See: https://github.com/open-telemetry/opentelemetry-python-contrib/issues/4344

Skipped when opentelemetry-instrumentation-openai-v2 is not installed.
"""

import os
import pathlib
import subprocess
import sys

import pytest

pytest.importorskip(
    "opentelemetry.instrumentation.openai_v2",
    reason="opentelemetry-instrumentation-openai-v2 not installed",
)

_SUBPROCESS_SCRIPT = str(pathlib.Path(__file__).parent / "_otel_stream_subprocess.py")


class TestAutoInstrumentation:
    """Proves the bug and fix under real ``opentelemetry-instrument`` auto-instrumentation."""

    @staticmethod
    def _run(*, patch: bool) -> subprocess.CompletedProcess:
        otel_bin = pathlib.Path(sys.executable).parent / "opentelemetry-instrument"
        if not otel_bin.exists():
            pytest.skip("opentelemetry-instrument CLI not found")
        return subprocess.run(
            [str(otel_bin), sys.executable, _SUBPROCESS_SCRIPT],
            capture_output=True,
            text=True,
            timeout=30,
            env={
                **os.environ,
                "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT": "true",
                "OTEL_TRACES_EXPORTER": "none",
                "OTEL_METRICS_EXPORTER": "none",
                "OTEL_LOGS_EXPORTER": "none",
                "APPLY_OTEL_PATCH": "1" if patch else "0",
            },
        )

    @pytest.mark.xfail(
        strict=True,
        reason="Auto-instrumented stream crashes on arguments=None. "
        "If this xfail starts failing, opentelemetry-python-contrib #4344 is fixed.",
    )
    def test_crashes_without_patch(self):
        result = self._run(patch=False)
        assert result.returncode == 0, result.stderr

    def test_works_with_patch(self):
        result = self._run(patch=True)
        assert result.returncode == 0, result.stderr
