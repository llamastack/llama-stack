# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Tests for Eval API backward compatibility.

These tests verify that both old-style (individual parameters) and new-style
(request objects) calling conventions work correctly, and that old-style usage
emits appropriate deprecation warnings.
"""

import warnings

import pytest

from llama_stack_api import (
    BenchmarkConfig,
    EvaluateRowsRequest,
    JobCancelRequest,
    JobResultRequest,
    JobStatusRequest,
    ModelCandidate,
    RunEvalRequest,
    resolve_evaluate_rows_request,
    resolve_job_cancel_request,
    resolve_job_result_request,
    resolve_job_status_request,
    resolve_run_eval_request,
)
from llama_stack_api.inference import SamplingParams, TopPSamplingStrategy


@pytest.fixture
def sample_benchmark_config():
    return BenchmarkConfig(
        eval_candidate=ModelCandidate(
            model="test-model",
            sampling_params=SamplingParams(max_tokens=100, strategy=TopPSamplingStrategy(temperature=0.7)),
        )
    )


class TestResolveRunEvalRequest:
    """Tests for resolve_run_eval_request."""

    def test_new_style_with_request_object(self, sample_benchmark_config):
        """Test that new-style (request object) works without deprecation warning."""
        request = RunEvalRequest(benchmark_id="bench-123", benchmark_config=sample_benchmark_config)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = resolve_run_eval_request(request)

            # No deprecation warning should be emitted
            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(deprecation_warnings) == 0

        assert result.benchmark_id == "bench-123"
        assert result.benchmark_config == sample_benchmark_config

    def test_old_style_with_individual_params(self, sample_benchmark_config):
        """Test that old-style (individual parameters) works and emits deprecation warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = resolve_run_eval_request(
                benchmark_id="bench-123",
                benchmark_config=sample_benchmark_config,
            )

            # Deprecation warning should be emitted
            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(deprecation_warnings) == 1
            assert "run_eval" in str(deprecation_warnings[0].message)
            assert "RunEvalRequest" in str(deprecation_warnings[0].message)

        assert result.benchmark_id == "bench-123"
        assert result.benchmark_config == sample_benchmark_config

    def test_request_object_takes_precedence_over_individual_params(self, sample_benchmark_config):
        """Test that request object takes precedence when both are provided."""
        request = RunEvalRequest(benchmark_id="from-request", benchmark_config=sample_benchmark_config)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = resolve_run_eval_request(
                request,
                benchmark_id="from-param",  # Should be ignored
                benchmark_config=sample_benchmark_config,
            )

            # No deprecation warning since request object is used
            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(deprecation_warnings) == 0

        # Request object values should be used
        assert result.benchmark_id == "from-request"

    def test_missing_parameters_raises_error(self, sample_benchmark_config):
        """Test that missing parameters raises ValueError with helpful message."""
        with pytest.raises(ValueError) as exc_info:
            resolve_run_eval_request()
        assert "Either 'request'" in str(exc_info.value)
        assert "missing:" in str(exc_info.value)

        with pytest.raises(ValueError) as exc_info:
            resolve_run_eval_request(benchmark_id="bench-123")  # missing benchmark_config
        assert "missing: benchmark_config" in str(exc_info.value)
        assert "provided: benchmark_id" in str(exc_info.value)

        with pytest.raises(ValueError) as exc_info:
            resolve_run_eval_request(benchmark_config=sample_benchmark_config)  # missing benchmark_id
        assert "missing: benchmark_id" in str(exc_info.value)
        assert "provided: benchmark_config" in str(exc_info.value)


class TestResolveEvaluateRowsRequest:
    """Tests for resolve_evaluate_rows_request."""

    def test_new_style_with_request_object(self, sample_benchmark_config):
        """Test that new-style (request object) works without deprecation warning."""
        request = EvaluateRowsRequest(
            benchmark_id="bench-123",
            input_rows=[{"test": "data"}],
            scoring_functions=["func1"],
            benchmark_config=sample_benchmark_config,
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = resolve_evaluate_rows_request(request)

            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(deprecation_warnings) == 0

        assert result.benchmark_id == "bench-123"
        assert result.input_rows == [{"test": "data"}]
        assert result.scoring_functions == ["func1"]

    def test_old_style_with_individual_params(self, sample_benchmark_config):
        """Test that old-style (individual parameters) works and emits deprecation warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = resolve_evaluate_rows_request(
                benchmark_id="bench-123",
                input_rows=[{"test": "data"}],
                scoring_functions=["func1"],
                benchmark_config=sample_benchmark_config,
            )

            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(deprecation_warnings) == 1
            assert "evaluate_rows" in str(deprecation_warnings[0].message)
            assert "EvaluateRowsRequest" in str(deprecation_warnings[0].message)

        assert result.benchmark_id == "bench-123"
        assert result.input_rows == [{"test": "data"}]
        assert result.scoring_functions == ["func1"]

    def test_request_object_takes_precedence_over_individual_params(self, sample_benchmark_config):
        """Test that request object takes precedence when both are provided."""
        request = EvaluateRowsRequest(
            benchmark_id="from-request",
            input_rows=[{"from": "request"}],
            scoring_functions=["request-func"],
            benchmark_config=sample_benchmark_config,
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = resolve_evaluate_rows_request(
                request,
                benchmark_id="from-param",
                input_rows=[{"from": "param"}],
                scoring_functions=["param-func"],
                benchmark_config=sample_benchmark_config,
            )

            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(deprecation_warnings) == 0

        assert result.benchmark_id == "from-request"
        assert result.input_rows == [{"from": "request"}]

    def test_missing_parameters_raises_error(self, sample_benchmark_config):
        """Test that missing parameters raises ValueError with helpful message."""
        with pytest.raises(ValueError) as exc_info:
            resolve_evaluate_rows_request()
        assert "missing:" in str(exc_info.value)

        with pytest.raises(ValueError) as exc_info:
            resolve_evaluate_rows_request(
                benchmark_id="bench-123",
                input_rows=[{"test": "data"}],
                # missing scoring_functions and benchmark_config
            )
        assert "missing: scoring_functions, benchmark_config" in str(exc_info.value)


class TestResolveJobStatusRequest:
    """Tests for resolve_job_status_request."""

    def test_new_style_with_request_object(self):
        """Test that new-style (request object) works without deprecation warning."""
        request = JobStatusRequest(benchmark_id="bench-123", job_id="job-456")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = resolve_job_status_request(request)

            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(deprecation_warnings) == 0

        assert result.benchmark_id == "bench-123"
        assert result.job_id == "job-456"

    def test_old_style_with_individual_params(self):
        """Test that old-style (individual parameters) works and emits deprecation warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = resolve_job_status_request(benchmark_id="bench-123", job_id="job-456")

            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(deprecation_warnings) == 1
            assert "job_status" in str(deprecation_warnings[0].message)
            assert "JobStatusRequest" in str(deprecation_warnings[0].message)

        assert result.benchmark_id == "bench-123"
        assert result.job_id == "job-456"

    def test_request_object_takes_precedence_over_individual_params(self):
        """Test that request object takes precedence when both are provided."""
        request = JobStatusRequest(benchmark_id="from-request", job_id="job-from-request")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = resolve_job_status_request(
                request,
                benchmark_id="from-param",
                job_id="job-from-param",
            )

            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(deprecation_warnings) == 0

        assert result.benchmark_id == "from-request"
        assert result.job_id == "job-from-request"

    def test_missing_parameters_raises_error(self):
        """Test that missing parameters raises ValueError with helpful message."""
        with pytest.raises(ValueError) as exc_info:
            resolve_job_status_request()
        assert "missing:" in str(exc_info.value)

        with pytest.raises(ValueError) as exc_info:
            resolve_job_status_request(benchmark_id="bench-123")  # missing job_id
        assert "missing: job_id" in str(exc_info.value)
        assert "provided: benchmark_id" in str(exc_info.value)

        with pytest.raises(ValueError) as exc_info:
            resolve_job_status_request(job_id="job-456")  # missing benchmark_id
        assert "missing: benchmark_id" in str(exc_info.value)
        assert "provided: job_id" in str(exc_info.value)


class TestResolveJobCancelRequest:
    """Tests for resolve_job_cancel_request."""

    def test_new_style_with_request_object(self):
        """Test that new-style (request object) works without deprecation warning."""
        request = JobCancelRequest(benchmark_id="bench-123", job_id="job-456")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = resolve_job_cancel_request(request)

            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(deprecation_warnings) == 0

        assert result.benchmark_id == "bench-123"
        assert result.job_id == "job-456"

    def test_old_style_with_individual_params(self):
        """Test that old-style (individual parameters) works and emits deprecation warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = resolve_job_cancel_request(benchmark_id="bench-123", job_id="job-456")

            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(deprecation_warnings) == 1
            assert "job_cancel" in str(deprecation_warnings[0].message)
            assert "JobCancelRequest" in str(deprecation_warnings[0].message)

        assert result.benchmark_id == "bench-123"
        assert result.job_id == "job-456"

    def test_request_object_takes_precedence_over_individual_params(self):
        """Test that request object takes precedence when both are provided."""
        request = JobCancelRequest(benchmark_id="from-request", job_id="job-from-request")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = resolve_job_cancel_request(
                request,
                benchmark_id="from-param",
                job_id="job-from-param",
            )

            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(deprecation_warnings) == 0

        assert result.benchmark_id == "from-request"
        assert result.job_id == "job-from-request"

    def test_missing_parameters_raises_error(self):
        """Test that missing parameters raises ValueError with helpful message."""
        with pytest.raises(ValueError) as exc_info:
            resolve_job_cancel_request()
        assert "missing:" in str(exc_info.value)

        with pytest.raises(ValueError) as exc_info:
            resolve_job_cancel_request(benchmark_id="bench-123")  # missing job_id
        assert "missing: job_id" in str(exc_info.value)
        assert "provided: benchmark_id" in str(exc_info.value)

        with pytest.raises(ValueError) as exc_info:
            resolve_job_cancel_request(job_id="job-456")  # missing benchmark_id
        assert "missing: benchmark_id" in str(exc_info.value)
        assert "provided: job_id" in str(exc_info.value)


class TestResolveJobResultRequest:
    """Tests for resolve_job_result_request."""

    def test_new_style_with_request_object(self):
        """Test that new-style (request object) works without deprecation warning."""
        request = JobResultRequest(benchmark_id="bench-123", job_id="job-456")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = resolve_job_result_request(request)

            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(deprecation_warnings) == 0

        assert result.benchmark_id == "bench-123"
        assert result.job_id == "job-456"

    def test_old_style_with_individual_params(self):
        """Test that old-style (individual parameters) works and emits deprecation warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = resolve_job_result_request(benchmark_id="bench-123", job_id="job-456")

            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(deprecation_warnings) == 1
            assert "job_result" in str(deprecation_warnings[0].message)
            assert "JobResultRequest" in str(deprecation_warnings[0].message)

        assert result.benchmark_id == "bench-123"
        assert result.job_id == "job-456"

    def test_request_object_takes_precedence_over_individual_params(self):
        """Test that request object takes precedence when both are provided."""
        request = JobResultRequest(benchmark_id="from-request", job_id="job-from-request")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = resolve_job_result_request(
                request,
                benchmark_id="from-param",
                job_id="job-from-param",
            )

            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(deprecation_warnings) == 0

        assert result.benchmark_id == "from-request"
        assert result.job_id == "job-from-request"

    def test_missing_parameters_raises_error(self):
        """Test that missing parameters raises ValueError with helpful message."""
        with pytest.raises(ValueError) as exc_info:
            resolve_job_result_request()
        assert "missing:" in str(exc_info.value)

        with pytest.raises(ValueError) as exc_info:
            resolve_job_result_request(benchmark_id="bench-123")  # missing job_id
        assert "missing: job_id" in str(exc_info.value)
        assert "provided: benchmark_id" in str(exc_info.value)

        with pytest.raises(ValueError) as exc_info:
            resolve_job_result_request(job_id="job-456")  # missing benchmark_id
        assert "missing: benchmark_id" in str(exc_info.value)
        assert "provided: job_id" in str(exc_info.value)


class TestEmptyValueValidation:
    """Tests for validation of None, empty strings, and empty lists."""

    def test_empty_benchmark_id_old_style(self, sample_benchmark_config):
        """Empty benchmark_id is rejected when using old-style parameters (treated as missing)."""
        with pytest.raises(ValueError) as exc_info:
            resolve_run_eval_request(benchmark_id="", benchmark_config=sample_benchmark_config)
        # Empty string is falsy, so it's treated as missing
        assert "benchmark_id" in str(exc_info.value)

    def test_empty_benchmark_id_in_request_object(self, sample_benchmark_config):
        """Empty benchmark_id in request object (via model_construct) is rejected."""
        request = RunEvalRequest.model_construct(
            benchmark_id="",
            benchmark_config=sample_benchmark_config,
        )
        with pytest.raises(ValueError) as exc_info:
            resolve_run_eval_request(request)
        assert "benchmark_id" in str(exc_info.value)
        assert "cannot be None or empty" in str(exc_info.value)

    def test_none_benchmark_id_in_request_object(self, sample_benchmark_config):
        """None benchmark_id in request object (via model_construct) is rejected."""
        request = RunEvalRequest.model_construct(
            benchmark_id=None,
            benchmark_config=sample_benchmark_config,
        )
        with pytest.raises(ValueError) as exc_info:
            resolve_run_eval_request(request)
        assert "benchmark_id" in str(exc_info.value)
        assert "cannot be None or empty" in str(exc_info.value)

    @pytest.mark.parametrize(
        "resolver,request_class",
        [
            (resolve_job_status_request, JobStatusRequest),
            (resolve_job_cancel_request, JobCancelRequest),
            (resolve_job_result_request, JobResultRequest),
        ],
    )
    def test_empty_job_id_old_style(self, resolver, request_class):
        """Empty job_id is rejected when using old-style parameters (treated as missing)."""
        with pytest.raises(ValueError) as exc_info:
            resolver(benchmark_id="bench-123", job_id="")
        # Empty string is falsy, so it's treated as missing
        assert "job_id" in str(exc_info.value)

    @pytest.mark.parametrize(
        "resolver,request_class",
        [
            (resolve_job_status_request, JobStatusRequest),
            (resolve_job_cancel_request, JobCancelRequest),
            (resolve_job_result_request, JobResultRequest),
        ],
    )
    def test_empty_job_id_in_request_object(self, resolver, request_class):
        """Empty job_id in request object (via model_construct) is rejected."""
        request = request_class.model_construct(
            benchmark_id="bench-123",
            job_id="",
        )
        with pytest.raises(ValueError) as exc_info:
            resolver(request)
        assert "job_id" in str(exc_info.value)
        assert "cannot be None or empty" in str(exc_info.value)

    @pytest.mark.parametrize(
        "resolver,request_class",
        [
            (resolve_job_status_request, JobStatusRequest),
            (resolve_job_cancel_request, JobCancelRequest),
            (resolve_job_result_request, JobResultRequest),
        ],
    )
    def test_none_job_id_in_request_object(self, resolver, request_class):
        """None job_id in request object (via model_construct) is rejected."""
        request = request_class.model_construct(
            benchmark_id="bench-123",
            job_id=None,
        )
        with pytest.raises(ValueError) as exc_info:
            resolver(request)
        assert "job_id" in str(exc_info.value)
        assert "cannot be None or empty" in str(exc_info.value)

    def test_empty_input_rows_old_style(self, sample_benchmark_config):
        """Empty input_rows is rejected when using old-style parameters (treated as missing)."""
        with pytest.raises(ValueError) as exc_info:
            resolve_evaluate_rows_request(
                benchmark_id="bench-123",
                input_rows=[],
                scoring_functions=["func1"],
                benchmark_config=sample_benchmark_config,
            )
        # Empty list is falsy, so it's treated as missing
        assert "input_rows" in str(exc_info.value)

    def test_empty_scoring_functions_old_style(self, sample_benchmark_config):
        """Empty scoring_functions is rejected when using old-style parameters (treated as missing)."""
        with pytest.raises(ValueError) as exc_info:
            resolve_evaluate_rows_request(
                benchmark_id="bench-123",
                input_rows=[{"test": "data"}],
                scoring_functions=[],
                benchmark_config=sample_benchmark_config,
            )
        # Empty list is falsy, so it's treated as missing
        assert "scoring_functions" in str(exc_info.value)

    def test_empty_input_rows_in_request_object(self, sample_benchmark_config):
        """Empty input_rows in request object (via model_construct) is rejected."""
        request = EvaluateRowsRequest.model_construct(
            benchmark_id="bench-123",
            input_rows=[],
            scoring_functions=["func1"],
            benchmark_config=sample_benchmark_config,
        )
        with pytest.raises(ValueError) as exc_info:
            resolve_evaluate_rows_request(request)
        assert "input_rows" in str(exc_info.value)
        assert "cannot be None or empty" in str(exc_info.value)

    def test_none_input_rows_in_request_object(self, sample_benchmark_config):
        """None input_rows in request object (via model_construct) is rejected."""
        request = EvaluateRowsRequest.model_construct(
            benchmark_id="bench-123",
            input_rows=None,
            scoring_functions=["func1"],
            benchmark_config=sample_benchmark_config,
        )
        with pytest.raises(ValueError) as exc_info:
            resolve_evaluate_rows_request(request)
        assert "input_rows" in str(exc_info.value)
        assert "cannot be None or empty" in str(exc_info.value)
