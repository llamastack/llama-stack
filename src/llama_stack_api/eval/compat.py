# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Backward compatibility helpers for the Eval API.

This module provides utilities to support both the old-style (individual parameters)
and new-style (request objects) calling conventions for Eval API methods.

The old-style parameters are deprecated and will be removed in a future release.

Note: When both a request object AND individual parameters are provided, the request
object takes precedence and individual parameters are ignored.
"""

import warnings
from typing import Any

from .models import (
    BenchmarkConfig,
    EvaluateRowsRequest,
    JobCancelRequest,
    JobResultRequest,
    JobStatusRequest,
    RunEvalRequest,
)

_DEPRECATION_TARGET = "0.6.0"

_DEPRECATION_MESSAGE = (
    "Passing individual parameters to {method_name}() is deprecated. "
    "Please use {request_class}(benchmark_id=..., ...) instead. "
    "This will be removed in version {target}."
)


def _emit_deprecation_warning(method_name: str, request_class: str) -> None:
    """Emit a deprecation warning for old-style parameter usage."""
    warnings.warn(
        _DEPRECATION_MESSAGE.format(method_name=method_name, request_class=request_class, target=_DEPRECATION_TARGET),
        DeprecationWarning,
        stacklevel=4,
    )


def _format_missing_params(required: list[str], provided: dict[str, Any]) -> str:
    """Format error message showing which parameters are missing."""
    missing = [p for p in required if provided.get(p) is None]
    provided_names = [p for p in required if provided.get(p) is not None]

    parts = []
    if missing:
        parts.append(f"missing: {', '.join(missing)}")
    if provided_names:
        parts.append(f"provided: {', '.join(provided_names)}")

    return "; ".join(parts)


def _validate_not_empty(value: Any, name: str) -> None:
    """Validate that a value is not None, empty string, or empty list."""
    if not value:
        raise ValueError(f"'{name}' cannot be None or empty. Provided: {value}")


def resolve_run_eval_request(
    request: RunEvalRequest | None = None,
    *,
    benchmark_id: str | None = None,
    benchmark_config: BenchmarkConfig | None = None,
) -> RunEvalRequest:
    """
    Resolve run_eval parameters to a RunEvalRequest object.

    Supports both new-style (request object) and old-style (individual parameters).
    Old-style usage emits a DeprecationWarning.

    Note: If both request object and individual parameters are provided, the request
    object takes precedence and individual parameters are ignored.

    Args:
        request: The new-style request object (preferred)
        benchmark_id: (Deprecated) The benchmark ID
        benchmark_config: (Deprecated) The benchmark configuration

    Returns:
        RunEvalRequest object
    """
    if request is not None:
        _validate_not_empty(request.benchmark_id, "benchmark_id")
        _validate_not_empty(request.benchmark_config, "benchmark_config")
        return request

    # Old-style parameters
    if benchmark_id and benchmark_config:
        _emit_deprecation_warning("run_eval", "RunEvalRequest")
        return RunEvalRequest(
            benchmark_id=benchmark_id,
            benchmark_config=benchmark_config,
        )

    required = ["benchmark_id", "benchmark_config"]
    provided = {"benchmark_id": benchmark_id, "benchmark_config": benchmark_config}
    raise ValueError(
        f"Either 'request' (RunEvalRequest) or both 'benchmark_id' and 'benchmark_config' "
        f"must be provided. {_format_missing_params(required, provided)}"
    )


def resolve_evaluate_rows_request(
    request: EvaluateRowsRequest | None = None,
    *,
    benchmark_id: str | None = None,
    input_rows: list[dict[str, Any]] | None = None,
    scoring_functions: list[str] | None = None,
    benchmark_config: BenchmarkConfig | None = None,
) -> EvaluateRowsRequest:
    """
    Resolve evaluate_rows parameters to an EvaluateRowsRequest object.

    Supports both new-style (request object) and old-style (individual parameters).
    Old-style usage emits a DeprecationWarning.

    Note: If both request object and individual parameters are provided, the request
    object takes precedence and individual parameters are ignored.

    Args:
        request: The new-style request object (preferred)
        benchmark_id: (Deprecated) The benchmark ID
        input_rows: (Deprecated) The rows to evaluate
        scoring_functions: (Deprecated) The scoring functions to use
        benchmark_config: (Deprecated) The benchmark configuration

    Returns:
        EvaluateRowsRequest object
    """
    if request is not None:
        _validate_not_empty(request.benchmark_id, "benchmark_id")
        _validate_not_empty(request.input_rows, "input_rows")
        _validate_not_empty(request.scoring_functions, "scoring_functions")
        _validate_not_empty(request.benchmark_config, "benchmark_config")
        return request

    # Old-style parameters
    if benchmark_id and input_rows and scoring_functions and benchmark_config:
        _emit_deprecation_warning("evaluate_rows", "EvaluateRowsRequest")
        return EvaluateRowsRequest(
            benchmark_id=benchmark_id,
            input_rows=input_rows,
            scoring_functions=scoring_functions,
            benchmark_config=benchmark_config,
        )

    required = ["benchmark_id", "input_rows", "scoring_functions", "benchmark_config"]
    provided = {
        "benchmark_id": benchmark_id,
        "input_rows": input_rows,
        "scoring_functions": scoring_functions,
        "benchmark_config": benchmark_config,
    }
    raise ValueError(
        f"Either 'request' (EvaluateRowsRequest) or all of 'benchmark_id', 'input_rows', "
        f"'scoring_functions', and 'benchmark_config' must be provided. "
        f"{_format_missing_params(required, provided)}"
    )


def resolve_job_status_request(
    request: JobStatusRequest | None = None,
    *,
    benchmark_id: str | None = None,
    job_id: str | None = None,
) -> JobStatusRequest:
    """
    Resolve job_status parameters to a JobStatusRequest object.

    Supports both new-style (request object) and old-style (individual parameters).
    Old-style usage emits a DeprecationWarning.

    Note: If both request object and individual parameters are provided, the request
    object takes precedence and individual parameters are ignored.

    Args:
        request: The new-style request object (preferred)
        benchmark_id: (Deprecated) The benchmark ID
        job_id: (Deprecated) The job ID

    Returns:
        JobStatusRequest object
    """
    if request is not None:
        _validate_not_empty(request.benchmark_id, "benchmark_id")
        _validate_not_empty(request.job_id, "job_id")
        return request

    # Old-style parameters
    if benchmark_id and job_id:
        _emit_deprecation_warning("job_status", "JobStatusRequest")
        return JobStatusRequest(
            benchmark_id=benchmark_id,
            job_id=job_id,
        )

    required = ["benchmark_id", "job_id"]
    provided = {"benchmark_id": benchmark_id, "job_id": job_id}
    raise ValueError(
        f"Either 'request' (JobStatusRequest) or both 'benchmark_id' and 'job_id' "
        f"must be provided. {_format_missing_params(required, provided)}"
    )


def resolve_job_cancel_request(
    request: JobCancelRequest | None = None,
    *,
    benchmark_id: str | None = None,
    job_id: str | None = None,
) -> JobCancelRequest:
    """
    Resolve job_cancel parameters to a JobCancelRequest object.

    Supports both new-style (request object) and old-style (individual parameters).
    Old-style usage emits a DeprecationWarning.

    Note: If both request object and individual parameters are provided, the request
    object takes precedence and individual parameters are ignored.

    Args:
        request: The new-style request object (preferred)
        benchmark_id: (Deprecated) The benchmark ID
        job_id: (Deprecated) The job ID

    Returns:
        JobCancelRequest object
    """
    if request is not None:
        _validate_not_empty(request.benchmark_id, "benchmark_id")
        _validate_not_empty(request.job_id, "job_id")
        return request

    # Old-style parameters
    if benchmark_id and job_id:
        _emit_deprecation_warning("job_cancel", "JobCancelRequest")
        return JobCancelRequest(
            benchmark_id=benchmark_id,
            job_id=job_id,
        )

    required = ["benchmark_id", "job_id"]
    provided = {"benchmark_id": benchmark_id, "job_id": job_id}
    raise ValueError(
        f"Either 'request' (JobCancelRequest) or both 'benchmark_id' and 'job_id' "
        f"must be provided. {_format_missing_params(required, provided)}"
    )


def resolve_job_result_request(
    request: JobResultRequest | None = None,
    *,
    benchmark_id: str | None = None,
    job_id: str | None = None,
) -> JobResultRequest:
    """
    Resolve job_result parameters to a JobResultRequest object.

    Supports both new-style (request object) and old-style (individual parameters).
    Old-style usage emits a DeprecationWarning.

    Note: If both request object and individual parameters are provided, the request
    object takes precedence and individual parameters are ignored.

    Args:
        request: The new-style request object (preferred)
        benchmark_id: (Deprecated) The benchmark ID
        job_id: (Deprecated) The job ID

    Returns:
        JobResultRequest object
    """
    if request is not None:
        _validate_not_empty(request.benchmark_id, "benchmark_id")
        _validate_not_empty(request.job_id, "job_id")
        return request

    # Old-style parameters
    if benchmark_id and job_id:
        _emit_deprecation_warning("job_result", "JobResultRequest")
        return JobResultRequest(
            benchmark_id=benchmark_id,
            job_id=job_id,
        )

    required = ["benchmark_id", "job_id"]
    provided = {"benchmark_id": benchmark_id, "job_id": job_id}
    raise ValueError(
        f"Either 'request' (JobResultRequest) or both 'benchmark_id' and 'job_id' "
        f"must be provided. {_format_missing_params(required, provided)}"
    )
