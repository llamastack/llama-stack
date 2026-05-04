# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from ogx_api import SafetyViolation


class SafetyException(Exception):  # noqa: N818
    """Raised when a safety shield detects a policy violation."""

    def __init__(self, violation: SafetyViolation):
        self.violation = violation
        super().__init__(violation.user_message)
