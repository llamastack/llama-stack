# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Root-level conftest.py for pytest plugin loading.

In pytest 8.4+, pytest_plugins must be defined at the rootdir conftest.py level,
not in subdirectory conftest files. This file dynamically loads the appropriate
fixture plugins based on which tests are being collected.
"""


def pytest_configure(config):
    """Dynamically import fixture plugins based on test collection paths."""
    # Get the args to determine which tests are being run
    args = config.invocation_params.args

    # Check if we're running unit tests
    running_unit = any("tests/unit" in str(arg) or "tests\\unit" in str(arg) for arg in args)
    # Check if we're running integration tests
    running_integration = any("tests/integration" in str(arg) or "tests\\integration" in str(arg) for arg in args)

    # If no specific path given, check if collecting from root
    if not args or args == (".",) or args == ("tests",) or args == ("tests/",):
        running_unit = True
        running_integration = True

    # Import plugins dynamically
    if running_unit:
        config.pluginmanager.import_plugin("tests.unit.fixtures")
        # Load shared fixtures from openai_responses test file (used by conversations tests)
        config.pluginmanager.import_plugin("tests.unit.providers.agents.meta_reference.test_openai_responses")

    if running_integration:
        config.pluginmanager.import_plugin("tests.integration.fixtures.common")
