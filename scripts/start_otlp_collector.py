#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Start OTLP HTTP test collector for integration tests."""

import os
import sys
import time

# Add ROOT_DIR to path (not tests/integration, which shadows stdlib inspect module)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from tests.integration.telemetry.collectors import OtlpHttpTestCollector

if __name__ == "__main__":
    collector_port = os.environ.get("LLAMA_STACK_TEST_COLLECTOR_PORT", "4318")
    print(f"Starting OTLP HTTP Test Collector on port {collector_port}...")

    collector = OtlpHttpTestCollector()
    print(f"OTLP Collector started at {collector.endpoint}")

    # Keep the collector running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping OTLP Collector...")
        collector.shutdown()
