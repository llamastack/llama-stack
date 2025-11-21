#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Manual test script for MLflow Prompts Provider.

This script provides a quick way to manually test the MLflow prompts provider
without running the full test suite.

Prerequisites:
    1. MLflow server running (default: http://localhost:5555)
    2. MLflow package installed: pip install 'mlflow>=3.4.0'

Usage:
    # Start MLflow server (in separate terminal)
    mlflow server --host 127.0.0.1 --port 5555

    # Run this script
    uv run python scripts/test_mlflow_prompts_manual.py

    # Or with custom MLflow URI
    MLFLOW_TRACKING_URI=http://localhost:8080 uv run python scripts/test_mlflow_prompts_manual.py
"""

import asyncio
import os
import sys


async def main():
    """Run manual tests for MLflow prompts provider."""
    print("=" * 80)
    print("MLflow Prompts Provider - Manual Test")
    print("=" * 80)
    print()

    # Get MLflow URI from environment or use default
    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5555")
    print(f"MLflow Server: {mlflow_uri}")
    print()

    # Check MLflow server availability
    print("1. Checking MLflow server availability...")
    try:
        import requests

        response = requests.get(f"{mlflow_uri}/health", timeout=5)
        if response.status_code == 200:
            print(f"   ✅ MLflow server is running at {mlflow_uri}")
        else:
            print(f"   ❌ MLflow server returned status {response.status_code}")
            print(f"   Start MLflow with: mlflow server --host 127.0.0.1 --port 5555")
            sys.exit(1)
    except Exception as e:
        print(f"   ❌ Failed to connect to MLflow server: {e}")
        print(f"   Start MLflow with: mlflow server --host 127.0.0.1 --port 5555")
        sys.exit(1)
    print()

    # Import MLflow provider
    print("2. Importing MLflow provider...")
    try:
        from llama_stack.providers.remote.prompts.mlflow import MLflowPromptsAdapter
        from llama_stack.providers.remote.prompts.mlflow.config import MLflowPromptsConfig

        print("   ✅ Import successful")
    except Exception as e:
        print(f"   ❌ Import failed: {e}")
        sys.exit(1)
    print()

    # Create and initialize adapter
    print("3. Initializing MLflow adapter...")
    try:
        config = MLflowPromptsConfig(
            mlflow_tracking_uri=mlflow_uri,
            experiment_name="manual-test-llama-stack",
        )
        adapter = MLflowPromptsAdapter(config=config)
        await adapter.initialize()
        print("   ✅ Adapter initialized")
    except Exception as e:
        print(f"   ❌ Initialization failed: {e}")
        sys.exit(1)
    print()

    # Test 1: Create prompt
    print("4. Creating a new prompt...")
    try:
        created = await adapter.create_prompt(
            prompt="Summarize the following text in {{ num_sentences }} sentences:\n\n{{ text }}",
            variables=["num_sentences", "text"],
        )
        print(f"   ✅ Created prompt: {created.prompt_id}")
        print(f"      - Version: {created.version}")
        print(f"      - Variables: {created.variables}")
        print(f"      - Is default: {created.is_default}")
    except Exception as e:
        print(f"   ❌ Create failed: {e}")
        sys.exit(1)
    print()

    # Test 2: Retrieve prompt
    print("5. Retrieving the prompt...")
    try:
        retrieved = await adapter.get_prompt(created.prompt_id)
        print(f"   ✅ Retrieved prompt: {retrieved.prompt_id}")
        print(f"      - Version: {retrieved.version}")
        print(f"      - Match: {retrieved.prompt_id == created.prompt_id}")
    except Exception as e:
        print(f"   ❌ Retrieve failed: {e}")
        sys.exit(1)
    print()

    # Test 3: Update prompt (create new version)
    print("6. Updating prompt (creates version 2)...")
    try:
        updated = await adapter.update_prompt(
            prompt_id=created.prompt_id,
            prompt="Summarize in exactly {{ num_sentences }} sentences:\n\n{{ text }}",
            version=1,
            variables=["num_sentences", "text"],
            set_as_default=True,
        )
        print(f"   ✅ Updated prompt: {updated.prompt_id}")
        print(f"      - New version: {updated.version}")
        print(f"      - Is default: {updated.is_default}")
    except Exception as e:
        print(f"   ❌ Update failed: {e}")
        sys.exit(1)
    print()

    # Test 4: List all versions
    print("7. Listing all versions...")
    try:
        versions_response = await adapter.list_prompt_versions(created.prompt_id)
        versions = versions_response.data
        print(f"   ✅ Found {len(versions)} versions:")
        for v in versions:
            default_marker = " (default)" if v.is_default else ""
            print(f"      - Version {v.version}{default_marker}")
    except Exception as e:
        print(f"   ❌ List versions failed: {e}")
        sys.exit(1)
    print()

    # Test 5: Set default version
    print("8. Setting version 1 as default...")
    try:
        await adapter.set_default_version(created.prompt_id, 1)
        default = await adapter.get_prompt(created.prompt_id)
        print(f"   ✅ Default version changed:")
        print(f"      - Current default: version {default.version}")
    except Exception as e:
        print(f"   ❌ Set default failed: {e}")
        sys.exit(1)
    print()

    # Test 6: List all prompts
    print("9. Listing all prompts...")
    try:
        prompts_response = await adapter.list_prompts()
        prompts = prompts_response.data
        print(f"   ✅ Found {len(prompts)} prompts (showing last 5):")
        for p in prompts[-5:]:
            print(f"      - {p.prompt_id[:20]}... (v{p.version})")
    except Exception as e:
        print(f"   ❌ List prompts failed: {e}")
        sys.exit(1)
    print()

    # Test 7: Auto-extract variables
    print("10. Testing auto-extract variables...")
    try:
        auto_created = await adapter.create_prompt(
            prompt="Role: {{ role }}, Task: {{ task }}, Format: {{ format }}",
        )
        print(f"   ✅ Auto-extracted variables: {auto_created.variables}")
        expected = {"role", "task", "format"}
        if set(auto_created.variables) == expected:
            print(f"      - Matches expected: {expected}")
        else:
            print(f"      - Warning: Expected {expected}")
    except Exception as e:
        print(f"   ❌ Auto-extract failed: {e}")
        sys.exit(1)
    print()

    # Test 8: Get cache stats
    print("11. Getting adapter statistics...")
    try:
        stats = await adapter.get_cache_stats()
        print(f"   ✅ Adapter stats:")
        print(f"      - Enabled: {stats['enabled']}")
        print(f"      - Backend: {stats['backend']}")
        print(f"      - Circuit breaker: {stats['circuit_breaker_state']}")
    except Exception as e:
        print(f"   ❌ Get stats failed: {e}")
        # Non-critical, continue
        pass
    print()

    # Cleanup
    print("12. Cleanup...")
    try:
        await adapter.shutdown()
        print("   ✅ Adapter shutdown complete")
    except Exception as e:
        print(f"   ⚠️  Shutdown warning: {e}")
    print()

    # Summary
    print("=" * 80)
    print("✅ All manual tests passed!")
    print("=" * 80)
    print()
    print("Next steps:")
    print("  - View prompts in MLflow UI: {}/".format(mlflow_uri))
    print("  - Run integration tests: uv run --group test pytest -sv tests/integration/providers/remote/prompts/mlflow/")
    print()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Test failed with unexpected error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
