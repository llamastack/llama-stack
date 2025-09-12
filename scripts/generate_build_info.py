#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# This file is auto-generated during build time
# DO NOT EDIT MANUALLY

import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path


def get_git_info():
    """Get git information for build"""
    git_info = {
        "git_commit": "unknown",
        "git_commit_date": "unknown",
        "git_branch": "unknown",
        "git_tag": "unknown",
    }

    try:
        # Get current directory - assume script is in scripts/ directory
        repo_root = Path(__file__).parent.parent

        # Get commit hash
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            git_info["git_commit"] = result.stdout.strip()[:12]  # Short hash

        # Get commit date
        result = subprocess.run(
            ["git", "log", "-1", "--format=%ci"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            git_info["git_commit_date"] = result.stdout.strip()

        # Get current branch
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            branch = result.stdout.strip()
            # If we're in detached HEAD state (common in CI), try to get branch from env
            if branch == "HEAD":
                # Try common CI environment variables
                import os

                branch = (
                    os.getenv("GITHUB_REF_NAME")
                    or os.getenv("CI_COMMIT_REF_NAME")  # GitLab
                    or os.getenv("BUILDKITE_BRANCH")
                    or os.getenv("TRAVIS_BRANCH")
                    or "HEAD"
                )
            git_info["git_branch"] = branch

        # Get latest tag
        result = subprocess.run(
            ["git", "describe", "--tags", "--abbrev=0"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            git_info["git_tag"] = result.stdout.strip()
        else:
            # If no tags, try to get the closest tag with distance
            result = subprocess.run(
                ["git", "describe", "--tags", "--always"],
                cwd=repo_root,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                git_info["git_tag"] = result.stdout.strip()

    except (subprocess.TimeoutExpired, subprocess.SubprocessError, FileNotFoundError) as e:
        print(f"Warning: Could not get git information: {e}", file=sys.stderr)

    return git_info


def generate_build_info_file():
    """Generate the build_info.py file with current git information"""
    git_info = get_git_info()

    # Add build timestamp
    build_timestamp = datetime.now(UTC).isoformat()

    # Get the target file path
    script_dir = Path(__file__).parent
    target_file = script_dir.parent / "llama_stack" / "cli" / "build_info.py"

    # Generate the content
    content = f"""#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# This file is auto-generated during build time
# DO NOT EDIT MANUALLY

BUILD_INFO = {{
    "git_commit": "{git_info["git_commit"]}",
    "git_commit_date": "{git_info["git_commit_date"]}",
    "git_branch": "{git_info["git_branch"]}",
    "git_tag": "{git_info["git_tag"]}",
    "build_timestamp": "{build_timestamp}",
}}
"""

    # Write the file
    try:
        target_file.write_text(content)
        print(f"Generated build info file: {target_file}")
        print(f"Git commit: {git_info['git_commit']}")
        print(f"Git branch: {git_info['git_branch']}")
        print(f"Git tag: {git_info['git_tag']}")
        print(f"Build timestamp: {build_timestamp}")
    except Exception as e:
        print(f"Error writing build info file: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    generate_build_info_file()
