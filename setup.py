#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import subprocess
import sys
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py


class BuildWithBuildInfo(build_py):
    """Custom build command that generates build info before building"""

    def run(self):
        # Generate build info before building
        self.generate_build_info()
        # Run the standard build
        super().run()

    def generate_build_info(self):
        """Generate build_info.py with current git information"""
        script_path = Path(__file__).parent / "scripts" / "generate_build_info.py"

        try:
            # Run the build info generation script
            result = subprocess.run(
                [sys.executable, str(script_path)],
                check=True,
                capture_output=True,
                text=True,
            )
            print("Build info generation completed successfully")
            if result.stdout:
                print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"Warning: Failed to generate build info: {e}")
            if e.stderr:
                print(f"Error output: {e.stderr}")
            # Don't fail the build, just continue with default values


if __name__ == "__main__":
    setup(
        cmdclass={
            "build_py": BuildWithBuildInfo,
        },
    )
