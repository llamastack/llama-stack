# Build Scripts

This directory contains scripts used during the build process.

## generate_build_info.py

This script generates `llama_stack/cli/build_info.py` with hardcoded git information at build time. This ensures that version information is captured at build time rather than being read dynamically at runtime.

### Usage

The script is automatically run during the build process via the custom `BuildWithBuildInfo` class in `setup.py`. You can also run it manually:

```bash
python scripts/generate_build_info.py
```

### Generated Information

The script captures the following information:
- Git commit hash (short form)
- Git commit date
- Git branch name
- Git tag (latest)
- Build timestamp

### CI/CD Integration

The script handles common CI/CD scenarios:
- Detached HEAD state (common in CI environments)
- Missing git information (fallback to environment variables)
- Git not available (fallback to "unknown" values)

### Environment Variables

When running in CI/CD environments where the branch name might not be available via git (detached HEAD), the script checks these environment variables:
- `GITHUB_REF_NAME` (GitHub Actions)
- `CI_COMMIT_REF_NAME` (GitLab CI)
- `BUILDKITE_BRANCH` (Buildkite)
- `TRAVIS_BRANCH` (Travis CI)

### Build Integration

The build info generation is integrated into the build process via:
1. `setup.py` - Custom build command that runs the script before building
2. `pyproject.toml` - Standard Python package configuration

This ensures that every build includes up-to-date git information without requiring runtime git access.
