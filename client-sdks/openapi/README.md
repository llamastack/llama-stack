# OpenAPI Generator SDK

Alternative SDK generation using [OpenAPI Generator](https://github.com/OpenAPITools/openapi-generator) instead of Stainless. See [#4609](https://github.com/llamastack/llama-stack/issues/4609) for context.

## Prerequisites

### Java 11+
openapi-generator-cli requires Java 11 or higher.

```bash
# macOS
brew install openjdk

# Fedora/RHEL/CentOS
sudo dnf install java-11-openjdk
# For other Linux distributions, use your package manager (apt, yum, pacman, etc.).
```

### OpenAPI Generator CLI

For more installation options, see: https://openapi-generator.tech/docs/installation

```bash
# macOS
brew install openapi-generator

# Linux (also possible for macOS)
npm install -g @openapitools/openapi-generator-cli
```

### Python Dependencies

```bash
uv pip install ruamel.yaml
```

## Usage

**From the client-sdks/openapi directory:**

```bash
cd client-sdks/openapi

make openapi  # Generate OpenAPI spec from Stainless config
make sdk      # Generate Python SDK
make version  # Show version that will be used
make clean    # Remove generated files
```

The `make sdk` target will automatically check for required dependencies (openapi-generator-cli and java) before generating.

## How it Works

1. Reads base spec from `../stainless/openapi.yml`
2. Enriches with resource mappings from `../stainless/config.yml`
3. Applies patches from `patches.yml`
4. Generates Python SDK using openapi-generator

**Generated files (git-ignored):**
- `openapi.yml` - Enriched OpenAPI specification
- `sdks/python/` - Generated Python SDK
- `.openapi-generator/` - Generator metadata

## Files

- `merge_stainless_to_openapi.py` - Script to merge Stainless config into OpenAPI spec
- `Makefile` - Build orchestration
- `patches.yml` - OpenAPI spec patches
- `openapi-config.json` - Python SDK generation config
- `openapitools.json` - OpenAPI Generator CLI config
- `.openapi-generator-ignore` - Exclusion patterns for generation
