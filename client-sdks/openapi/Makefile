SHELL := /bin/bash
STAINLESS_DIR := ../stainless
PYTHON := uv run python
# Auto-detect openapi-generator command (npm installs as openapi-generator-cli, brew as openapi-generator)
OPENAPI_GENERATOR := $(shell command -v openapi-generator-cli 2>/dev/null || command -v openapi-generator 2>/dev/null || echo openapi-generator-cli)

MERGE_SCRIPT := ./merge_stainless_to_openapi.py
PATCHES_FILE := ./patches.yml
STAINLESS_CONFIG := $(STAINLESS_DIR)/config.yml
STAINLESS_OPENAPI := $(STAINLESS_DIR)/openapi.yml
OPENAPI_OUTPUT := openapi.yml
OPENAPI_CONFIG := openapi-config.json
SDK_OUTPUT_DIR := sdks/python

# Extract version from root pyproject.toml
VERSION := $(shell grep 'fallback_version' ../../pyproject.toml | cut -d'"' -f2)

.PHONY: all sdk openapi clean help version check-generator

all: sdk

help:
	@echo "Available targets:"
	@echo "  all       - Generate SDK (default)"
	@echo "  openapi   - Generate OpenAPI spec from Stainless config"
	@echo "  sdk       - Generate Python SDK from OpenAPI spec"
	@echo "  version   - Show version that will be used for SDK"
	@echo "  clean     - Remove generated files"
	@echo "  help      - Show this help message"

version:
	@echo "SDK version: $(VERSION)"

check-generator:
	@if ! command -v openapi-generator-cli >/dev/null 2>&1 && ! command -v openapi-generator >/dev/null 2>&1; then \
		echo "Error: openapi-generator not found in PATH"; \
		echo ""; \
		echo "Please install it first:"; \
		echo ""; \
		echo "macOS:"; \
		echo "  brew install openapi-generator"; \
		echo "  OR"; \
		echo "  npm install -g @openapitools/openapi-generator-cli"; \
		echo ""; \
		echo "Linux:"; \
		echo "  npm install -g @openapitools/openapi-generator-cli"; \
		echo ""; \
		echo "For more options: https://openapi-generator.tech/docs/installation"; \
		exit 1; \
	fi
	@command -v java >/dev/null 2>&1 || \
		{ echo "Error: java not found in PATH"; \
		  echo ""; \
		  echo "openapi-generator requires Java 11+"; \
		  echo ""; \
		  echo "macOS:"; \
		  echo "  brew install openjdk"; \
		  echo ""; \
		  echo "Linux:"; \
		  echo "  sudo dnf install java-11-openjdk"; \
		  echo "  (or use your package manager: apt, yum, etc.)"; \
		  echo ""; \
		  exit 1; }

openapi: $(OPENAPI_OUTPUT)

$(OPENAPI_OUTPUT): $(MERGE_SCRIPT) $(PATCHES_FILE) $(STAINLESS_CONFIG) $(STAINLESS_OPENAPI)
	@echo "Generating OpenAPI spec..."
	$(PYTHON) $(MERGE_SCRIPT) --patch $(PATCHES_FILE) --stainless $(STAINLESS_CONFIG) --openapi $(STAINLESS_OPENAPI) --output $(OPENAPI_OUTPUT)

sdk: check-generator $(OPENAPI_OUTPUT)
	@echo "Generating Python SDK (version: $(VERSION))..."
	@mkdir -p $(SDK_OUTPUT_DIR)
	$(OPENAPI_GENERATOR) generate -i $(OPENAPI_OUTPUT) -g python -c $(OPENAPI_CONFIG) -o $(SDK_OUTPUT_DIR) \
		--additional-properties=packageVersion=$(VERSION)

clean:
	@echo "Cleaning generated files..."
	@rm -f $(OPENAPI_OUTPUT)
	@rm -rf sdks
