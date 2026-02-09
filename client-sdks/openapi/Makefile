STAINLESS_DIR := ../stainless
PYTHON := python
OPENAPI_GENERATOR := openapi-generator-cli

MERGE_SCRIPT := ./merge_stainless_to_openapi.py
PATCHES_FILE := ./patches.yml
STAINLESS_CONFIG := $(STAINLESS_DIR)/config.yml
STAINLESS_OPENAPI := $(STAINLESS_DIR)/openapi.yml
OPENAPI_OUTPUT := openapi.yml
OPENAPI_CONFIG := openapi-config.json
SDK_OUTPUT_DIR := sdks/python

# Extract version from root pyproject.toml
VERSION := $(shell grep 'fallback_version' ../../pyproject.toml | sed 's/.*"\(.*\)".*/\1/')

.PHONY: all sdk openapi clean help version

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

openapi: $(OPENAPI_OUTPUT)

$(OPENAPI_OUTPUT): $(MERGE_SCRIPT) $(PATCHES_FILE) $(STAINLESS_CONFIG) $(STAINLESS_OPENAPI)
	@echo "Generating OpenAPI spec..."
	$(PYTHON) $(MERGE_SCRIPT) --patch $(PATCHES_FILE) --stainless $(STAINLESS_CONFIG) --openapi $(STAINLESS_OPENAPI)

sdk: $(OPENAPI_OUTPUT)
	@echo "Generating Python SDK (version: $(VERSION))..."
	@mkdir -p $(SDK_OUTPUT_DIR)
	$(OPENAPI_GENERATOR) generate -i $(OPENAPI_OUTPUT) -g python -c $(OPENAPI_CONFIG) -o $(SDK_OUTPUT_DIR) \
		--additional-properties=packageVersion=$(VERSION)

clean:
	@echo "Cleaning generated files..."
	@rm -f $(OPENAPI_OUTPUT)
	@rm -rf sdks
