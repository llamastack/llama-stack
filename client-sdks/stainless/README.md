These are the source-of-truth configuration files used to generate the Stainless client SDKs via Stainless.

- `openapi.json`: this is the OpenAPI specification for the Llama Stack API (JSON format).
- `openapi.stainless.yml`: this is the Stainless _configuration_ which instructs Stainless how to generate the client SDKs.

A small side note: the OpenAPI spec uses `.json` format, while the Stainless configuration uses `.yml` suffix since Stainless uses that suffix typically for its configuration files.

These files go hand-in-hand. As of now, only the `openapi.json` file is automatically generated using the `run_openapi_generator.sh` script.
