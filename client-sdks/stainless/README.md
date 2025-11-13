These are the source-of-truth configuration files used to generate the Stainless client SDKs via Stainless.

- `openapi.yml`: this is the OpenAPI specification for the Llama Stack API.
- `config.yml`: this is the Stainless _configuration_ which instructs Stainless how to generate the client SDKs.

A small side note: notice the `.yml` suffixes since Stainless uses that suffix typically for its configuration files.

These files go hand-in-hand. As of now, only the `openapi.yml` file is automatically generated using the `run_openapi_generator.sh` script.
