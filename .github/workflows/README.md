# Llama Stack CI

Llama Stack uses GitHub Actions for Continuous Integration (CI). Below is a table detailing what CI the project includes and the purpose.

| Name | File | Purpose |
| ---- | ---- | ------- |
| Backward Compatibility Check | [backward-compat.yml](backward-compat.yml) | Check backward compatibility for run.yaml configs |
| Update Changelog | [changelog.yml](changelog.yml) | Creates PR for updating the CHANGELOG.md |
| API Conformance Tests | [conformance.yml](conformance.yml) | Run the API Conformance test suite on the changes. |
| Installer CI | [install-script-ci.yml](install-script-ci.yml) | Test the installation script |
| Integration Auth Tests | [integration-auth-tests.yml](integration-auth-tests.yml) | Run the integration test suite with Kubernetes authentication |
| SqlStore Integration Tests | [integration-sql-store-tests.yml](integration-sql-store-tests.yml) | Run the integration test suite with SqlStore |
| Integration Tests (Replay) | [integration-tests.yml](integration-tests.yml) | Run the integration test suites from tests/integration in replay mode |
| Vector IO Integration Tests | [integration-vector-io-tests.yml](integration-vector-io-tests.yml) | Run the integration test suite with various VectorIO providers |
| Pre-commit | [pre-commit.yml](pre-commit.yml) | Run pre-commit checks |
| Test Llama Stack Build | [providers-build.yml](providers-build.yml) | Test llama stack build |
| Test llama stack list-deps | [providers-list-deps.yml](providers-list-deps.yml) | Test llama stack list-deps |
| Python Package Build Test | [python-build-test.yml](python-build-test.yml) | Test building the llama-stack PyPI project |
| Integration Tests (Record) | [record-integration-tests.yml](record-integration-tests.yml) | Run the integration test suite from tests/integration |
| Check semantic PR titles | [semantic-pr.yml](semantic-pr.yml) | Ensure that PR titles follow the conventional commit spec |
| Stainless SDK Builds | [stainless-builds.yml](stainless-builds.yml) | Build Stainless SDK from OpenAPI spec changes |
| Close stale issues and PRs | [stale_bot.yml](stale_bot.yml) | Run the Stale Bot action |
| Test External Providers Installed via Module | [test-external-provider-module.yml](test-external-provider-module.yml) | Test External Provider installation via Python module |
| Test External API and Providers | [test-external.yml](test-external.yml) | Test the External API and Provider mechanisms |
| UI Tests | [ui-unit-tests.yml](ui-unit-tests.yml) | Run the UI test suite |
| Unit Tests | [unit-tests.yml](unit-tests.yml) | Run the unit test suite |
