import os
import warnings


def pytest_sessionstart(session) -> None:
    if "LLAMA_STACK_LOGGING" not in os.environ:
        os.environ["LLAMA_STACK_LOGGING"] = "all=WARNING"

    # Silence common deprecation spam during unit tests.
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=PendingDeprecationWarning)


pytest_plugins = ["tests.unit.fixtures"]
