# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import logging  # allow-direct-logging
import os
import re
from logging.config import dictConfig  # allow-direct-logging

from rich.console import Console
from rich.errors import MarkupError
from rich.logging import RichHandler

from llama_stack.core.datatypes import LoggingConfig

# Default log level
DEFAULT_LOG_LEVEL = logging.INFO

# Predefined categories
CATEGORIES = [
    "core",
    "server",
    "router",
    "inference",
    "agents",
    "safety",
    "eval",
    "tools",
    "client",
    "telemetry",
    "openai",
    "openai_responses",
    "openai_conversations",
    "testing",
    "providers",
    "models",
    "files",
    "vector_io",
    "tool_runtime",
    "cli",
    "post_training",
    "scoring",
    "tests",
]
UNCATEGORIZED = "uncategorized"

# Initialize category levels with default level
_category_levels: dict[str, int] = dict.fromkeys(CATEGORIES, DEFAULT_LOG_LEVEL)


def config_to_category_levels(category: str, level: str):
    """
    Helper function to be called either by environment parsing or yaml parsing to go from a list of categories and levels to a dictionary ready to be
    used by the logger dictConfig.

    Parameters:
        category (str): logging category to apply the level to
        level (str): logging level to be used in the category

    Returns:
        Dict[str, int]: A dictionary mapping categories to their log levels.
    """

    category_levels: dict[str, int] = {}
    level_value = logging._nameToLevel.get(str(level).upper())
    if level_value is None:
        logging.warning(f"Unknown log level '{level}' for category '{category}'. Falling back to default 'INFO'.")
        return category_levels

    if category == "all":
        # Apply the log level to all categories and the root logger
        for cat in CATEGORIES:
            category_levels[cat] = level_value
        # Set the root logger's level to the specified level
        category_levels["root"] = level_value
    elif category in CATEGORIES:
        category_levels[category] = level_value
    else:
        logging.warning(f"Unknown logging category: {category}. No changes made.")
    return category_levels


def parse_yaml_config(yaml_config: LoggingConfig) -> dict[str, int]:
    """
    Helper function to parse a yaml logging configuration found in the run.yaml

    Parameters:
        yaml_config (Logging): the logger config object found in the run.yaml

    Returns:
        Dict[str, int]: A dictionary mapping categories to their log levels.
    """
    category_levels = {}
    for category, level in yaml_config.category_levels.items():
        category_levels.update(config_to_category_levels(category=category, level=level))

    return category_levels


def parse_environment_config(env_config: str) -> dict[str, int]:
    """
    Parse the LLAMA_STACK_LOGGING environment variable and return a dictionary of category log levels.

    Parameters:
        env_config (str): The value of the LLAMA_STACK_LOGGING environment variable.

    Returns:
        Dict[str, int]: A dictionary mapping categories to their log levels.
    """
    category_levels = {}
    delimiter = ","
    for pair in env_config.split(delimiter):
        if not pair.strip():
            continue

        try:
            category, level = pair.split("=", 1)
            category = category.strip().lower()
            level = level.strip().upper()  # Convert to uppercase for logging._nameToLevel
            category_levels.update(config_to_category_levels(category=category, level=level))

        except ValueError:
            logging.warning(f"Invalid logging configuration: '{pair}'. Expected format: 'category=level'.")

    return category_levels


def strip_rich_markup(text):
    """Remove Rich markup tags like [dim], [bold magenta], etc."""
    return re.sub(r"\[/?[a-zA-Z0-9 _#=,]+\]", "", text)


class CustomRichHandler(RichHandler):
    def __init__(self, *args, **kwargs):
        # Set a reasonable default width for console output, especially when redirected to files
        console_width = int(os.environ.get("LLAMA_STACK_LOG_WIDTH", "120"))
        # Don't force terminal codes to avoid ANSI escape codes in log files
        kwargs["console"] = Console(width=console_width)
        super().__init__(*args, **kwargs)

    def emit(self, record):
        """Override emit to handle markup errors gracefully."""
        try:
            super().emit(record)
        except MarkupError:
            original_markup = self.markup
            self.markup = False
            try:
                super().emit(record)
            finally:
                self.markup = original_markup


class CustomFileHandler(logging.FileHandler):
    def __init__(self, filename, mode="a", encoding=None, delay=False):
        super().__init__(filename, mode, encoding, delay)
        # Default formatter to match console output
        self.default_formatter = logging.Formatter("%(asctime)s %(name)s:%(lineno)d %(category)s: %(message)s")
        self.setFormatter(self.default_formatter)

    def emit(self, record):
        if hasattr(record, "msg"):
            record.msg = strip_rich_markup(str(record.msg))
        super().emit(record)


def setup_logging(category_levels: dict[str, int], log_file: str | None) -> None:
    """
    Configure logging based on the provided category log levels and an optional log file.

    Parameters:
        category_levels (Dict[str, int]): A dictionary mapping categories to their log levels.
        log_file (str): Path to a log file to additionally pipe the logs into
    """
    log_format = "%(asctime)s %(name)s:%(lineno)d %(category)s: %(message)s"

    class CategoryFilter(logging.Filter):
        """Ensure category is always present in log records."""

        def filter(self, record):
            if not hasattr(record, "category"):
                record.category = UNCATEGORIZED  # Default to 'uncategorized' if no category found
            return True

    # Determine the root logger's level (default to WARNING if not specified)
    root_level = category_levels.get("root", logging.WARNING)

    handlers = {
        "console": {
            "()": CustomRichHandler,  # Use custom console handler
            "formatter": "rich",
            "rich_tracebacks": True,
            "show_time": False,
            "show_path": False,
            "markup": True,
            "filters": ["category_filter"],
        }
    }

    # Add a file handler if log_file is set
    if log_file:
        handlers["file"] = {
            "()": CustomFileHandler,
            "filename": log_file,
            "mode": "a",
            "encoding": "utf-8",
        }

    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "rich": {
                "()": logging.Formatter,
                "format": log_format,
            }
        },
        "handlers": handlers,
        "filters": {
            "category_filter": {
                "()": CategoryFilter,
            }
        },
        "loggers": {
            category: {
                "handlers": list(handlers.keys()),  # Apply all handlers
                "level": category_levels.get(category, DEFAULT_LOG_LEVEL),
                "propagate": False,  # Disable propagation to root logger
            }
            for category in CATEGORIES
        },
        "root": {
            "handlers": list(handlers.keys()),
            "level": root_level,  # Set root logger's level dynamically
        },
    }
    dictConfig(logging_config)

    # Ensure third-party libraries follow the root log level
    for _, logger in logging.root.manager.loggerDict.items():
        if isinstance(logger, logging.Logger):
            logger.setLevel(root_level)


def get_logger(
    name: str, category: str = "uncategorized", config: LoggingConfig | None | None = None
) -> logging.LoggerAdapter:
    """
    Returns a logger with the specified name and category.
    If no category is provided, defaults to 'uncategorized'.

    Parameters:
        name (str): The name of the logger (e.g., module or filename).
        category (str): The category of the logger (default 'uncategorized').
        config (Logging): optional yaml config to override the existing logger configuration

    Returns:
        logging.LoggerAdapter: Configured logger with category support.
    """
    if config:
        _category_levels.update(parse_yaml_config(config))

    logger = logging.getLogger(name)
    if category in _category_levels:
        log_level = _category_levels[category]
    else:
        root_category = category.split("::")[0]
        if root_category in _category_levels:
            log_level = _category_levels[root_category]
        else:
            if category != UNCATEGORIZED:
                raise ValueError(
                    f"Unknown logging category: {category}. To resolve, choose a valid category from the CATEGORIES list "
                    f"or add it to the CATEGORIES list. Available categories: {CATEGORIES}"
                )
            log_level = _category_levels.get("root", DEFAULT_LOG_LEVEL)
    logger.setLevel(log_level)
    return logging.LoggerAdapter(logger, {"category": category})


env_config = os.environ.get("LLAMA_STACK_LOGGING", "")
if env_config:
    _category_levels.update(parse_environment_config(env_config))

log_file = os.environ.get("LLAMA_STACK_LOG_FILE")

setup_logging(_category_levels, log_file)
