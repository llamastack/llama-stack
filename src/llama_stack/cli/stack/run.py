# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import argparse
import os
import ssl
import subprocess
import sys
from pathlib import Path

import uvicorn
import yaml

from llama_stack.cli.stack.utils import ImageType
from llama_stack.cli.subcommand import Subcommand
from llama_stack.core.datatypes import StackRunConfig
from llama_stack.core.stack import cast_image_name_to_string, replace_env_vars
from llama_stack.core.utils.config_resolution import Mode, resolve_config_or_distro
from llama_stack.log import LoggingConfig, get_logger

REPO_ROOT = Path(__file__).parent.parent.parent.parent

logger = get_logger(name=__name__, category="cli")


class StackRun(Subcommand):
    def __init__(self, subparsers: argparse._SubParsersAction):
        super().__init__()
        self.parser = subparsers.add_parser(
            "run",
            prog="llama stack run",
            description="""Start the server for a Llama Stack Distribution. You should have already built (or downloaded) and configured the distribution.""",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        self._add_arguments()
        self.parser.set_defaults(func=self._run_stack_run_cmd)

    def _add_arguments(self):
        self.parser.add_argument(
            "config",
            type=str,
            nargs="?",  # Make it optional
            metavar="config | distro",
            help="Path to config file to use for the run or name of known distro (`llama stack list` for a list).",
        )
        self.parser.add_argument(
            "--port",
            type=int,
            help="Port to run the server on. It can also be passed via the env var LLAMA_STACK_PORT.",
            default=int(os.getenv("LLAMA_STACK_PORT", 8321)),
        )
        self.parser.add_argument(
            "--image-name",
            type=str,
            default=None,
            help="[DEPRECATED] This flag is no longer supported. Please activate your virtual environment before running.",
        )
        self.parser.add_argument(
            "--image-type",
            type=str,
            help="[DEPRECATED] This flag is no longer supported. Please activate your virtual environment before running.",
            choices=[e.value for e in ImageType if e.value != ImageType.CONTAINER.value],
        )
        self.parser.add_argument(
            "--enable-ui",
            action="store_true",
            help="Start the UI server",
        )

    def _run_stack_run_cmd(self, args: argparse.Namespace) -> None:
        import yaml

        from llama_stack.core.configure import parse_and_maybe_upgrade_config

        if args.image_type or args.image_name:
            self.parser.error(
                "The --image-type and --image-name flags are no longer supported.\n\n"
                "Please activate your virtual environment manually before running `llama stack run`.\n\n"
                "For example:\n"
                "  source /path/to/venv/bin/activate\n"
                "  llama stack run <config>\n"
            )

        if args.enable_ui:
            self._start_ui_development_server(args.port)

        if args.config:
            try:
                from llama_stack.core.utils.config_resolution import Mode, resolve_config_or_distro

                config_file = resolve_config_or_distro(args.config, Mode.RUN)
            except ValueError as e:
                self.parser.error(str(e))
        else:
            config_file = None

        if config_file:
            logger.info(f"Using run configuration: {config_file}")

            try:
                config_dict = yaml.safe_load(config_file.read_text())
            except yaml.parser.ParserError as e:
                self.parser.error(f"failed to load config file '{config_file}':\n {e}")

            try:
                config = parse_and_maybe_upgrade_config(config_dict)
                if not os.path.exists(str(config.external_providers_dir)):
                    os.makedirs(str(config.external_providers_dir), exist_ok=True)
            except AttributeError as e:
                self.parser.error(f"failed to parse config file '{config_file}':\n {e}")

        self._uvicorn_run(config_file, args)

    def _uvicorn_run(self, config_file: Path | None, args: argparse.Namespace) -> None:
        if not config_file:
            self.parser.error("Config file is required")

        config_file = resolve_config_or_distro(str(config_file), Mode.RUN)
        with open(config_file) as fp:
            config_contents = yaml.safe_load(fp)
            if isinstance(config_contents, dict) and (cfg := config_contents.get("logging_config")):
                logger_config = LoggingConfig(**cfg)
            else:
                logger_config = None
            config = StackRunConfig(**cast_image_name_to_string(replace_env_vars(config_contents)))

        port = args.port or config.server.port
        host = config.server.host or ["::", "0.0.0.0"]

        # Set the config file in environment so create_app can find it
        os.environ["LLAMA_STACK_CONFIG"] = str(config_file)

        uvicorn_config = {
            "factory": True,
            "host": host,
            "port": port,
            "lifespan": "on",
            "log_level": logger.getEffectiveLevel(),
            "log_config": logger_config,
        }

        keyfile = config.server.tls_keyfile
        certfile = config.server.tls_certfile
        if keyfile and certfile:
            uvicorn_config["ssl_keyfile"] = config.server.tls_keyfile
            uvicorn_config["ssl_certfile"] = config.server.tls_certfile
            if config.server.tls_cafile:
                uvicorn_config["ssl_ca_certs"] = config.server.tls_cafile
                uvicorn_config["ssl_cert_reqs"] = ssl.CERT_REQUIRED

            logger.info(
                f"HTTPS enabled with certificates:\n  Key: {keyfile}\n  Cert: {certfile}\n  CA: {config.server.tls_cafile}"
            )
        else:
            logger.info(f"HTTPS enabled with certificates:\n  Key: {keyfile}\n  Cert: {certfile}")

        logger.info(f"Listening on {host}:{port}")

        # We need to catch KeyboardInterrupt because uvicorn's signal handling
        # re-raises SIGINT signals using signal.raise_signal(), which Python
        # converts to KeyboardInterrupt. Without this catch, we'd get a confusing
        # stack trace when using Ctrl+C or kill -2 (SIGINT).
        # SIGTERM (kill -15) works fine without this because Python doesn't
        # have a default handler for it.
        #
        # Another approach would be to ignore SIGINT entirely - let uvicorn handle it through its own
        # signal handling but this is quite intrusive and not worth the effort.
        try:
            # Check if Gunicorn should be disabled (for testing or debugging)
            disable_gunicorn = os.getenv("LLAMA_STACK_DISABLE_GUNICORN", "false").lower() == "true"

            if not disable_gunicorn and sys.platform in ("linux", "darwin"):
                # On Unix-like systems, use Gunicorn with Uvicorn workers for production-grade performance
                self._run_with_gunicorn(host, port, uvicorn_config)
            else:
                # On other systems (e.g., Windows), fall back to Uvicorn directly
                # Also used when LLAMA_STACK_DISABLE_GUNICORN=true (for tests)
                if disable_gunicorn:
                    logger.info("Gunicorn disabled via LLAMA_STACK_DISABLE_GUNICORN environment variable")
                uvicorn.run("llama_stack.core.server.server:create_app", **uvicorn_config)  # type: ignore[arg-type]
        except (KeyboardInterrupt, SystemExit):
            logger.info("Received interrupt signal, shutting down gracefully...")

    def _run_with_gunicorn(self, host: str | list[str], port: int, uvicorn_config: dict) -> None:
        """
        Run the server using Gunicorn with Uvicorn workers.

        This provides production-grade multi-process performance on Unix systems.
        """
        import logging  # allow-direct-logging
        import multiprocessing

        # Calculate number of workers: (2 * CPU cores) + 1 is a common formula
        # Can be overridden by WEB_CONCURRENCY or GUNICORN_WORKERS environment variable
        default_workers = (multiprocessing.cpu_count() * 2) + 1
        num_workers = int(os.getenv("GUNICORN_WORKERS") or os.getenv("WEB_CONCURRENCY") or default_workers)

        # Handle host configuration - Gunicorn expects a single bind address
        # Uvicorn can accept a list of hosts, but Gunicorn binds to one address
        bind_host = host[0] if isinstance(host, list) else host

        # IPv6 addresses need to be wrapped in brackets
        if ":" in bind_host and not bind_host.startswith("["):
            bind_address = f"[{bind_host}]:{port}"
        else:
            bind_address = f"{bind_host}:{port}"

        # Map Python logging level to Gunicorn log level string (from uvicorn_config)
        log_level_map = {
            logging.CRITICAL: "critical",
            logging.ERROR: "error",
            logging.WARNING: "warning",
            logging.INFO: "info",
            logging.DEBUG: "debug",
        }
        log_level = uvicorn_config.get("log_level", logging.INFO)
        gunicorn_log_level = log_level_map.get(log_level, "info")

        # Worker timeout - longer for async workers, configurable via env var
        timeout = int(os.getenv("GUNICORN_TIMEOUT", "120"))

        # Worker connections - concurrent connections per worker
        worker_connections = int(os.getenv("GUNICORN_WORKER_CONNECTIONS", "1000"))

        # Worker recycling to prevent memory leaks
        max_requests = int(os.getenv("GUNICORN_MAX_REQUESTS", "10000"))
        max_requests_jitter = int(os.getenv("GUNICORN_MAX_REQUESTS_JITTER", "1000"))

        # Keep-alive for connection reuse
        keepalive = int(os.getenv("GUNICORN_KEEPALIVE", "5"))

        # Build Gunicorn command
        gunicorn_command = [
            "gunicorn",
            "-k",
            "uvicorn.workers.UvicornWorker",
            "--workers",
            str(num_workers),
            "--worker-connections",
            str(worker_connections),
            "--bind",
            bind_address,
            "--timeout",
            str(timeout),
            "--keep-alive",
            str(keepalive),
            "--max-requests",
            str(max_requests),
            "--max-requests-jitter",
            str(max_requests_jitter),
            "--log-level",
            gunicorn_log_level,
            "--access-logfile",
            "-",  # Log to stdout
            "--error-logfile",
            "-",  # Log to stderr
        ]

        # Preload app for memory efficiency (disabled by default to avoid import issues)
        # Enable with GUNICORN_PRELOAD=true for production deployments
        if os.getenv("GUNICORN_PRELOAD", "true").lower() == "true":
            gunicorn_command.append("--preload")

        # Add SSL configuration if present (from uvicorn_config)
        if uvicorn_config.get("ssl_keyfile") and uvicorn_config.get("ssl_certfile"):
            gunicorn_command.extend(
                [
                    "--keyfile",
                    uvicorn_config["ssl_keyfile"],
                    "--certfile",
                    uvicorn_config["ssl_certfile"],
                ]
            )
            if uvicorn_config.get("ssl_ca_certs"):
                gunicorn_command.extend(["--ca-certs", uvicorn_config["ssl_ca_certs"]])

        # Add the application
        gunicorn_command.append("llama_stack.core.server.server:create_app()")

        # Log comprehensive configuration
        logger.info(f"Starting Gunicorn server with {num_workers} workers on {bind_address}...")
        logger.info("Using Uvicorn workers for ASGI application support")
        logger.info(
            f"Configuration: {worker_connections} connections/worker, {timeout}s timeout, {keepalive}s keepalive"
        )
        logger.info(f"Worker recycling: every {max_requests}±{max_requests_jitter} requests (prevents memory leaks)")
        logger.info(f"Total concurrent capacity: {num_workers * worker_connections} connections")

        try:
            # Execute the Gunicorn command
            subprocess.run(gunicorn_command, check=True)
        except FileNotFoundError:
            logger.error("Error: 'gunicorn' command not found. Please ensure Gunicorn is installed.")
            logger.error("Falling back to Uvicorn...")
            uvicorn.run("llama_stack.core.server.server:create_app", **uvicorn_config)  # type: ignore[arg-type]
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to start Gunicorn server. Error: {e}")
            sys.exit(1)

    def _start_ui_development_server(self, stack_server_port: int):
        logger.info("Attempting to start UI development server...")
        # Check if npm is available
        npm_check = subprocess.run(["npm", "--version"], capture_output=True, text=True, check=False)
        if npm_check.returncode != 0:
            logger.warning(
                f"'npm' command not found or not executable. UI development server will not be started. Error: {npm_check.stderr}"
            )
            return

        ui_dir = REPO_ROOT / "llama_stack" / "ui"
        logs_dir = Path("~/.llama/ui/logs").expanduser()
        try:
            # Create logs directory if it doesn't exist
            logs_dir.mkdir(parents=True, exist_ok=True)

            ui_stdout_log_path = logs_dir / "stdout.log"
            ui_stderr_log_path = logs_dir / "stderr.log"

            # Open log files in append mode
            stdout_log_file = open(ui_stdout_log_path, "a")
            stderr_log_file = open(ui_stderr_log_path, "a")

            process = subprocess.Popen(
                ["npm", "run", "dev"],
                cwd=str(ui_dir),
                stdout=stdout_log_file,
                stderr=stderr_log_file,
                env={**os.environ, "NEXT_PUBLIC_LLAMA_STACK_BASE_URL": f"http://localhost:{stack_server_port}"},
            )
            logger.info(f"UI development server process started in {ui_dir} with PID {process.pid}.")
            logger.info(f"Logs: stdout -> {ui_stdout_log_path}, stderr -> {ui_stderr_log_path}")
            logger.info(f"UI will be available at http://localhost:{os.getenv('LLAMA_STACK_UI_PORT', 8322)}")

        except FileNotFoundError:
            logger.error(
                "Failed to start UI development server: 'npm' command not found. Make sure npm is installed and in your PATH."
            )
        except Exception as e:
            logger.error(f"Failed to start UI development server in {ui_dir}: {e}")
