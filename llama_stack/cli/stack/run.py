# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import argparse
import os
import ssl
import subprocess
from pathlib import Path

import uvicorn
import yaml

from llama_stack.cli.stack.utils import ImageType
from llama_stack.cli.subcommand import Subcommand
from llama_stack.core.datatypes import LoggingConfig, StackRunConfig
from llama_stack.core.stack import cast_image_name_to_string, replace_env_vars, validate_env_pair
from llama_stack.core.utils.config_resolution import Mode, resolve_config_or_distro
from llama_stack.log import get_logger

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
            help="Name of the image to run. Defaults to the current environment",
        )
        self.parser.add_argument(
            "--env",
            action="append",
            help="Environment variables to pass to the server in KEY=VALUE format. Can be specified multiple times.",
            metavar="KEY=VALUE",
        )
        self.parser.add_argument(
            "--image-type",
            type=str,
            help="Image Type used during the build. This can be only venv.",
            choices=[e.value for e in ImageType if e.value != ImageType.CONTAINER.value],
        )
        self.parser.add_argument(
            "--enable-ui",
            action="store_true",
            help="Start the UI server",
        )

    def _resolve_config_and_distro(self, args: argparse.Namespace) -> tuple[Path | None, str | None]:
        """Resolve config file path and distribution name from args.config"""
        from llama_stack.core.utils.config_dirs import DISTRIBS_BASE_DIR

        if not args.config:
            return None, None

        config_file = Path(args.config)
        has_yaml_suffix = args.config.endswith(".yaml")
        distro_name = None

        if not config_file.exists() and not has_yaml_suffix:
            # check if this is a distribution
            config_file = Path(REPO_ROOT) / "llama_stack" / "distributions" / args.config / "run.yaml"
            if config_file.exists():
                distro_name = args.config

        if not config_file.exists() and not has_yaml_suffix:
            # check if it's a build config saved to ~/.llama dir
            config_file = Path(DISTRIBS_BASE_DIR / f"llamastack-{args.config}" / f"{args.config}-run.yaml")

        if not config_file.exists():
            self.parser.error(
                f"File {str(config_file)} does not exist.\n\nPlease run `llama stack build` to generate (and optionally edit) a run.yaml file"
            )

        if not config_file.is_file():
            self.parser.error(
                f"Config file must be a valid file path, '{config_file}' is not a file: type={type(config_file)}"
            )

        return config_file, distro_name

    def _run_stack_run_cmd(self, args: argparse.Namespace) -> None:
        import yaml

        from llama_stack.core.configure import parse_and_maybe_upgrade_config
        from llama_stack.core.utils.exec import formulate_run_args, run_command

        if args.enable_ui:
            self._start_ui_development_server(args.port)
        image_type, image_name = args.image_type, args.image_name

        if args.config:
            try:
                from llama_stack.core.utils.config_resolution import Mode, resolve_config_or_distro

                config_file = resolve_config_or_distro(args.config, Mode.RUN)
            except ValueError as e:
                self.parser.error(str(e))
        else:
            config_file = None

        # Check if config is required based on image type
        if image_type == ImageType.VENV.value and not config_file:
            self.parser.error("Config file is required for venv environment")

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
        else:
            config = None

        # If neither image type nor image name is provided, assume the server should be run directly
        # using the current environment packages.
        if not image_type and not image_name:
            logger.info("No image type or image name provided. Assuming environment packages.")
            self._uvicorn_run(config_file, args)
        else:
            run_args = formulate_run_args(image_type, image_name)

            run_args.extend([str(args.port)])

            if config_file:
                run_args.extend(["--config", str(config_file)])

            if args.env:
                for env_var in args.env:
                    if "=" not in env_var:
                        self.parser.error(f"Environment variable '{env_var}' must be in KEY=VALUE format")
                        return
                    key, value = env_var.split("=", 1)  # split on first = only
                    if not key:
                        self.parser.error(f"Environment variable '{env_var}' has empty key")
                        return
                    run_args.extend(["--env", f"{key}={value}"])

            run_command(run_args)

    def _uvicorn_run(self, config_file: Path | None, args: argparse.Namespace) -> None:
        if not config_file:
            self.parser.error("Config file is required")

        # Set environment variables if provided
        if args.env:
            for env_pair in args.env:
                try:
                    key, value = validate_env_pair(env_pair)
                    logger.info(f"Setting environment variable {key} => {value}")
                    os.environ[key] = value
                except ValueError as e:
                    logger.error(f"Error: {str(e)}")
                    self.parser.error(f"Invalid environment variable format: {env_pair}")

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
            uvicorn.run("llama_stack.core.server.server:create_app", **uvicorn_config)
        except (KeyboardInterrupt, SystemExit):
            logger.info("Received interrupt signal, shutting down gracefully...")

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
