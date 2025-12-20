# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""MLflow Prompt Registry provider implementation.

This module implements the Llama Stack Prompts protocol using MLflow's Prompt Registry
as the backend for centralized prompt management and versioning.
"""

import os
import re

from llama_stack.core.request_headers import NeedsRequestProviderData
from llama_stack.log import get_logger
from llama_stack.providers.remote.prompts.mlflow.config import MLflowPromptsConfig
from llama_stack.providers.remote.prompts.mlflow.mapping import PromptIDMapper
from llama_stack_api import ListPromptsResponse, Prompt, Prompts

# Try importing mlflow at module level
try:
    import mlflow
except ImportError:
    # Fail gracefully when provider is instantiated during initialize()
    mlflow = None

logger = get_logger(__name__)

class MLflowPromptsAdapter(NeedsRequestProviderData, Prompts):
    """MLflow Prompt Registry adapter for Llama Stack.

    This adapter implements the Llama Stack Prompts protocol using MLflow's
    Prompt Registry as the backend storage system. It handles:

    - Bidirectional ID mapping (prompt_id <-> MLflow name)
    - Version management via MLflow versioning
    - Variable extraction from prompt templates
    - Metadata storage in MLflow tags
    - Default version management via MLflow aliases
    - Credential management via provider data (backstopped by config)

    Credentials can be provided via:
    1. Per-request provider data header (preferred for security)
    2. Configuration auth_credential (fallback)
    3. Environment variables (MLFLOW_TRACKING_TOKEN, etc.)

    Attributes:
        config: MLflow provider configuration
        mapper: ID mapping utility
    """

    def __init__(self, config: MLflowPromptsConfig):
        """Initialize MLflow prompts adapter.

        Args:
            config: MLflow provider configuration
        """
        self.config = config
        self.mapper = PromptIDMapper()
        logger.info(
            f"MLflowPromptsAdapter initialized: tracking_uri={config.mlflow_tracking_uri}, "
            f"experiment={config.experiment_name}"
        )

    def _setup_auth(self) -> None:
        """Set up authentication for MLflow operations.

        Checks for per-request credentials in provider data (preferred),
        then falls back to config credentials. Sets MLFLOW_TRACKING_TOKEN
        environment variable which MLflow reads for authentication.
        """
        # Try to get per-request token from provider data
        provider_data = self.get_request_provider_data()
        if provider_data and provider_data.mlflow_api_token:
            os.environ["MLFLOW_TRACKING_TOKEN"] = provider_data.mlflow_api_token
            logger.debug("Using MLflow token from provider data")
            return

        # Fall back to config token (reset in case per-request token was used previously)
        if self.config.auth_credential is not None:
            os.environ["MLFLOW_TRACKING_TOKEN"] = self.config.auth_credential.get_secret_value()
            logger.debug("Using MLflow token from config")

    async def initialize(self) -> None:
        """Initialize MLflow client and set up experiment.

        Sets up MLflow connection with optional authentication via token.
        Token can be provided via config or will be read from environment variables
        (MLFLOW_TRACKING_TOKEN, etc.) as per MLflow's standard behavior.

        Raises:
            ImportError: If mlflow package is not installed
            Exception: If MLflow connection fails
        """
        if mlflow is None:
            raise ImportError(
                "mlflow package is required for MLflow prompts provider. "
                "Install with: pip install 'mlflow>=3.4.0'"
            )

        # Set MLflow URIs
        mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)

        if self.config.mlflow_registry_uri:
            mlflow.set_registry_uri(self.config.mlflow_registry_uri)
        else:
            # Default to tracking URI if registry not specified
            mlflow.set_registry_uri(self.config.mlflow_tracking_uri)

        # Set up authentication
        self._setup_auth()

        # Validate experiment exists (don't create during initialization)
        try:
            mlflow.set_experiment(self.config.experiment_name)
            logger.info(f"Using MLflow experiment: {self.config.experiment_name}")
        except Exception as e:
            logger.warning(
                f"Experiment '{self.config.experiment_name}' not found: {e}. "
                f"It will be created automatically on first prompt creation."
            )

    def _ensure_experiment(self) -> None:
        """Ensure MLflow experiment exists, creating it if necessary.

        This is called lazily on first write operation to avoid creating
        external resources during initialization.
        """
        self._setup_auth()
        try:
            mlflow.set_experiment(self.config.experiment_name)
        except Exception:
            # Experiment doesn't exist, create it
            try:
                mlflow.create_experiment(self.config.experiment_name)
                mlflow.set_experiment(self.config.experiment_name)
                logger.info(f"Created MLflow experiment: {self.config.experiment_name}")
            except Exception as e:
                raise ValueError(
                    f"Failed to create experiment '{self.config.experiment_name}': {e}"
                ) from e

    def _extract_variables(self, template: str) -> list[str]:
        """Extract variables from prompt template.

        Extracts variables in {{ variable }} format from the template.

        Args:
            template: Prompt template string

        Returns:
            List of unique variable names in order of appearance

        Example:
            >>> adapter._extract_variables("Hello {{ name }}, your score is {{ score }}")
            ["name", "score"]
        """
        if not template:
            return []

        # Find all {{ variable }} patterns
        matches = re.findall(r"{{\s*(\w+)\s*}}", template)

        # Return unique variables in order of appearance
        seen = set()
        variables = []
        for var in matches:
            if var not in seen:
                variables.append(var)
                seen.add(var)

        return variables

    async def create_prompt(
        self,
        prompt: str,
        variables: list[str] | None = None,
    ) -> Prompt:
        """Create a new prompt in MLflow registry.

        Args:
            prompt: Prompt template text with {{ variable }} placeholders
            variables: List of variable names (auto-extracted if not provided)

        Returns:
            Created Prompt resource with prompt_id and version=1

        Raises:
            ValueError: If prompt validation fails
            Exception: If MLflow registration fails
        """
        # Ensure experiment exists (lazy creation on first write)
        self._ensure_experiment()

        # Auto-extract variables if not provided
        if variables is None:
            variables = self._extract_variables(prompt)
        else:
            # Validate declared variables match template
            template_vars = set(self._extract_variables(prompt))
            declared_vars = set(variables)
            undeclared = template_vars - declared_vars
            if undeclared:
                raise ValueError(f"Template contains undeclared variables: {sorted(undeclared)}")

        # Generate Llama Stack prompt_id
        prompt_id = Prompt.generate_prompt_id()

        # Convert to MLflow name
        mlflow_name = self.mapper.to_mlflow_name(prompt_id)

        # Prepare metadata tags
        tags = self.mapper.get_metadata_tags(prompt_id, variables)

        # Register in MLflow
        try:
            mlflow.genai.register_prompt(
                name=mlflow_name,
                template=prompt,
                commit_message="Created via Llama Stack",
                tags=tags,
            )
            logger.info(f"Created prompt {prompt_id} (MLflow: {mlflow_name})")
        except Exception as e:
            logger.error(f"Failed to register prompt in MLflow: {e}")
            raise

        # Set as default (first version is always default)
        try:
            mlflow.genai.set_prompt_alias(
                name=mlflow_name,
                version=1,
                alias="default",
            )
        except Exception as e:
            logger.warning(f"Failed to set default alias for {prompt_id}: {e}")

        return Prompt(
            prompt_id=prompt_id,
            prompt=prompt,
            version=1,
            variables=variables,
            is_default=True,
        )

    async def get_prompt(
        self,
        prompt_id: str,
        version: int | None = None,
    ) -> Prompt:
        """Get prompt from MLflow registry.

        Args:
            prompt_id: Llama Stack prompt ID
            version: Version number (defaults to default version)

        Returns:
            Prompt resource

        Raises:
            ValueError: If prompt not found
        """
        self._setup_auth()
        mlflow_name = self.mapper.to_mlflow_name(prompt_id)

        # Build MLflow URI
        if version:
            uri = f"prompts:/{mlflow_name}/{version}"
        else:
            uri = f"prompts:/{mlflow_name}@default"

        # Load from MLflow
        try:
            mlflow_prompt = mlflow.genai.load_prompt(uri)
        except Exception as e:
            raise ValueError(f"Prompt {prompt_id} (version {version if version else 'default'}) not found: {e}") from e

        # Extract template
        template = mlflow_prompt.template if hasattr(mlflow_prompt, "template") else str(mlflow_prompt)

        # Extract variables from template
        variables = self._extract_variables(template)

        # Get version number
        prompt_version = 1
        if hasattr(mlflow_prompt, "version"):
            prompt_version = int(mlflow_prompt.version)
        elif version:
            prompt_version = version

        # Check if this is the default version
        is_default = await self._is_default_version(mlflow_name, prompt_version)

        return Prompt(
            prompt_id=prompt_id,
            prompt=template,
            version=prompt_version,
            variables=variables,
            is_default=is_default,
        )

    async def update_prompt(
        self,
        prompt_id: str,
        prompt: str,
        version: int,
        variables: list[str] | None = None,
        set_as_default: bool = True,
    ) -> Prompt:
        """Update prompt (creates new version in MLflow).

        Args:
            prompt_id: Llama Stack prompt ID
            prompt: Updated prompt template
            version: Current version being updated
            variables: Updated variables list (auto-extracted if not provided)
            set_as_default: Set new version as default

        Returns:
            Updated Prompt resource with incremented version

        Raises:
            ValueError: If current version not found or validation fails
        """
        # Ensure experiment exists (edge case: updating prompts created outside Llama Stack)
        self._ensure_experiment()

        # Auto-extract variables if not provided
        if variables is None:
            variables = self._extract_variables(prompt)
        else:
            # Validate variables
            template_vars = set(self._extract_variables(prompt))
            declared_vars = set(variables)
            undeclared = template_vars - declared_vars
            if undeclared:
                raise ValueError(f"Template contains undeclared variables: {sorted(undeclared)}")

        mlflow_name = self.mapper.to_mlflow_name(prompt_id)

        # Get all versions to determine the latest and next version number
        versions_response = await self.list_prompt_versions(prompt_id)
        if not versions_response.data:
            raise ValueError(f"Prompt {prompt_id} not found")

        max_version = max(p.version for p in versions_response.data)

        # Verify the provided version is the latest
        if version != max_version:
            raise ValueError(
                f"Version {version} is not the latest version. Use latest version {max_version} to update."
            )

        new_version = max_version + 1

        # Prepare metadata tags
        tags = self.mapper.get_metadata_tags(prompt_id, variables)

        # Register new version in MLflow
        try:
            mlflow.genai.register_prompt(
                name=mlflow_name,
                template=prompt,
                commit_message=f"Updated from version {version} via Llama Stack",
                tags=tags,
            )
            logger.info(f"Updated prompt {prompt_id} to version {new_version}")
        except Exception as e:
            logger.error(f"Failed to update prompt in MLflow: {e}")
            raise

        # Set as default if requested
        if set_as_default:
            try:
                mlflow.genai.set_prompt_alias(
                    name=mlflow_name,
                    version=new_version,
                    alias="default",
                )
            except Exception as e:
                logger.warning(f"Failed to set default alias: {e}")

        return Prompt(
            prompt_id=prompt_id,
            prompt=prompt,
            version=new_version,
            variables=variables,
            is_default=set_as_default,
        )

    async def delete_prompt(self, prompt_id: str) -> None:
        """Delete prompt from MLflow registry.

        Note: MLflow Prompt Registry does not support deletion of registered prompts.
        This method will raise NotImplementedError.

        Args:
            prompt_id: Llama Stack prompt ID

        Raises:
            NotImplementedError: MLflow doesn't support prompt deletion
        """
        # MLflow doesn't support deletion of registered prompts
        # Options:
        # 1. Raise NotImplementedError (current approach)
        # 2. Mark as deleted with tag (soft delete)
        # 3. Delete all versions individually (if API exists)

        raise NotImplementedError(
            "MLflow Prompt Registry does not support deletion. Consider using tags to mark prompts as archived/deleted."
        )

    async def list_prompts(self) -> ListPromptsResponse:
        """List all prompts (default versions only).

        Returns:
            ListPromptsResponse with default version of each prompt

        Note:
            Only lists prompts created/managed by Llama Stack
            (those with llama_stack_managed=true tag)
        """
        self._setup_auth()
        try:
            # Search for Llama Stack managed prompts using metadata tags
            prompts = mlflow.genai.search_prompts(filter_string="tag.llama_stack_managed='true'")
        except Exception as e:
            logger.error(f"Failed to search prompts in MLflow: {e}")
            return ListPromptsResponse(data=[])

        llama_prompts = []
        for mlflow_prompt in prompts:
            try:
                # Convert MLflow name to Llama Stack ID
                prompt_id = self.mapper.to_llama_id(mlflow_prompt.name)

                # Get default version
                llama_prompt = await self.get_prompt(prompt_id)
                llama_prompts.append(llama_prompt)
            except (ValueError, Exception) as e:
                # Skip prompts that can't be converted or retrieved
                logger.warning(f"Skipping prompt {mlflow_prompt.name}: {e}")
                continue

        # Sort by prompt_id
        llama_prompts.sort(key=lambda p: p.prompt_id, reverse=True)

        return ListPromptsResponse(data=llama_prompts)

    async def list_prompt_versions(self, prompt_id: str) -> ListPromptsResponse:
        """List all versions of a specific prompt.

        Args:
            prompt_id: Llama Stack prompt ID

        Returns:
            ListPromptsResponse with all versions of the prompt

        Raises:
            ValueError: If prompt not found
        """
        # MLflow doesn't have a direct "list versions" API for prompts
        # We need to iterate and try to load each version
        versions = []
        version_num = 1
        max_attempts = 100  # Safety limit

        while version_num <= max_attempts:
            try:
                prompt = await self.get_prompt(prompt_id, version_num)
                versions.append(prompt)
                version_num += 1
            except ValueError:
                # No more versions
                break
            except Exception as e:
                logger.warning(f"Error loading version {version_num} of {prompt_id}: {e}")
                break

        if not versions:
            raise ValueError(f"Prompt {prompt_id} not found")

        # Sort by version number
        versions.sort(key=lambda p: p.version)

        return ListPromptsResponse(data=versions)

    async def set_default_version(self, prompt_id: str, version: int) -> Prompt:
        """Set default version using MLflow alias.

        Args:
            prompt_id: Llama Stack prompt ID
            version: Version number to set as default

        Returns:
            Prompt resource with is_default=True

        Raises:
            ValueError: If version not found
        """
        # Ensure experiment exists (edge case: managing prompts created outside Llama Stack)
        self._ensure_experiment()

        mlflow_name = self.mapper.to_mlflow_name(prompt_id)

        # Verify version exists (get_prompt calls _setup_auth internally)
        try:
            prompt = await self.get_prompt(prompt_id, version)
        except ValueError as e:
            raise ValueError(f"Cannot set default: {e}") from e

        # Set "default" alias in MLflow
        self._setup_auth()
        try:
            mlflow.genai.set_prompt_alias(
                name=mlflow_name,
                version=version,
                alias="default",
            )
            logger.info(f"Set version {version} as default for {prompt_id}")
        except Exception as e:
            logger.error(f"Failed to set default version: {e}")
            raise

        # Update is_default flag
        prompt.is_default = True

        return prompt

    async def _is_default_version(self, mlflow_name: str, version: int) -> bool:
        """Check if a version is the default version.

        Args:
            mlflow_name: MLflow prompt name
            version: Version number

        Returns:
            True if this version is the default, False otherwise
        """
        self._setup_auth()
        try:
            # Try to load with @default alias
            default_uri = f"prompts:/{mlflow_name}@default"
            default_prompt = mlflow.genai.load_prompt(default_uri)

            # Get default version number
            default_version = 1
            if hasattr(default_prompt, "version"):
                default_version = int(default_prompt.version)

            return version == default_version
        except Exception:
            # If default doesn't exist or can't be determined, assume False
            return False

    async def shutdown(self) -> None:
        """Cleanup resources (no-op for MLflow)."""
        pass
