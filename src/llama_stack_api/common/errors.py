# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# Custom Llama Stack Exception classes should follow the following schema
#   1. All classes should inherit from an existing Built-In Exception class: https://docs.python.org/3/library/exceptions.html
#   2. All classes should have a custom error message with the goal of informing the Llama Stack user specifically
#   3. All classes should propogate the inherited __init__ function otherwise via 'super().__init__(message)'

from typing import Literal

import httpx


class LlamaStackError(Exception):
    """A base class for all Llama Stack errors with an HTTP status code for API responses."""

    status_code: httpx.codes

    def __init__(self, message: str):
        super().__init__(message)


class ClientListCommand:
    """
    A formatted client list command string.
    Args:
        command: The command to list the resources.
        arguments: The arguments to the command.
        resource_name_plural: The plural name of the resource.

    Returns:
        A formatted client list command string: "Use 'client.files.list()' to list available files."
    """

    def __init__(
        self,
        command: str,
        arguments: list[str] | str | None = None,
        resource_name_plural: str | None = None,
    ):
        self.resource_name_plural = resource_name_plural
        self.command = command
        self.arguments = arguments

    def __str__(self) -> str:
        args_str = ""
        resource_name_str = ""
        if self.arguments:
            if isinstance(self.arguments, list):
                args_str = ", ".join(f'"{arg}"' for arg in self.arguments)
            else:
                args_str = f'"{self.arguments}"'
        if self.resource_name_plural:
            resource_name_str = f" to list available {self.resource_name_plural.lower()}"

        return f"Use 'client.{self.command}({args_str})'{resource_name_str}."


class ResourceNotFoundError(ValueError, LlamaStackError):
    """generic exception for a missing Llama Stack resource"""

    status_code: httpx.codes = httpx.codes.NOT_FOUND

    def __init__(self, resource_name: str, resource_type: str, client_list: ClientListCommand | None = None) -> None:
        message = f"{resource_type} '{resource_name}' not found."
        if client_list:
            if not client_list.resource_name_plural:
                client_list.resource_name_plural = f"{resource_type}s"
            message += f" {client_list}"
        super().__init__(message)


class ModelNotFoundError(ResourceNotFoundError):
    """raised when Llama Stack cannot find a referenced model"""

    def __init__(self, model_name: str) -> None:
        super().__init__(model_name, "Model", ClientListCommand("models.list"))


class VectorStoreNotFoundError(ResourceNotFoundError):
    """raised when Llama Stack cannot find a referenced vector store"""

    def __init__(self, vector_store_name: str) -> None:
        super().__init__(vector_store_name, "Vector Store", ClientListCommand("vector_dbs.list"))


class DatasetNotFoundError(ResourceNotFoundError):
    """raised when Llama Stack cannot find a referenced dataset"""

    def __init__(self, dataset_name: str) -> None:
        super().__init__(dataset_name, "Dataset", ClientListCommand("datasets.list"))


class ToolGroupNotFoundError(ResourceNotFoundError):
    """raised when Llama Stack cannot find a referenced tool group"""

    def __init__(self, toolgroup_name: str) -> None:
        super().__init__(toolgroup_name, "Tool Group", ClientListCommand("toolgroups.list"))


class ConversationNotFoundError(ResourceNotFoundError):
    """raised when Llama Stack cannot find a referenced conversation"""

    def __init__(self, conversation_id: str) -> None:
        super().__init__(conversation_id, "Conversation")


class ConversationItemNotFoundError(ResourceNotFoundError):
    """raised when Llama Stack cannot find a referenced conversation"""

    def __init__(self, item_id: str, conversation_id: str) -> None:
        super().__init__(item_id, "Conversation Item", ClientListCommand("conversations.items.list", conversation_id))


class ResponseNotFoundError(ResourceNotFoundError):
    """raised when Llama Stack cannot find a referenced response"""

    def __init__(self, response_id: str) -> None:
        super().__init__(response_id, "Response")


class ConnectorNotFoundError(ResourceNotFoundError):
    """raised when Llama Stack cannot find a referenced connector"""

    def __init__(self, connector_id: str) -> None:
        super().__init__(connector_id, "Connector", ClientListCommand("connectors.list"))


class ConnectorToolNotFoundError(ResourceNotFoundError):
    """raised when Llama Stack cannot find a referenced tool in a connector"""

    def __init__(self, connector_id: str, tool_name: str) -> None:
        super().__init__(
            resource_name=f"{connector_id}.{tool_name}",
            resource_type="Connector Tool",
            client_list=ClientListCommand("connectors.list_tools", connector_id),
        )


class OpenAIFileObjectNotFoundError(ResourceNotFoundError):
    """raised when Llama Stack cannot find a referenced file"""

    def __init__(self, file_id: str) -> None:
        super().__init__(file_id, "File", ClientListCommand("files.list"))


class BatchNotFoundError(ResourceNotFoundError):
    """raised when Llama Stack cannot find a referenced batch"""

    def __init__(self, batch_id: str) -> None:
        self.batch_id = batch_id
        super().__init__(batch_id, "Batch", ClientListCommand("batches.list", resource_name_plural="batches"))


class UnsupportedModelError(LlamaStackError):
    """raised when model is not present in the list of supported models"""

    status_code: httpx.codes = httpx.codes.BAD_REQUEST

    def __init__(self, model_name: str, supported_models_list: list[str]):
        message = f"'{model_name}' model is not supported. Supported models are: {', '.join(supported_models_list)}"
        super().__init__(message)


class ModelTypeError(LlamaStackError):
    """raised when a model is present but not the correct type"""

    status_code: httpx.codes = httpx.codes.BAD_REQUEST

    def __init__(self, model_name: str, model_type: str, expected_model_type: str) -> None:
        message = (
            f"Model '{model_name}' is of type '{model_type}' rather than the expected type '{expected_model_type}'"
        )
        super().__init__(message)


class ConflictError(LlamaStackError):
    """raised when an operation cannot be performed due to a conflict with the current state"""

    status_code: httpx.codes = httpx.codes.CONFLICT

    def __init__(self, message: str) -> None:
        super().__init__(message)


class TokenValidationError(LlamaStackError):
    """raised when token validation fails during authentication"""

    status_code: httpx.codes = httpx.codes.UNAUTHORIZED

    def __init__(self, message: str) -> None:
        super().__init__(message)


class InvalidConversationIdError(LlamaStackError):
    """raised when a conversation ID has an invalid format"""

    status_code: httpx.codes = httpx.codes.BAD_REQUEST

    def __init__(self, conversation_id: str, hint: str | None = None) -> None:
        message = f"Invalid conversation ID '{conversation_id}'." + (f" {hint}" if hint else "")
        super().__init__(message)


class InvalidParameterError(LlamaStackError):
    """raised when a request parameter has an invalid value"""

    status_code: httpx.codes = httpx.codes.BAD_REQUEST

    def __init__(self, param_name: str, value: object, constraint: str) -> None:
        message = f"Invalid value for '{param_name}': {value}. {constraint}"
        super().__init__(message)


class MissingRequiredParameterError(LlamaStackError):
    """raised when a required request parameter is missing"""

    status_code: httpx.codes = httpx.codes.BAD_REQUEST

    def __init__(self, param_name: str) -> None:
        message = f"Missing required parameter: '{param_name}'"
        super().__init__(message)


class ServiceNotEnabledError(LlamaStackError):
    """raised when a llama stack service is not enabled"""

    status_code: httpx.codes = httpx.codes.SERVICE_UNAVAILABLE

    def __init__(
        self,
        service_name: str,
        service_config_name: str | None = None,
        *,
        provider_specific_hint_override: str | None = None,
    ) -> None:
        """
        Args:
            service_name: The name of the service that is not enabled.
            service_config_name: The name of the service configuration to enable in the run config. This is used to give a specific hint to the user on what to enable.
            provider_specific_hint_override: A provider specific hint to override the default hint. This will completely override the default hint if provided.
        """
        message = f"Service '{service_name}' is not enabled."
        if provider_specific_hint_override:
            message += f" {provider_specific_hint_override}"
        else:
            message += f" Please check your configuration and enable {service_config_name.strip() if service_config_name else 'the service'} before trying again."
        super().__init__(message)


class InternalServerError(LlamaStackError):
    """
    A generic server side error that is not caused by the user's request. Sensitive data
    or details of the internal workings of the server should never be exposed to the user.
    Instead, sanitized error information should be logged for debugging purposes.
    """

    status_code: httpx.codes = httpx.codes.INTERNAL_SERVER_ERROR

    def __init__(self) -> None:
        message = "An internal error occurred while processing your request."
        super().__init__(message)


class ConfigurationError(Exception):
    """
    An error that gets raised during server initialization when the server configuration is invalid.
    ConfigurationErrors are shielded from returning sensitive information to API users because they
    are configured to return a generic 500 Internal Server Error response if they get raised during
    normal operation of the server.
    """

    status_code: httpx.codes = httpx.codes.INTERNAL_SERVER_ERROR
    sanitized_message: str = "An internal error occurred while processing your request."

    def __init__(self, message: str) -> None:
        super().__init__(message)


class ServiceMisconfiguredError(ConfigurationError):
    """
    An error that gets raised during server initialization when a service is misconfigured.
    """

    def __init__(
        self,
        service_config_name: str,
        error_condition: Literal["misconfigured", "not configured"],
        hint: str | None = None,
    ) -> None:
        """
        Args:
            service_config_name: The name of the service configuration that is misconfigured.
            misconfiguration_hint: The hint to describe the misconfiguration. Formatted as: 'service_config_name' is 'misconfiguration_hint'.
            hint: A sentence or short paragraph hint to the user on how to fix the misconfiguration.
        """
        message = f"Service '{service_config_name}' is {error_condition}." + (f" {hint}" if hint else "")
        super().__init__(message)
