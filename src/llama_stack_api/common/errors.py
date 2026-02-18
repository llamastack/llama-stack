# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# Custom Llama Stack Exception classes should follow the following schema
#   1. All classes should inherit from LlamaStackError (single inheritance only)
#   2. All classes should have a custom error message with the goal of informing the Llama Stack user specifically
#   3. All classes should set a status_code class attribute for HTTP response mapping

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
            resource_name_str = f" to list available {self.resource_name_plural}"

        return f"Use 'client.{self.command}({args_str})'{resource_name_str}."


class ResourceNotFoundError(LlamaStackError):
    """generic exception for a missing Llama Stack resource"""

    status_code: httpx.codes = httpx.codes.NOT_FOUND

    def __init__(
        self,
        resource_name: str,
        resource_type: str = "Resource",
        client_command: str | None = None,
        client_command_args: list[str] | str | None = None,
        resource_name_plural: str | None = None,
    ) -> None:
        resource_name_plural = resource_name_plural or f"{resource_type}s"

        message = f"{resource_type} '{resource_name}' not found."
        if client_command:
            client_list = ClientListCommand(client_command, client_command_args, resource_name_plural)
            message += f" {client_list}"
        super().__init__(message)


class ModelNotFoundError(ResourceNotFoundError):
    """raised when Llama Stack cannot find a referenced model"""

    def __init__(self, model_name: str) -> None:
        super().__init__(model_name, resource_type="Model", client_command="models.list")


class VectorStoreNotFoundError(ResourceNotFoundError):
    """raised when Llama Stack cannot find a referenced vector store"""

    def __init__(self, vector_store_name: str) -> None:
        super().__init__(vector_store_name, resource_type="Vector Store", client_command="vector_dbs.list")


class DatasetNotFoundError(ResourceNotFoundError):
    """raised when Llama Stack cannot find a referenced dataset"""

    def __init__(self, dataset_name: str) -> None:
        super().__init__(dataset_name, resource_type="Dataset", client_command="datasets.list")


class ToolGroupNotFoundError(ResourceNotFoundError):
    """raised when Llama Stack cannot find a referenced tool group"""

    def __init__(self, toolgroup_name: str) -> None:
        super().__init__(toolgroup_name, resource_type="Tool Group", client_command="toolgroups.list")


class ConversationNotFoundError(ResourceNotFoundError):
    """raised when Llama Stack cannot find a referenced conversation"""

    def __init__(self, conversation_id: str) -> None:
        super().__init__(conversation_id, resource_type="Conversation")


class ConnectorNotFoundError(ResourceNotFoundError):
    """raised when Llama Stack cannot find a referenced connector"""

    def __init__(self, connector_id: str) -> None:
        super().__init__(connector_id, resource_type="Connector", client_command="connectors.list")


class ConnectorToolNotFoundError(ResourceNotFoundError):
    """raised when Llama Stack cannot find a referenced tool in a connector"""

    def __init__(self, connector_id: str, tool_name: str) -> None:
        super().__init__(
            f"{connector_id}.{tool_name}",
            resource_type="Connector Tool",
            client_command="connectors.list_tools",
            client_command_args=connector_id,
        )


class OpenAIFileObjectNotFoundError(ResourceNotFoundError):
    """raised when Llama Stack cannot find a referenced file"""

    def __init__(self, file_id: str) -> None:
        super().__init__(file_id, resource_type="File", client_command="files.list")


class BatchNotFoundError(ResourceNotFoundError):
    """raised when Llama Stack cannot find a referenced batch"""

    def __init__(self, batch_id: str) -> None:
        super().__init__(batch_id, resource_type="Batch", client_command="batches.list", resource_name_plural="batches")


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

    def __init__(self, conversation_id: str) -> None:
        message = f"Invalid conversation ID '{conversation_id}'. Expected an ID that begins with 'conv_'."
        super().__init__(message)
