# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# Custom Llama Stack Exception classes should follow the following schema
#   1. All classes should inherit from an existing Built-In Exception class: https://docs.python.org/3/library/exceptions.html
#   2. All classes should have a custom error message with the goal of informing the Llama Stack user specifically
#   3. All classes should propogate the inherited __init__ function otherwise via 'super().__init__(message)'

from abc import ABC, abstractmethod

import httpx
from fastapi import HTTPException


class LlamaStackError(Exception, ABC):
    """A Llama Stack error that can be translated to a fastapi HTTPException"""

    @property
    @abstractmethod
    def status_code(self) -> httpx.codes:
        """The HTTP status code for this exception"""
        ...

    def http_exception(self) -> HTTPException:
        """A fastapi HTTPException with the appropriate status code and detail"""
        return HTTPException(status_code=self.status_code, detail=str(self))


class ResourceNotFoundError(ValueError, LlamaStackError):
    """generic exception for a missing Llama Stack resource"""

    def __init__(self, resource_name: str, resource_type: str, client_list: str | None = None) -> None:
        message = f"{resource_type} '{resource_name}' not found."
        if client_list:
            message += f" Use '{client_list}' to list available {resource_type}s."
        super().__init__(message)

    @property
    def status_code(self) -> httpx.codes:
        return httpx.codes.NOT_FOUND


class ModelNotFoundError(ResourceNotFoundError):
    """raised when Llama Stack cannot find a referenced model"""

    def __init__(self, model_name: str) -> None:
        super().__init__(model_name, "Model", "client.models.list()")


class VectorStoreNotFoundError(ResourceNotFoundError):
    """raised when Llama Stack cannot find a referenced vector store"""

    def __init__(self, vector_store_name: str) -> None:
        super().__init__(vector_store_name, "Vector Store", "client.vector_dbs.list()")


class DatasetNotFoundError(ResourceNotFoundError):
    """raised when Llama Stack cannot find a referenced dataset"""

    def __init__(self, dataset_name: str) -> None:
        super().__init__(dataset_name, "Dataset", "client.datasets.list()")


class ToolGroupNotFoundError(ResourceNotFoundError):
    """raised when Llama Stack cannot find a referenced tool group"""

    def __init__(self, toolgroup_name: str) -> None:
        super().__init__(toolgroup_name, "Tool Group", "client.toolgroups.list()")


class ConversationNotFoundError(ResourceNotFoundError):
    """raised when Llama Stack cannot find a referenced conversation"""

    def __init__(self, conversation_id: str) -> None:
        super().__init__(conversation_id, "Conversation")

    @property
    def status_code(self) -> httpx.codes:
        return httpx.codes.NOT_FOUND


class ResponseNotFoundError(ResourceNotFoundError):
    """raised when Llama Stack cannot find a referenced response"""

    def __init__(self, response_id: str) -> None:
        super().__init__(response_id, "Response")

    @property
    def status_code(self) -> httpx.codes:
        return httpx.codes.NOT_FOUND


class UnsupportedModelError(ValueError, LlamaStackError):
    """raised when model is not present in the list of supported models"""

    def __init__(self, model_name: str, supported_models_list: list[str]):
        message = f"'{model_name}' model is not supported. Supported models are: {', '.join(supported_models_list)}"
        super().__init__(message)

    @property
    def status_code(self) -> httpx.codes:
        return httpx.codes.BAD_REQUEST


class ModelTypeError(TypeError, LlamaStackError):
    """raised when a model is present but not the correct type"""

    def __init__(self, model_name: str, model_type: str, expected_model_type: str) -> None:
        message = (
            f"Model '{model_name}' is of type '{model_type}' rather than the expected type '{expected_model_type}'"
        )
        super().__init__(message)

    @property
    def status_code(self) -> httpx.codes:
        return httpx.codes.BAD_REQUEST


class ConflictError(ValueError, LlamaStackError):
    """raised when an operation cannot be performed due to a conflict with the current state"""

    def __init__(self, message: str) -> None:
        super().__init__(message)

    @property
    def status_code(self) -> httpx.codes:
        return httpx.codes.CONFLICT


class TokenValidationError(ValueError, LlamaStackError):
    """raised when token validation fails during authentication"""

    def __init__(self, message: str) -> None:
        super().__init__(message)

    @property
    def status_code(self) -> httpx.codes:
        return httpx.codes.UNAUTHORIZED


class InvalidConversationIdError(ValueError, LlamaStackError):
    """raised when a conversation ID has an invalid format"""

    def __init__(self, conversation_id: str) -> None:
        message = f"Invalid conversation ID '{conversation_id}'. Expected an ID that begins with 'conv_'."
        super().__init__(message)
    
    @property
    def status_code(self) -> httpx.codes:
        return httpx.codes.BAD_REQUEST


class ServiceNotEnabledError(LlamaStackError, ValueError):
    """raised when a llama stack service is not enabled"""

    def __init__(self, service_name: str, *, provider_specific_message: str | None = None) -> None:
        message = f"Service '{service_name}' is not enabled. Please check your configuration and enable the service before trying again."
        if provider_specific_message:
            message += f"\n\n{provider_specific_message}"
        super().__init__(message)

    @property
    def status_code(self) -> httpx.codes:
        return httpx.codes.SERVICE_UNAVAILABLE


class ConnectorNotFoundError(ResourceNotFoundError):
    """raised when Llama Stack cannot find a referenced connector"""

    def __init__(self, connector_id: str) -> None:
        super().__init__(connector_id, "Connector", "client.connectors.list()")


class ConnectorToolNotFoundError(ValueError):
    """raised when Llama Stack cannot find a referenced tool in a connector"""

    def __init__(self, connector_id: str, tool_name: str) -> None:
        message = f"Tool '{tool_name}' not found in connector '{connector_id}'. Use 'client.connectors.list_tools(\"{connector_id}\")' to list available tools."
        super().__init__(message)


class InternalServerError(LlamaStackError):
    """
    A generic server side error that is not caused by the user's request. Sensitive data
    or details of the internal workings of the server should never be exposed to the user.
    Instead, sanitized error information should be logged for debugging purposes.
    """

    def __init__(self) -> None:
        message = "An internal error occurred while processing your request."
        super().__init__(message)

    @property
    def status_code(self) -> httpx.codes:
        return httpx.codes.INTERNAL_SERVER_ERROR
