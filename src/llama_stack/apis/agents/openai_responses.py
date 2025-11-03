# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from collections.abc import Sequence
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, model_validator
from typing_extensions import TypedDict

from llama_stack.apis.vector_io import SearchRankingOptions as FileSearchRankingOptions
from llama_stack.schema_utils import json_schema_type, register_schema

# NOTE(ashwin): this file is literally a copy of the OpenAI responses API schema. We should probably
# take their YAML and generate this file automatically. Their YAML is available.


@json_schema_type
class OpenAIResponseError(BaseModel):
    """Error details for failed OpenAI response requests.

    :param code: Error code identifying the type of failure
    :param message: Human-readable error message describing the failure
    """

    code: str = Field(description="Error code identifying the type of failure")
    message: str = Field(description="Human-readable error message describing the failure")


@json_schema_type
class OpenAIResponseInputMessageContentText(BaseModel):
    """Text content for input messages in OpenAI response format.

    :param text: The text content of the input message
    :param type: Content type identifier, always "input_text"
    """

    text: str = Field(description="The text content of the input message")
    type: Literal["input_text"] = Field(
        default="input_text", description='Content type identifier, always "input_text"'
    )


@json_schema_type
class OpenAIResponseInputMessageContentImage(BaseModel):
    """Image content for input messages in OpenAI response format.

    :param detail: Level of detail for image processing, can be "low", "high", or "auto"
    :param type: Content type identifier, always "input_image"
    :param file_id: (Optional) The ID of the file to be sent to the model.
    :param image_url: (Optional) URL of the image content
    """

    detail: Literal["low"] | Literal["high"] | Literal["auto"] = Field(
        default="auto", description='Level of detail for image processing, can be "low", "high", or "auto"'
    )
    type: Literal["input_image"] = Field(
        default="input_image", description='Content type identifier, always "input_image"'
    )
    file_id: str | None = Field(default=None, description="The ID of the file to be sent to the model.")
    image_url: str | None = Field(default=None, description="URL of the image content")


@json_schema_type
class OpenAIResponseInputMessageContentFile(BaseModel):
    """File content for input messages in OpenAI response format.

    :param type: The type of the input item. Always `input_file`.
    :param file_data: The data of the file to be sent to the model.
    :param file_id: (Optional) The ID of the file to be sent to the model.
    :param file_url: The URL of the file to be sent to the model.
    :param filename: The name of the file to be sent to the model.
    """

    type: Literal["input_file"] = Field(
        default="input_file", description="The type of the input item. Always `input_file`."
    )
    file_data: str | None = Field(default=None, description="The data of the file to be sent to the model.")
    file_id: str | None = Field(default=None, description="The ID of the file to be sent to the model.")
    file_url: str | None = Field(default=None, description="The URL of the file to be sent to the model.")
    filename: str | None = Field(default=None, description="The name of the file to be sent to the model.")

    @model_validator(mode="after")
    def validate_file_source(self) -> "OpenAIResponseInputMessageContentFile":
        if not any([self.file_data, self.file_id, self.file_url, self.filename]):
            raise ValueError(
                "At least one of 'file_data', 'file_id', 'file_url', or 'filename' must be provided for file content"
            )
        return self


OpenAIResponseInputMessageContent = Annotated[
    OpenAIResponseInputMessageContentText
    | OpenAIResponseInputMessageContentImage
    | OpenAIResponseInputMessageContentFile,
    Field(discriminator="type"),
]
register_schema(OpenAIResponseInputMessageContent, name="OpenAIResponseInputMessageContent")


@json_schema_type
class OpenAIResponsePrompt(BaseModel):
    """OpenAI compatible Prompt object that is used in OpenAI responses.

    :param id: Unique identifier of the prompt template
    :param variables: Dictionary of variable names to OpenAIResponseInputMessageContent structure for template substitution. The substitution values can either be strings, or other Response input types
    like images or files.
    :param version: Version number of the prompt to use (defaults to latest if not specified)
    """

    id: str = Field(description="Unique identifier of the prompt template")
    variables: dict[str, OpenAIResponseInputMessageContent] | None = Field(
        default=None,
        description="Dictionary of variable names to OpenAIResponseInputMessageContent structure for template substitution. The substitution values can either be strings, or other Response input types like images or files.",
    )
    version: str | None = Field(
        default=None, description="Version number of the prompt to use (defaults to latest if not specified)"
    )


@json_schema_type
class OpenAIResponseAnnotationFileCitation(BaseModel):
    """File citation annotation for referencing specific files in response content.

    :param type: Annotation type identifier, always "file_citation"
    :param file_id: Unique identifier of the referenced file
    :param filename: Name of the referenced file
    :param index: Position index of the citation within the content
    """

    type: Literal["file_citation"] = Field(
        default="file_citation", description='Annotation type identifier, always "file_citation"'
    )
    file_id: str = Field(description="Unique identifier of the referenced file")
    filename: str = Field(description="Name of the referenced file")
    index: int = Field(description="Position index of the citation within the content")


@json_schema_type
class OpenAIResponseAnnotationCitation(BaseModel):
    """URL citation annotation for referencing external web resources.

    :param type: Annotation type identifier, always "url_citation"
    :param end_index: End position of the citation span in the content
    :param start_index: Start position of the citation span in the content
    :param title: Title of the referenced web resource
    :param url: URL of the referenced web resource
    """

    type: Literal["url_citation"] = Field(
        default="url_citation", description='Annotation type identifier, always "url_citation"'
    )
    end_index: int = Field(description="End position of the citation span in the content")
    start_index: int = Field(description="Start position of the citation span in the content")
    title: str = Field(description="Title of the referenced web resource")
    url: str = Field(description="URL of the referenced web resource")


@json_schema_type
class OpenAIResponseAnnotationContainerFileCitation(BaseModel):
    type: Literal["container_file_citation"] = Field(default="container_file_citation")
    container_id: str = Field()
    end_index: int = Field()
    file_id: str = Field()
    filename: str = Field()
    start_index: int = Field()


@json_schema_type
class OpenAIResponseAnnotationFilePath(BaseModel):
    type: Literal["file_path"] = Field(default="file_path")
    file_id: str = Field()
    index: int = Field()


OpenAIResponseAnnotations = Annotated[
    OpenAIResponseAnnotationFileCitation
    | OpenAIResponseAnnotationCitation
    | OpenAIResponseAnnotationContainerFileCitation
    | OpenAIResponseAnnotationFilePath,
    Field(discriminator="type"),
]
register_schema(OpenAIResponseAnnotations, name="OpenAIResponseAnnotations")


@json_schema_type
class OpenAIResponseOutputMessageContentOutputText(BaseModel):
    text: str = Field()
    type: Literal["output_text"] = Field(default="output_text")
    annotations: list[OpenAIResponseAnnotations] = Field(default_factory=list)


@json_schema_type
class OpenAIResponseContentPartRefusal(BaseModel):
    """Refusal content within a streamed response part.

    :param type: Content part type identifier, always "refusal"
    :param refusal: Refusal text supplied by the model
    """

    type: Literal["refusal"] = Field(default="refusal", description='Content part type identifier, always "refusal"')
    refusal: str = Field(description="Refusal text supplied by the model")


OpenAIResponseOutputMessageContent = Annotated[
    OpenAIResponseOutputMessageContentOutputText | OpenAIResponseContentPartRefusal,
    Field(discriminator="type"),
]
register_schema(OpenAIResponseOutputMessageContent, name="OpenAIResponseOutputMessageContent")


@json_schema_type
class OpenAIResponseMessage(BaseModel):
    """
    Corresponds to the various Message types in the Responses API.
    They are all under one type because the Responses API gives them all
    the same "type" value, and there is no way to tell them apart in certain
    scenarios.
    """

    content: str | Sequence[OpenAIResponseInputMessageContent] | Sequence[OpenAIResponseOutputMessageContent] = Field()
    role: Literal["system"] | Literal["developer"] | Literal["user"] | Literal["assistant"] = Field()
    type: Literal["message"] = Field(default="message")

    # The fields below are not used in all scenarios, but are required in others.
    id: str | None = Field(default=None)
    status: str | None = Field(default=None)


@json_schema_type
class OpenAIResponseOutputMessageWebSearchToolCall(BaseModel):
    """Web search tool call output message for OpenAI responses.

    :param id: Unique identifier for this tool call
    :param status: Current status of the web search operation
    :param type: Tool call type identifier, always "web_search_call"
    """

    id: str = Field(description="Unique identifier for this tool call")
    status: str = Field(description="Current status of the web search operation")
    type: Literal["web_search_call"] = Field(
        default="web_search_call", description='Tool call type identifier, always "web_search_call"'
    )


class OpenAIResponseOutputMessageFileSearchToolCallResults(BaseModel):
    """Search results returned by the file search operation.

    :param attributes: (Optional) Key-value attributes associated with the file
    :param file_id: Unique identifier of the file containing the result
    :param filename: Name of the file containing the result
    :param score: Relevance score for this search result (between 0 and 1)
    :param text: Text content of the search result
    """

    attributes: dict[str, Any] = Field(description="Key-value attributes associated with the file")
    file_id: str = Field(description="Unique identifier of the file containing the result")
    filename: str = Field(description="Name of the file containing the result")
    score: float = Field(description="Relevance score for this search result (between 0 and 1)")
    text: str = Field(description="Text content of the search result")


@json_schema_type
class OpenAIResponseOutputMessageFileSearchToolCall(BaseModel):
    """File search tool call output message for OpenAI responses.

    :param id: Unique identifier for this tool call
    :param queries: List of search queries executed
    :param status: Current status of the file search operation
    :param type: Tool call type identifier, always "file_search_call"
    :param results: (Optional) Search results returned by the file search operation
    """

    id: str = Field(description="Unique identifier for this tool call")
    queries: Sequence[str] = Field(description="List of search queries executed")
    status: str = Field(description="Current status of the file search operation")
    type: Literal["file_search_call"] = Field(
        default="file_search_call", description='Tool call type identifier, always "file_search_call"'
    )
    results: Sequence[OpenAIResponseOutputMessageFileSearchToolCallResults] | None = Field(
        default=None, description="Search results returned by the file search operation"
    )


@json_schema_type
class OpenAIResponseOutputMessageFunctionToolCall(BaseModel):
    """Function tool call output message for OpenAI responses.

    :param call_id: Unique identifier for the function call
    :param name: Name of the function being called
    :param arguments: JSON string containing the function arguments
    :param type: Tool call type identifier, always "function_call"
    :param id: (Optional) Additional identifier for the tool call
    :param status: (Optional) Current status of the function call execution
    """

    call_id: str = Field(description="Unique identifier for the function call")
    name: str = Field(description="Name of the function being called")
    arguments: str = Field(description="JSON string containing the function arguments")
    type: Literal["function_call"] = Field(
        default="function_call", description='Tool call type identifier, always "function_call"'
    )
    id: str | None = Field(default=None, description="Additional identifier for the tool call")
    status: str | None = Field(default=None, description="Current status of the function call execution")


@json_schema_type
class OpenAIResponseOutputMessageMCPCall(BaseModel):
    """Model Context Protocol (MCP) call output message for OpenAI responses.

    :param id: Unique identifier for this MCP call
    :param type: Tool call type identifier, always "mcp_call"
    :param arguments: JSON string containing the MCP call arguments
    :param name: Name of the MCP method being called
    :param server_label: Label identifying the MCP server handling the call
    :param error: (Optional) Error message if the MCP call failed
    :param output: (Optional) Output result from the successful MCP call
    """

    id: str = Field(description="Unique identifier for this MCP call")
    type: Literal["mcp_call"] = Field(default="mcp_call", description='Tool call type identifier, always "mcp_call"')
    arguments: str = Field(description="JSON string containing the MCP call arguments")
    name: str = Field(description="Name of the MCP method being called")
    server_label: str = Field(description="Label identifying the MCP server handling the call")
    error: str | None = Field(default=None, description="Error message if the MCP call failed")
    output: str | None = Field(default=None, description="Output result from the successful MCP call")


class MCPListToolsTool(BaseModel):
    """Tool definition returned by MCP list tools operation.

    :param input_schema: JSON schema defining the tool's input parameters
    :param name: Name of the tool
    :param description: (Optional) Description of what the tool does
    """

    input_schema: dict[str, Any] = Field(description="JSON schema defining the tool's input parameters")
    name: str = Field(description="Name of the tool")
    description: str | None = Field(default=None, description="Description of what the tool does")


@json_schema_type
class OpenAIResponseOutputMessageMCPListTools(BaseModel):
    """MCP list tools output message containing available tools from an MCP server.

    :param id: Unique identifier for this MCP list tools operation
    :param type: Tool call type identifier, always "mcp_list_tools"
    :param server_label: Label identifying the MCP server providing the tools
    :param tools: List of available tools provided by the MCP server
    """

    id: str = Field(description="Unique identifier for this MCP list tools operation")
    type: Literal["mcp_list_tools"] = Field(
        default="mcp_list_tools", description='Tool call type identifier, always "mcp_list_tools"'
    )
    server_label: str = Field(description="Label identifying the MCP server providing the tools")
    tools: list[MCPListToolsTool] = Field(description="List of available tools provided by the MCP server")


@json_schema_type
class OpenAIResponseMCPApprovalRequest(BaseModel):
    """
    A request for human approval of a tool invocation.
    """

    arguments: str = Field()
    id: str = Field()
    name: str = Field()
    server_label: str = Field()
    type: Literal["mcp_approval_request"] = Field(default="mcp_approval_request")


@json_schema_type
class OpenAIResponseMCPApprovalResponse(BaseModel):
    """
    A response to an MCP approval request.
    """

    approval_request_id: str = Field()
    approve: bool = Field()
    type: Literal["mcp_approval_response"] = Field(default="mcp_approval_response")
    id: str | None = Field(default=None)
    reason: str | None = Field(default=None)


OpenAIResponseOutput = Annotated[
    OpenAIResponseMessage
    | OpenAIResponseOutputMessageWebSearchToolCall
    | OpenAIResponseOutputMessageFileSearchToolCall
    | OpenAIResponseOutputMessageFunctionToolCall
    | OpenAIResponseOutputMessageMCPCall
    | OpenAIResponseOutputMessageMCPListTools
    | OpenAIResponseMCPApprovalRequest,
    Field(discriminator="type"),
]
register_schema(OpenAIResponseOutput, name="OpenAIResponseOutput")


# This has to be a TypedDict because we need a "schema" field and our strong
# typing code in the schema generator doesn't support Pydantic aliases. That also
# means we can't use a discriminator field here, because TypedDicts don't support
# default values which the strong typing code requires for discriminators.
class OpenAIResponseTextFormat(TypedDict, total=False):
    """Configuration for Responses API text format.

    :param type: Must be "text", "json_schema", or "json_object" to identify the format type
    :param name: The name of the response format. Only used for json_schema.
    :param schema: The JSON schema the response should conform to. In a Python SDK, this is often a `pydantic` model. Only used for json_schema.
    :param description: (Optional) A description of the response format. Only used for json_schema.
    :param strict: (Optional) Whether to strictly enforce the JSON schema. If true, the response must match the schema exactly. Only used for json_schema.
    """

    type: Literal["text"] | Literal["json_schema"] | Literal["json_object"]
    name: str | None
    schema: dict[str, Any] | None
    description: str | None
    strict: bool | None


@json_schema_type
class OpenAIResponseText(BaseModel):
    """Text response configuration for OpenAI responses.

    :param format: (Optional) Text format configuration specifying output format requirements
    """

    format: OpenAIResponseTextFormat | None = Field(
        default=None, description="Text format configuration specifying output format requirements"
    )


# Must match type Literals of OpenAIResponseInputToolWebSearch below
WebSearchToolTypes = ["web_search", "web_search_preview", "web_search_preview_2025_03_11"]


@json_schema_type
class OpenAIResponseInputToolWebSearch(BaseModel):
    """Web search tool configuration for OpenAI response inputs.

    :param type: Web search tool type variant to use
    :param search_context_size: (Optional) Size of search context, must be "low", "medium", or "high"
    """

    # Must match values of WebSearchToolTypes above
    type: Literal["web_search"] | Literal["web_search_preview"] | Literal["web_search_preview_2025_03_11"] = Field(
        default="web_search", description="Web search tool type variant to use"
    )
    # TODO: actually use search_context_size somewhere...
    search_context_size: str | None = Field(
        default="medium",
        pattern="^low|medium|high$",
        description='Size of search context, must be "low", "medium", or "high"',
    )
    # TODO: add user_location


@json_schema_type
class OpenAIResponseInputToolFunction(BaseModel):
    """Function tool configuration for OpenAI response inputs.

    :param type: Tool type identifier, always "function"
    :param name: Name of the function that can be called
    :param description: (Optional) Description of what the function does
    :param parameters: (Optional) JSON schema defining the function's parameters
    :param strict: (Optional) Whether to enforce strict parameter validation
    """

    type: Literal["function"] = Field(default="function", description='Tool type identifier, always "function"')
    name: str = Field(description="Name of the function that can be called")
    description: str | None = Field(default=None, description="Description of what the function does")
    parameters: dict[str, Any] | None = Field(
        default=None, description="JSON schema defining the function's parameters"
    )
    strict: bool | None = Field(default=None, description="Whether to enforce strict parameter validation")


@json_schema_type
class OpenAIResponseInputToolFileSearch(BaseModel):
    """File search tool configuration for OpenAI response inputs.

    :param type: Tool type identifier, always "file_search"
    :param vector_store_ids: List of vector store identifiers to search within
    :param filters: (Optional) Additional filters to apply to the search
    :param max_num_results: (Optional) Maximum number of search results to return (1-50)
    :param ranking_options: (Optional) Options for ranking and scoring search results
    """

    type: Literal["file_search"] = Field(
        default="file_search", description='Tool type identifier, always "file_search"'
    )
    vector_store_ids: list[str] = Field(description="List of vector store identifiers to search within")
    filters: dict[str, Any] | None = Field(default=None, description="Additional filters to apply to the search")
    max_num_results: int | None = Field(
        default=10, ge=1, le=50, description="Maximum number of search results to return (1-50)"
    )
    ranking_options: FileSearchRankingOptions | None = Field(
        default=None, description="Options for ranking and scoring search results"
    )


class ApprovalFilter(BaseModel):
    """Filter configuration for MCP tool approval requirements.

    :param always: (Optional) List of tool names that always require approval
    :param never: (Optional) List of tool names that never require approval
    """

    always: list[str] | None = Field(default=None, description="List of tool names that always require approval")
    never: list[str] | None = Field(default=None, description="List of tool names that never require approval")


class AllowedToolsFilter(BaseModel):
    """Filter configuration for restricting which MCP tools can be used.

    :param tool_names: (Optional) List of specific tool names that are allowed
    """

    tool_names: list[str] | None = Field(default=None, description="List of specific tool names that are allowed")


@json_schema_type
class OpenAIResponseInputToolMCP(BaseModel):
    """Model Context Protocol (MCP) tool configuration for OpenAI response inputs.

    :param type: Tool type identifier, always "mcp"
    :param server_label: Label to identify this MCP server
    :param server_url: URL endpoint of the MCP server
    :param headers: (Optional) HTTP headers to include when connecting to the server
    :param require_approval: Approval requirement for tool calls ("always", "never", or filter)
    :param allowed_tools: (Optional) Restriction on which tools can be used from this server
    """

    type: Literal["mcp"] = Field(default="mcp", description='Tool type identifier, always "mcp"')
    server_label: str = Field(description="Label to identify this MCP server")
    server_url: str = Field(description="URL endpoint of the MCP server")
    headers: dict[str, Any] | None = Field(
        default=None, description="HTTP headers to include when connecting to the server"
    )

    require_approval: Literal["always"] | Literal["never"] | ApprovalFilter = Field(
        default="never", description='Approval requirement for tool calls ("always", "never", or filter)'
    )
    allowed_tools: list[str] | AllowedToolsFilter | None = Field(
        default=None, description="Restriction on which tools can be used from this server"
    )


OpenAIResponseInputTool = Annotated[
    OpenAIResponseInputToolWebSearch
    | OpenAIResponseInputToolFileSearch
    | OpenAIResponseInputToolFunction
    | OpenAIResponseInputToolMCP,
    Field(discriminator="type"),
]
register_schema(OpenAIResponseInputTool, name="OpenAIResponseInputTool")


@json_schema_type
class OpenAIResponseToolMCP(BaseModel):
    """Model Context Protocol (MCP) tool configuration for OpenAI response object.

    :param type: Tool type identifier, always "mcp"
    :param server_label: Label to identify this MCP server
    :param allowed_tools: (Optional) Restriction on which tools can be used from this server
    """

    type: Literal["mcp"] = Field(default="mcp", description='Tool type identifier, always "mcp"')
    server_label: str = Field(description="Label to identify this MCP server")
    allowed_tools: list[str] | AllowedToolsFilter | None = Field(
        default=None, description="Restriction on which tools can be used from this server"
    )


OpenAIResponseTool = Annotated[
    OpenAIResponseInputToolWebSearch
    | OpenAIResponseInputToolFileSearch
    | OpenAIResponseInputToolFunction
    | OpenAIResponseToolMCP,  # The only type that differes from that in the inputs is the MCP tool
    Field(discriminator="type"),
]
register_schema(OpenAIResponseTool, name="OpenAIResponseTool")


class OpenAIResponseUsageOutputTokensDetails(BaseModel):
    """Token details for output tokens in OpenAI response usage.

    :param reasoning_tokens: Number of tokens used for reasoning (o1/o3 models)
    """

    reasoning_tokens: int | None = Field(default=None, description="Number of tokens used for reasoning (o1/o3 models)")


class OpenAIResponseUsageInputTokensDetails(BaseModel):
    """Token details for input tokens in OpenAI response usage.

    :param cached_tokens: Number of tokens retrieved from cache
    """

    cached_tokens: int | None = Field(default=None, description="Number of tokens retrieved from cache")


@json_schema_type
class OpenAIResponseUsage(BaseModel):
    """Usage information for OpenAI response.

    :param input_tokens: Number of tokens in the input
    :param output_tokens: Number of tokens in the output
    :param total_tokens: Total tokens used (input + output)
    :param input_tokens_details: Detailed breakdown of input token usage
    :param output_tokens_details: Detailed breakdown of output token usage
    """

    input_tokens: int = Field(description="Number of tokens in the input")
    output_tokens: int = Field(description="Number of tokens in the output")
    total_tokens: int = Field(description="Total tokens used (input + output)")
    input_tokens_details: OpenAIResponseUsageInputTokensDetails | None = Field(
        default=None, description="Detailed breakdown of input token usage"
    )
    output_tokens_details: OpenAIResponseUsageOutputTokensDetails | None = Field(
        default=None, description="Detailed breakdown of output token usage"
    )


@json_schema_type
class OpenAIResponseObject(BaseModel):
    """Complete OpenAI response object containing generation results and metadata.

    :param created_at: Unix timestamp when the response was created
    :param error: (Optional) Error details if the response generation failed
    :param id: Unique identifier for this response
    :param model: Model identifier used for generation
    :param object: Object type identifier, always "response"
    :param output: List of generated output items (messages, tool calls, etc.)
    :param parallel_tool_calls: Whether tool calls can be executed in parallel
    :param previous_response_id: (Optional) ID of the previous response in a conversation
    :param prompt: (Optional) Reference to a prompt template and its variables.
    :param status: Current status of the response generation
    :param temperature: (Optional) Sampling temperature used for generation
    :param text: Text formatting configuration for the response
    :param top_p: (Optional) Nucleus sampling parameter used for generation
    :param tools: (Optional) An array of tools the model may call while generating a response.
    :param truncation: (Optional) Truncation strategy applied to the response
    :param usage: (Optional) Token usage information for the response
    :param instructions: (Optional) System message inserted into the model's context
    """

    created_at: int = Field(description="Unix timestamp when the response was created")
    error: OpenAIResponseError | None = Field(
        default=None, description="Error details if the response generation failed"
    )
    id: str = Field(description="Unique identifier for this response")
    model: str = Field(description="Model identifier used for generation")
    object: Literal["response"] = Field(default="response", description='Object type identifier, always "response"')
    output: Sequence[OpenAIResponseOutput] = Field(
        description="List of generated output items (messages, tool calls, etc.)"
    )
    parallel_tool_calls: bool = Field(default=False, description="Whether tool calls can be executed in parallel")
    previous_response_id: str | None = Field(default=None, description="ID of the previous response in a conversation")
    prompt: OpenAIResponsePrompt | None = Field(
        default=None, description="Reference to a prompt template and its variables."
    )
    status: str = Field(description="Current status of the response generation")
    temperature: float | None = Field(default=None, description="Sampling temperature used for generation")
    # Default to text format to avoid breaking the loading of old responses
    # before the field was added. New responses will have this set always.
    text: OpenAIResponseText = Field(
        default_factory=lambda: OpenAIResponseText(format=OpenAIResponseTextFormat(type="text")),
        description="Text formatting configuration for the response",
    )
    top_p: float | None = Field(default=None, description="Nucleus sampling parameter used for generation")
    tools: Sequence[OpenAIResponseTool] | None = Field(
        default=None, description="An array of tools the model may call while generating a response."
    )
    truncation: str | None = Field(default=None, description="Truncation strategy applied to the response")
    usage: OpenAIResponseUsage | None = Field(default=None, description="Token usage information for the response")
    instructions: str | None = Field(default=None, description="System message inserted into the model's context")


@json_schema_type
class OpenAIDeleteResponseObject(BaseModel):
    """Response object confirming deletion of an OpenAI response.

    :param id: Unique identifier of the deleted response
    :param object: Object type identifier, always "response"
    :param deleted: Deletion confirmation flag, always True
    """

    id: str = Field(description="Unique identifier of the deleted response")
    object: Literal["response"] = Field(default="response", description='Object type identifier, always "response"')
    deleted: bool = Field(default=True, description="Deletion confirmation flag, always True")


@json_schema_type
class OpenAIResponseObjectStreamResponseCreated(BaseModel):
    """Streaming event indicating a new response has been created.

    :param response: The response object that was created
    :param type: Event type identifier, always "response.created"
    """

    response: OpenAIResponseObject = Field(description="The response object that was created")
    type: Literal["response.created"] = Field(
        default="response.created", description='Event type identifier, always "response.created"'
    )


@json_schema_type
class OpenAIResponseObjectStreamResponseInProgress(BaseModel):
    """Streaming event indicating the response remains in progress.

    :param response: Current response state while in progress
    :param sequence_number: Sequential number for ordering streaming events
    :param type: Event type identifier, always "response.in_progress"
    """

    response: OpenAIResponseObject = Field(description="Current response state while in progress")
    sequence_number: int = Field(description="Sequential number for ordering streaming events")
    type: Literal["response.in_progress"] = Field(
        default="response.in_progress", description='Event type identifier, always "response.in_progress"'
    )


@json_schema_type
class OpenAIResponseObjectStreamResponseCompleted(BaseModel):
    """Streaming event indicating a response has been completed.

    :param response: Completed response object
    :param type: Event type identifier, always "response.completed"
    """

    response: OpenAIResponseObject = Field(description="Completed response object")
    type: Literal["response.completed"] = Field(
        default="response.completed", description='Event type identifier, always "response.completed"'
    )


@json_schema_type
class OpenAIResponseObjectStreamResponseIncomplete(BaseModel):
    """Streaming event emitted when a response ends in an incomplete state.

    :param response: Response object describing the incomplete state
    :param sequence_number: Sequential number for ordering streaming events
    :param type: Event type identifier, always "response.incomplete"
    """

    response: OpenAIResponseObject = Field(description="Response object describing the incomplete state")
    sequence_number: int = Field(description="Sequential number for ordering streaming events")
    type: Literal["response.incomplete"] = Field(
        default="response.incomplete", description='Event type identifier, always "response.incomplete"'
    )


@json_schema_type
class OpenAIResponseObjectStreamResponseFailed(BaseModel):
    """Streaming event emitted when a response fails.

    :param response: Response object describing the failure
    :param sequence_number: Sequential number for ordering streaming events
    :param type: Event type identifier, always "response.failed"
    """

    response: OpenAIResponseObject = Field(description="Response object describing the failure")
    sequence_number: int = Field(description="Sequential number for ordering streaming events")
    type: Literal["response.failed"] = Field(
        default="response.failed", description='Event type identifier, always "response.failed"'
    )


@json_schema_type
class OpenAIResponseObjectStreamResponseOutputItemAdded(BaseModel):
    """Streaming event for when a new output item is added to the response.

    :param response_id: Unique identifier of the response containing this output
    :param item: The output item that was added (message, tool call, etc.)
    :param output_index: Index position of this item in the output list
    :param sequence_number: Sequential number for ordering streaming events
    :param type: Event type identifier, always "response.output_item.added"
    """

    response_id: str = Field(description="Unique identifier of the response containing this output")
    item: OpenAIResponseOutput = Field(description="The output item that was added (message, tool call, etc.)")
    output_index: int = Field(description="Index position of this item in the output list")
    sequence_number: int = Field(description="Sequential number for ordering streaming events")
    type: Literal["response.output_item.added"] = Field(
        default="response.output_item.added", description='Event type identifier, always "response.output_item.added"'
    )


@json_schema_type
class OpenAIResponseObjectStreamResponseOutputItemDone(BaseModel):
    """Streaming event for when an output item is completed.

    :param response_id: Unique identifier of the response containing this output
    :param item: The completed output item (message, tool call, etc.)
    :param output_index: Index position of this item in the output list
    :param sequence_number: Sequential number for ordering streaming events
    :param type: Event type identifier, always "response.output_item.done"
    """

    response_id: str = Field(description="Unique identifier of the response containing this output")
    item: OpenAIResponseOutput = Field(description="The completed output item (message, tool call, etc.)")
    output_index: int = Field(description="Index position of this item in the output list")
    sequence_number: int = Field(description="Sequential number for ordering streaming events")
    type: Literal["response.output_item.done"] = Field(
        default="response.output_item.done", description='Event type identifier, always "response.output_item.done"'
    )


@json_schema_type
class OpenAIResponseObjectStreamResponseOutputTextDelta(BaseModel):
    """Streaming event for incremental text content updates.

    :param content_index: Index position within the text content
    :param delta: Incremental text content being added
    :param item_id: Unique identifier of the output item being updated
    :param output_index: Index position of the item in the output list
    :param sequence_number: Sequential number for ordering streaming events
    :param type: Event type identifier, always "response.output_text.delta"
    """

    content_index: int = Field(description="Index position within the text content")
    delta: str = Field(description="Incremental text content being added")
    item_id: str = Field(description="Unique identifier of the output item being updated")
    output_index: int = Field(description="Index position of the item in the output list")
    sequence_number: int = Field(description="Sequential number for ordering streaming events")
    type: Literal["response.output_text.delta"] = Field(
        default="response.output_text.delta", description='Event type identifier, always "response.output_text.delta"'
    )


@json_schema_type
class OpenAIResponseObjectStreamResponseOutputTextDone(BaseModel):
    """Streaming event for when text output is completed.

    :param content_index: Index position within the text content
    :param text: Final complete text content of the output item
    :param item_id: Unique identifier of the completed output item
    :param output_index: Index position of the item in the output list
    :param sequence_number: Sequential number for ordering streaming events
    :param type: Event type identifier, always "response.output_text.done"
    """

    content_index: int = Field(description="Index position within the text content")
    text: str = Field(description="Final complete text content of the output item")  # final text of the output item
    item_id: str = Field(description="Unique identifier of the completed output item")
    output_index: int = Field(description="Index position of the item in the output list")
    sequence_number: int = Field(description="Sequential number for ordering streaming events")
    type: Literal["response.output_text.done"] = Field(
        default="response.output_text.done", description='Event type identifier, always "response.output_text.done"'
    )


@json_schema_type
class OpenAIResponseObjectStreamResponseFunctionCallArgumentsDelta(BaseModel):
    """Streaming event for incremental function call argument updates.

    :param delta: Incremental function call arguments being added
    :param item_id: Unique identifier of the function call being updated
    :param output_index: Index position of the item in the output list
    :param sequence_number: Sequential number for ordering streaming events
    :param type: Event type identifier, always "response.function_call_arguments.delta"
    """

    delta: str = Field(description="Incremental function call arguments being added")
    item_id: str = Field(description="Unique identifier of the function call being updated")
    output_index: int = Field(description="Index position of the item in the output list")
    sequence_number: int = Field(description="Sequential number for ordering streaming events")
    type: Literal["response.function_call_arguments.delta"] = Field(
        default="response.function_call_arguments.delta",
        description='Event type identifier, always "response.function_call_arguments.delta"',
    )


@json_schema_type
class OpenAIResponseObjectStreamResponseFunctionCallArgumentsDone(BaseModel):
    """Streaming event for when function call arguments are completed.

    :param arguments: Final complete arguments JSON string for the function call
    :param item_id: Unique identifier of the completed function call
    :param output_index: Index position of the item in the output list
    :param sequence_number: Sequential number for ordering streaming events
    :param type: Event type identifier, always "response.function_call_arguments.done"
    """

    arguments: str = Field(
        description="Final complete arguments JSON string for the function call"
    )  # final arguments of the function call
    item_id: str = Field(description="Unique identifier of the completed function call")
    output_index: int = Field(description="Index position of the item in the output list")
    sequence_number: int = Field(description="Sequential number for ordering streaming events")
    type: Literal["response.function_call_arguments.done"] = Field(
        default="response.function_call_arguments.done",
        description='Event type identifier, always "response.function_call_arguments.done"',
    )


@json_schema_type
class OpenAIResponseObjectStreamResponseWebSearchCallInProgress(BaseModel):
    """Streaming event for web search calls in progress.

    :param item_id: Unique identifier of the web search call
    :param output_index: Index position of the item in the output list
    :param sequence_number: Sequential number for ordering streaming events
    :param type: Event type identifier, always "response.web_search_call.in_progress"
    """

    item_id: str = Field(description="Unique identifier of the web search call")
    output_index: int = Field(description="Index position of the item in the output list")
    sequence_number: int = Field(description="Sequential number for ordering streaming events")
    type: Literal["response.web_search_call.in_progress"] = Field(
        default="response.web_search_call.in_progress",
        description='Event type identifier, always "response.web_search_call.in_progress"',
    )


@json_schema_type
class OpenAIResponseObjectStreamResponseWebSearchCallSearching(BaseModel):
    item_id: str = Field()
    output_index: int = Field()
    sequence_number: int = Field()
    type: Literal["response.web_search_call.searching"] = Field(default="response.web_search_call.searching")


@json_schema_type
class OpenAIResponseObjectStreamResponseWebSearchCallCompleted(BaseModel):
    """Streaming event for completed web search calls.

    :param item_id: Unique identifier of the completed web search call
    :param output_index: Index position of the item in the output list
    :param sequence_number: Sequential number for ordering streaming events
    :param type: Event type identifier, always "response.web_search_call.completed"
    """

    item_id: str = Field(description="Unique identifier of the completed web search call")
    output_index: int = Field(description="Index position of the item in the output list")
    sequence_number: int = Field(description="Sequential number for ordering streaming events")
    type: Literal["response.web_search_call.completed"] = Field(
        default="response.web_search_call.completed",
        description='Event type identifier, always "response.web_search_call.completed"',
    )


@json_schema_type
class OpenAIResponseObjectStreamResponseMcpListToolsInProgress(BaseModel):
    sequence_number: int = Field()
    type: Literal["response.mcp_list_tools.in_progress"] = Field(default="response.mcp_list_tools.in_progress")


@json_schema_type
class OpenAIResponseObjectStreamResponseMcpListToolsFailed(BaseModel):
    sequence_number: int = Field()
    type: Literal["response.mcp_list_tools.failed"] = Field(default="response.mcp_list_tools.failed")


@json_schema_type
class OpenAIResponseObjectStreamResponseMcpListToolsCompleted(BaseModel):
    sequence_number: int = Field()
    type: Literal["response.mcp_list_tools.completed"] = Field(default="response.mcp_list_tools.completed")


@json_schema_type
class OpenAIResponseObjectStreamResponseMcpCallArgumentsDelta(BaseModel):
    delta: str = Field()
    item_id: str = Field()
    output_index: int = Field()
    sequence_number: int = Field()
    type: Literal["response.mcp_call.arguments.delta"] = Field(default="response.mcp_call.arguments.delta")


@json_schema_type
class OpenAIResponseObjectStreamResponseMcpCallArgumentsDone(BaseModel):
    arguments: str = Field()  # final arguments of the MCP call
    item_id: str = Field()
    output_index: int = Field()
    sequence_number: int = Field()
    type: Literal["response.mcp_call.arguments.done"] = Field(default="response.mcp_call.arguments.done")


@json_schema_type
class OpenAIResponseObjectStreamResponseMcpCallInProgress(BaseModel):
    """Streaming event for MCP calls in progress.

    :param item_id: Unique identifier of the MCP call
    :param output_index: Index position of the item in the output list
    :param sequence_number: Sequential number for ordering streaming events
    :param type: Event type identifier, always "response.mcp_call.in_progress"
    """

    item_id: str = Field(description="Unique identifier of the MCP call")
    output_index: int = Field(description="Index position of the item in the output list")
    sequence_number: int = Field(description="Sequential number for ordering streaming events")
    type: Literal["response.mcp_call.in_progress"] = Field(
        default="response.mcp_call.in_progress",
        description='Event type identifier, always "response.mcp_call.in_progress"',
    )


@json_schema_type
class OpenAIResponseObjectStreamResponseMcpCallFailed(BaseModel):
    """Streaming event for failed MCP calls.

    :param sequence_number: Sequential number for ordering streaming events
    :param type: Event type identifier, always "response.mcp_call.failed"
    """

    sequence_number: int = Field(description="Sequential number for ordering streaming events")
    type: Literal["response.mcp_call.failed"] = Field(
        default="response.mcp_call.failed", description='Event type identifier, always "response.mcp_call.failed"'
    )


@json_schema_type
class OpenAIResponseObjectStreamResponseMcpCallCompleted(BaseModel):
    """Streaming event for completed MCP calls.

    :param sequence_number: Sequential number for ordering streaming events
    :param type: Event type identifier, always "response.mcp_call.completed"
    """

    sequence_number: int = Field(description="Sequential number for ordering streaming events")
    type: Literal["response.mcp_call.completed"] = Field(
        default="response.mcp_call.completed", description='Event type identifier, always "response.mcp_call.completed"'
    )


@json_schema_type
class OpenAIResponseContentPartOutputText(BaseModel):
    """Text content within a streamed response part.

    :param type: Content part type identifier, always "output_text"
    :param text: Text emitted for this content part
    :param annotations: Structured annotations associated with the text
    :param logprobs: (Optional) Token log probability details
    """

    type: Literal["output_text"] = Field(
        default="output_text", description='Content part type identifier, always "output_text"'
    )
    text: str = Field(description="Text emitted for this content part")
    annotations: list[OpenAIResponseAnnotations] = Field(
        default_factory=list, description="Structured annotations associated with the text"
    )
    logprobs: list[dict[str, Any]] | None = Field(default=None, description="Token log probability details")


@json_schema_type
class OpenAIResponseContentPartReasoningText(BaseModel):
    """Reasoning text emitted as part of a streamed response.

    :param type: Content part type identifier, always "reasoning_text"
    :param text: Reasoning text supplied by the model
    """

    type: Literal["reasoning_text"] = Field(
        default="reasoning_text", description='Content part type identifier, always "reasoning_text"'
    )
    text: str = Field(description="Reasoning text supplied by the model")


OpenAIResponseContentPart = Annotated[
    OpenAIResponseContentPartOutputText | OpenAIResponseContentPartRefusal | OpenAIResponseContentPartReasoningText,
    Field(discriminator="type"),
]
register_schema(OpenAIResponseContentPart, name="OpenAIResponseContentPart")


@json_schema_type
class OpenAIResponseObjectStreamResponseContentPartAdded(BaseModel):
    """Streaming event for when a new content part is added to a response item.

    :param content_index: Index position of the part within the content array
    :param response_id: Unique identifier of the response containing this content
    :param item_id: Unique identifier of the output item containing this content part
    :param output_index: Index position of the output item in the response
    :param part: The content part that was added
    :param sequence_number: Sequential number for ordering streaming events
    :param type: Event type identifier, always "response.content_part.added"
    """

    content_index: int = Field(description="Index position of the part within the content array")
    response_id: str = Field(description="Unique identifier of the response containing this content")
    item_id: str = Field(description="Unique identifier of the output item containing this content part")
    output_index: int = Field(description="Index position of the output item in the response")
    part: OpenAIResponseContentPart = Field(description="The content part that was added")
    sequence_number: int = Field(description="Sequential number for ordering streaming events")
    type: Literal["response.content_part.added"] = Field(
        default="response.content_part.added", description='Event type identifier, always "response.content_part.added"'
    )


@json_schema_type
class OpenAIResponseObjectStreamResponseContentPartDone(BaseModel):
    """Streaming event for when a content part is completed.

    :param content_index: Index position of the part within the content array
    :param response_id: Unique identifier of the response containing this content
    :param item_id: Unique identifier of the output item containing this content part
    :param output_index: Index position of the output item in the response
    :param part: The completed content part
    :param sequence_number: Sequential number for ordering streaming events
    :param type: Event type identifier, always "response.content_part.done"
    """

    content_index: int = Field(description="Index position of the part within the content array")
    response_id: str = Field(description="Unique identifier of the response containing this content")
    item_id: str = Field(description="Unique identifier of the output item containing this content part")
    output_index: int = Field(description="Index position of the output item in the response")
    part: OpenAIResponseContentPart = Field(description="The completed content part")
    sequence_number: int = Field(description="Sequential number for ordering streaming events")
    type: Literal["response.content_part.done"] = Field(
        default="response.content_part.done", description='Event type identifier, always "response.content_part.done"'
    )


@json_schema_type
class OpenAIResponseObjectStreamResponseReasoningTextDelta(BaseModel):
    """Streaming event for incremental reasoning text updates.

    :param content_index: Index position of the reasoning content part
    :param delta: Incremental reasoning text being added
    :param item_id: Unique identifier of the output item being updated
    :param output_index: Index position of the item in the output list
    :param sequence_number: Sequential number for ordering streaming events
    :param type: Event type identifier, always "response.reasoning_text.delta"
    """

    content_index: int = Field(description="Index position of the reasoning content part")
    delta: str = Field(description="Incremental reasoning text being added")
    item_id: str = Field(description="Unique identifier of the output item being updated")
    output_index: int = Field(description="Index position of the item in the output list")
    sequence_number: int = Field(description="Sequential number for ordering streaming events")
    type: Literal["response.reasoning_text.delta"] = Field(
        default="response.reasoning_text.delta",
        description='Event type identifier, always "response.reasoning_text.delta"',
    )


@json_schema_type
class OpenAIResponseObjectStreamResponseReasoningTextDone(BaseModel):
    """Streaming event for when reasoning text is completed.

    :param content_index: Index position of the reasoning content part
    :param text: Final complete reasoning text
    :param item_id: Unique identifier of the completed output item
    :param output_index: Index position of the item in the output list
    :param sequence_number: Sequential number for ordering streaming events
    :param type: Event type identifier, always "response.reasoning_text.done"
    """

    content_index: int = Field(description="Index position of the reasoning content part")
    text: str = Field(description="Final complete reasoning text")
    item_id: str = Field(description="Unique identifier of the completed output item")
    output_index: int = Field(description="Index position of the item in the output list")
    sequence_number: int = Field(description="Sequential number for ordering streaming events")
    type: Literal["response.reasoning_text.done"] = Field(
        default="response.reasoning_text.done",
        description='Event type identifier, always "response.reasoning_text.done"',
    )


@json_schema_type
class OpenAIResponseContentPartReasoningSummary(BaseModel):
    """Reasoning summary part in a streamed response.

    :param type: Content part type identifier, always "summary_text"
    :param text: Summary text
    """

    type: Literal["summary_text"] = Field(
        default="summary_text", description='Content part type identifier, always "summary_text"'
    )
    text: str = Field(description="Summary text")


@json_schema_type
class OpenAIResponseObjectStreamResponseReasoningSummaryPartAdded(BaseModel):
    """Streaming event for when a new reasoning summary part is added.

    :param item_id: Unique identifier of the output item
    :param output_index: Index position of the output item
    :param part: The summary part that was added
    :param sequence_number: Sequential number for ordering streaming events
    :param summary_index: Index of the summary part within the reasoning summary
    :param type: Event type identifier, always "response.reasoning_summary_part.added"
    """

    item_id: str = Field(description="Unique identifier of the output item")
    output_index: int = Field(description="Index position of the output item")
    part: OpenAIResponseContentPartReasoningSummary = Field(description="The summary part that was added")
    sequence_number: int = Field(description="Sequential number for ordering streaming events")
    summary_index: int = Field(description="Index of the summary part within the reasoning summary")
    type: Literal["response.reasoning_summary_part.added"] = Field(
        default="response.reasoning_summary_part.added",
        description='Event type identifier, always "response.reasoning_summary_part.added"',
    )


@json_schema_type
class OpenAIResponseObjectStreamResponseReasoningSummaryPartDone(BaseModel):
    """Streaming event for when a reasoning summary part is completed.

    :param item_id: Unique identifier of the output item
    :param output_index: Index position of the output item
    :param part: The completed summary part
    :param sequence_number: Sequential number for ordering streaming events
    :param summary_index: Index of the summary part within the reasoning summary
    :param type: Event type identifier, always "response.reasoning_summary_part.done"
    """

    item_id: str = Field(description="Unique identifier of the output item")
    output_index: int = Field(description="Index position of the output item")
    part: OpenAIResponseContentPartReasoningSummary = Field(description="The completed summary part")
    sequence_number: int = Field(description="Sequential number for ordering streaming events")
    summary_index: int = Field(description="Index of the summary part within the reasoning summary")
    type: Literal["response.reasoning_summary_part.done"] = Field(
        default="response.reasoning_summary_part.done",
        description='Event type identifier, always "response.reasoning_summary_part.done"',
    )


@json_schema_type
class OpenAIResponseObjectStreamResponseReasoningSummaryTextDelta(BaseModel):
    """Streaming event for incremental reasoning summary text updates.

    :param delta: Incremental summary text being added
    :param item_id: Unique identifier of the output item
    :param output_index: Index position of the output item
    :param sequence_number: Sequential number for ordering streaming events
    :param summary_index: Index of the summary part within the reasoning summary
    :param type: Event type identifier, always "response.reasoning_summary_text.delta"
    """

    delta: str = Field(description="Incremental summary text being added")
    item_id: str = Field(description="Unique identifier of the output item")
    output_index: int = Field(description="Index position of the output item")
    sequence_number: int = Field(description="Sequential number for ordering streaming events")
    summary_index: int = Field(description="Index of the summary part within the reasoning summary")
    type: Literal["response.reasoning_summary_text.delta"] = Field(
        default="response.reasoning_summary_text.delta",
        description='Event type identifier, always "response.reasoning_summary_text.delta"',
    )


@json_schema_type
class OpenAIResponseObjectStreamResponseReasoningSummaryTextDone(BaseModel):
    """Streaming event for when reasoning summary text is completed.

    :param text: Final complete summary text
    :param item_id: Unique identifier of the output item
    :param output_index: Index position of the output item
    :param sequence_number: Sequential number for ordering streaming events
    :param summary_index: Index of the summary part within the reasoning summary
    :param type: Event type identifier, always "response.reasoning_summary_text.done"
    """

    text: str = Field(description="Final complete summary text")
    item_id: str = Field(description="Unique identifier of the output item")
    output_index: int = Field(description="Index position of the output item")
    sequence_number: int = Field(description="Sequential number for ordering streaming events")
    summary_index: int = Field(description="Index of the summary part within the reasoning summary")
    type: Literal["response.reasoning_summary_text.done"] = Field(
        default="response.reasoning_summary_text.done",
        description='Event type identifier, always "response.reasoning_summary_text.done"',
    )


@json_schema_type
class OpenAIResponseObjectStreamResponseRefusalDelta(BaseModel):
    """Streaming event for incremental refusal text updates.

    :param content_index: Index position of the content part
    :param delta: Incremental refusal text being added
    :param item_id: Unique identifier of the output item
    :param output_index: Index position of the item in the output list
    :param sequence_number: Sequential number for ordering streaming events
    :param type: Event type identifier, always "response.refusal.delta"
    """

    content_index: int = Field(description="Index position of the content part")
    delta: str = Field(description="Incremental refusal text being added")
    item_id: str = Field(description="Unique identifier of the output item")
    output_index: int = Field(description="Index position of the item in the output list")
    sequence_number: int = Field(description="Sequential number for ordering streaming events")
    type: Literal["response.refusal.delta"] = Field(
        default="response.refusal.delta", description='Event type identifier, always "response.refusal.delta"'
    )


@json_schema_type
class OpenAIResponseObjectStreamResponseRefusalDone(BaseModel):
    """Streaming event for when refusal text is completed.

    :param content_index: Index position of the content part
    :param refusal: Final complete refusal text
    :param item_id: Unique identifier of the output item
    :param output_index: Index position of the item in the output list
    :param sequence_number: Sequential number for ordering streaming events
    :param type: Event type identifier, always "response.refusal.done"
    """

    content_index: int = Field(description="Index position of the content part")
    refusal: str = Field(description="Final complete refusal text")
    item_id: str = Field(description="Unique identifier of the output item")
    output_index: int = Field(description="Index position of the item in the output list")
    sequence_number: int = Field(description="Sequential number for ordering streaming events")
    type: Literal["response.refusal.done"] = Field(
        default="response.refusal.done", description='Event type identifier, always "response.refusal.done"'
    )


@json_schema_type
class OpenAIResponseObjectStreamResponseOutputTextAnnotationAdded(BaseModel):
    """Streaming event for when an annotation is added to output text.

    :param item_id: Unique identifier of the item to which the annotation is being added
    :param output_index: Index position of the output item in the response's output array
    :param content_index: Index position of the content part within the output item
    :param annotation_index: Index of the annotation within the content part
    :param annotation: The annotation object being added
    :param sequence_number: Sequential number for ordering streaming events
    :param type: Event type identifier, always "response.output_text.annotation.added"
    """

    item_id: str = Field(description="Unique identifier of the item to which the annotation is being added")
    output_index: int = Field(description="Index position of the output item in the response's output array")
    content_index: int = Field(description="Index position of the content part within the output item")
    annotation_index: int = Field(description="Index of the annotation within the content part")
    annotation: OpenAIResponseAnnotations = Field(description="The annotation object being added")
    sequence_number: int = Field(description="Sequential number for ordering streaming events")
    type: Literal["response.output_text.annotation.added"] = Field(
        default="response.output_text.annotation.added",
        description='Event type identifier, always "response.output_text.annotation.added"',
    )


@json_schema_type
class OpenAIResponseObjectStreamResponseFileSearchCallInProgress(BaseModel):
    """Streaming event for file search calls in progress.

    :param item_id: Unique identifier of the file search call
    :param output_index: Index position of the item in the output list
    :param sequence_number: Sequential number for ordering streaming events
    :param type: Event type identifier, always "response.file_search_call.in_progress"
    """

    item_id: str = Field(description="Unique identifier of the file search call")
    output_index: int = Field(description="Index position of the item in the output list")
    sequence_number: int = Field(description="Sequential number for ordering streaming events")
    type: Literal["response.file_search_call.in_progress"] = Field(
        default="response.file_search_call.in_progress",
        description='Event type identifier, always "response.file_search_call.in_progress"',
    )


@json_schema_type
class OpenAIResponseObjectStreamResponseFileSearchCallSearching(BaseModel):
    """Streaming event for file search currently searching.

    :param item_id: Unique identifier of the file search call
    :param output_index: Index position of the item in the output list
    :param sequence_number: Sequential number for ordering streaming events
    :param type: Event type identifier, always "response.file_search_call.searching"
    """

    item_id: str = Field(description="Unique identifier of the file search call")
    output_index: int = Field(description="Index position of the item in the output list")
    sequence_number: int = Field(description="Sequential number for ordering streaming events")
    type: Literal["response.file_search_call.searching"] = Field(
        default="response.file_search_call.searching",
        description='Event type identifier, always "response.file_search_call.searching"',
    )


@json_schema_type
class OpenAIResponseObjectStreamResponseFileSearchCallCompleted(BaseModel):
    """Streaming event for completed file search calls.

    :param item_id: Unique identifier of the completed file search call
    :param output_index: Index position of the item in the output list
    :param sequence_number: Sequential number for ordering streaming events
    :param type: Event type identifier, always "response.file_search_call.completed"
    """

    item_id: str = Field(description="Unique identifier of the completed file search call")
    output_index: int = Field(description="Index position of the item in the output list")
    sequence_number: int = Field(description="Sequential number for ordering streaming events")
    type: Literal["response.file_search_call.completed"] = Field(
        default="response.file_search_call.completed",
        description='Event type identifier, always "response.file_search_call.completed"',
    )


OpenAIResponseObjectStream = Annotated[
    OpenAIResponseObjectStreamResponseCreated
    | OpenAIResponseObjectStreamResponseInProgress
    | OpenAIResponseObjectStreamResponseOutputItemAdded
    | OpenAIResponseObjectStreamResponseOutputItemDone
    | OpenAIResponseObjectStreamResponseOutputTextDelta
    | OpenAIResponseObjectStreamResponseOutputTextDone
    | OpenAIResponseObjectStreamResponseFunctionCallArgumentsDelta
    | OpenAIResponseObjectStreamResponseFunctionCallArgumentsDone
    | OpenAIResponseObjectStreamResponseWebSearchCallInProgress
    | OpenAIResponseObjectStreamResponseWebSearchCallSearching
    | OpenAIResponseObjectStreamResponseWebSearchCallCompleted
    | OpenAIResponseObjectStreamResponseMcpListToolsInProgress
    | OpenAIResponseObjectStreamResponseMcpListToolsFailed
    | OpenAIResponseObjectStreamResponseMcpListToolsCompleted
    | OpenAIResponseObjectStreamResponseMcpCallArgumentsDelta
    | OpenAIResponseObjectStreamResponseMcpCallArgumentsDone
    | OpenAIResponseObjectStreamResponseMcpCallInProgress
    | OpenAIResponseObjectStreamResponseMcpCallFailed
    | OpenAIResponseObjectStreamResponseMcpCallCompleted
    | OpenAIResponseObjectStreamResponseContentPartAdded
    | OpenAIResponseObjectStreamResponseContentPartDone
    | OpenAIResponseObjectStreamResponseReasoningTextDelta
    | OpenAIResponseObjectStreamResponseReasoningTextDone
    | OpenAIResponseObjectStreamResponseReasoningSummaryPartAdded
    | OpenAIResponseObjectStreamResponseReasoningSummaryPartDone
    | OpenAIResponseObjectStreamResponseReasoningSummaryTextDelta
    | OpenAIResponseObjectStreamResponseReasoningSummaryTextDone
    | OpenAIResponseObjectStreamResponseRefusalDelta
    | OpenAIResponseObjectStreamResponseRefusalDone
    | OpenAIResponseObjectStreamResponseOutputTextAnnotationAdded
    | OpenAIResponseObjectStreamResponseFileSearchCallInProgress
    | OpenAIResponseObjectStreamResponseFileSearchCallSearching
    | OpenAIResponseObjectStreamResponseFileSearchCallCompleted
    | OpenAIResponseObjectStreamResponseIncomplete
    | OpenAIResponseObjectStreamResponseFailed
    | OpenAIResponseObjectStreamResponseCompleted,
    Field(discriminator="type"),
]
register_schema(OpenAIResponseObjectStream, name="OpenAIResponseObjectStream")


@json_schema_type
class OpenAIResponseInputFunctionToolCallOutput(BaseModel):
    """
    This represents the output of a function call that gets passed back to the model.
    """

    call_id: str = Field()
    output: str = Field()
    type: Literal["function_call_output"] = Field(default="function_call_output")
    id: str | None = Field(default=None)
    status: str | None = Field(default=None)


OpenAIResponseInput = Annotated[
    # Responses API allows output messages to be passed in as input
    OpenAIResponseOutput
    | OpenAIResponseInputFunctionToolCallOutput
    | OpenAIResponseMCPApprovalResponse
    | OpenAIResponseMessage,
    Field(union_mode="left_to_right"),
]
register_schema(OpenAIResponseInput, name="OpenAIResponseInput")


@json_schema_type
class ListOpenAIResponseInputItem(BaseModel):
    """List container for OpenAI response input items.

    :param data: List of input items
    :param object: Object type identifier, always "list"
    """

    data: Sequence[OpenAIResponseInput] = Field(description="List of input items")
    object: Literal["list"] = Field(default="list", description='Object type identifier, always "list"')


@json_schema_type
class OpenAIResponseObjectWithInput(OpenAIResponseObject):
    """OpenAI response object extended with input context information.

    :param input: List of input items that led to this response
    """

    input: Sequence[OpenAIResponseInput] = Field(description="List of input items that led to this response")

    def to_response_object(self) -> OpenAIResponseObject:
        """Convert to OpenAIResponseObject by excluding input field."""
        return OpenAIResponseObject(**{k: v for k, v in self.model_dump().items() if k != "input"})


@json_schema_type
class ListOpenAIResponseObject(BaseModel):
    """Paginated list of OpenAI response objects with navigation metadata.

    :param data: List of response objects with their input context
    :param has_more: Whether there are more results available beyond this page
    :param first_id: Identifier of the first item in this page
    :param last_id: Identifier of the last item in this page
    :param object: Object type identifier, always "list"
    """

    data: Sequence[OpenAIResponseObjectWithInput] = Field(
        description="List of response objects with their input context"
    )
    has_more: bool = Field(description="Whether there are more results available beyond this page")
    first_id: str = Field(description="Identifier of the first item in this page")
    last_id: str = Field(description="Identifier of the last item in this page")
    object: Literal["list"] = Field(default="list", description='Object type identifier, always "list"')
