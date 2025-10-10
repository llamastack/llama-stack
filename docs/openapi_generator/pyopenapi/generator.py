# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import hashlib
import inspect
import ipaddress
import os
import types
import typing
from dataclasses import make_dataclass
from pathlib import Path
from typing import Annotated, Any, Dict, get_args, get_origin, Set, Union

from fastapi import UploadFile

from llama_stack.apis.datatypes import Error
from llama_stack.strong_typing.core import JsonType
from llama_stack.strong_typing.docstring import Docstring, parse_type
from llama_stack.strong_typing.inspection import (
    is_generic_list,
    is_type_optional,
    is_type_union,
    is_unwrapped_body_param,
    unwrap_generic_list,
    unwrap_optional_type,
    unwrap_union_types,
)
from llama_stack.strong_typing.name import python_type_to_name
from llama_stack.strong_typing.schema import (
    get_schema_identifier,
    JsonSchemaGenerator,
    register_schema,
    Schema,
    SchemaOptions,
)
from llama_stack.strong_typing.serialization import json_dump_string, object_to_json
from pydantic import BaseModel

from .operations import (
    EndpointOperation,
    get_endpoint_events,
    get_endpoint_operations,
    HTTPMethod,
)
from .options import *
from .specification import (
    Components,
    Document,
    Example,
    ExampleRef,
    ExtraBodyParameter,
    MediaType,
    Operation,
    Parameter,
    ParameterLocation,
    PathItem,
    RequestBody,
    Response,
    ResponseRef,
    SchemaOrRef,
    SchemaRef,
    Tag,
    TagGroup,
)

register_schema(
    ipaddress.IPv4Address,
    schema={
        "type": "string",
        "format": "ipv4",
        "title": "IPv4 address",
        "description": "IPv4 address, according to dotted-quad ABNF syntax as defined in RFC 2673, section 3.2.",
    },
    examples=["192.0.2.0", "198.51.100.1", "203.0.113.255"],
)

register_schema(
    ipaddress.IPv6Address,
    schema={
        "type": "string",
        "format": "ipv6",
        "title": "IPv6 address",
        "description": "IPv6 address, as defined in RFC 2373, section 2.2.",
    },
    examples=[
        "FEDC:BA98:7654:3210:FEDC:BA98:7654:3210",
        "1080:0:0:0:8:800:200C:417A",
        "1080::8:800:200C:417A",
        "FF01::101",
        "::1",
    ],
)


def http_status_to_string(status_code: HTTPStatusCode) -> str:
    "Converts an HTTP status code to a string."

    if isinstance(status_code, HTTPStatus):
        return str(status_code.value)
    elif isinstance(status_code, int):
        return str(status_code)
    elif isinstance(status_code, str):
        return status_code
    else:
        raise TypeError("expected: HTTP status code")


class SchemaBuilder:
    schema_generator: JsonSchemaGenerator
    schemas: Dict[str, Schema]

    def __init__(self, schema_generator: JsonSchemaGenerator) -> None:
        self.schema_generator = schema_generator
        self.schemas = {}

    def classdef_to_schema(self, typ: type) -> Schema:
        """
        Converts a type to a JSON schema.
        For nested types found in the type hierarchy, adds the type to the schema registry in the OpenAPI specification section `components`.
        """

        type_schema, type_definitions = self.schema_generator.classdef_to_schema(typ)

        # append schema to list of known schemas, to be used in OpenAPI's Components Object section
        for ref, schema in type_definitions.items():
            self._add_ref(ref, schema)

        return type_schema

    def classdef_to_named_schema(self, name: str, typ: type) -> Schema:
        schema = self.classdef_to_schema(typ)
        self._add_ref(name, schema)
        return schema

    def classdef_to_ref(self, typ: type) -> SchemaOrRef:
        """
        Converts a type to a JSON schema, and if possible, returns a schema reference.
        For composite types (such as classes), adds the type to the schema registry in the OpenAPI specification section `components`.
        """

        type_schema = self.classdef_to_schema(typ)
        if typ is str or typ is int or typ is float:
            # represent simple types as themselves
            return type_schema

        type_name = get_schema_identifier(typ)
        if type_name is not None:
            return self._build_ref(type_name, type_schema)

        try:
            type_name = python_type_to_name(typ)
            return self._build_ref(type_name, type_schema)
        except TypeError:
            pass

        return type_schema

    def _build_ref(self, type_name: str, type_schema: Schema) -> SchemaRef:
        self._add_ref(type_name, type_schema)
        return SchemaRef(type_name)

    def _add_ref(self, type_name: str, type_schema: Schema) -> None:
        if type_name not in self.schemas:
            self.schemas[type_name] = type_schema


class ContentBuilder:
    schema_builder: SchemaBuilder
    schema_transformer: Optional[Callable[[SchemaOrRef], SchemaOrRef]]
    sample_transformer: Optional[Callable[[JsonType], JsonType]]

    def __init__(
        self,
        schema_builder: SchemaBuilder,
        schema_transformer: Optional[Callable[[SchemaOrRef], SchemaOrRef]] = None,
        sample_transformer: Optional[Callable[[JsonType], JsonType]] = None,
    ) -> None:
        self.schema_builder = schema_builder
        self.schema_transformer = schema_transformer
        self.sample_transformer = sample_transformer

    def build_content(
        self, payload_type: type, examples: Optional[List[Any]] = None
    ) -> Dict[str, MediaType]:
        "Creates the content subtree for a request or response."

        def is_iterator_type(t):
            return "StreamChunk" in str(t) or "OpenAIResponseObjectStream" in str(t)

        def get_media_type(t):
            if is_generic_list(t):
                return "application/jsonl"
            elif is_iterator_type(t):
                return "text/event-stream"
            else:
                return "application/json"

        if typing.get_origin(payload_type) in (typing.Union, types.UnionType):
            media_types = []
            item_types = []
            for x in typing.get_args(payload_type):
                media_types.append(get_media_type(x))
                item_types.append(x)

            if len(set(media_types)) == 1:
                # all types have the same media type
                return {media_types[0]: self.build_media_type(payload_type, examples)}
            else:
                # different types have different media types
                return {
                    media_type: self.build_media_type(item_type, examples)
                    for media_type, item_type in zip(media_types, item_types)
                }

        if is_generic_list(payload_type):
            media_type = "application/jsonl"
            item_type = unwrap_generic_list(payload_type)
        else:
            media_type = "application/json"
            item_type = payload_type

        return {media_type: self.build_media_type(item_type, examples)}

    def build_media_type(
        self, item_type: type, examples: Optional[List[Any]] = None
    ) -> MediaType:
        schema = self.schema_builder.classdef_to_ref(item_type)
        if self.schema_transformer:
            schema_transformer: Callable[[SchemaOrRef], SchemaOrRef] = (
                self.schema_transformer
            )
            schema = schema_transformer(schema)

        if not examples:
            return MediaType(schema=schema)

        if len(examples) == 1:
            return MediaType(schema=schema, example=self._build_example(examples[0]))

        return MediaType(
            schema=schema,
            examples=self._build_examples(examples),
        )

    def _build_examples(
        self, examples: List[Any]
    ) -> Dict[str, Union[Example, ExampleRef]]:
        "Creates a set of several examples for a media type."

        if self.sample_transformer:
            sample_transformer: Callable[[JsonType], JsonType] = self.sample_transformer  # type: ignore
        else:
            sample_transformer = lambda sample: sample

        results: Dict[str, Union[Example, ExampleRef]] = {}
        for example in examples:
            value = sample_transformer(object_to_json(example))

            hash_string = (
                hashlib.sha256(json_dump_string(value).encode("utf-8"))
                .digest()
                .hex()[:16]
            )
            name = f"ex-{hash_string}"

            results[name] = Example(value=value)

        return results

    def _build_example(self, example: Any) -> Any:
        "Creates a single example for a media type."

        if self.sample_transformer:
            sample_transformer: Callable[[JsonType], JsonType] = self.sample_transformer  # type: ignore
        else:
            sample_transformer = lambda sample: sample

        return sample_transformer(object_to_json(example))


@dataclass
class ResponseOptions:
    """
    Configuration options for building a response for an operation.

    :param type_descriptions: Maps each response type to a textual description (if available).
    :param examples: A list of response examples.
    :param status_catalog: Maps each response type to an HTTP status code.
    :param default_status_code: HTTP status code assigned to responses that have no mapping.
    """

    type_descriptions: Dict[type, str]
    examples: Optional[List[Any]]
    status_catalog: Dict[type, HTTPStatusCode]
    default_status_code: HTTPStatusCode


@dataclass
class StatusResponse:
    status_code: str
    types: List[type] = dataclasses.field(default_factory=list)
    examples: List[Any] = dataclasses.field(default_factory=list)


def create_docstring_for_request(
    request_name: str, fields: List[Tuple[str, type, Any]], doc_params: Dict[str, str]
) -> str:
    """Creates a ReST-style docstring for a dynamically generated request dataclass."""
    lines = ["\n"]  # Short description

    # Add parameter documentation in ReST format
    for name, type_ in fields:
        desc = doc_params.get(name, "")
        lines.append(f":param {name}: {desc}")

    return "\n".join(lines)


class ResponseBuilder:
    content_builder: ContentBuilder

    def __init__(self, content_builder: ContentBuilder) -> None:
        self.content_builder = content_builder

    def _get_status_responses(
        self, options: ResponseOptions
    ) -> Dict[str, StatusResponse]:
        status_responses: Dict[str, StatusResponse] = {}

        for response_type in options.type_descriptions.keys():
            status_code = http_status_to_string(
                options.status_catalog.get(response_type, options.default_status_code)
            )

            # look up response for status code
            if status_code not in status_responses:
                status_responses[status_code] = StatusResponse(status_code)
            status_response = status_responses[status_code]

            # append response types that are assigned the given status code
            status_response.types.append(response_type)

            # append examples that have the matching response type
            if options.examples:
                status_response.examples.extend(
                    example
                    for example in options.examples
                    if isinstance(example, response_type)
                )

        return dict(sorted(status_responses.items()))

    def build_response(
        self, options: ResponseOptions
    ) -> Dict[str, Union[Response, ResponseRef]]:
        """
        Groups responses that have the same status code.
        """

        responses: Dict[str, Union[Response, ResponseRef]] = {}
        status_responses = self._get_status_responses(options)
        for status_code, status_response in status_responses.items():
            response_types = tuple(status_response.types)
            if len(response_types) > 1:
                composite_response_type: type = Union[response_types]  # type: ignore
            else:
                (response_type,) = response_types
                composite_response_type = response_type

            description = " **OR** ".join(
                filter(
                    None,
                    (
                        options.type_descriptions[response_type]
                        for response_type in response_types
                    ),
                )
            )

            responses[status_code] = self._build_response(
                response_type=composite_response_type,
                description=description,
                examples=status_response.examples or None,
            )

        return responses

    def _build_response(
        self,
        response_type: type,
        description: str,
        examples: Optional[List[Any]] = None,
    ) -> Response:
        "Creates a response subtree."

        if response_type is not None:
            return Response(
                description=description,
                content=self.content_builder.build_content(response_type, examples),
            )
        else:
            return Response(description=description)


def schema_error_wrapper(schema: SchemaOrRef) -> Schema:
    "Wraps an error output schema into a top-level error schema."

    return {
        "type": "object",
        "properties": {
            "error": schema,  # type: ignore
        },
        "additionalProperties": False,
        "required": [
            "error",
        ],
    }


def sample_error_wrapper(error: JsonType) -> JsonType:
    "Wraps an error output sample into a top-level error sample."

    return {"error": error}


class Generator:
    endpoint: type
    options: Options
    schema_builder: SchemaBuilder
    responses: Dict[str, Response]

    def __init__(self, endpoint: type, options: Options) -> None:
        self.endpoint = endpoint
        self.options = options
        schema_generator = JsonSchemaGenerator(
            SchemaOptions(
                definitions_path="#/components/schemas/",
                use_examples=self.options.use_examples,
                property_description_fun=options.property_description_fun,
            )
        )
        self.schema_builder = SchemaBuilder(schema_generator)
        self.responses = {}

        # Create standard error responses
        self._create_standard_error_responses()

    def _create_standard_error_responses(self) -> None:
        """
        Creates standard error responses that can be reused across operations.
        These will be added to the components.responses section of the OpenAPI document.
        """
        # Get the Error schema
        error_schema = self.schema_builder.classdef_to_ref(Error)

        # Create standard error responses
        self.responses["BadRequest400"] = Response(
            description="The request was invalid or malformed",
            content={
                "application/json": MediaType(
                    schema=error_schema,
                    example={
                        "status": 400,
                        "title": "Bad Request",
                        "detail": "The request was invalid or malformed",
                    },
                )
            },
        )

        self.responses["TooManyRequests429"] = Response(
            description="The client has sent too many requests in a given amount of time",
            content={
                "application/json": MediaType(
                    schema=error_schema,
                    example={
                        "status": 429,
                        "title": "Too Many Requests",
                        "detail": "You have exceeded the rate limit. Please try again later.",
                    },
                )
            },
        )

        self.responses["InternalServerError500"] = Response(
            description="The server encountered an unexpected error",
            content={
                "application/json": MediaType(
                    schema=error_schema,
                    example={
                        "status": 500,
                        "title": "Internal Server Error",
                        "detail": "An unexpected error occurred. Our team has been notified.",
                    },
                )
            },
        )

        # Add a default error response for any unhandled error cases
        self.responses["DefaultError"] = Response(
            description="An unexpected error occurred",
            content={
                "application/json": MediaType(
                    schema=error_schema,
                    example={
                        "status": 0,
                        "title": "Error",
                        "detail": "An unexpected error occurred",
                    },
                )
            },
        )

    def _build_type_tag(self, ref: str, schema: Schema) -> Tag:
        # Don't include schema definition in the tag description because for one,
        # it is not very valuable and for another, it causes string formatting
        # discrepancies via the Stainless Studio.
        #
        # definition = f'<SchemaDefinition schemaRef="#/components/schemas/{ref}" />'
        title = typing.cast(str, schema.get("title"))
        description = typing.cast(str, schema.get("description"))
        return Tag(
            name=ref,
            description="\n\n".join(s for s in (title, description) if s is not None),
        )

    def _build_extra_tag_groups(
        self, extra_types: Dict[str, Dict[str, type]]
    ) -> Dict[str, List[Tag]]:
        """
        Creates a dictionary of tag group captions as keys, and tag lists as values.

        :param extra_types: A dictionary of type categories and list of types in that category.
        """

        extra_tags: Dict[str, List[Tag]] = {}

        for category_name, category_items in extra_types.items():
            tag_list: List[Tag] = []

            for name, extra_type in category_items.items():
                schema = self.schema_builder.classdef_to_schema(extra_type)
                tag_list.append(self._build_type_tag(name, schema))

            if tag_list:
                extra_tags[category_name] = tag_list

        return extra_tags

    def _get_api_group_for_operation(self, op) -> str | None:
        """
        Determine the API group for an operation based on its route path.

        Args:
            op: The endpoint operation

        Returns:
            The API group name derived from the route, or None if unable to determine
        """
        if not hasattr(op, 'webmethod') or not op.webmethod or not hasattr(op.webmethod, 'route'):
            return None

        route = op.webmethod.route
        if not route or not route.startswith('/'):
            return None

        # Extract API group from route path
        # Examples: /v1/agents/list -> agents-api
        #          /v1/responses -> responses-api
        #          /v1/models -> models-api
        path_parts = route.strip('/').split('/')

        if len(path_parts) < 2:
            return None

        # Skip version prefix (v1, v1alpha, v1beta, etc.)
        if path_parts[0].startswith('v1'):
            if len(path_parts) < 2:
                return None
            api_segment = path_parts[1]
        else:
            api_segment = path_parts[0]

        # Convert to supplementary file naming convention
        # agents -> agents-api, responses -> responses-api, etc.
        return f"{api_segment}-api"

    def _load_supplemental_content(self, api_group: str | None) -> str:
        """
        Load supplemental content for an API group based on stability level.

        Follows this resolution order:
        1. docs/supplementary/{stability}/{api_group}.md
        2. docs/supplementary/shared/{api_group}.md (fallback)
        3. Empty string if no files found

        Args:
            api_group: The API group name (e.g., "agents-responses-api"), or None if no mapping exists

        Returns:
            The supplemental content as markdown string, or empty string if not found
        """
        if not api_group:
            return ""

        base_path = Path(__file__).parent.parent.parent / "supplementary"

        # Try stability-specific content first if stability filter is set
        if self.options.stability_filter:
            stability_path = base_path / self.options.stability_filter / f"{api_group}.md"
            if stability_path.exists():
                try:
                    return stability_path.read_text(encoding="utf-8")
                except Exception as e:
                    print(f"Warning: Could not read stability-specific supplemental content from {stability_path}: {e}")

        # Fall back to shared content
        shared_path = base_path / "shared" / f"{api_group}.md"
        if shared_path.exists():
            try:
                return shared_path.read_text(encoding="utf-8")
            except Exception as e:
                print(f"Warning: Could not read shared supplemental content from {shared_path}: {e}")

        # No supplemental content found
        return ""

    def _build_operation(self, op: EndpointOperation) -> Operation:
        if op.defining_class.__name__ in [
            "SyntheticDataGeneration",
            "PostTraining",
        ]:
            op.defining_class.__name__ = f"{op.defining_class.__name__} (Coming Soon)"
            print(op.defining_class.__name__)

        # TODO (xiyan): temporary fix for datasetio inner impl + datasets api
        # if op.defining_class.__name__ in ["DatasetIO"]:
        #     op.defining_class.__name__ = "Datasets"

        doc_string = parse_type(op.func_ref)
        doc_params = dict(
            (param.name, param.description) for param in doc_string.params.values()
        )

        # parameters passed in URL component path
        path_parameters = [
            Parameter(
                name=param_name,
                in_=ParameterLocation.Path,
                description=doc_params.get(param_name),
                required=True,
                schema=self.schema_builder.classdef_to_ref(param_type),
            )
            for param_name, param_type in op.path_params
        ]

        # parameters passed in URL component query string
        query_parameters = []
        for param_name, param_type in op.query_params:
            if is_type_optional(param_type):
                inner_type: type = unwrap_optional_type(param_type)
                required = False
            else:
                inner_type = param_type
                required = True

            query_parameter = Parameter(
                name=param_name,
                in_=ParameterLocation.Query,
                description=doc_params.get(param_name),
                required=required,
                schema=self.schema_builder.classdef_to_ref(inner_type),
            )
            query_parameters.append(query_parameter)

        # parameters passed anywhere
        parameters = path_parameters + query_parameters

        # Build extra body parameters documentation
        extra_body_parameters = []
        for param_name, param_type, description in op.extra_body_params:
            if is_type_optional(param_type):
                inner_type: type = unwrap_optional_type(param_type)
                required = False
            else:
                inner_type = param_type
                required = True

            # Use description from ExtraBodyField if available, otherwise from docstring
            param_description = description or doc_params.get(param_name)

            extra_body_param = ExtraBodyParameter(
                name=param_name,
                schema=self.schema_builder.classdef_to_ref(inner_type),
                description=param_description,
                required=required,
            )
            extra_body_parameters.append(extra_body_param)

        webmethod = getattr(op.func_ref, "__webmethod__", None)
        raw_bytes_request_body = False
        if webmethod:
            raw_bytes_request_body = getattr(webmethod, "raw_bytes_request_body", False)

        # data passed in request body as raw bytes cannot have request parameters
        if raw_bytes_request_body and op.request_params:
            raise ValueError(
                "Cannot have both raw bytes request body and request parameters"
            )

        # data passed in request body as raw bytes
        if raw_bytes_request_body:
            requestBody = RequestBody(
                content={
                    "application/octet-stream": {
                        "schema": {
                            "type": "string",
                            "format": "binary",
                        }
                    }
                },
                required=True,
            )
        # data passed in request body as multipart/form-data
        elif op.multipart_params:
            builder = ContentBuilder(self.schema_builder)

            # Create schema properties for multipart form fields
            properties = {}
            required_fields = []

            for name, param_type in op.multipart_params:
                if get_origin(param_type) is Annotated:
                    base_type = get_args(param_type)[0]
                else:
                    base_type = param_type

                # Check if the type is optional
                is_optional = is_type_optional(base_type)
                if is_optional:
                    base_type = unwrap_optional_type(base_type)

                if base_type is UploadFile:
                    # File upload
                    properties[name] = {"type": "string", "format": "binary"}
                else:
                    # All other types - generate schema reference
                    # This includes enums, BaseModels, and simple types
                    properties[name] = self.schema_builder.classdef_to_ref(base_type)

                if not is_optional:
                    required_fields.append(name)

            multipart_schema = {
                "type": "object",
                "properties": properties,
                "required": required_fields,
            }

            requestBody = RequestBody(
                content={"multipart/form-data": {"schema": multipart_schema}},
                required=True,
            )
        # data passed in payload as JSON and mapped to request parameters
        elif op.request_params:
            builder = ContentBuilder(self.schema_builder)
            first = next(iter(op.request_params))
            request_name, request_type = first

            # Special case: if there's a single parameter with Body(embed=False) that's a BaseModel,
            # unwrap it to show the flat structure in the OpenAPI spec
            # Example: openai_chat_completion()
            if (len(op.request_params) == 1 and is_unwrapped_body_param(request_type)):
                pass
            else:
                op_name = "".join(word.capitalize() for word in op.name.split("_"))
                request_name = f"{op_name}Request"
                fields = [
                    (
                        name,
                        type_,
                    )
                    for name, type_ in op.request_params
                ]
                request_type = make_dataclass(
                    request_name,
                    fields,
                    namespace={
                        "__doc__": create_docstring_for_request(
                            request_name, fields, doc_params
                        )
                    },
                )

            requestBody = RequestBody(
                content={
                    "application/json": builder.build_media_type(
                        request_type, op.request_examples
                    )
                },
                description=doc_params.get(request_name),
                required=True,
            )
        else:
            requestBody = None

        # success response types
        if doc_string.returns is None and is_type_union(op.response_type):
            # split union of return types into a list of response types
            success_type_docstring: Dict[type, Docstring] = {
                typing.cast(type, item): parse_type(item)
                for item in unwrap_union_types(op.response_type)
            }
            success_type_descriptions = {
                item: doc_string.short_description
                for item, doc_string in success_type_docstring.items()
            }
        else:
            # use return type as a single response type
            success_type_descriptions = {
                op.response_type: (
                    doc_string.returns.description if doc_string.returns else "OK"
                )
            }

        response_examples = op.response_examples or []
        success_examples = [
            example
            for example in response_examples
            if not isinstance(example, Exception)
        ]

        content_builder = ContentBuilder(self.schema_builder)
        response_builder = ResponseBuilder(content_builder)
        response_options = ResponseOptions(
            success_type_descriptions,
            success_examples if self.options.use_examples else None,
            self.options.success_responses,
            "200",
        )
        responses = response_builder.build_response(response_options)

        # failure response types
        if doc_string.raises:
            exception_types: Dict[type, str] = {
                item.raise_type: item.description for item in doc_string.raises.values()
            }
            exception_examples = [
                example
                for example in response_examples
                if isinstance(example, Exception)
            ]

            if self.options.error_wrapper:
                schema_transformer = schema_error_wrapper
                sample_transformer = sample_error_wrapper
            else:
                schema_transformer = None
                sample_transformer = None

            content_builder = ContentBuilder(
                self.schema_builder,
                schema_transformer=schema_transformer,
                sample_transformer=sample_transformer,
            )
            response_builder = ResponseBuilder(content_builder)
            response_options = ResponseOptions(
                exception_types,
                exception_examples if self.options.use_examples else None,
                self.options.error_responses,
                "500",
            )
            responses.update(response_builder.build_response(response_options))

        assert len(responses.keys()) > 0, f"No responses found for {op.name}"

        # Add standard error response references
        if self.options.include_standard_error_responses:
            if "400" not in responses:
                responses["400"] = ResponseRef("BadRequest400")
            if "429" not in responses:
                responses["429"] = ResponseRef("TooManyRequests429")
            if "500" not in responses:
                responses["500"] = ResponseRef("InternalServerError500")
            if "default" not in responses:
                responses["default"] = ResponseRef("DefaultError")

        if op.event_type is not None:
            builder = ContentBuilder(self.schema_builder)
            callbacks = {
                f"{op.func_name}_callback": {
                    "{$request.query.callback}": PathItem(
                        post=Operation(
                            requestBody=RequestBody(
                                content=builder.build_content(op.event_type)
                            ),
                            responses={"200": Response(description="OK")},
                        )
                    )
                }
            }

        else:
            callbacks = None

        # Build base description from docstring
        base_description = "\n".join(
            filter(None, [doc_string.short_description, doc_string.long_description])
        )

        # Individual endpoints get clean descriptions only
        description = base_description

        return Operation(
            tags=[
                getattr(op.defining_class, "API_NAMESPACE", op.defining_class.__name__)
            ],
            summary=doc_string.short_description,
            description=description,
            parameters=parameters,
            requestBody=requestBody,
            responses=responses,
            callbacks=callbacks,
            deprecated=getattr(op.webmethod, "deprecated", False)
            or "DEPRECATED" in op.func_name,
            security=[] if op.public else None,
            extraBodyParameters=extra_body_parameters if extra_body_parameters else None,
        )

    def _get_api_stability_priority(self, api_level: str) -> int:
        """
        Return sorting priority for API stability levels.
        Lower numbers = higher priority (appear first)

        :param api_level: The API level (e.g., "v1", "v1beta", "v1alpha")
        :return: Priority number for sorting
        """
        stability_order = {
            "v1": 0,  # Stable - highest priority
            "v1beta": 1,  # Beta - medium priority
            "v1alpha": 2,  # Alpha - lowest priority
        }
        return stability_order.get(api_level, 999)  # Unknown levels go last

    def generate(self) -> Document:
        paths: Dict[str, PathItem] = {}
        endpoint_classes: Set[type] = set()

        # Collect all operations and filter by stability if specified
        operations = list(
            get_endpoint_operations(
                self.endpoint, use_examples=self.options.use_examples
            )
        )

        # Filter operations by stability level if requested
        if self.options.stability_filter:
            filtered_operations = []
            for op in operations:
                deprecated = (
                    getattr(op.webmethod, "deprecated", False)
                    or "DEPRECATED" in op.func_name
                )
                stability_level = op.webmethod.level

                if self.options.stability_filter == "stable":
                    # Include v1 non-deprecated endpoints
                    if stability_level == "v1" and not deprecated:
                        filtered_operations.append(op)
                elif self.options.stability_filter == "experimental":
                    # Include v1alpha and v1beta endpoints (deprecated or not)
                    if stability_level in ["v1alpha", "v1beta"]:
                        filtered_operations.append(op)
                elif self.options.stability_filter == "deprecated":
                    # Include only deprecated endpoints
                    if deprecated:
                        filtered_operations.append(op)
                elif self.options.stability_filter == "stainless":
                    # Include both stable (v1 non-deprecated) and experimental (v1alpha, v1beta) endpoints
                    if (stability_level == "v1" and not deprecated) or stability_level in ["v1alpha", "v1beta"]:
                        filtered_operations.append(op)

            operations = filtered_operations
            print(
                f"Filtered to {len(operations)} operations for stability level: {self.options.stability_filter}"
            )

        # Sort operations by multiple criteria for consistent ordering:
        # 1. Stability level with deprecation handling (global priority):
        #    - Active stable (v1) comes first
        #    - Beta (v1beta) comes next
        #    - Alpha (v1alpha) comes next
        #    - Deprecated stable (v1 deprecated) comes last
        # 2. Route path (group related endpoints within same stability level)
        # 3. HTTP method (GET, POST, PUT, DELETE, PATCH)
        # 4. Operation name (alphabetical)
        def sort_key(op):
            http_method_order = {
                HTTPMethod.GET: 0,
                HTTPMethod.POST: 1,
                HTTPMethod.PUT: 2,
                HTTPMethod.DELETE: 3,
                HTTPMethod.PATCH: 4,
            }

            # Enhanced stability priority for migration pattern support
            deprecated = getattr(op.webmethod, "deprecated", False)
            stability_priority = self._get_api_stability_priority(op.webmethod.level)

            # Deprecated versions should appear after everything else
            # This ensures deprecated stable endpoints come last globally
            if deprecated:
                stability_priority += 10  # Push deprecated endpoints to the end

            return (
                stability_priority,  # Global stability handling comes first
                op.get_route(
                    op.webmethod
                ),  # Group by route path within stability level
                http_method_order.get(op.http_method, 999),
                op.func_name,
            )

        operations.sort(key=sort_key)

        # Debug output for migration pattern tracking
        migration_routes = {}
        for op in operations:
            route_key = (op.get_route(op.webmethod), op.http_method)
            if route_key not in migration_routes:
                migration_routes[route_key] = []
            migration_routes[route_key].append(
                (op.webmethod.level, getattr(op.webmethod, "deprecated", False))
            )

        for route_key, versions in migration_routes.items():
            if len(versions) > 1:
                print(f"Migration pattern detected for {route_key[1]} {route_key[0]}:")
                for level, deprecated in versions:
                    status = "DEPRECATED" if deprecated else "ACTIVE"
                    print(f"  - {level} ({status})")

        for op in operations:
            endpoint_classes.add(op.defining_class)

            operation = self._build_operation(op)

            if op.http_method is HTTPMethod.GET:
                pathItem = PathItem(get=operation)
            elif op.http_method is HTTPMethod.PUT:
                pathItem = PathItem(put=operation)
            elif op.http_method is HTTPMethod.POST:
                pathItem = PathItem(post=operation)
            elif op.http_method is HTTPMethod.DELETE:
                pathItem = PathItem(delete=operation)
            elif op.http_method is HTTPMethod.PATCH:
                pathItem = PathItem(patch=operation)
            else:
                raise NotImplementedError(f"unknown HTTP method: {op.http_method}")

            route = op.get_route(op.webmethod)
            route = route.replace(":path", "")
            print(f"route: {route}")
            if route in paths:
                paths[route].update(pathItem)
            else:
                paths[route] = pathItem

        operation_tags: List[Tag] = []
        for cls in endpoint_classes:
            doc_string = parse_type(cls)
            if hasattr(cls, "API_NAMESPACE") and cls.API_NAMESPACE != cls.__name__:
                continue

            # Add supplemental content to tag pages
            api_group = f"{cls.__name__.lower()}-api"
            supplemental_content = self._load_supplemental_content(api_group)

            tag_description = doc_string.long_description or ""
            if supplemental_content:
                if tag_description:
                    tag_description = f"{tag_description}\n\n{supplemental_content}"
                else:
                    tag_description = supplemental_content

            operation_tags.append(
                Tag(
                    name=cls.__name__,
                    description=tag_description,
                    displayName=doc_string.short_description,
                )
            )

        # types that are emitted by events
        event_tags: List[Tag] = []
        events = get_endpoint_events(self.endpoint)
        for ref, event_type in events.items():
            event_schema = self.schema_builder.classdef_to_named_schema(ref, event_type)
            event_tags.append(self._build_type_tag(ref, event_schema))

        # types that are explicitly declared
        extra_tag_groups: Dict[str, List[Tag]] = {}
        if self.options.extra_types is not None:
            if isinstance(self.options.extra_types, list):
                extra_tag_groups = self._build_extra_tag_groups(
                    {"AdditionalTypes": self.options.extra_types}
                )
            elif isinstance(self.options.extra_types, dict):
                extra_tag_groups = self._build_extra_tag_groups(
                    self.options.extra_types
                )
            else:
                raise TypeError(
                    f"type mismatch for collection of extra types: {type(self.options.extra_types)}"
                )

        # list all operations and types
        tags: List[Tag] = []
        tags.extend(operation_tags)
        tags.extend(event_tags)
        for extra_tag_group in extra_tag_groups.values():
            tags.extend(extra_tag_group)

        tags = sorted(tags, key=lambda t: t.name)

        tag_groups = []
        if operation_tags:
            tag_groups.append(
                TagGroup(
                    name=self.options.map("Operations"),
                    tags=sorted(tag.name for tag in operation_tags),
                )
            )
        if event_tags:
            tag_groups.append(
                TagGroup(
                    name=self.options.map("Events"),
                    tags=sorted(tag.name for tag in event_tags),
                )
            )
        for caption, extra_tag_group in extra_tag_groups.items():
            tag_groups.append(
                TagGroup(
                    name=caption,
                    tags=sorted(tag.name for tag in extra_tag_group),
                )
            )

        if self.options.default_security_scheme:
            securitySchemes = {"Default": self.options.default_security_scheme}
        else:
            securitySchemes = None

        return Document(
            openapi=".".join(str(item) for item in self.options.version),
            info=self.options.info,
            jsonSchemaDialect=(
                "https://json-schema.org/draft/2020-12/schema"
                if self.options.version >= (3, 1, 0)
                else None
            ),
            servers=[self.options.server],
            paths=paths,
            components=Components(
                schemas=self.schema_builder.schemas,
                responses=self.responses,
                securitySchemes=securitySchemes,
            ),
            security=[{"Default": []}],
            tags=tags,
            tagGroups=tag_groups,
        )
