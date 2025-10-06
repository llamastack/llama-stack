# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import dataclasses
import enum
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, List, Optional, Union

from llama_stack.strong_typing.schema import JsonType, Schema, StrictJsonType

URL = str


@dataclass
class Ref:
    ref_type: ClassVar[str]
    id: str

    def to_json(self) -> StrictJsonType:
        return {"$ref": f"#/components/{self.ref_type}/{self.id}"}


@dataclass
class SchemaRef(Ref):
    ref_type: ClassVar[str] = "schemas"


SchemaOrRef = Union[Schema, SchemaRef]


@dataclass
class ResponseRef(Ref):
    ref_type: ClassVar[str] = "responses"


@dataclass
class ParameterRef(Ref):
    ref_type: ClassVar[str] = "parameters"


@dataclass
class ExampleRef(Ref):
    ref_type: ClassVar[str] = "examples"


@dataclass
class Contact:
    name: Optional[str] = None
    url: Optional[URL] = None
    email: Optional[str] = None


@dataclass
class License:
    name: str
    url: Optional[URL] = None


@dataclass
class Info:
    title: str
    version: str
    description: Optional[str] = None
    termsOfService: Optional[str] = None
    contact: Optional[Contact] = None
    license: Optional[License] = None


@dataclass
class MediaType:
    schema: Optional[SchemaOrRef] = None
    example: Optional[Any] = None
    examples: Optional[Dict[str, Union["Example", ExampleRef]]] = None


@dataclass
class RequestBody:
    content: Dict[str, MediaType | Dict[str, Any]]
    description: Optional[str] = None
    required: Optional[bool] = None


@dataclass
class Response:
    description: str
    content: Optional[Dict[str, MediaType]] = None


class ParameterLocation(enum.Enum):
    Query = "query"
    Header = "header"
    Path = "path"
    Cookie = "cookie"


@dataclass
class Parameter:
    name: str
    in_: ParameterLocation
    description: Optional[str] = None
    required: Optional[bool] = None
    schema: Optional[SchemaOrRef] = None
    example: Optional[Any] = None


@dataclass
class ExtraBodyParameter:
    """Represents a parameter that arrives via extra_body in the request."""
    name: str
    schema: SchemaOrRef
    description: Optional[str] = None
    required: Optional[bool] = None


@dataclass
class Operation:
    responses: Dict[str, Union[Response, ResponseRef]]
    tags: Optional[List[str]] = None
    summary: Optional[str] = None
    description: Optional[str] = None
    operationId: Optional[str] = None
    parameters: Optional[List[Parameter]] = None
    requestBody: Optional[RequestBody] = None
    callbacks: Optional[Dict[str, "Callback"]] = None
    security: Optional[List["SecurityRequirement"]] = None
    deprecated: Optional[bool] = None
    extraBodyParameters: Optional[List[ExtraBodyParameter]] = None


@dataclass
class PathItem:
    summary: Optional[str] = None
    description: Optional[str] = None
    get: Optional[Operation] = None
    put: Optional[Operation] = None
    post: Optional[Operation] = None
    delete: Optional[Operation] = None
    options: Optional[Operation] = None
    head: Optional[Operation] = None
    patch: Optional[Operation] = None
    trace: Optional[Operation] = None

    def update(self, other: "PathItem") -> None:
        "Merges another instance of this class into this object."

        for field in dataclasses.fields(self.__class__):
            value = getattr(other, field.name)
            if value is not None:
                setattr(self, field.name, value)


# maps run-time expressions such as "$request.body#/url" to path items
Callback = Dict[str, PathItem]


@dataclass
class Example:
    summary: Optional[str] = None
    description: Optional[str] = None
    value: Optional[Any] = None
    externalValue: Optional[URL] = None


@dataclass
class Server:
    url: URL
    description: Optional[str] = None


class SecuritySchemeType(enum.Enum):
    ApiKey = "apiKey"
    HTTP = "http"
    OAuth2 = "oauth2"
    OpenIDConnect = "openIdConnect"


@dataclass
class SecurityScheme:
    type: SecuritySchemeType
    description: str


@dataclass(init=False)
class SecuritySchemeAPI(SecurityScheme):
    name: str
    in_: ParameterLocation

    def __init__(self, description: str, name: str, in_: ParameterLocation) -> None:
        super().__init__(SecuritySchemeType.ApiKey, description)
        self.name = name
        self.in_ = in_


@dataclass(init=False)
class SecuritySchemeHTTP(SecurityScheme):
    scheme: str
    bearerFormat: Optional[str] = None

    def __init__(
        self, description: str, scheme: str, bearerFormat: Optional[str] = None
    ) -> None:
        super().__init__(SecuritySchemeType.HTTP, description)
        self.scheme = scheme
        self.bearerFormat = bearerFormat


@dataclass(init=False)
class SecuritySchemeOpenIDConnect(SecurityScheme):
    openIdConnectUrl: str

    def __init__(self, description: str, openIdConnectUrl: str) -> None:
        super().__init__(SecuritySchemeType.OpenIDConnect, description)
        self.openIdConnectUrl = openIdConnectUrl


@dataclass
class Components:
    schemas: Optional[Dict[str, Schema]] = None
    responses: Optional[Dict[str, Response]] = None
    parameters: Optional[Dict[str, Parameter]] = None
    examples: Optional[Dict[str, Example]] = None
    requestBodies: Optional[Dict[str, RequestBody]] = None
    securitySchemes: Optional[Dict[str, SecurityScheme]] = None
    callbacks: Optional[Dict[str, Callback]] = None


SecurityScope = str
SecurityRequirement = Dict[str, List[SecurityScope]]


@dataclass
class Tag:
    name: str
    description: Optional[str] = None
    displayName: Optional[str] = None


@dataclass
class TagGroup:
    """
    A ReDoc extension to provide information about groups of tags.

    Exposed via the vendor-specific property "x-tagGroups" of the top-level object.
    """

    name: str
    tags: List[str]


@dataclass
class Document:
    """
    This class is a Python dataclass adaptation of the OpenAPI Specification.

    For details, see <https://swagger.io/specification/>
    """

    openapi: str
    info: Info
    servers: List[Server]
    paths: Dict[str, PathItem]
    jsonSchemaDialect: Optional[str] = None
    components: Optional[Components] = None
    security: Optional[List[SecurityRequirement]] = None
    tags: Optional[List[Tag]] = None
    tagGroups: Optional[List[TagGroup]] = None
