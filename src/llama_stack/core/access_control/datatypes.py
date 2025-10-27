# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from enum import StrEnum
from typing import Self

from pydantic import BaseModel, model_validator

from .conditions import parse_conditions


class Action(StrEnum):
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"


class Scope(BaseModel):
    principal: str | None = None
    actions: Action | list[Action]
    resource: str | None = None


def _mutually_exclusive(obj, a: str, b: str):
    if getattr(obj, a) and getattr(obj, b):
        raise ValueError(f"{a} and {b} are mutually exclusive")


def _require_one_of(obj, a: str, b: str):
    if not getattr(obj, a) and not getattr(obj, b):
        raise ValueError(f"on of {a} or {b} is required")


class AccessRule(BaseModel):
    """Access rule based loosely on cedar policy language

    A rule defines a list of action either to permit or to forbid. It may specify a
    principal or a resource that must match for the rule to take effect. The resource
    to match should be specified in the form of a type qualified identifier, e.g.
    model::my-model or vector_store::some-db, or a wildcard for all resources of a type,
    e.g. model::*. If the principal or resource are not specified, they will match all
    requests.

    A rule may also specify a condition, either a 'when' or an 'unless', with additional
    constraints as to where the rule applies. The constraints supported at present are:

    - 'user with <attr-value> in <attr-name>'
    - 'user with <attr-value> not in <attr-name>'
    - 'user is owner'
    - 'user is not owner'
    - 'user in owners <attr-name>'
    - 'user not in owners <attr-name>'

    Rules are tested in order to find a match. If a match is found, the request is
    permitted or forbidden depending on the type of rule. If no match is found, the
    request is denied. If no rules are specified, a rule that allows any action as
    long as the resource attributes match the user attributes is added
    (i.e. the previous behaviour is the default).

    Some examples in yaml:

    - permit:
        principal: user-1
        actions: [create, read, delete]
        resource: model::*
      description: user-1 has full access to all models
    - permit:
        principal: user-2
        actions: [read]
        resource: model::model-1
      description: user-2 has read access to model-1 only
    - permit:
        actions: [read]
      when: user in owner teams
      description: any user has read access to any resource created by a member of their team
    - forbid:
        actions: [create, read, delete]
        resource: vector_store::*
      unless: user with admin in roles
      description: only user with admin role can use vector_store resources

    """

    permit: Scope | None = None
    forbid: Scope | None = None
    when: str | list[str] | None = None
    unless: str | list[str] | None = None
    description: str | None = None

    @model_validator(mode="after")
    def validate_rule_format(self) -> Self:
        _require_one_of(self, "permit", "forbid")
        _mutually_exclusive(self, "permit", "forbid")
        _mutually_exclusive(self, "when", "unless")
        if isinstance(self.when, list):
            parse_conditions(self.when)
        elif self.when:
            parse_conditions([self.when])
        if isinstance(self.unless, list):
            parse_conditions(self.unless)
        elif self.unless:
            parse_conditions([self.unless])
        return self
