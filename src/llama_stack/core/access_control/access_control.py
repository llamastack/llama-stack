# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from llama_stack.core.datatypes import User
from llama_stack.log import get_logger

from .conditions import (
    Condition,
    ProtectedResource,
    parse_conditions,
)
from .datatypes import (
    AccessRule,
    Action,
    Scope,
)

logger = get_logger(name=__name__, category="core::auth")


def matches_resource(resource_scope: str, actual_resource: str) -> bool:
    if resource_scope == actual_resource:
        return True
    return resource_scope.endswith("::*") and actual_resource.startswith(resource_scope[:-1])


def matches_scope(
    scope: Scope,
    action: Action,
    resource: str,
    user: str | None,
) -> bool:
    if scope.resource and not matches_resource(scope.resource, resource):
        return False
    if scope.principal and scope.principal != user:
        return False
    return action in scope.actions


def as_list(obj: Any) -> list[Any]:
    if isinstance(obj, list):
        return obj
    return [obj]


def matches_conditions(
    conditions: list[Condition],
    resource: ProtectedResource,
    user: User,
) -> bool:
    for condition in conditions:
        # must match all conditions
        if not condition.matches(resource, user):
            return False
    return True


def default_policy() -> list[AccessRule]:
    # for backwards compatibility, if no rules are provided, assume
    # full access subject to previous attribute matching rules
    return [
        AccessRule(
            permit=Scope(actions=list(Action)),
            when=["user in owners " + name],
        )
        for name in ["roles", "teams", "projects", "namespaces"]
    ] + [
        AccessRule(
            permit=Scope(actions=list(Action)),
            when=["user is owner"],
        ),
        AccessRule(
            permit=Scope(actions=list(Action)),
            when=["resource is unowned"],
        ),
    ]


def is_action_allowed(
    policy: list[AccessRule],
    action: Action,
    resource: ProtectedResource,
    user: User | None,
) -> bool:
    qualified_resource_id = f"{resource.type}::{resource.identifier}"
    decision = False
    reason = ""
    index = -1

    # If user is not set, assume authentication is not enabled
    if not user:
        decision = True
        reason = "no auth"
    else:
        if not len(policy):
            policy = default_policy()

        for index, rule in enumerate(policy):  # noqa: B007
            if rule.forbid and matches_scope(rule.forbid, action, qualified_resource_id, user.principal):
                if rule.when:
                    if matches_conditions(parse_conditions(as_list(rule.when)), resource, user):
                        decision = False
                        reason = rule.description or ""
                        break
                elif rule.unless:
                    if not matches_conditions(parse_conditions(as_list(rule.unless)), resource, user):
                        decision = False
                        reason = rule.description or ""
                        break
                else:
                    decision = False
                    reason = rule.description or ""
                    break
            elif rule.permit and matches_scope(rule.permit, action, qualified_resource_id, user.principal):
                if rule.when:
                    if matches_conditions(parse_conditions(as_list(rule.when)), resource, user):
                        decision = True
                        reason = rule.description or ""
                        break
                elif rule.unless:
                    if not matches_conditions(parse_conditions(as_list(rule.unless)), resource, user):
                        decision = True
                        reason = rule.description or ""
                        break
                else:
                    decision = True
                    reason = rule.description or ""
                    break
        else:
            reason = "no matching rule"
            index = -1

    # print apprived or denied
    decision_str = "APPROVED" if decision else "DENIED"
    user_str = user.principal if user else "none"
    logger.debug(
        f"AUTHZ,decision={decision_str},user={user_str},"
        f"resource_id={qualified_resource_id},action={action},"
        f"rule_index={index},reason={reason!r}"
    )
    return decision


class AccessDeniedError(RuntimeError):
    def __init__(self, action: str | None = None, resource: ProtectedResource | None = None, user: User | None = None):
        self.action = action
        self.resource = resource
        self.user = user

        message = _build_access_denied_message(action, resource, user)
        super().__init__(message)


def _build_access_denied_message(action: str | None, resource: ProtectedResource | None, user: User | None) -> str:
    """Build detailed error message for access denied scenarios."""
    if action and resource and user:
        resource_info = f"{resource.type}::{resource.identifier}"
        user_info = f"'{user.principal}'"
        if user.attributes:
            attrs = ", ".join([f"{k}={v}" for k, v in user.attributes.items()])
            user_info += f" (attributes: {attrs})"

        message = f"User {user_info} cannot perform action '{action}' on resource '{resource_info}'"
    else:
        message = "Insufficient permissions"

    return message
