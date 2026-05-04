# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from enum import StrEnum
from typing import Protocol

from ogx.log import get_logger

logger = get_logger(name=__name__, category="core::access_control")


class User(Protocol):
    """Protocol for user identity with principal and attribute information."""

    principal: str
    attributes: dict[str, list[str]] | None


class ProtectedResource(Protocol):
    """Protocol for resources subject to access control."""

    type: str
    identifier: str
    owner: User | None


class Condition(Protocol):
    """Protocol for access control conditions that evaluate resource-user relationships."""

    def evaluate(self, resource: ProtectedResource, user: User) -> "ConditionEvaluation": ...

    def matches(self, resource: ProtectedResource, user: User) -> bool: ...


class ConditionEvaluation(StrEnum):
    """Tri-state result for evaluating access control conditions."""

    MATCH = "match"
    NO_MATCH = "no_match"
    INDETERMINATE = "indeterminate"


def _to_evaluation(matched: bool) -> ConditionEvaluation:
    return ConditionEvaluation.MATCH if matched else ConditionEvaluation.NO_MATCH


class UserInOwnersList:
    """Condition that checks if the user has any matching attribute values in the resource owner's attribute list."""

    def __init__(self, name: str):
        self.name = name

    def owners_values(self, resource: ProtectedResource) -> list[str] | None:
        if (
            hasattr(resource, "owner")
            and resource.owner
            and resource.owner.attributes
            and self.name in resource.owner.attributes
        ):
            return resource.owner.attributes[self.name]
        else:
            return None

    def evaluate(self, resource: ProtectedResource, user: User) -> ConditionEvaluation:
        defined = self.owners_values(resource)
        if not defined:
            return ConditionEvaluation.NO_MATCH
        if not user.attributes or self.name not in user.attributes or not user.attributes[self.name]:
            return ConditionEvaluation.NO_MATCH
        user_values = user.attributes[self.name]
        for value in defined:
            if value in user_values:
                return ConditionEvaluation.MATCH
        return ConditionEvaluation.NO_MATCH

    def matches(self, resource: ProtectedResource, user: User) -> bool:
        return self.evaluate(resource, user) is ConditionEvaluation.MATCH

    def __repr__(self) -> str:
        return f"user in owners {self.name}"


class UserNotInOwnersList(UserInOwnersList):
    """Condition that checks if the user does NOT have matching attribute values in the resource owner's attribute list."""

    def __init__(self, name: str):
        super().__init__(name)

    def evaluate(self, resource: ProtectedResource, user: User) -> ConditionEvaluation:
        if self.owners_values(resource) is None:
            logger.warning(
                "Negation condition denied access due to missing owner data",
                condition=repr(self),
            )
            return ConditionEvaluation.INDETERMINATE
        if not user.attributes or self.name not in user.attributes or not user.attributes[self.name]:
            logger.warning(
                "Negation condition denied access due to missing user attributes",
                condition=repr(self),
                attribute=self.name,
            )
            return ConditionEvaluation.INDETERMINATE
        return _to_evaluation(super().evaluate(resource, user) is ConditionEvaluation.NO_MATCH)

    def matches(self, resource: ProtectedResource, user: User) -> bool:
        return self.evaluate(resource, user) is ConditionEvaluation.MATCH

    def __repr__(self) -> str:
        return f"user not in owners {self.name}"


class UserWithValueInList:
    """Condition that checks if the user has a specific value in a named attribute list."""

    def __init__(self, name: str, value: str):
        self.name = name
        self.value = value

    def evaluate(self, resource: ProtectedResource, user: User) -> ConditionEvaluation:
        if user.attributes and self.name in user.attributes:
            return _to_evaluation(self.value in user.attributes[self.name])
        logger.debug("User does not have attribute", value=self.value, attribute=self.name)
        return ConditionEvaluation.NO_MATCH

    def matches(self, resource: ProtectedResource, user: User) -> bool:
        return self.evaluate(resource, user) is ConditionEvaluation.MATCH

    def __repr__(self) -> str:
        return f"user with {self.value} in {self.name}"


class UserWithValueNotInList(UserWithValueInList):
    """Condition that checks if the user does NOT have a specific value in a named attribute list."""

    def __init__(self, name: str, value: str):
        super().__init__(name, value)

    def evaluate(self, resource: ProtectedResource, user: User) -> ConditionEvaluation:
        if not user.attributes or self.name not in user.attributes:
            logger.warning(
                "Negation condition denied access due to missing user attributes",
                condition=repr(self),
                attribute=self.name,
            )
            return ConditionEvaluation.INDETERMINATE
        return _to_evaluation(self.value not in user.attributes[self.name])

    def matches(self, resource: ProtectedResource, user: User) -> bool:
        return self.evaluate(resource, user) is ConditionEvaluation.MATCH

    def __repr__(self) -> str:
        return f"user with {self.value} not in {self.name}"


class UserIsOwner:
    """Condition that checks if the user is the owner of the resource."""

    def evaluate(self, resource: ProtectedResource, user: User) -> ConditionEvaluation:
        return _to_evaluation(resource.owner.principal == user.principal if resource.owner else False)

    def matches(self, resource: ProtectedResource, user: User) -> bool:
        return self.evaluate(resource, user) is ConditionEvaluation.MATCH

    def __repr__(self) -> str:
        return "user is owner"


class UserIsNotOwner:
    """Condition that checks if the user is NOT the owner of the resource."""

    def evaluate(self, resource: ProtectedResource, user: User) -> ConditionEvaluation:
        if not resource.owner:
            logger.warning(
                "Negation condition denied access due to missing resource owner",
                condition=repr(self),
            )
            return ConditionEvaluation.INDETERMINATE
        return _to_evaluation(resource.owner.principal != user.principal)

    def matches(self, resource: ProtectedResource, user: User) -> bool:
        return self.evaluate(resource, user) is ConditionEvaluation.MATCH

    def __repr__(self) -> str:
        return "user is not owner"


class ResourceIsUnowned:
    """Condition that checks if the resource has no owner."""

    def evaluate(self, resource: ProtectedResource, user: User) -> ConditionEvaluation:
        return _to_evaluation(not resource.owner)

    def matches(self, resource: ProtectedResource, user: User) -> bool:
        return self.evaluate(resource, user) is ConditionEvaluation.MATCH

    def __repr__(self) -> str:
        return "resource is unowned"


def parse_condition(condition: str) -> Condition:
    """Parse a condition string into a Condition object.

    Args:
        condition: A natural language condition string (e.g., 'user is owner', 'user in owners roles').

    Returns:
        A Condition instance matching the parsed expression.

    Raises:
        ValueError: If the condition string is not recognized.
    """
    words = condition.split()
    match words:
        case ["user", "is", "owner"]:
            return UserIsOwner()
        case ["user", "is", "not", "owner"]:
            return UserIsNotOwner()
        case ["user", "with", value, "in", name]:
            return UserWithValueInList(name, value)
        case ["user", "with", value, "not", "in", name]:
            return UserWithValueNotInList(name, value)
        case ["user", "in", "owners", name]:
            return UserInOwnersList(name)
        case ["user", "not", "in", "owners", name]:
            return UserNotInOwnersList(name)
        case ["resource", "is", "unowned"]:
            return ResourceIsUnowned()
        case _:
            raise ValueError(f"Invalid condition: {condition}")


def parse_conditions(conditions: list[str]) -> list[Condition]:
    """Parse a list of condition strings into Condition objects.

    Args:
        conditions: List of natural language condition strings.

    Returns:
        List of corresponding Condition instances.
    """
    return [parse_condition(c) for c in conditions]
