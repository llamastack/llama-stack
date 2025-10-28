# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Protocol


class User(Protocol):
    principal: str
    attributes: dict[str, list[str]] | None


class ProtectedResource(Protocol):
    type: str
    identifier: str
    owner: User


class Condition(Protocol):
    def matches(self, resource: ProtectedResource, user: User) -> bool: ...


class UserInOwnersList:
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

    def matches(self, resource: ProtectedResource, user: User) -> bool:
        required = self.owners_values(resource)
        if not required:
            return True
        if not user.attributes or self.name not in user.attributes or not user.attributes[self.name]:
            return False
        user_values = user.attributes[self.name]
        for value in required:
            if value in user_values:
                return True
        return False

    def __repr__(self):
        return f"user in owners {self.name}"


class UserNotInOwnersList(UserInOwnersList):
    def __init__(self, name: str):
        super().__init__(name)

    def matches(self, resource: ProtectedResource, user: User) -> bool:
        return not super().matches(resource, user)

    def __repr__(self):
        return f"user not in owners {self.name}"


class UserWithValueInList:
    def __init__(self, name: str, value: str):
        self.name = name
        self.value = value

    def matches(self, resource: ProtectedResource, user: User) -> bool:
        if user.attributes and self.name in user.attributes:
            return self.value in user.attributes[self.name]
        print(f"User does not have {self.value} in {self.name}")
        return False

    def __repr__(self):
        return f"user with {self.value} in {self.name}"


class UserWithValueNotInList(UserWithValueInList):
    def __init__(self, name: str, value: str):
        super().__init__(name, value)

    def matches(self, resource: ProtectedResource, user: User) -> bool:
        return not super().matches(resource, user)

    def __repr__(self):
        return f"user with {self.value} not in {self.name}"


class UserIsOwner:
    def matches(self, resource: ProtectedResource, user: User) -> bool:
        return resource.owner.principal == user.principal if resource.owner else False

    def __repr__(self):
        return "user is owner"


class UserIsNotOwner:
    def matches(self, resource: ProtectedResource, user: User) -> bool:
        return not resource.owner or resource.owner.principal != user.principal

    def __repr__(self):
        return "user is not owner"


def parse_condition(condition: str) -> Condition:
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
        case _:
            raise ValueError(f"Invalid condition: {condition}")


def parse_conditions(conditions: list[str]) -> list[Condition]:
    return [parse_condition(c) for c in conditions]
