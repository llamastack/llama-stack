# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from dataclasses import dataclass, field

import pytest

from ogx.core.access_control.conditions import (
    ResourceIsUnowned,
    UserInOwnersList,
    UserIsNotOwner,
    UserIsOwner,
    UserNotInOwnersList,
    UserWithValueInList,
    UserWithValueNotInList,
    parse_condition,
)


@dataclass
class MockUser:
    principal: str
    attributes: dict[str, list[str]] | None = None


@dataclass
class MockResource:
    type: str = "model"
    identifier: str = "test-model"
    owner: MockUser | None = None


@dataclass
class MockOwner:
    principal: str = "owner-1"
    attributes: dict[str, list[str]] = field(default_factory=lambda: {"roles": ["admin"]})


# --- UserIsOwner ---


class TestUserIsOwner:
    def test_user_is_owner(self):
        user = MockUser(principal="user-1")
        resource = MockResource(owner=MockUser(principal="user-1"))
        assert UserIsOwner().matches(resource, user) is True

    def test_user_is_not_the_owner(self):
        user = MockUser(principal="user-1")
        resource = MockResource(owner=MockUser(principal="user-2"))
        assert UserIsOwner().matches(resource, user) is False

    def test_resource_has_no_owner(self):
        user = MockUser(principal="user-1")
        resource = MockResource(owner=None)
        assert UserIsOwner().matches(resource, user) is False


# --- UserIsNotOwner ---


class TestUserIsNotOwner:
    def test_user_is_not_the_owner(self):
        user = MockUser(principal="user-1")
        resource = MockResource(owner=MockUser(principal="user-2"))
        assert UserIsNotOwner().matches(resource, user) is True

    def test_user_is_the_owner(self):
        user = MockUser(principal="user-1")
        resource = MockResource(owner=MockUser(principal="user-1"))
        assert UserIsNotOwner().matches(resource, user) is False

    def test_missing_owner_denies_access(self):
        """Regression: must fail-closed when resource has no owner."""
        user = MockUser(principal="user-1")
        resource = MockResource(owner=None)
        assert UserIsNotOwner().matches(resource, user) is False

    def test_missing_owner_in_route_context(self):
        """Regression: _RouteContext has owner=None, must not grant access."""
        user = MockUser(principal="user-1")
        route_ctx = MockResource(type="route", identifier="route", owner=None)
        assert UserIsNotOwner().matches(route_ctx, user) is False


# --- UserInOwnersList ---


class TestUserInOwnersList:
    def test_user_in_owners_list(self):
        owner = MockOwner(attributes={"teams": ["ml-team", "nlp-team"]})
        user = MockUser(principal="user-1", attributes={"teams": ["ml-team"]})
        resource = MockResource(owner=owner)
        assert UserInOwnersList("teams").matches(resource, user) is True

    def test_user_not_in_owners_list(self):
        owner = MockOwner(attributes={"teams": ["ml-team"]})
        user = MockUser(principal="user-1", attributes={"teams": ["infra-team"]})
        resource = MockResource(owner=owner)
        assert UserInOwnersList("teams").matches(resource, user) is False

    def test_no_owner(self):
        user = MockUser(principal="user-1", attributes={"teams": ["ml-team"]})
        resource = MockResource(owner=None)
        assert UserInOwnersList("teams").matches(resource, user) is False

    def test_owner_missing_attribute(self):
        owner = MockOwner(attributes={"roles": ["admin"]})
        user = MockUser(principal="user-1", attributes={"teams": ["ml-team"]})
        resource = MockResource(owner=owner)
        assert UserInOwnersList("teams").matches(resource, user) is False

    def test_user_missing_attributes(self):
        owner = MockOwner(attributes={"teams": ["ml-team"]})
        user = MockUser(principal="user-1", attributes=None)
        resource = MockResource(owner=owner)
        assert UserInOwnersList("teams").matches(resource, user) is False

    def test_user_missing_attribute_key(self):
        owner = MockOwner(attributes={"teams": ["ml-team"]})
        user = MockUser(principal="user-1", attributes={"roles": ["admin"]})
        resource = MockResource(owner=owner)
        assert UserInOwnersList("teams").matches(resource, user) is False

    def test_owner_empty_attribute_list(self):
        owner = MockOwner(attributes={"teams": []})
        user = MockUser(principal="user-1", attributes={"teams": ["ml-team"]})
        resource = MockResource(owner=owner)
        assert UserInOwnersList("teams").matches(resource, user) is False

    def test_user_empty_attribute_list(self):
        owner = MockOwner(attributes={"teams": ["ml-team"]})
        user = MockUser(principal="user-1", attributes={"teams": []})
        resource = MockResource(owner=owner)
        assert UserInOwnersList("teams").matches(resource, user) is False


# --- UserNotInOwnersList ---


class TestUserNotInOwnersList:
    def test_user_not_in_owners_list(self):
        owner = MockOwner(attributes={"teams": ["ml-team"]})
        user = MockUser(principal="user-1", attributes={"teams": ["infra-team"]})
        resource = MockResource(owner=owner)
        assert UserNotInOwnersList("teams").matches(resource, user) is True

    def test_user_in_owners_list(self):
        owner = MockOwner(attributes={"teams": ["ml-team"]})
        user = MockUser(principal="user-1", attributes={"teams": ["ml-team"]})
        resource = MockResource(owner=owner)
        assert UserNotInOwnersList("teams").matches(resource, user) is False

    def test_missing_owner_denies_access(self):
        """Regression: must fail-closed when resource has no owner."""
        user = MockUser(principal="user-1", attributes={"teams": ["ml-team"]})
        resource = MockResource(owner=None)
        assert UserNotInOwnersList("teams").matches(resource, user) is False

    def test_owner_missing_attribute_denies_access(self):
        """Regression: must fail-closed when owner lacks the attribute."""
        owner = MockOwner(attributes={"roles": ["admin"]})
        user = MockUser(principal="user-1", attributes={"teams": ["ml-team"]})
        resource = MockResource(owner=owner)
        assert UserNotInOwnersList("teams").matches(resource, user) is False

    def test_user_missing_attributes_denies_access(self):
        """Regression: must fail-closed when user has no attributes."""
        owner = MockOwner(attributes={"teams": ["ml-team"]})
        user = MockUser(principal="user-1", attributes=None)
        resource = MockResource(owner=owner)
        assert UserNotInOwnersList("teams").matches(resource, user) is False

    def test_user_missing_attribute_key_denies_access(self):
        """Regression: must fail-closed when user lacks the attribute key."""
        owner = MockOwner(attributes={"teams": ["ml-team"]})
        user = MockUser(principal="user-1", attributes={"roles": ["admin"]})
        resource = MockResource(owner=owner)
        assert UserNotInOwnersList("teams").matches(resource, user) is False

    def test_user_empty_attribute_list_denies_access(self):
        """Regression: must fail-closed when user has empty attribute list."""
        owner = MockOwner(attributes={"teams": ["ml-team"]})
        user = MockUser(principal="user-1", attributes={"teams": []})
        resource = MockResource(owner=owner)
        assert UserNotInOwnersList("teams").matches(resource, user) is False

    def test_route_context_missing_owner_denies_access(self):
        """Regression: _RouteContext has owner=None, must not grant access."""
        user = MockUser(principal="user-1", attributes={"teams": ["ml-team"]})
        route_ctx = MockResource(type="route", identifier="route", owner=None)
        assert UserNotInOwnersList("teams").matches(route_ctx, user) is False


# --- UserWithValueInList ---


class TestUserWithValueInList:
    def test_user_has_value(self):
        user = MockUser(principal="user-1", attributes={"roles": ["admin", "user"]})
        resource = MockResource()
        assert UserWithValueInList("roles", "admin").matches(resource, user) is True

    def test_user_does_not_have_value(self):
        user = MockUser(principal="user-1", attributes={"roles": ["user"]})
        resource = MockResource()
        assert UserWithValueInList("roles", "admin").matches(resource, user) is False

    def test_user_missing_attributes(self):
        user = MockUser(principal="user-1", attributes=None)
        resource = MockResource()
        assert UserWithValueInList("roles", "admin").matches(resource, user) is False

    def test_user_missing_attribute_key(self):
        user = MockUser(principal="user-1", attributes={"teams": ["ml-team"]})
        resource = MockResource()
        assert UserWithValueInList("roles", "admin").matches(resource, user) is False


# --- UserWithValueNotInList ---


class TestUserWithValueNotInList:
    def test_user_does_not_have_value(self):
        user = MockUser(principal="user-1", attributes={"roles": ["user"]})
        resource = MockResource()
        assert UserWithValueNotInList("roles", "admin").matches(resource, user) is True

    def test_user_has_value(self):
        user = MockUser(principal="user-1", attributes={"roles": ["admin", "user"]})
        resource = MockResource()
        assert UserWithValueNotInList("roles", "admin").matches(resource, user) is False

    def test_missing_attributes_denies_access(self):
        """Regression: must fail-closed when user has no attributes."""
        user = MockUser(principal="user-1", attributes=None)
        resource = MockResource()
        assert UserWithValueNotInList("roles", "admin").matches(resource, user) is False

    def test_missing_attribute_key_denies_access(self):
        """Regression: must fail-closed when user lacks the attribute key."""
        user = MockUser(principal="user-1", attributes={"teams": ["ml-team"]})
        resource = MockResource()
        assert UserWithValueNotInList("roles", "admin").matches(resource, user) is False


# --- ResourceIsUnowned ---


class TestResourceIsUnowned:
    def test_resource_is_unowned(self):
        user = MockUser(principal="user-1")
        resource = MockResource(owner=None)
        assert ResourceIsUnowned().matches(resource, user) is True

    def test_resource_is_owned(self):
        user = MockUser(principal="user-1")
        resource = MockResource(owner=MockUser(principal="owner-1"))
        assert ResourceIsUnowned().matches(resource, user) is False


# --- parse_condition ---


class TestParseCondition:
    def test_user_is_owner(self):
        condition = parse_condition("user is owner")
        assert isinstance(condition, UserIsOwner)

    def test_user_is_not_owner(self):
        condition = parse_condition("user is not owner")
        assert isinstance(condition, UserIsNotOwner)

    def test_user_with_value_in_list(self):
        condition = parse_condition("user with admin in roles")
        assert isinstance(condition, UserWithValueInList)

    def test_user_with_value_not_in_list(self):
        condition = parse_condition("user with admin not in roles")
        assert isinstance(condition, UserWithValueNotInList)

    def test_user_in_owners_list(self):
        condition = parse_condition("user in owners teams")
        assert isinstance(condition, UserInOwnersList)

    def test_user_not_in_owners_list(self):
        condition = parse_condition("user not in owners teams")
        assert isinstance(condition, UserNotInOwnersList)

    def test_resource_is_unowned(self):
        condition = parse_condition("resource is unowned")
        assert isinstance(condition, ResourceIsUnowned)

    def test_invalid_condition(self):
        with pytest.raises(ValueError, match="Invalid condition"):
            parse_condition("something invalid")

    def test_empty_string(self):
        with pytest.raises(ValueError, match="Invalid condition"):
            parse_condition("")

    def test_partial_match(self):
        with pytest.raises(ValueError, match="Invalid condition"):
            parse_condition("user is")
