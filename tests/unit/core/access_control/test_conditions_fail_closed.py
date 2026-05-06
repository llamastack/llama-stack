"""Tests that negation access-control conditions fail closed on missing data (issue #5700)."""
from ogx.core.access_control.conditions import (
    UserIsNotOwner,
    UserNotInOwnersList,
    UserWithValueNotInList,
)


class _User:
    def __init__(self, principal: str, attributes: dict | None = None):
        self.principal = principal
        self.attributes = attributes


class _Owner:
    def __init__(self, principal: str, attributes: dict | None = None):
        self.principal = principal
        self.attributes = attributes


class _Resource:
    def __init__(self, owner=None):
        self.type = "test"
        self.identifier = "res-1"
        self.owner = owner


class TestUserNotInOwnersListFailClosed:
    def test_no_owner_denies(self):
        cond = UserNotInOwnersList("roles")
        user = _User("alice", {"roles": ["admin"]})
        resource = _Resource(owner=None)
        assert cond.matches(resource, user) is False, "no owner → must deny"

    def test_owner_no_attribute_denies(self):
        cond = UserNotInOwnersList("roles")
        user = _User("alice", {"roles": ["admin"]})
        owner = _Owner("bob", attributes=None)
        resource = _Resource(owner=owner)
        assert cond.matches(resource, user) is False, "owner has no attributes → must deny"

    def test_user_no_attribute_denies(self):
        cond = UserNotInOwnersList("roles")
        user = _User("alice", attributes=None)
        owner = _Owner("bob", {"roles": ["admin"]})
        resource = _Resource(owner=owner)
        assert cond.matches(resource, user) is False, "user has no attributes → must deny"

    def test_user_genuinely_not_in_list_grants(self):
        cond = UserNotInOwnersList("roles")
        user = _User("alice", {"roles": ["viewer"]})
        owner = _Owner("bob", {"roles": ["admin", "editor"]})
        resource = _Resource(owner=owner)
        assert cond.matches(resource, user) is True, "user not in list → should grant"

    def test_user_in_list_denies(self):
        cond = UserNotInOwnersList("roles")
        user = _User("alice", {"roles": ["admin"]})
        owner = _Owner("bob", {"roles": ["admin"]})
        resource = _Resource(owner=owner)
        assert cond.matches(resource, user) is False, "user in list → should deny"


class TestUserWithValueNotInListFailClosed:
    def test_user_no_attributes_denies(self):
        cond = UserWithValueNotInList("roles", "admin")
        user = _User("alice", attributes=None)
        resource = _Resource()
        assert cond.matches(resource, user) is False, "user has no attributes → must deny"

    def test_user_missing_attribute_key_denies(self):
        cond = UserWithValueNotInList("roles", "admin")
        user = _User("alice", {"teams": ["eng"]})
        resource = _Resource()
        assert cond.matches(resource, user) is False, "user missing the key → must deny"

    def test_user_genuinely_not_having_value_grants(self):
        cond = UserWithValueNotInList("roles", "admin")
        user = _User("alice", {"roles": ["viewer"]})
        resource = _Resource()
        assert cond.matches(resource, user) is True, "user lacks value → should grant"

    def test_user_having_value_denies(self):
        cond = UserWithValueNotInList("roles", "admin")
        user = _User("alice", {"roles": ["admin"]})
        resource = _Resource()
        assert cond.matches(resource, user) is False, "user has value → should deny"


class TestUserIsNotOwnerFailClosed:
    def test_no_owner_denies(self):
        cond = UserIsNotOwner()
        user = _User("alice")
        resource = _Resource(owner=None)
        assert cond.matches(resource, user) is False, "no owner → must deny"

    def test_different_owner_grants(self):
        cond = UserIsNotOwner()
        user = _User("alice")
        owner = _Owner("bob")
        resource = _Resource(owner=owner)
        assert cond.matches(resource, user) is True, "different owner → should grant"

    def test_same_owner_denies(self):
        cond = UserIsNotOwner()
        user = _User("alice")
        owner = _Owner("alice")
        resource = _Resource(owner=owner)
        assert cond.matches(resource, user) is False, "user is owner → should deny"
