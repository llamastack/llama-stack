# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from unittest.mock import MagicMock, Mock, patch

import pytest
import yaml
from pydantic import TypeAdapter, ValidationError

from llama_stack.apis.datatypes import Api
from llama_stack.apis.models import ModelType
from llama_stack.core.access_control.access_control import AccessDeniedError, is_action_allowed
from llama_stack.core.datatypes import AccessRule, ModelWithOwner, User
from llama_stack.core.routing_tables.models import ModelsRoutingTable


class AsyncMock(MagicMock):
    async def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)


def _return_model(model):
    return model


@pytest.fixture
async def test_setup(cached_disk_dist_registry):
    mock_inference = Mock()
    mock_inference.__provider_spec__ = MagicMock()
    mock_inference.__provider_spec__.api = Api.inference
    mock_inference.register_model = AsyncMock(side_effect=_return_model)
    routing_table = ModelsRoutingTable(
        impls_by_provider_id={"test_provider": mock_inference},
        dist_registry=cached_disk_dist_registry,
        policy={},
    )
    yield cached_disk_dist_registry, routing_table


@patch("llama_stack.core.routing_tables.common.get_authenticated_user")
async def test_access_control_with_cache(mock_get_authenticated_user, test_setup):
    registry, routing_table = test_setup
    model_public = ModelWithOwner(
        identifier="model-public",
        provider_id="test_provider",
        provider_resource_id="model-public",
        model_type=ModelType.llm,
    )
    model_admin_only = ModelWithOwner(
        identifier="model-admin",
        provider_id="test_provider",
        provider_resource_id="model-admin",
        model_type=ModelType.llm,
        owner=User("testuser", {"roles": ["admin"]}),
    )
    model_data_scientist = ModelWithOwner(
        identifier="model-data-scientist",
        provider_id="test_provider",
        provider_resource_id="model-data-scientist",
        model_type=ModelType.llm,
        owner=User("testuser", {"roles": ["data-scientist", "researcher"], "teams": ["ml-team"]}),
    )
    await registry.register(model_public)
    await registry.register(model_admin_only)
    await registry.register(model_data_scientist)

    mock_get_authenticated_user.return_value = User("test-user", {"roles": ["admin"], "teams": ["management"]})
    all_models = await routing_table.list_models()
    assert len(all_models.data) == 2

    model = await routing_table.get_model("model-public")
    assert model.identifier == "model-public"
    model = await routing_table.get_model("model-admin")
    assert model.identifier == "model-admin"
    with pytest.raises(ValueError):
        await routing_table.get_model("model-data-scientist")

    mock_get_authenticated_user.return_value = User("test-user", {"roles": ["data-scientist"], "teams": ["other-team"]})
    all_models = await routing_table.list_models()
    assert len(all_models.data) == 1
    assert all_models.data[0].identifier == "model-public"
    model = await routing_table.get_model("model-public")
    assert model.identifier == "model-public"
    with pytest.raises(ValueError):
        await routing_table.get_model("model-admin")
    with pytest.raises(ValueError):
        await routing_table.get_model("model-data-scientist")

    mock_get_authenticated_user.return_value = User("test-user", {"roles": ["data-scientist"], "teams": ["ml-team"]})
    all_models = await routing_table.list_models()
    assert len(all_models.data) == 2
    model_ids = [m.identifier for m in all_models.data]
    assert "model-public" in model_ids
    assert "model-data-scientist" in model_ids
    assert "model-admin" not in model_ids
    model = await routing_table.get_model("model-public")
    assert model.identifier == "model-public"
    model = await routing_table.get_model("model-data-scientist")
    assert model.identifier == "model-data-scientist"
    with pytest.raises(ValueError):
        await routing_table.get_model("model-admin")


@patch("llama_stack.core.routing_tables.common.get_authenticated_user")
async def test_access_control_and_updates(mock_get_authenticated_user, test_setup):
    registry, routing_table = test_setup
    model_public = ModelWithOwner(
        identifier="model-updates",
        provider_id="test_provider",
        provider_resource_id="model-updates",
        model_type=ModelType.llm,
    )
    await registry.register(model_public)
    mock_get_authenticated_user.return_value = User(
        "test-user",
        {
            "roles": ["user"],
        },
    )
    model = await routing_table.get_model("model-updates")
    assert model.identifier == "model-updates"
    model_public.owner = User("testuser", {"roles": ["admin"]})
    await registry.update(model_public)
    mock_get_authenticated_user.return_value = User(
        "test-user",
        {
            "roles": ["user"],
        },
    )
    with pytest.raises(ValueError):
        await routing_table.get_model("model-updates")
    mock_get_authenticated_user.return_value = User(
        "test-user",
        {
            "roles": ["admin"],
        },
    )
    model = await routing_table.get_model("model-updates")
    assert model.identifier == "model-updates"


@patch("llama_stack.core.routing_tables.common.get_authenticated_user")
async def test_access_control_empty_attributes(mock_get_authenticated_user, test_setup):
    registry, routing_table = test_setup
    model = ModelWithOwner(
        identifier="model-empty-attrs",
        provider_id="test_provider",
        provider_resource_id="model-empty-attrs",
        model_type=ModelType.llm,
        owner=User("testuser", {}),
    )
    await registry.register(model)
    mock_get_authenticated_user.return_value = User(
        "test-user",
        {
            "roles": [],
        },
    )
    result = await routing_table.get_model("model-empty-attrs")
    assert result.identifier == "model-empty-attrs"
    all_models = await routing_table.list_models()
    model_ids = [m.identifier for m in all_models.data]
    assert "model-empty-attrs" in model_ids


@patch("llama_stack.core.routing_tables.common.get_authenticated_user")
async def test_no_user_attributes(mock_get_authenticated_user, test_setup):
    registry, routing_table = test_setup
    model_public = ModelWithOwner(
        identifier="model-public-2",
        provider_id="test_provider",
        provider_resource_id="model-public-2",
        model_type=ModelType.llm,
    )
    model_restricted = ModelWithOwner(
        identifier="model-restricted",
        provider_id="test_provider",
        provider_resource_id="model-restricted",
        model_type=ModelType.llm,
        owner=User("testuser", {"roles": ["admin"]}),
    )
    await registry.register(model_public)
    await registry.register(model_restricted)
    mock_get_authenticated_user.return_value = User("test-user", None)
    model = await routing_table.get_model("model-public-2")
    assert model.identifier == "model-public-2"

    with pytest.raises(ValueError):
        await routing_table.get_model("model-restricted")

    all_models = await routing_table.list_models()
    assert len(all_models.data) == 1
    assert all_models.data[0].identifier == "model-public-2"


@patch("llama_stack.core.routing_tables.common.get_authenticated_user")
async def test_automatic_access_attributes(mock_get_authenticated_user, test_setup):
    """Test that newly created resources inherit access attributes from their creator."""
    registry, routing_table = test_setup

    # Set creator's attributes
    creator_attributes = {"roles": ["data-scientist"], "teams": ["ml-team"], "projects": ["llama-3"]}
    mock_get_authenticated_user.return_value = User("test-user", creator_attributes)

    # Create model without explicit access attributes
    model = ModelWithOwner(
        identifier="auto-access-model",
        provider_id="test_provider",
        provider_resource_id="auto-access-model",
        model_type=ModelType.llm,
    )
    await routing_table.register_object(model)

    # Verify the model got creator's attributes
    registered_model = await routing_table.get_model("auto-access-model")
    assert registered_model.owner is not None
    assert registered_model.owner.attributes is not None
    assert registered_model.owner.attributes["roles"] == ["data-scientist"]
    assert registered_model.owner.attributes["teams"] == ["ml-team"]
    assert registered_model.owner.attributes["projects"] == ["llama-3"]

    # Verify another user without matching attributes can't access it
    mock_get_authenticated_user.return_value = User("test-user", {"roles": ["engineer"], "teams": ["infra-team"]})
    with pytest.raises(ValueError):
        await routing_table.get_model("auto-access-model")

    # But a user with matching attributes can
    mock_get_authenticated_user.return_value = User(
        "test-user",
        {
            "roles": ["data-scientist", "engineer"],
            "teams": ["ml-team", "platform-team"],
            "projects": ["llama-3"],
        },
    )
    model = await routing_table.get_model("auto-access-model")
    assert model.identifier == "auto-access-model"


@pytest.fixture
async def test_setup_with_access_policy(cached_disk_dist_registry):
    mock_inference = Mock()
    mock_inference.__provider_spec__ = MagicMock()
    mock_inference.__provider_spec__.api = Api.inference
    mock_inference.register_model = AsyncMock(side_effect=_return_model)
    mock_inference.unregister_model = AsyncMock(side_effect=_return_model)

    config = """
                - permit:
                    principal: user-1
                    actions: [create, read, delete]
                    description: user-1 has full access to all models
                - permit:
                    principal: user-2
                    actions: [read]
                    resource: model::test_provider/model-1
                    description: user-2 has read access to model-1 only
                - permit:
                    principal: user-3
                    actions: [read]
                    resource: model::test_provider/model-2
                    description: user-3 has read access to model-2 only
                - forbid:
                    actions: [create, read, delete]
             """
    policy = TypeAdapter(list[AccessRule]).validate_python(yaml.safe_load(config))
    routing_table = ModelsRoutingTable(
        impls_by_provider_id={"test_provider": mock_inference},
        dist_registry=cached_disk_dist_registry,
        policy=policy,
    )
    yield routing_table


@patch("llama_stack.core.routing_tables.common.get_authenticated_user")
async def test_access_policy(mock_get_authenticated_user, test_setup_with_access_policy):
    routing_table = test_setup_with_access_policy
    mock_get_authenticated_user.return_value = User(
        "user-1",
        {
            "roles": ["admin"],
            "projects": ["foo", "bar"],
        },
    )
    await routing_table.register_model("model-1", provider_model_id="model-1", provider_id="test_provider")
    await routing_table.register_model("model-2", provider_model_id="model-2", provider_id="test_provider")
    await routing_table.register_model("model-3", provider_model_id="model-3", provider_id="test_provider")
    model = await routing_table.get_model("test_provider/model-1")
    assert model.identifier == "test_provider/model-1"
    model = await routing_table.get_model("test_provider/model-2")
    assert model.identifier == "test_provider/model-2"
    model = await routing_table.get_model("test_provider/model-3")
    assert model.identifier == "test_provider/model-3"

    mock_get_authenticated_user.return_value = User(
        "user-2",
        {
            "roles": ["user"],
            "projects": ["foo"],
        },
    )
    model = await routing_table.get_model("test_provider/model-1")
    assert model.identifier == "test_provider/model-1"
    with pytest.raises(ValueError):
        await routing_table.get_model("test_provider/model-2")
    with pytest.raises(ValueError):
        await routing_table.get_model("test_provider/model-3")
    with pytest.raises(AccessDeniedError):
        await routing_table.register_model("model-4", provider_id="test_provider")
    with pytest.raises(AccessDeniedError):
        await routing_table.unregister_model("test_provider/model-1")

    mock_get_authenticated_user.return_value = User(
        "user-3",
        {
            "roles": ["user"],
            "projects": ["bar"],
        },
    )
    model = await routing_table.get_model("test_provider/model-2")
    assert model.identifier == "test_provider/model-2"
    with pytest.raises(ValueError):
        await routing_table.get_model("test_provider/model-1")
    with pytest.raises(ValueError):
        await routing_table.get_model("test_provider/model-3")
    with pytest.raises(AccessDeniedError):
        await routing_table.register_model("model-5", provider_id="test_provider")
    with pytest.raises(AccessDeniedError):
        await routing_table.unregister_model("test_provider/model-2")

    mock_get_authenticated_user.return_value = User(
        "user-1",
        {
            "roles": ["admin"],
            "projects": ["foo", "bar"],
        },
    )
    await routing_table.unregister_model("test_provider/model-3")
    with pytest.raises(ValueError):
        await routing_table.get_model("test_provider/model-3")


def test_permit_when():
    config = """
    - permit:
        principal: user-1
        actions: [read]
      when: user in owners namespaces
    """
    policy = TypeAdapter(list[AccessRule]).validate_python(yaml.safe_load(config))
    model = ModelWithOwner(
        identifier="mymodel",
        provider_id="myprovider",
        model_type=ModelType.llm,
        owner=User("testuser", {"namespaces": ["foo"]}),
    )
    assert is_action_allowed(policy, "read", model, User("user-1", {"namespaces": ["foo"]}))
    assert not is_action_allowed(policy, "read", model, User("user-1", {"namespaces": ["bar"]}))
    assert not is_action_allowed(policy, "read", model, User("user-2", {"namespaces": ["foo"]}))


def test_permit_unless():
    config = """
    - permit:
        principal: user-1
        actions: [read]
        resource: model::*
      unless:
        - user not in owners namespaces
        - user in owners teams
    """
    policy = TypeAdapter(list[AccessRule]).validate_python(yaml.safe_load(config))
    model = ModelWithOwner(
        identifier="mymodel",
        provider_id="myprovider",
        model_type=ModelType.llm,
        owner=User("testuser", {"namespaces": ["foo"]}),
    )
    assert is_action_allowed(policy, "read", model, User("user-1", {"namespaces": ["foo"]}))
    assert not is_action_allowed(policy, "read", model, User("user-1", {"namespaces": ["bar"]}))
    assert not is_action_allowed(policy, "read", model, User("user-2", {"namespaces": ["foo"]}))


def test_forbid_when():
    config = """
    - forbid:
        principal: user-1
        actions: [read]
      when:
        user in owners namespaces
    - permit:
        actions: [read]
    """
    policy = TypeAdapter(list[AccessRule]).validate_python(yaml.safe_load(config))
    model = ModelWithOwner(
        identifier="mymodel",
        provider_id="myprovider",
        model_type=ModelType.llm,
        owner=User("testuser", {"namespaces": ["foo"]}),
    )
    assert not is_action_allowed(policy, "read", model, User("user-1", {"namespaces": ["foo"]}))
    assert is_action_allowed(policy, "read", model, User("user-1", {"namespaces": ["bar"]}))
    assert is_action_allowed(policy, "read", model, User("user-2", {"namespaces": ["foo"]}))


def test_forbid_unless():
    config = """
    - forbid:
        principal: user-1
        actions: [read]
      unless:
        user in owners namespaces
    - permit:
        actions: [read]
    """
    policy = TypeAdapter(list[AccessRule]).validate_python(yaml.safe_load(config))
    model = ModelWithOwner(
        identifier="mymodel",
        provider_id="myprovider",
        model_type=ModelType.llm,
        owner=User("testuser", {"namespaces": ["foo"]}),
    )
    assert is_action_allowed(policy, "read", model, User("user-1", {"namespaces": ["foo"]}))
    assert not is_action_allowed(policy, "read", model, User("user-1", {"namespaces": ["bar"]}))
    assert is_action_allowed(policy, "read", model, User("user-2", {"namespaces": ["foo"]}))


def test_user_has_attribute():
    config = """
    - permit:
        actions: [read]
      when: user with admin in roles
    """
    policy = TypeAdapter(list[AccessRule]).validate_python(yaml.safe_load(config))
    model = ModelWithOwner(
        identifier="mymodel",
        provider_id="myprovider",
        model_type=ModelType.llm,
    )
    assert not is_action_allowed(policy, "read", model, User("user-1", {"roles": ["basic"]}))
    assert is_action_allowed(policy, "read", model, User("user-2", {"roles": ["admin"]}))
    assert not is_action_allowed(policy, "read", model, User("user-3", {"namespaces": ["foo"]}))
    assert not is_action_allowed(policy, "read", model, User("user-4", None))


def test_user_does_not_have_attribute():
    config = """
    - permit:
        actions: [read]
      unless: user with admin not in roles
    """
    policy = TypeAdapter(list[AccessRule]).validate_python(yaml.safe_load(config))
    model = ModelWithOwner(
        identifier="mymodel",
        provider_id="myprovider",
        model_type=ModelType.llm,
    )
    assert not is_action_allowed(policy, "read", model, User("user-1", {"roles": ["basic"]}))
    assert is_action_allowed(policy, "read", model, User("user-2", {"roles": ["admin"]}))
    assert not is_action_allowed(policy, "read", model, User("user-3", {"namespaces": ["foo"]}))
    assert not is_action_allowed(policy, "read", model, User("user-4", None))


def test_is_owner():
    config = """
    - permit:
        actions: [read]
      when: user is owner
    """
    policy = TypeAdapter(list[AccessRule]).validate_python(yaml.safe_load(config))
    model = ModelWithOwner(
        identifier="mymodel",
        provider_id="myprovider",
        model_type=ModelType.llm,
        owner=User("user-2", {"namespaces": ["foo"]}),
    )
    assert not is_action_allowed(policy, "read", model, User("user-1", {"roles": ["basic"]}))
    assert is_action_allowed(policy, "read", model, User("user-2", {"roles": ["admin"]}))
    assert not is_action_allowed(policy, "read", model, User("user-3", {"namespaces": ["foo"]}))
    assert not is_action_allowed(policy, "read", model, User("user-4", None))


def test_is_not_owner():
    config = """
    - permit:
        actions: [read]
      unless: user is not owner
    """
    policy = TypeAdapter(list[AccessRule]).validate_python(yaml.safe_load(config))
    model = ModelWithOwner(
        identifier="mymodel",
        provider_id="myprovider",
        model_type=ModelType.llm,
        owner=User("user-2", {"namespaces": ["foo"]}),
    )
    assert not is_action_allowed(policy, "read", model, User("user-1", {"roles": ["basic"]}))
    assert is_action_allowed(policy, "read", model, User("user-2", {"roles": ["admin"]}))
    assert not is_action_allowed(policy, "read", model, User("user-3", {"namespaces": ["foo"]}))
    assert not is_action_allowed(policy, "read", model, User("user-4", None))


def test_invalid_rule_permit_and_forbid_both_specified():
    config = """
    - permit:
        actions: [read]
      forbid:
        actions: [create]
    """
    with pytest.raises(ValidationError):
        TypeAdapter(list[AccessRule]).validate_python(yaml.safe_load(config))


def test_invalid_rule_neither_permit_or_forbid_specified():
    config = """
    - when: user is owner
      unless: user with admin in roles
    """
    with pytest.raises(ValidationError):
        TypeAdapter(list[AccessRule]).validate_python(yaml.safe_load(config))


def test_invalid_rule_when_and_unless_both_specified():
    config = """
    - permit:
        actions: [read]
      when: user is owner
      unless: user with admin in roles
    """
    with pytest.raises(ValidationError):
        TypeAdapter(list[AccessRule]).validate_python(yaml.safe_load(config))


def test_invalid_condition():
    config = """
    - permit:
        actions: [read]
      when: random words that are not valid
    """
    with pytest.raises(ValidationError):
        TypeAdapter(list[AccessRule]).validate_python(yaml.safe_load(config))


@pytest.mark.parametrize(
    "condition",
    [
        "user is owner",
        "user is not owner",
        "user with dev in teams",
        "user with default not in namespaces",
        "user in owners roles",
        "user not in owners projects",
    ],
)
def test_condition_reprs(condition):
    from llama_stack.core.access_control.conditions import parse_condition

    assert condition == str(parse_condition(condition))
