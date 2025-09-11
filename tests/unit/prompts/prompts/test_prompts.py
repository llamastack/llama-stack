# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


import pytest


class TestPrompts:
    async def test_create_and_get_prompt(self, temp_prompt_store):
        prompt = await temp_prompt_store.create_prompt("Hello world!", ["name"])
        assert prompt.prompt == "Hello world!"
        assert prompt.version == 1
        assert prompt.prompt_id.startswith("pmpt_")
        assert prompt.variables == ["name"]

        retrieved = await temp_prompt_store.get_prompt(prompt.prompt_id)
        assert retrieved.prompt_id == prompt.prompt_id
        assert retrieved.prompt == prompt.prompt

    async def test_update_prompt(self, temp_prompt_store):
        prompt = await temp_prompt_store.create_prompt("Original")
        updated = await temp_prompt_store.update_prompt(prompt.prompt_id, "Updated", 1, ["v"])
        assert updated.version == 2
        assert updated.prompt == "Updated"

    async def test_update_prompt_with_version(self, temp_prompt_store):
        version_for_update = 1

        prompt = await temp_prompt_store.create_prompt("Original")
        assert prompt.version == 1
        prompt = await temp_prompt_store.update_prompt(prompt.prompt_id, "Updated", version_for_update, ["v"])
        assert prompt.version == 2

        with pytest.raises(ValueError):
            # now this is a stale version
            await temp_prompt_store.update_prompt(prompt.prompt_id, "Another Update", version_for_update, ["v"])

        with pytest.raises(ValueError):
            # this version does not exist
            await temp_prompt_store.update_prompt(prompt.prompt_id, "Another Update", 99, ["v"])

    async def test_delete_prompt(self, temp_prompt_store):
        prompt = await temp_prompt_store.create_prompt("to be deleted")
        await temp_prompt_store.delete_prompt(prompt.prompt_id)
        with pytest.raises(ValueError):
            await temp_prompt_store.get_prompt(prompt.prompt_id)

    async def test_list_prompts(self, temp_prompt_store):
        response = await temp_prompt_store.list_prompts()
        assert response.data == []

        await temp_prompt_store.create_prompt("first")
        await temp_prompt_store.create_prompt("second")

        response = await temp_prompt_store.list_prompts()
        assert len(response.data) == 2

    async def test_version(self, temp_prompt_store):
        prompt = await temp_prompt_store.create_prompt("V1")
        await temp_prompt_store.update_prompt(prompt.prompt_id, "V2", 1)

        v1 = await temp_prompt_store.get_prompt(prompt.prompt_id, version=1)
        assert v1.version == 1 and v1.prompt == "V1"

        latest = await temp_prompt_store.get_prompt(prompt.prompt_id)
        assert latest.version == 2 and latest.prompt == "V2"

    async def test_set_default_version(self, temp_prompt_store):
        prompt0 = await temp_prompt_store.create_prompt("V1")
        prompt1 = await temp_prompt_store.update_prompt(prompt0.prompt_id, "V2", 1)

        assert (await temp_prompt_store.get_prompt(prompt0.prompt_id)).version == 2
        prompt_default = await temp_prompt_store.set_default_version(prompt0.prompt_id, 1)
        assert (await temp_prompt_store.get_prompt(prompt0.prompt_id)).version == 1
        assert prompt_default.version == 1

        prompt2 = await temp_prompt_store.update_prompt(prompt0.prompt_id, "V3", prompt1.version)
        assert prompt2.version == 3

    async def test_prompt_id_generation_and_validation(self, temp_prompt_store):
        prompt = await temp_prompt_store.create_prompt("Test")
        assert prompt.prompt_id.startswith("pmpt_")
        assert len(prompt.prompt_id) == 53

        with pytest.raises(ValueError):
            await temp_prompt_store.get_prompt("invalid_id")

    async def test_list_shows_default_versions(self, temp_prompt_store):
        prompt = await temp_prompt_store.create_prompt("V1")
        await temp_prompt_store.update_prompt(prompt.prompt_id, "V2", 1)
        await temp_prompt_store.update_prompt(prompt.prompt_id, "V3", 2)

        response = await temp_prompt_store.list_prompts()
        listed_prompt = response.data[0]
        assert listed_prompt.version == 3 and listed_prompt.prompt == "V3"

        await temp_prompt_store.set_default_version(prompt.prompt_id, 1)

        response = await temp_prompt_store.list_prompts()
        listed_prompt = response.data[0]
        assert listed_prompt.version == 1 and listed_prompt.prompt == "V1"
        assert not (await temp_prompt_store.get_prompt(prompt.prompt_id, 3)).is_default

    async def test_get_all_prompt_versions(self, temp_prompt_store):
        prompt = await temp_prompt_store.create_prompt("V1")
        await temp_prompt_store.update_prompt(prompt.prompt_id, "V2", 1)
        await temp_prompt_store.update_prompt(prompt.prompt_id, "V3", 2)

        versions = (await temp_prompt_store.list_prompt_versions(prompt.prompt_id)).data
        assert len(versions) == 3
        assert [v.version for v in versions] == [1, 2, 3]
        assert [v.is_default for v in versions] == [False, False, True]

        await temp_prompt_store.set_default_version(prompt.prompt_id, 2)
        versions = (await temp_prompt_store.list_prompt_versions(prompt.prompt_id)).data
        assert [v.is_default for v in versions] == [False, True, False]

        with pytest.raises(ValueError):
            await temp_prompt_store.list_prompt_versions("nonexistent")

    async def test_prompt_variable_validation(self, temp_prompt_store):
        prompt = await temp_prompt_store.create_prompt("Hello {{ name }}, you live in {{ city }}!", ["name", "city"])
        assert prompt.variables == ["name", "city"]

        prompt_no_vars = await temp_prompt_store.create_prompt("Hello world!", [])
        assert prompt_no_vars.variables == []

        with pytest.raises(ValueError, match="undeclared variables"):
            await temp_prompt_store.create_prompt("Hello {{ name }}, invalid {{ unknown }}!", ["name"])

    async def test_update_prompt_set_as_default_behavior(self, temp_prompt_store):
        prompt = await temp_prompt_store.create_prompt("V1")
        assert (await temp_prompt_store.get_prompt(prompt.prompt_id)).version == 1

        prompt_v2 = await temp_prompt_store.update_prompt(prompt.prompt_id, "V2", 1, [], set_as_default=True)
        assert prompt_v2.version == 2
        assert (await temp_prompt_store.get_prompt(prompt.prompt_id)).version == 2

        prompt_v3 = await temp_prompt_store.update_prompt(prompt.prompt_id, "V3", 2, [], set_as_default=False)
        assert prompt_v3.version == 3
        assert (await temp_prompt_store.get_prompt(prompt.prompt_id)).version == 2
