# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from tempfile import TemporaryDirectory
from unittest.mock import patch

from llama_stack.core.access_control.access_control import default_policy, is_action_allowed
from llama_stack.core.access_control.datatypes import Action
from llama_stack.core.datatypes import User
from llama_stack.providers.utils.sqlstore.api import ColumnType
from llama_stack.providers.utils.sqlstore.authorized_sqlstore import AuthorizedSqlStore, SqlRecord
from llama_stack.providers.utils.sqlstore.sqlalchemy_sqlstore import SqlAlchemySqlStoreImpl
from llama_stack.providers.utils.sqlstore.sqlstore import SqliteSqlStoreConfig


@patch("llama_stack.providers.utils.sqlstore.authorized_sqlstore.get_authenticated_user")
async def test_authorized_fetch_with_where_sql_access_control(mock_get_authenticated_user):
    """Test that fetch_all works correctly with where_sql for access control"""
    with TemporaryDirectory() as tmp_dir:
        db_name = "test_access_control.db"
        base_sqlstore = SqlAlchemySqlStoreImpl(
            SqliteSqlStoreConfig(
                db_path=tmp_dir + "/" + db_name,
            )
        )
        sqlstore = AuthorizedSqlStore(base_sqlstore, default_policy())

        # Create table with access control
        await sqlstore.create_table(
            table="documents",
            schema={
                "id": ColumnType.INTEGER,
                "title": ColumnType.STRING,
                "content": ColumnType.TEXT,
            },
        )

        admin_user = User("admin-user", {"roles": ["admin"], "teams": ["engineering"]})
        regular_user = User("regular-user", {"roles": ["user"], "teams": ["marketing"]})

        # Set user attributes for creating documents
        mock_get_authenticated_user.return_value = admin_user

        # Insert documents with access attributes
        await sqlstore.insert("documents", {"id": 1, "title": "Admin Document", "content": "This is admin content"})

        # Change user attributes
        mock_get_authenticated_user.return_value = regular_user

        await sqlstore.insert("documents", {"id": 2, "title": "User Document", "content": "Public user content"})

        # Test that access control works with where parameter
        mock_get_authenticated_user.return_value = admin_user

        # Admin should see both documents
        result = await sqlstore.fetch_all("documents", where={"id": 1})
        assert len(result.data) == 1
        assert result.data[0]["title"] == "Admin Document"

        # User should only see their document
        mock_get_authenticated_user.return_value = regular_user

        result = await sqlstore.fetch_all("documents", where={"id": 1})
        assert len(result.data) == 0

        result = await sqlstore.fetch_all("documents", where={"id": 2})
        assert len(result.data) == 1
        assert result.data[0]["title"] == "User Document"

        row = await sqlstore.fetch_one("documents", where={"id": 1})
        assert row is None

        row = await sqlstore.fetch_one("documents", where={"id": 2})
        assert row is not None
        assert row["title"] == "User Document"


@patch("llama_stack.providers.utils.sqlstore.authorized_sqlstore.get_authenticated_user")
async def test_sql_policy_consistency(mock_get_authenticated_user):
    """Test that SQL WHERE clause logic exactly matches is_action_allowed policy logic"""
    with TemporaryDirectory() as tmp_dir:
        db_name = "test_consistency.db"
        base_sqlstore = SqlAlchemySqlStoreImpl(
            SqliteSqlStoreConfig(
                db_path=tmp_dir + "/" + db_name,
            )
        )
        sqlstore = AuthorizedSqlStore(base_sqlstore, default_policy())

        await sqlstore.create_table(
            table="resources",
            schema={
                "id": ColumnType.STRING,
                "name": ColumnType.STRING,
            },
        )

        # Test scenarios with different access control patterns
        test_scenarios = [
            # Scenario 1: Public record (no access control - represents None user insert)
            {"id": "1", "name": "public", "access_attributes": None},
            # Scenario 2: Record with roles requirement
            {"id": "2", "name": "admin-only", "access_attributes": {"roles": ["admin"]}},
            # Scenario 3: Record with multiple attribute categories
            {"id": "3", "name": "admin-ml-team", "access_attributes": {"roles": ["admin"], "teams": ["ml-team"]}},
            # Scenario 4: Record with teams only (missing roles category)
            {"id": "4", "name": "ml-team-only", "access_attributes": {"teams": ["ml-team"]}},
            # Scenario 5: Record with roles and projects
            {
                "id": "5",
                "name": "admin-project-x",
                "access_attributes": {"roles": ["admin"], "projects": ["project-x"]},
            },
        ]

        mock_get_authenticated_user.return_value = User("test-user", {"roles": ["admin"]})
        for scenario in test_scenarios:
            await base_sqlstore.insert("resources", scenario)

        # Test with different user configurations
        user_scenarios = [
            # User 1: No attributes (should only see public records)
            {"principal": "user1", "attributes": None},
            # User 2: Empty attributes (should only see public records)
            {"principal": "user2", "attributes": {}},
            # User 3: Admin role only
            {"principal": "user3", "attributes": {"roles": ["admin"]}},
            # User 4: ML team only
            {"principal": "user4", "attributes": {"teams": ["ml-team"]}},
            # User 5: Admin + ML team
            {"principal": "user5", "attributes": {"roles": ["admin"], "teams": ["ml-team"]}},
            # User 6: Admin + Project X
            {"principal": "user6", "attributes": {"roles": ["admin"], "projects": ["project-x"]}},
            # User 7: Different role (should only see public)
            {"principal": "user7", "attributes": {"roles": ["viewer"]}},
        ]

        policy = default_policy()

        for user_data in user_scenarios:
            user = User(principal=user_data["principal"], attributes=user_data["attributes"])
            mock_get_authenticated_user.return_value = user

            sql_results = await sqlstore.fetch_all("resources")
            sql_ids = {row["id"] for row in sql_results.data}
            policy_ids = set()
            for scenario in test_scenarios:
                sql_record = SqlRecord(
                    record_id=scenario["id"],
                    table_name="resources",
                    owner=User(principal="test-user", attributes=scenario["access_attributes"]),
                )

                if is_action_allowed(policy, Action.READ, sql_record, user):
                    policy_ids.add(scenario["id"])
            assert sql_ids == policy_ids, (
                f"Consistency failure for user {user.principal} with attributes {user.attributes}:\n"
                f"SQL returned: {sorted(sql_ids)}\n"
                f"Policy allows: {sorted(policy_ids)}\n"
                f"Difference: SQL only: {sql_ids - policy_ids}, Policy only: {policy_ids - sql_ids}"
            )


@patch("llama_stack.providers.utils.sqlstore.authorized_sqlstore.get_authenticated_user")
async def test_authorized_store_user_attribute_capture(mock_get_authenticated_user):
    """Test that user attributes are properly captured during insert"""
    with TemporaryDirectory() as tmp_dir:
        db_name = "test_attributes.db"
        base_sqlstore = SqlAlchemySqlStoreImpl(
            SqliteSqlStoreConfig(
                db_path=tmp_dir + "/" + db_name,
            )
        )
        authorized_store = AuthorizedSqlStore(base_sqlstore, default_policy())

        await authorized_store.create_table(
            table="user_data",
            schema={
                "id": ColumnType.STRING,
                "content": ColumnType.STRING,
            },
        )

        mock_get_authenticated_user.return_value = User(
            "user-with-attrs", {"roles": ["editor"], "teams": ["content"], "projects": ["blog"]}
        )

        await authorized_store.insert("user_data", {"id": "item1", "content": "User content"})

        mock_get_authenticated_user.return_value = User("user-no-attrs", None)

        await authorized_store.insert("user_data", {"id": "item2", "content": "Public content"})

        mock_get_authenticated_user.return_value = None

        await authorized_store.insert("user_data", {"id": "item3", "content": "Anonymous content"})
        result = await base_sqlstore.fetch_all("user_data", order_by=[("id", "asc")])
        assert len(result.data) == 3

        # First item should have full attributes
        assert result.data[0]["id"] == "item1"
        assert result.data[0]["access_attributes"] == {"roles": ["editor"], "teams": ["content"], "projects": ["blog"]}

        # Second item should have null attributes (user with no attributes)
        assert result.data[1]["id"] == "item2"
        assert result.data[1]["access_attributes"] is None

        # Third item should have null attributes (no authenticated user)
        assert result.data[2]["id"] == "item3"
        assert result.data[2]["access_attributes"] is None
