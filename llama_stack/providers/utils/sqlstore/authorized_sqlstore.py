# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from collections.abc import Mapping, Sequence
from typing import Any, Literal

from llama_stack.core.access_control.access_control import default_policy, is_action_allowed
from llama_stack.core.access_control.conditions import ProtectedResource
from llama_stack.core.access_control.datatypes import AccessRule, Action, Scope
from llama_stack.core.datatypes import User
from llama_stack.core.request_headers import get_authenticated_user
from llama_stack.core.storage.datatypes import StorageBackendType
from llama_stack.log import get_logger

from .api import ColumnDefinition, ColumnType, PaginatedResponse, SqlStore

logger = get_logger(name=__name__, category="providers::utils")

# Hardcoded copy of the default policy that our SQL filtering implements
# WARNING: If default_policy() changes, this constant must be updated accordingly
# or SQL filtering will fall back to conservative mode (safe but less performant)
#
# This policy represents: "Permit all actions when user is in owners list for ALL attribute categories"
# The corresponding SQL logic is implemented in _build_default_policy_where_clause():
# - Public records (no access_attributes) are always accessible
# - Records with access_attributes require user to match ALL categories that exist in the resource
# - Missing categories in the resource are treated as "no restriction" (allow)
# - Within each category, user needs ANY matching value (OR logic)
# - Between categories, user needs ALL categories to match (AND logic)
SQL_OPTIMIZED_POLICY = [
    AccessRule(
        permit=Scope(actions=list(Action)),
        when=["user in owners roles", "user in owners teams", "user in owners projects", "user in owners namespaces"],
    ),
]


def _enhance_item_with_access_control(item: Mapping[str, Any], current_user: User | None) -> Mapping[str, Any]:
    """Add access control attributes to a data item."""
    enhanced = dict(item)
    if current_user:
        enhanced["owner_principal"] = current_user.principal
        enhanced["access_attributes"] = current_user.attributes
    else:
        enhanced["owner_principal"] = None
        enhanced["access_attributes"] = None
    return enhanced


class SqlRecord(ProtectedResource):
    def __init__(self, record_id: str, table_name: str, owner: User):
        self.type = f"sql_record::{table_name}"
        self.identifier = record_id
        self.owner = owner


class AuthorizedSqlStore:
    """
    Authorization layer for SqlStore that provides access control functionality.

    This class composes a base SqlStore and adds authorization methods that handle
    access control policies, user attribute capture, and SQL filtering optimization.
    """

    def __init__(self, sql_store: SqlStore, policy: list[AccessRule]):
        """
        Initialize the authorization layer.

        :param sql_store: Base SqlStore implementation to wrap
        :param policy: Access control policy to use for authorization
        """
        self.sql_store = sql_store
        self.policy = policy
        self._detect_database_type()
        self._validate_sql_optimized_policy()

    def _detect_database_type(self) -> None:
        """Detect the database type from the underlying SQL store."""
        if not hasattr(self.sql_store, "config"):
            raise ValueError("SqlStore must have a config attribute to be used with AuthorizedSqlStore")

        self.database_type = self.sql_store.config.type.value
        if self.database_type not in [StorageBackendType.SQL_POSTGRES.value, StorageBackendType.SQL_SQLITE.value]:
            raise ValueError(f"Unsupported database type: {self.database_type}")

    def _validate_sql_optimized_policy(self) -> None:
        """Validate that SQL_OPTIMIZED_POLICY matches the actual default_policy().

        This ensures that if default_policy() changes, we detect the mismatch and
        can update our SQL filtering logic accordingly.
        """
        actual_default = default_policy()

        if SQL_OPTIMIZED_POLICY != actual_default:
            logger.warning(
                f"SQL_OPTIMIZED_POLICY does not match default_policy(). "
                f"SQL filtering will use conservative mode. "
                f"Expected: {SQL_OPTIMIZED_POLICY}, Got: {actual_default}",
            )

    async def create_table(self, table: str, schema: Mapping[str, ColumnType | ColumnDefinition]) -> None:
        """Create a table with built-in access control support."""

        enhanced_schema = dict(schema)
        if "access_attributes" not in enhanced_schema:
            enhanced_schema["access_attributes"] = ColumnType.JSON
        if "owner_principal" not in enhanced_schema:
            enhanced_schema["owner_principal"] = ColumnType.STRING

        await self.sql_store.create_table(table, enhanced_schema)
        await self.sql_store.add_column_if_not_exists(table, "access_attributes", ColumnType.JSON)
        await self.sql_store.add_column_if_not_exists(table, "owner_principal", ColumnType.STRING)

    async def insert(self, table: str, data: Mapping[str, Any] | Sequence[Mapping[str, Any]]) -> None:
        """Insert a row or batch of rows with automatic access control attribute capture."""
        current_user = get_authenticated_user()
        enhanced_data: Mapping[str, Any] | Sequence[Mapping[str, Any]]
        if isinstance(data, Mapping):
            enhanced_data = _enhance_item_with_access_control(data, current_user)
        else:
            enhanced_data = [_enhance_item_with_access_control(item, current_user) for item in data]
        await self.sql_store.insert(table, enhanced_data)

    async def fetch_all(
        self,
        table: str,
        where: Mapping[str, Any] | None = None,
        limit: int | None = None,
        order_by: list[tuple[str, Literal["asc", "desc"]]] | None = None,
        cursor: tuple[str, str] | None = None,
    ) -> PaginatedResponse:
        """Fetch all rows with automatic access control filtering."""
        access_where = self._build_access_control_where_clause(self.policy)
        rows = await self.sql_store.fetch_all(
            table=table,
            where=where,
            where_sql=access_where,
            limit=limit,
            order_by=order_by,
            cursor=cursor,
        )

        current_user = get_authenticated_user()
        filtered_rows = []

        for row in rows.data:
            stored_access_attrs = row.get("access_attributes")
            stored_owner_principal = row.get("owner_principal") or ""

            record_id = row.get("id", "unknown")
            sql_record = SqlRecord(
                str(record_id), table, User(principal=stored_owner_principal, attributes=stored_access_attrs)
            )

            if is_action_allowed(self.policy, Action.READ, sql_record, current_user):
                filtered_rows.append(row)

        return PaginatedResponse(
            data=filtered_rows,
            has_more=rows.has_more,
        )

    async def fetch_one(
        self,
        table: str,
        where: Mapping[str, Any] | None = None,
        order_by: list[tuple[str, Literal["asc", "desc"]]] | None = None,
    ) -> dict[str, Any] | None:
        """Fetch one row with automatic access control checking."""
        results = await self.fetch_all(
            table=table,
            where=where,
            limit=1,
            order_by=order_by,
        )

        return results.data[0] if results.data else None

    async def update(self, table: str, data: Mapping[str, Any], where: Mapping[str, Any]) -> None:
        """Update rows with automatic access control attribute capture."""
        enhanced_data = dict(data)

        current_user = get_authenticated_user()
        if current_user:
            enhanced_data["owner_principal"] = current_user.principal
            enhanced_data["access_attributes"] = current_user.attributes
        else:
            enhanced_data["owner_principal"] = None
            enhanced_data["access_attributes"] = None

        await self.sql_store.update(table, enhanced_data, where)

    async def delete(self, table: str, where: Mapping[str, Any]) -> None:
        """Delete rows with automatic access control filtering."""
        await self.sql_store.delete(table, where)

    def _build_access_control_where_clause(self, policy: list[AccessRule]) -> str:
        """Build SQL WHERE clause for access control filtering.

        Only applies SQL filtering for the default policy to ensure correctness.
        For custom policies, uses conservative filtering to avoid blocking legitimate access.
        """
        current_user = get_authenticated_user()

        if not policy or policy == SQL_OPTIMIZED_POLICY:
            return self._build_default_policy_where_clause(current_user)
        else:
            return self._build_conservative_where_clause()

    def _json_extract(self, column: str, path: str) -> str:
        """Extract JSON value (keeping JSON type).

        Args:
            column: The JSON column name
            path: The JSON path (e.g., 'roles', 'teams')

        Returns:
            SQL expression to extract JSON value
        """
        if self.database_type == StorageBackendType.SQL_POSTGRES.value:
            return f"{column}->'{path}'"
        elif self.database_type == StorageBackendType.SQL_SQLITE.value:
            return f"JSON_EXTRACT({column}, '$.{path}')"
        else:
            raise ValueError(f"Unsupported database type: {self.database_type}")

    def _json_extract_text(self, column: str, path: str) -> str:
        """Extract JSON value as text.

        Args:
            column: The JSON column name
            path: The JSON path (e.g., 'roles', 'teams')

        Returns:
            SQL expression to extract JSON value as text
        """
        if self.database_type == StorageBackendType.SQL_POSTGRES.value:
            return f"{column}->>'{path}'"
        elif self.database_type == StorageBackendType.SQL_SQLITE.value:
            return f"JSON_EXTRACT({column}, '$.{path}')"
        else:
            raise ValueError(f"Unsupported database type: {self.database_type}")

    def _get_public_access_conditions(self) -> list[str]:
        """Get the SQL conditions for public access."""
        # Public records are records that have no owner_principal or access_attributes
        conditions = ["owner_principal = ''"]
        if self.database_type == StorageBackendType.SQL_POSTGRES.value:
            # Postgres stores JSON null as 'null'
            conditions.append("access_attributes::text = 'null'")
        elif self.database_type == StorageBackendType.SQL_SQLITE.value:
            conditions.append("access_attributes = 'null'")
        else:
            raise ValueError(f"Unsupported database type: {self.database_type}")
        return conditions

    def _build_default_policy_where_clause(self, current_user: User | None) -> str:
        """Build SQL WHERE clause for the default policy.

        Default policy: permit all actions when user in owners [roles, teams, projects, namespaces]
        This means user must match ALL attribute categories that exist in the resource.
        """
        base_conditions = self._get_public_access_conditions()
        user_attr_conditions = []

        if current_user and current_user.attributes:
            for attr_key, user_values in current_user.attributes.items():
                if user_values:
                    value_conditions = []
                    for value in user_values:
                        # Check if JSON array contains the value
                        escaped_value = value.replace("'", "''")
                        json_text = self._json_extract_text("access_attributes", attr_key)
                        value_conditions.append(f"({json_text} LIKE '%\"{escaped_value}\"%')")

                    if value_conditions:
                        # Check if the category is missing (NULL)
                        category_missing = f"{self._json_extract('access_attributes', attr_key)} IS NULL"
                        user_matches_category = f"({' OR '.join(value_conditions)})"
                        user_attr_conditions.append(f"({category_missing} OR {user_matches_category})")

            if user_attr_conditions:
                all_requirements_met = f"({' AND '.join(user_attr_conditions)})"
                base_conditions.append(all_requirements_met)

        return f"({' OR '.join(base_conditions)})"

    def _build_conservative_where_clause(self) -> str:
        """Conservative SQL filtering for custom policies.

        Only filters records we're 100% certain would be denied by any reasonable policy.
        """
        current_user = get_authenticated_user()

        if not current_user:
            # Only allow public records
            base_conditions = self._get_public_access_conditions()
            return f"({' OR '.join(base_conditions)})"

        return "1=1"
