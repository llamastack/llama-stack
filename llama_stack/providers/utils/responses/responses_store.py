# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
from llama_stack.apis.agents import (
    Order,
)
from llama_stack.apis.agents.openai_responses import (
    ListOpenAIResponseInputItem,
    ListOpenAIResponseObject,
    OpenAIDeleteResponseObject,
    OpenAIResponseInput,
    OpenAIResponseObject,
    OpenAIResponseObjectWithInput,
)
from llama_stack.core.datatypes import AccessRule
from llama_stack.core.utils.config_dirs import RUNTIME_BASE_DIR

from ..sqlstore.api import ColumnDefinition, ColumnType
from ..sqlstore.authorized_sqlstore import AuthorizedSqlStore
from ..sqlstore.sqlstore import SqliteSqlStoreConfig, SqlStoreConfig, sqlstore_impl


class ResponsesStore:
    def __init__(self, sql_store_config: SqlStoreConfig, policy: list[AccessRule]):
        if not sql_store_config:
            sql_store_config = SqliteSqlStoreConfig(
                db_path=(RUNTIME_BASE_DIR / "sqlstore.db").as_posix(),
            )
        self.sql_store = AuthorizedSqlStore(sqlstore_impl(sql_store_config))
        self.policy = policy

    async def initialize(self):
        """Create the necessary tables if they don't exist."""
        await self.sql_store.create_table(
            "openai_responses",
            {
                "id": ColumnDefinition(type=ColumnType.STRING, primary_key=True),
                "created_at": ColumnType.INTEGER,
                "response_object": ColumnType.JSON,
                "model": ColumnType.STRING,
            },
        )

    async def store_response_object(
        self, response_object: OpenAIResponseObject, input: list[OpenAIResponseInput]
    ) -> None:
        data = response_object.model_dump()
        data["input"] = [input_item.model_dump() for input_item in input]

        await self.sql_store.insert(
            "openai_responses",
            {
                "id": data["id"],
                "created_at": data["created_at"],
                "model": data["model"],
                "response_object": data,
            },
        )

    async def list_responses(
        self,
        after: str | None = None,
        limit: int | None = 50,
        model: str | None = None,
        order: Order | None = Order.desc,
    ) -> ListOpenAIResponseObject:
        """
        List responses from the database.

        :param after: The ID of the last response to return.
        :param limit: The maximum number of responses to return.
        :param model: The model to filter by.
        :param order: The order to sort the responses by.
        """
        if not order:
            order = Order.desc

        where_conditions = {}
        if model:
            where_conditions["model"] = model

        paginated_result = await self.sql_store.fetch_all(
            table="openai_responses",
            where=where_conditions if where_conditions else None,
            order_by=[("created_at", order.value)],
            cursor=("id", after) if after else None,
            limit=limit,
            policy=self.policy,
        )

        data = [OpenAIResponseObjectWithInput(**row["response_object"]) for row in paginated_result.data]
        return ListOpenAIResponseObject(
            data=data,
            has_more=paginated_result.has_more,
            first_id=data[0].id if data else "",
            last_id=data[-1].id if data else "",
        )

    async def get_response_object(self, response_id: str) -> OpenAIResponseObjectWithInput:
        """
        Get a response object with automatic access control checking.
        """
        row = await self.sql_store.fetch_one(
            "openai_responses",
            where={"id": response_id},
            policy=self.policy,
        )

        if not row:
            # SecureSqlStore will return None if record doesn't exist OR access is denied
            # This provides security by not revealing whether the record exists
            raise ValueError(f"Response with id {response_id} not found") from None

        return OpenAIResponseObjectWithInput(**row["response_object"])

    async def delete_response_object(self, response_id: str) -> OpenAIDeleteResponseObject:
        row = await self.sql_store.fetch_one("openai_responses", where={"id": response_id}, policy=self.policy)
        if not row:
            raise ValueError(f"Response with id {response_id} not found")
        await self.sql_store.delete("openai_responses", where={"id": response_id})
        return OpenAIDeleteResponseObject(id=response_id)

    async def list_response_input_items(
        self,
        response_id: str,
        after: str | None = None,
        before: str | None = None,
        include: list[str] | None = None,
        limit: int | None = 20,
        order: Order | None = Order.desc,
    ) -> ListOpenAIResponseInputItem:
        """
        List input items for a given response.

        :param response_id: The ID of the response to retrieve input items for.
        :param after: An item ID to list items after, used for pagination.
        :param before: An item ID to list items before, used for pagination.
        :param include: Additional fields to include in the response.
        :param limit: A limit on the number of objects to be returned.
        :param order: The order to return the input items in.
        """
        if include:
            raise NotImplementedError("Include is not supported yet")
        if before and after:
            raise ValueError("Cannot specify both 'before' and 'after' parameters")

        response_with_input = await self.get_response_object(response_id)
        items = response_with_input.input

        if order == Order.desc:
            items = list(reversed(items))

        start_index = 0
        end_index = len(items)

        if after or before:
            for i, item in enumerate(items):
                item_id = getattr(item, "id", None)
                if after and item_id == after:
                    start_index = i + 1
                if before and item_id == before:
                    end_index = i
                    break

            if after and start_index == 0:
                raise ValueError(f"Input item with id '{after}' not found for response '{response_id}'")
            if before and end_index == len(items):
                raise ValueError(f"Input item with id '{before}' not found for response '{response_id}'")

        items = items[start_index:end_index]

        # Apply limit
        if limit is not None:
            items = items[:limit]

        return ListOpenAIResponseInputItem(data=items)
