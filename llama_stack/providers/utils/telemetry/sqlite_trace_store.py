# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
from datetime import UTC, datetime
from typing import Protocol

import aiosqlite

from llama_stack.apis.telemetry import (
    MetricDataPoint,
    MetricLabel,
    MetricLabelMatcher,
    MetricQueryType,
    MetricSeries,
    QueryCondition,
    QueryMetricsResponse,
    Span,
    SpanWithStatus,
    Trace,
)


class TraceStore(Protocol):
    async def query_traces(
        self,
        attribute_filters: list[QueryCondition] | None = None,
        limit: int | None = 100,
        offset: int | None = 0,
        order_by: list[str] | None = None,
    ) -> list[Trace]: ...

    async def get_span_tree(
        self,
        span_id: str,
        attributes_to_return: list[str] | None = None,
        max_depth: int | None = None,
    ) -> dict[str, SpanWithStatus]: ...

    async def query_metrics(
        self,
        metric_name: str,
        start_time: datetime,
        end_time: datetime | None = None,
        granularity: str | None = "1d",
        query_type: MetricQueryType = MetricQueryType.RANGE,
        label_matchers: list[MetricLabelMatcher] | None = None,
    ) -> QueryMetricsResponse: ...


class SQLiteTraceStore(TraceStore):
    def __init__(self, conn_string: str):
        self.conn_string = conn_string

    async def query_metrics(
        self,
        metric_name: str,
        start_time: datetime,
        end_time: datetime | None = None,
        granularity: str | None = None,
        query_type: MetricQueryType = MetricQueryType.RANGE,
        label_matchers: list[MetricLabelMatcher] | None = None,
    ) -> QueryMetricsResponse:
        if end_time is None:
            end_time = datetime.now(UTC)

        # Build base query
        if query_type == MetricQueryType.INSTANT:
            query = """
                SELECT
                    se.name,
                    SUM(CAST(json_extract(se.attributes, '$.value') AS REAL)) as value,
                    json_extract(se.attributes, '$.unit') as unit,
                    se.attributes
                FROM span_events se
                WHERE se.name = ?
                AND se.timestamp BETWEEN ? AND ?
            """
        else:
            if granularity:
                time_format = self._get_time_format_for_granularity(granularity)
                query = f"""
                    SELECT
                        se.name,
                        SUM(CAST(json_extract(se.attributes, '$.value') AS REAL)) as value,
                        json_extract(se.attributes, '$.unit') as unit,
                        se.attributes,
                        strftime('{time_format}', se.timestamp) as bucket_start
                    FROM span_events se
                    WHERE se.name = ?
                    AND se.timestamp BETWEEN ? AND ?
                """
            else:
                query = """
                    SELECT
                        se.name,
                        json_extract(se.attributes, '$.value') as value,
                        json_extract(se.attributes, '$.unit') as unit,
                        se.attributes,
                        se.timestamp
                    FROM span_events se
                    WHERE se.name = ?
                    AND se.timestamp BETWEEN ? AND ?
                """

        params = [f"metric.{metric_name}", start_time.isoformat(), end_time.isoformat()]

        # Labels that will be attached to the MetricSeries (preserve matcher labels)
        all_labels: list[MetricLabel] = []
        matcher_label_names = set()
        if label_matchers:
            for matcher in label_matchers:
                json_path = f"$.{matcher.name}"
                if matcher.operator == "=":
                    query += f" AND json_extract(se.attributes, '{json_path}') = ?"
                    params.append(matcher.value)
                elif matcher.operator == "!=":
                    query += f" AND json_extract(se.attributes, '{json_path}') != ?"
                    params.append(matcher.value)
                elif matcher.operator == "=~":
                    query += f" AND json_extract(se.attributes, '{json_path}') LIKE ?"
                    params.append(f"%{matcher.value}%")
                elif matcher.operator == "!~":
                    query += f" AND json_extract(se.attributes, '{json_path}') NOT LIKE ?"
                    params.append(f"%{matcher.value}%")
                # Preserve filter context in output
                all_labels.append(MetricLabel(name=matcher.name, value=str(matcher.value)))
                matcher_label_names.add(matcher.name)

        # GROUP BY / ORDER BY logic
        if query_type == MetricQueryType.RANGE and granularity:
            group_time_format = self._get_time_format_for_granularity(granularity)
            query += f" GROUP BY strftime('{group_time_format}', se.timestamp), json_extract(se.attributes, '$.unit')"
            query += " ORDER BY bucket_start"
        elif query_type == MetricQueryType.INSTANT:
            query += " GROUP BY json_extract(se.attributes, '$.unit')"
        else:
            query += " ORDER BY se.timestamp"

        # Execute query
        async with aiosqlite.connect(self.conn_string) as conn:
            conn.row_factory = aiosqlite.Row
            async with conn.execute(query, params) as cursor:
                rows = await cursor.fetchall()

                if not rows:
                    return QueryMetricsResponse(data=[])

                data_points = []
                # We want to add attribute labels, but only those not already present as matcher labels.
                attr_label_names = set()
                for row in rows:
                    # Parse JSON attributes safely, if there are no attributes (weird), just don't add the labels to the result.
                    try:
                        attributes = json.loads(row["attributes"] or "{}")
                    except (TypeError, json.JSONDecodeError):
                        attributes = {}

                    value = row["value"]
                    unit = row["unit"] or ""

                    # Add labels from attributes without duplicating matcher labels, if we don't do this, there will be a lot of duplicate label in the result.
                    for k, v in attributes.items():
                        if k not in ["value", "unit"] and k not in matcher_label_names and k not in attr_label_names:
                            all_labels.append(MetricLabel(name=k, value=str(v)))
                            attr_label_names.add(k)

                    # Determine timestamp
                    if query_type == MetricQueryType.RANGE and granularity:
                        try:
                            bucket_start_raw = row["bucket_start"]
                        except KeyError as e:
                            raise ValueError(
                                "DB did not have a bucket_start time in row when using granularity, this indicates improper formatting"
                            ) from e
                        # this value could also be there, but be NULL, I think.
                        if bucket_start_raw is None:
                            raise ValueError("bucket_start is None check time format and data")
                        bucket_start = datetime.fromisoformat(bucket_start_raw)
                        timestamp = int(bucket_start.timestamp())
                    elif query_type == MetricQueryType.INSTANT:
                        timestamp = int(datetime.now(UTC).timestamp())
                    else:
                        try:
                            timestamp_raw = row["timestamp"]
                        except KeyError as e:
                            raise ValueError(
                                "DB did not have a timestamp in row, this indicates improper formatting"
                            ) from e
                        # this value could also be there, but be NULL, I think.
                        if timestamp_raw is None:
                            raise ValueError("timestamp is None check time format and data")
                        timestamp_iso = datetime.fromisoformat(timestamp_raw)
                        timestamp = int(timestamp_iso.timestamp())

                    data_points.append(
                        MetricDataPoint(
                            timestamp=timestamp,
                            value=value,
                            unit=unit,
                        )
                    )

                metric_series = [MetricSeries(metric=metric_name, labels=all_labels, values=data_points)]
                return QueryMetricsResponse(data=metric_series)

    def _get_time_format_for_granularity(self, granularity: str | None) -> str:
        """Get the SQLite strftime format string for a given granularity.
        Args:
            granularity: Granularity string (e.g., "1m", "5m", "1h", "1d")
        Returns:
            SQLite strftime format string for the granularity
        """
        if granularity is None:
            raise ValueError("granularity cannot be None for this method - use separate logic for no aggregation")

        if granularity.endswith("d"):
            return "%Y-%m-%d 00:00:00"
        elif granularity.endswith("h"):
            return "%Y-%m-%d %H:00:00"
        elif granularity.endswith("m"):
            return "%Y-%m-%d %H:%M:00"
        else:
            return "%Y-%m-%d %H:%M:00"  # Default to most granular which will give us the most timestamps.

    async def query_traces(
        self,
        attribute_filters: list[QueryCondition] | None = None,
        limit: int | None = 100,
        offset: int | None = 0,
        order_by: list[str] | None = None,
    ) -> list[Trace]:
        def build_where_clause() -> tuple[str, list]:
            if not attribute_filters:
                return "", []

            ops_map = {"eq": "=", "ne": "!=", "gt": ">", "lt": "<"}

            conditions = [
                f"json_extract(s.attributes, '$.{condition.key}') {ops_map[condition.op.value]} ?"
                for condition in attribute_filters
            ]
            params = [condition.value for condition in attribute_filters]
            where_clause = " WHERE " + " AND ".join(conditions)
            return where_clause, params

        def build_order_clause() -> str:
            if not order_by:
                return ""

            order_clauses = []
            for field in order_by:
                desc = field.startswith("-")
                clean_field = field[1:] if desc else field
                order_clauses.append(f"t.{clean_field} {'DESC' if desc else 'ASC'}")
            return " ORDER BY " + ", ".join(order_clauses)

        # Build the main query
        base_query = """
            WITH matching_traces AS (
                SELECT DISTINCT t.trace_id
                FROM traces t
                JOIN spans s ON t.trace_id = s.trace_id
                {where_clause}
            ),
            filtered_traces AS (
                SELECT t.trace_id, t.root_span_id, t.start_time, t.end_time
                FROM matching_traces mt
                JOIN traces t ON mt.trace_id = t.trace_id
                LEFT JOIN spans s ON t.trace_id = s.trace_id
                {order_clause}
            )
            SELECT DISTINCT trace_id, root_span_id, start_time, end_time
            FROM filtered_traces
            WHERE root_span_id IS NOT NULL
            LIMIT {limit} OFFSET {offset}
        """

        where_clause, params = build_where_clause()
        query = base_query.format(
            where_clause=where_clause,
            order_clause=build_order_clause(),
            limit=limit,
            offset=offset,
        )

        # Execute query and return results
        async with aiosqlite.connect(self.conn_string) as conn:
            conn.row_factory = aiosqlite.Row
            async with conn.execute(query, params) as cursor:
                rows = await cursor.fetchall()
                return [
                    Trace(
                        trace_id=row["trace_id"],
                        root_span_id=row["root_span_id"],
                        start_time=datetime.fromisoformat(row["start_time"]),
                        end_time=datetime.fromisoformat(row["end_time"]),
                    )
                    for row in rows
                ]

    async def get_span_tree(
        self,
        span_id: str,
        attributes_to_return: list[str] | None = None,
        max_depth: int | None = None,
    ) -> dict[str, SpanWithStatus]:
        # Build the attributes selection
        attributes_select = "s.attributes"
        if attributes_to_return:
            json_object = ", ".join(f"'{key}', json_extract(s.attributes, '$.{key}')" for key in attributes_to_return)
            attributes_select = f"json_object({json_object})"

        # SQLite CTE query with filtered attributes
        query = f"""
        WITH RECURSIVE span_tree AS (
            SELECT s.*, 1 as depth, {attributes_select} as filtered_attributes
            FROM spans s
            WHERE s.span_id = ?

            UNION ALL

            SELECT s.*, st.depth + 1, {attributes_select} as filtered_attributes
            FROM spans s
            JOIN span_tree st ON s.parent_span_id = st.span_id
            WHERE (? IS NULL OR st.depth < ?)
        )
        SELECT *
        FROM span_tree
        ORDER BY depth, start_time
        """

        spans_by_id = {}
        async with aiosqlite.connect(self.conn_string) as conn:
            conn.row_factory = aiosqlite.Row
            async with conn.execute(query, (span_id, max_depth, max_depth)) as cursor:
                rows = await cursor.fetchall()

                if not rows:
                    raise ValueError(f"Span {span_id} not found")

                for row in rows:
                    span = SpanWithStatus(
                        span_id=row["span_id"],
                        trace_id=row["trace_id"],
                        parent_span_id=row["parent_span_id"],
                        name=row["name"],
                        start_time=datetime.fromisoformat(row["start_time"]),
                        end_time=datetime.fromisoformat(row["end_time"]),
                        attributes=json.loads(row["filtered_attributes"]),
                        status=row["status"].lower(),
                    )

                    spans_by_id[span.span_id] = span

                return spans_by_id

    async def get_trace(self, trace_id: str) -> Trace:
        query = """
            SELECT *
            FROM traces t
            WHERE t.trace_id = ?
        """
        async with aiosqlite.connect(self.conn_string) as conn:
            conn.row_factory = aiosqlite.Row
            async with conn.execute(query, (trace_id,)) as cursor:
                row = await cursor.fetchone()
                if row is None:
                    raise ValueError(f"Trace {trace_id} not found")
                return Trace(**row)

    async def get_span(self, trace_id: str, span_id: str) -> Span:
        query = "SELECT * FROM spans WHERE trace_id = ? AND span_id = ?"
        async with aiosqlite.connect(self.conn_string) as conn:
            conn.row_factory = aiosqlite.Row
            async with conn.execute(query, (trace_id, span_id)) as cursor:
                row = await cursor.fetchone()
                if row is None:
                    raise ValueError(f"Span {span_id} not found")
                return Span(**row)
