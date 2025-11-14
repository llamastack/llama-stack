# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Type-safe data interchange for Python data classes.

:see: https://github.com/hunyadi/strong_typing
"""

from collections.abc import Callable, Iterable
from typing import TypeVar

from .inspection import TypeCollector

T = TypeVar("T")


def topological_sort(graph: dict[T, set[T]]) -> list[T]:
    """
    Performs a topological sort of a graph.

    Nodes with no outgoing edges are first. Nodes with no incoming edges are last.
    The topological ordering is not unique.

    :param graph: A dictionary of mappings from nodes to adjacent nodes. Keys and set members must be hashable.
    :returns: The list of nodes in topological order.
    """

    # empty list that will contain the sorted nodes (in reverse order)
    ordered: list[T] = []

    seen: dict[T, bool] = {}

    def _visit(n: T) -> None:
        status = seen.get(n)
        if status is not None:
            if status:  # node has a permanent mark
                return
            else:  # node has a temporary mark
                raise RuntimeError(f"cycle detected in graph for node {n}")

        seen[n] = False  # apply temporary mark
        for m in graph[n]:  # visit all adjacent nodes
            if m != n:  # ignore self-referencing nodes
                _visit(m)

        seen[n] = True  # apply permanent mark
        ordered.append(n)

    for n in graph.keys():
        _visit(n)

    return ordered


def type_topological_sort(
    types: Iterable[type],
    dependency_fn: Callable[[type], Iterable[type]] | None = None,
) -> list[type]:
    """
    Performs a topological sort of a list of types.

    Types that don't depend on other types (i.e. fundamental types) are first. Types on which no other types depend
    are last. The topological ordering is not unique.

    :param types: A list of types (simple or composite).
    :param dependency_fn: Returns a list of additional dependencies for a class (e.g. classes referenced by a foreign key).
    :returns: The list of types in topological order.
    """

    if not all(isinstance(typ, type) for typ in types):
        raise TypeError("expected a list of types")

    collector = TypeCollector()
    collector.traverse_all(types)
    graph = collector.graph

    if dependency_fn:
        new_types: set[type] = set()
        for source_type, references in graph.items():
            dependent_types = dependency_fn(source_type)
            references.update(dependent_types)
            new_types.update(dependent_types)
        for new_type in new_types:
            graph[new_type] = set()

    return topological_sort(graph)
