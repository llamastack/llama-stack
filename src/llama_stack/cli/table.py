# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from collections.abc import Iterable

from rich.console import Console
from rich.table import Table


def print_table(rows, headers=None, separate_rows: bool = False, sort_by: Iterable[int] = tuple()):
    # Convert rows and handle None values
    rows = [[x or "" for x in row] for row in rows]

    # Sort rows if sort_by is specified
    if sort_by:
        rows.sort(key=lambda x: tuple(x[i] for i in sort_by))

    # Create Rich table
    table = Table(show_lines=separate_rows)

    # Add headers if provided
    if headers:
        for header in headers:
            table.add_column(header, style="bold white")
    else:
        # Add unnamed columns based on first row
        for _ in range(len(rows[0]) if rows else 0):
            table.add_column()

    # Add rows
    for row in rows:
        table.add_row(*row)

    # Print table
    console = Console()
    console.print(table)
