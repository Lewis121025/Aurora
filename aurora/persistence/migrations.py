from __future__ import annotations

import sqlite3


def apply_migrations(connection: sqlite3.Connection) -> None:
    with connection:
        connection.execute(
            "CREATE TABLE IF NOT EXISTS runtime_snapshots("
            "snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT, payload TEXT NOT NULL)"
        )
