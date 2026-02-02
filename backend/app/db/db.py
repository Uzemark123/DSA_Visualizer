import sqlite3
from typing import Any, Iterable
from ..config import settings


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(settings.db_file, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def run(sql: str, params: Iterable[Any] = ()):
    with get_connection() as conn:
        cur = conn.execute(sql, tuple(params))
        conn.commit()
        return {"id": cur.lastrowid, "changes": cur.rowcount}


def get(sql: str, params: Iterable[Any] = ()):
    with get_connection() as conn:
        cur = conn.execute(sql, tuple(params))
        return cur.fetchone()


def all(sql: str, params: Iterable[Any] = ()):
    with get_connection() as conn:
        cur = conn.execute(sql, tuple(params))
        return cur.fetchall()
