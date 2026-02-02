# app/tests/test_phase5_part01_query_builders.py

import importlib
import sqlite3

from .fixtures_db import build_test_db


def reload_query_builder(monkeypatch, db_path: str):
    monkeypatch.setenv("DB_FILE", db_path)

    import backend.app.config as config
    import backend.app.db.sqlalchemy_db as sqlalchemy_db
    import backend.app.db as db_pkg
    import backend.app.engine.PART01.database_query_builders as query_builders

    importlib.reload(config)
    importlib.reload(sqlalchemy_db)
    importlib.reload(db_pkg)
    importlib.reload(query_builders)

    return query_builders, db_pkg


def test_weighted_stmt_filters_deselected(monkeypatch, tmp_path):
    db_path = str(build_test_db(tmp_path))
    query_builders, db_pkg = reload_query_builder(monkeypatch, db_path)

    stmt = query_builders.build_weighted_stmt(
        problems=db_pkg.problems_table,
        user_difficulty_selection=[("easy", 1.0), ("medium", 1.0), ("hard", 0.0)],
        user_deselected_subpattern_ids=[10.1],
        quota_rows=[(10.1, 2), (10.2, 2)],
    )

    rows = db_pkg.execute_stmt(stmt)
    subpattern_ids = {float(row.subpattern_id) for row in rows}

    assert 10.1 not in subpattern_ids
    assert 10.2 in subpattern_ids


def test_weighted_stmt_respects_top_k_per_subpattern(monkeypatch, tmp_path):
    db_path = str(build_test_db(tmp_path))

    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            INSERT INTO problems (
                Difficulty, Title, combindedFrequency, Category, Pattern, pattern_id, subpattern,
                subpattern_id, url, curatedListFreq, Notes, Frequency, tier
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "easy",
                "TP Array A2",
                5.0,
                "arrays",
                "two pointers",
                10,
                "converging",
                10.1,
                "http://a2",
                1,
                "",
                1,
                1.0,
            ),
        )
        conn.commit()
    finally:
        conn.close()

    query_builders, db_pkg = reload_query_builder(monkeypatch, db_path)

    stmt = query_builders.build_weighted_stmt(
        problems=db_pkg.problems_table,
        user_difficulty_selection=[("easy", 1.0), ("medium", 1.0), ("hard", 0.0)],
        user_deselected_subpattern_ids=[],
        quota_rows=[(10.1, 1)],
    )

    rows = db_pkg.execute_stmt(stmt)

    assert len(rows) == 1
    assert float(rows[0].subpattern_id) == 10.1
    assert rows[0].title == "TP Array A"
