# app/tests/test_phase6_regressions.py

import importlib

from .fixtures_db import build_test_db


def reload_engine(monkeypatch, db_path: str):
    monkeypatch.setenv("DB_FILE", db_path)

    import backend.app.config as config
    import backend.app.db.sqlalchemy_db as sqlalchemy_db
    import backend.app.db as db_pkg
    import backend.app.engine.serialize_roadmap as serialize_roadmap

    importlib.reload(config)
    importlib.reload(sqlalchemy_db)
    importlib.reload(db_pkg)
    importlib.reload(serialize_roadmap)

    return serialize_roadmap, db_pkg


def _difficulty_selection():
    return [("easy", 1.0), ("medium", 1.0), ("hard", 0.0)]


def test_empty_include_returns_empty_roadmap(monkeypatch, tmp_path):
    db_path = str(build_test_db(tmp_path))
    serialize_roadmap, db_pkg = reload_engine(monkeypatch, db_path)

    result = serialize_roadmap.build_and_serialize_roadmap(
        problems_table=db_pkg.problems_table,
        pattern_stats_table=db_pkg.pattern_stats_table,
        execute_stmt=db_pkg.execute_stmt,
        user_requested_roadmap_length=6,
        difficulty_selection=_difficulty_selection(),
        user_included_patterns=[],
        user_priority_patterns=[],
        user_excluded_patterns=[],
        chunk_size=None,
        user_deselected_subpattern_ids=[],
        user_priority_subpattern_multipliers=[],
    )

    assert result["roadmap"] == []


def test_unknown_patterns_return_empty(monkeypatch, tmp_path):
    db_path = str(build_test_db(tmp_path))
    serialize_roadmap, db_pkg = reload_engine(monkeypatch, db_path)

    result = serialize_roadmap.build_and_serialize_roadmap(
        problems_table=db_pkg.problems_table,
        pattern_stats_table=db_pkg.pattern_stats_table,
        execute_stmt=db_pkg.execute_stmt,
        user_requested_roadmap_length=6,
        difficulty_selection=_difficulty_selection(),
        user_included_patterns=[999],
        user_priority_patterns=[],
        user_excluded_patterns=[],
        chunk_size=None,
        user_deselected_subpattern_ids=[],
        user_priority_subpattern_multipliers=[],
    )

    assert result["roadmap"] == []


def test_priority_and_exclude_overlap_exclude_wins(monkeypatch, tmp_path):
    db_path = str(build_test_db(tmp_path))
    serialize_roadmap, db_pkg = reload_engine(monkeypatch, db_path)

    result = serialize_roadmap.build_and_serialize_roadmap(
        problems_table=db_pkg.problems_table,
        pattern_stats_table=db_pkg.pattern_stats_table,
        execute_stmt=db_pkg.execute_stmt,
        user_requested_roadmap_length=6,
        difficulty_selection=_difficulty_selection(),
        user_included_patterns=[10],
        user_priority_patterns=[10],
        user_excluded_patterns=[10],
        chunk_size=None,
        user_deselected_subpattern_ids=[],
        user_priority_subpattern_multipliers=[],
    )

    assert result["roadmap"] == []


def test_requested_length_caps_to_available(monkeypatch, tmp_path):
    db_path = str(build_test_db(tmp_path))
    serialize_roadmap, db_pkg = reload_engine(monkeypatch, db_path)

    result = serialize_roadmap.build_and_serialize_roadmap(
        problems_table=db_pkg.problems_table,
        pattern_stats_table=db_pkg.pattern_stats_table,
        execute_stmt=db_pkg.execute_stmt,
        user_requested_roadmap_length=50,
        difficulty_selection=_difficulty_selection(),
        user_included_patterns=[10, 44],
        user_priority_patterns=[],
        user_excluded_patterns=[],
        chunk_size=None,
        user_deselected_subpattern_ids=[],
        user_priority_subpattern_multipliers=[],
    )

    assert len(result["roadmap"]) <= 3
