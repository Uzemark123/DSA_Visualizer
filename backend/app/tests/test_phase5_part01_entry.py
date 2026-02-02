# app/tests/test_phase5_part01_entry.py

import importlib

from .fixtures_db import build_test_db


def reload_part01(monkeypatch, db_path: str):
    monkeypatch.setenv("DB_FILE", db_path)

    import backend.app.config as config
    import backend.app.db.sqlalchemy_db as sqlalchemy_db
    import backend.app.db as db_pkg
    import backend.app.engine.PART01.part01_entry as part01_entry

    importlib.reload(config)
    importlib.reload(sqlalchemy_db)
    importlib.reload(db_pkg)
    importlib.reload(part01_entry)

    return part01_entry, db_pkg


def test_fetch_chunks_and_revisions_lists_returns_chunks(monkeypatch, tmp_path):
    db_path = str(build_test_db(tmp_path))
    part01_entry, db_pkg = reload_part01(monkeypatch, db_path)

    chunks_by_tier, _sorted, revisions, pattern_stats, tier_stats = part01_entry.fetch_chunks_and_revisions_lists(
        problems=db_pkg.problems_table,
        pattern_stats_table=db_pkg.pattern_stats_table,
        user_difficulty_selection=[("easy", 1.0), ("medium", 1.0), ("hard", 0.0)],
        user_included_subpattern_ids=[10.1, 10.2],
        user_deselected_subpattern_ids=[],
        user_priority_subpattern_multipliers=[],
        user_requested_roadmap_length=2,
        chunk_size=2,
        structured_ratio=1.0,
        execute_stmt=db_pkg.execute_stmt,
    )

    assert 1.0 in chunks_by_tier
    assert len(chunks_by_tier[1.0]) == 2
    assert revisions == []
    assert pattern_stats
    assert tier_stats[1.0] == 2


def test_fetch_chunks_and_revisions_lists_empty_when_no_quotas(monkeypatch, tmp_path):
    db_path = str(build_test_db(tmp_path))
    part01_entry, db_pkg = reload_part01(monkeypatch, db_path)

    chunks_by_tier, sorted_chunks, revisions, pattern_stats, tier_stats = part01_entry.fetch_chunks_and_revisions_lists(
        problems=db_pkg.problems_table,
        pattern_stats_table=db_pkg.pattern_stats_table,
        user_difficulty_selection=[("easy", 1.0), ("medium", 1.0), ("hard", 0.0)],
        user_included_subpattern_ids=[999.1],
        user_deselected_subpattern_ids=[],
        user_priority_subpattern_multipliers=[],
        user_requested_roadmap_length=2,
        chunk_size=2,
        structured_ratio=1.0,
        execute_stmt=db_pkg.execute_stmt,
    )

    assert dict(chunks_by_tier) == {}
    assert sorted_chunks == []
    assert revisions == []
    assert pattern_stats == {}
    assert tier_stats == {}
