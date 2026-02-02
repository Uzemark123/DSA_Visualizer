# app/tests/test_phase5_part01_allocation.py

import importlib

from .fixtures_db import build_test_db


def reload_allocation(monkeypatch, db_path: str):
    monkeypatch.setenv("DB_FILE", db_path)

    import backend.app.config as config
    import backend.app.db.sqlalchemy_db as sqlalchemy_db
    import backend.app.db as db_pkg
    import backend.app.engine.PART01.roadmap_pattern_allocation as allocation

    importlib.reload(config)
    importlib.reload(sqlalchemy_db)
    importlib.reload(db_pkg)
    importlib.reload(allocation)

    return allocation, db_pkg


def test_build_pattern_quotas_exclusion_wins(monkeypatch, tmp_path):
    db_path = str(build_test_db(tmp_path))
    allocation, db_pkg = reload_allocation(monkeypatch, db_path)

    quota_rows, share = allocation.build_pattern_quotas(
        pattern_stats_table=db_pkg.pattern_stats_table,
        execute_stmt=db_pkg.execute_stmt,
        included_subpattern_ids=[10.1, 10.2],
        excluded_subpattern_ids=[10.1],
        priority_multipliers=[(10.1, 2.0)],
        requested_total=5,
    )

    assert quota_rows == [(10.2, 5)]
    assert share == {10.2: 100.0}


def test_build_pattern_quotas_renormalizes_to_100(monkeypatch, tmp_path):
    db_path = str(build_test_db(tmp_path))
    allocation, db_pkg = reload_allocation(monkeypatch, db_path)

    _quota_rows, share = allocation.build_pattern_quotas(
        pattern_stats_table=db_pkg.pattern_stats_table,
        execute_stmt=db_pkg.execute_stmt,
        included_subpattern_ids=[10.1, 10.2, 44.1],
        excluded_subpattern_ids=[],
        priority_multipliers=[],
        requested_total=10,
    )

    total = sum(share.values())
    assert 99.9 <= total <= 100.1


def test_build_pattern_quotas_priority_increases_share(monkeypatch, tmp_path):
    db_path = str(build_test_db(tmp_path))
    allocation, db_pkg = reload_allocation(monkeypatch, db_path)

    _quota_rows, base_share = allocation.build_pattern_quotas(
        pattern_stats_table=db_pkg.pattern_stats_table,
        execute_stmt=db_pkg.execute_stmt,
        included_subpattern_ids=[10.1, 10.2],
        excluded_subpattern_ids=[],
        priority_multipliers=[],
        requested_total=10,
    )

    _quota_rows, boosted_share = allocation.build_pattern_quotas(
        pattern_stats_table=db_pkg.pattern_stats_table,
        execute_stmt=db_pkg.execute_stmt,
        included_subpattern_ids=[10.1, 10.2],
        excluded_subpattern_ids=[],
        priority_multipliers=[(10.1, 2.0)],
        requested_total=10,
    )

    assert boosted_share[10.1] > base_share[10.1]
