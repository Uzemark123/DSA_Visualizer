# app/tests/test_phase4_engine.py

import importlib

from backend.app.models.models import generateRoadMapRequest

from .fixtures_db import build_test_db
from .fixtures_payloads import VALID_PAYLOAD


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


def test_build_and_serialize_roadmap_smoke(monkeypatch, tmp_path):
    db_path = str(build_test_db(tmp_path))
    serialize_roadmap, db_pkg = reload_engine(monkeypatch, db_path)

    payload = generateRoadMapRequest(**VALID_PAYLOAD)
    difficulty_selection = [
        ("easy", 1.0 if payload.difficulty.easy else 0.0),
        ("medium", 1.0 if payload.difficulty.medium else 0.0),
        ("hard", 1.0 if payload.difficulty.hard else 0.0),
    ]

    result = serialize_roadmap.build_and_serialize_roadmap(
        problems_table=db_pkg.problems_table,
        pattern_stats_table=db_pkg.pattern_stats_table,
        execute_stmt=db_pkg.execute_stmt,
        user_requested_roadmap_length=payload.user_requested_roadmap_length,
        difficulty_selection=difficulty_selection,
        user_included_patterns=payload.user_difficulty_selection,
        user_priority_patterns=payload.user_priority_patterns,
        user_excluded_patterns=payload.user_excluded_patterns,
        chunk_size=payload.chunk_size,
        user_deselected_subpattern_ids=[],
        user_priority_subpattern_multipliers=[],
    )

    assert "roadmap" in result
    assert isinstance(result["roadmap"], list)
    assert len(result["roadmap"]) <= payload.user_requested_roadmap_length

    if result["roadmap"]:
        first = result["roadmap"][0]
        assert "roadmapIndex" in first
        assert "chunk" in first
        assert "revisionProblem" in first
