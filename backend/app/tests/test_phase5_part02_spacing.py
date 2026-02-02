# app/tests/test_phase5_part02_spacing.py

from backend.app.engine.PART02.desired_spacing import (
    get_desired_chunk_positions,
    get_patterns_with_spacing,
)


def test_get_patterns_with_spacing():
    stats = get_patterns_with_spacing(10, {10: 2, 44: 1})
    stats_map = {pattern: spacing for pattern, _count, spacing in stats}

    assert stats_map[10] == 5
    assert stats_map[44] == 10


def test_get_desired_chunk_positions_deterministic():
    positions = get_desired_chunk_positions(6, {10: 2, 44: 1})

    assert positions == [(10, 2), (44, 3), (10, 4)]
