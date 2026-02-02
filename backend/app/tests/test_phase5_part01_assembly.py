# app/tests/test_phase5_part01_assembly.py

from backend.app.engine.PART01.assemble_chunks_and_revisons import (
    assign_buckets_and_chunks,
    convert_rows_to_objects,
)
from backend.app.engine.PART01.types import WeightedProblemRow


def _row(
    *,
    subpattern_id: float,
    tier: float,
    final_frequency: float,
    title: str,
    url: str,
    chunk_number: int | None = None,
    bucket: str = "structured",
):
    return WeightedProblemRow(
        subpattern_id=subpattern_id,
        tier=tier,
        final_frequency=final_frequency,
        chunk_number=chunk_number,
        bucket=bucket,
        title=title,
        url=url,
    )


def test_assign_buckets_and_chunks_respects_structured_ratio():
    rows = [
        _row(subpattern_id=10.1, tier=1.0, final_frequency=9.0, title="A", url="http://a"),
        _row(subpattern_id=10.1, tier=1.0, final_frequency=8.0, title="B", url="http://b"),
        _row(subpattern_id=10.1, tier=1.0, final_frequency=7.0, title="C", url="http://c"),
        _row(subpattern_id=10.1, tier=1.0, final_frequency=6.0, title="D", url="http://d"),
    ]

    enriched = assign_buckets_and_chunks(rows, structured_ratio=0.5, chunk_size=2)

    structured = [row for row in enriched if row.bucket == "structured"]
    recap = [row for row in enriched if row.bucket == "recap"]

    assert len(structured) == 2
    assert len(recap) == 2


def test_assign_buckets_and_chunks_chunk_numbers_increment():
    rows = [
        _row(subpattern_id=10.1, tier=1.0, final_frequency=9.0, title="A", url="http://a"),
        _row(subpattern_id=10.1, tier=1.0, final_frequency=8.0, title="B", url="http://b"),
        _row(subpattern_id=10.1, tier=1.0, final_frequency=7.0, title="C", url="http://c"),
    ]

    enriched = assign_buckets_and_chunks(rows, structured_ratio=1.0, chunk_size=2)
    chunk_numbers = [row.chunk_number for row in enriched]

    assert chunk_numbers == [0, 0, 1]


def test_convert_rows_to_objects_groups_chunks_and_revisions():
    rows = [
        _row(
            subpattern_id=10.1,
            tier=1.0,
            final_frequency=9.0,
            title="A",
            url="http://a",
            chunk_number=0,
            bucket="structured",
        ),
        _row(
            subpattern_id=10.1,
            tier=1.0,
            final_frequency=5.0,
            title="B",
            url="http://b",
            chunk_number=0,
            bucket="structured",
        ),
        _row(
            subpattern_id=44.1,
            tier=3.0,
            final_frequency=8.0,
            title="C",
            url="http://c",
            chunk_number=0,
            bucket="recap",
        ),
    ]

    chunks_by_tier, _sorted, revisions, _pattern_stats, tier_stats = convert_rows_to_objects(
        rows,
        pattern_stats={10.1: 50.0, 44.1: 50.0},
    )

    assert 1.0 in chunks_by_tier
    assert len(chunks_by_tier[1.0]) == 1

    chunk = chunks_by_tier[1.0][0]
    assert chunk.final_frequency_sum == 14.0
    assert chunk.chunk_number == 0

    assert len(revisions) == 1
    assert revisions[0].subpattern_id == 44.1
    assert tier_stats[1.0] == 1
