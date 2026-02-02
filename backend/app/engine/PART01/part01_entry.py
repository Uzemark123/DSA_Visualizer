from __future__ import annotations

from collections import defaultdict
from typing import Callable, DefaultDict, Iterable, List, Sequence, Tuple

from .roadmap_pattern_allocation import build_pattern_quotas
from .assemble_chunks_and_revisons import assign_buckets_and_chunks, convert_rows_to_objects
from .database_query_builders import build_weighted_stmt
from .types import WeightedProblemRow
from ..models import Chunk, RevisionProblem


def fetch_chunks_and_revisions_lists(
    *,
    problems,
    pattern_stats_table,
    user_difficulty_selection: Sequence[tuple[str, float]],
    user_included_subpattern_ids: Sequence[float],
    user_deselected_subpattern_ids: Sequence[float],
    user_priority_subpattern_multipliers: Sequence[tuple[float, float]],
    user_requested_roadmap_length: int,
    chunk_size: int,
    structured_ratio: float,
    build_stmt: Callable[..., object] = build_weighted_stmt,
    execute_stmt: Callable[[object], Iterable],
) -> Tuple[DefaultDict[float, List[Chunk]], List[Chunk], List[RevisionProblem], dict, dict]:
    quotas, adjusted_shares = build_pattern_quotas(
        pattern_stats_table=pattern_stats_table,
        execute_stmt=execute_stmt,
        included_subpattern_ids=user_included_subpattern_ids,
        excluded_subpattern_ids=user_deselected_subpattern_ids,
        priority_multipliers=user_priority_subpattern_multipliers,
        requested_total=user_requested_roadmap_length,
    )

    if not quotas:
        return defaultdict(list), [], [], adjusted_shares, {}

    stmt = build_stmt(
        problems=problems,
        user_difficulty_selection=user_difficulty_selection,
        user_deselected_subpattern_ids=user_deselected_subpattern_ids,
        quota_rows=quotas,
    )

    raw_rows: List[WeightedProblemRow] = []
    for row in execute_stmt(stmt):
        raw_rows.append(
            WeightedProblemRow(
                subpattern_id=row.subpattern_id,
                tier=row.tier,
                final_frequency=row.final_frequency,
                chunk_number=0,
                bucket="raw",
                title=row.title,
                url=row.url,
            )
        )

    weighted_rows = assign_buckets_and_chunks(raw_rows, structured_ratio, chunk_size)
    return convert_rows_to_objects(weighted_rows, adjusted_shares)


__all__ = ["fetch_chunks_and_revisions_lists"]
