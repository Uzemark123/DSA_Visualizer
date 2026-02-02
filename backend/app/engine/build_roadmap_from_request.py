"""
Orchestration boundary: build a roadmap from request inputs.
Ties together Part01 (DB → chunks/revisions), Part02 (spacing/placement), and Part03 (revision placement).
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

from .PART01.part01_entry import fetch_chunks_and_revisions_lists
from .PART02.entry import run_spacing_and_allocation
from .models import RoadMap
from .PART03.entry import assign_revision_problems_random
from ..config import DEFAULT_PRIORITY_MULTIPLIER, PATTERN_ID_TO_SUBPATTERN_ID, STRUCTURED_RATIO


def build_roadmap_from_request(
    *,
    problems_table,
    difficulty_selection,
    user_included_patterns,
    user_deselected_subpattern_ids,
    user_priority_subpattern_multipliers,
    chunk_size,
    structured_ratio=STRUCTURED_RATIO,
    execute_stmt,
    pattern_stats_table,
    user_requested_roadmap_length,
    user_priority_patterns,
    user_excluded_patterns,
) -> RoadMap:
    """
    High-level entrypoint to construct a roadmap.
    - pulls weighted problems from DB, splits into chunks + revisions
    - spaces/places chunks into roadmap nodes (tier by tier)
    - assigns revision problems into the remaining eligible nodes
    """
    if user_requested_roadmap_length <= 50:
        chunk_size = 2
    else:
        chunk_size = 3

    roadmap = RoadMap(
        user_requested_roadmap_length,
        user_included_patterns,
        user_priority_patterns,
        user_excluded_patterns,
    )

    def expand_pattern_ids(pattern_ids: Sequence[int]) -> List[float]:
        expanded: List[float] = []
        for pattern_id in pattern_ids:
            expanded.extend(PATTERN_ID_TO_SUBPATTERN_ID.get(pattern_id, []))
        return expanded

    expanded_included = expand_pattern_ids(user_included_patterns)
    expanded_excluded = set(expand_pattern_ids(user_excluded_patterns))
    expanded_excluded.update(user_deselected_subpattern_ids)

    expanded_priority = expand_pattern_ids(user_priority_patterns)
    priority_by_id = {sub_id: mult for sub_id, mult in user_priority_subpattern_multipliers}
    for sub_id in expanded_priority:
        priority_by_id.setdefault(sub_id, DEFAULT_PRIORITY_MULTIPLIER)
    priority_multipliers: List[Tuple[float, float]] = list(priority_by_id.items())

    # Step 03–05: fetch weighted rows and split into objects/stat buckets.
    chunks_by_tier, _legacy_sorted_chunks, revision_problems, pattern_stats, tier_stats = fetch_chunks_and_revisions_lists(
        problems=problems_table,
        pattern_stats_table=pattern_stats_table,
        user_difficulty_selection=difficulty_selection,
        user_included_subpattern_ids=expanded_included,
        user_deselected_subpattern_ids=list(expanded_excluded),
        user_priority_subpattern_multipliers=priority_multipliers,
        user_requested_roadmap_length=user_requested_roadmap_length,
        chunk_size=chunk_size,
        structured_ratio=structured_ratio,
        execute_stmt=execute_stmt,
    )

    # Persist stats on the roadmap (set-once guards in the model).
    roadmap.pattern_stats = pattern_stats
    roadmap.tier_stats = tier_stats

    # Step 08-ish: spacing + union-find allocation, tier by tier.
    nodes_with_chunks = run_spacing_and_allocation(chunks_by_tier, start_offset=0)
    roadmap.roadmap_array = nodes_with_chunks
    roadmap.roadmap_array_length = len(nodes_with_chunks)

    # Step 09: place revision problems onto roadmap nodes (tier-aware).
    assign_revision_problems_random(
        roadmap_nodes=roadmap.roadmap_array,
        revision_problems=revision_problems,
    )

    return roadmap


__all__ = ["build_roadmap_from_request"]
