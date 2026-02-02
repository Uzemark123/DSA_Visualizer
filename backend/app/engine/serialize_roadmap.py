from __future__ import annotations

from .build_roadmap_from_request import build_roadmap_from_request


def _serialize_chunk(chunk):
    return {
        "subpatternId": chunk.subpattern_id,
        "tier": chunk.tier,
        "finalFrequencySum": chunk.final_frequency_sum,
        "chunkNumber": chunk.chunk_number,
        "roadmapIndexAllocation": chunk.roadmap_index_allocation,
        "problems": [
            {"finalFrequency": freq, "title": title, "url": url}
            for (freq, title, url) in chunk.problems
        ],
    }


def _serialize_revision_problem(rp):
    title, url = rp.problem
    return {
        "subpatternId": rp.subpattern_id,
        "tier": rp.tier,
        "finalFrequencySum": rp.final_frequency_sum,
        "problem": {"title": title, "url": url},
        "lastPosition": rp.last_position,
    }


def _serialize_node(node):
    return {
        "roadmapIndex": node.roadmap_index,
        "chunk": _serialize_chunk(node.chunk_pointer) if node.chunk_pointer else None,
        "revisionProblem": _serialize_revision_problem(node.revision_pointer)
        if node.revision_pointer
        else None,
    }


def build_and_serialize_roadmap(
    *,
    problems_table,
    pattern_stats_table,
    execute_stmt,
    user_requested_roadmap_length,
    difficulty_selection,
    user_included_patterns,
    user_priority_patterns,
    user_excluded_patterns,
    chunk_size,
    user_deselected_subpattern_ids=None,
    user_priority_subpattern_multipliers=None,
):
    if user_deselected_subpattern_ids is None:
        user_deselected_subpattern_ids = []
    if user_priority_subpattern_multipliers is None:
        user_priority_subpattern_multipliers = []

    roadmap = build_roadmap_from_request(
        problems_table=problems_table,
        pattern_stats_table=pattern_stats_table,
        difficulty_selection=difficulty_selection,
        user_included_patterns=user_included_patterns,
        user_deselected_subpattern_ids=user_deselected_subpattern_ids,
        user_priority_subpattern_multipliers=user_priority_subpattern_multipliers,
        chunk_size=chunk_size,
        execute_stmt=execute_stmt,
        user_requested_roadmap_length=user_requested_roadmap_length,
        user_priority_patterns=user_priority_patterns,
        user_excluded_patterns=user_excluded_patterns,
    )

    return {"roadmap": [_serialize_node(node) for node in roadmap.roadmap_array]}


__all__ = ["build_and_serialize_roadmap"]
