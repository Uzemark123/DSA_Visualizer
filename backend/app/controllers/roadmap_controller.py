from ..db import execute_stmt, problems_table, pattern_stats_table
from ..engine.serialize_roadmap import build_and_serialize_roadmap
from ..models.models import generateRoadMapRequest


def generate_roadmap(payload: generateRoadMapRequest):
    difficulty_selection = [
        ("easy", 1.0 if payload.difficulty.easy else 0.0),
        ("medium", 1.0 if payload.difficulty.medium else 0.0),
        ("hard", 1.0 if payload.difficulty.hard else 0.0),
    ]
    effective_chunk_size = 2 if payload.user_requested_roadmap_length <= 50 else 3

    returned_roadmap_payload = build_and_serialize_roadmap(
        problems_table=problems_table,
        pattern_stats_table=pattern_stats_table,
        difficulty_selection=difficulty_selection,
        user_included_patterns=payload.user_difficulty_selection,
        user_deselected_subpattern_ids=[],
        user_priority_subpattern_multipliers=[],
        chunk_size=effective_chunk_size,
        execute_stmt=execute_stmt,
        user_requested_roadmap_length=payload.user_requested_roadmap_length,
        user_priority_patterns=payload.user_priority_patterns,
        user_excluded_patterns=payload.user_excluded_patterns,
    )

    return {
        "message": "Roadmap generated",
        "request": {
            "user_requested_roadmap_length": payload.user_requested_roadmap_length,
            "user_difficulty_selection": payload.user_difficulty_selection,
            "user_priority_patterns": payload.user_priority_patterns,
            "user_excluded_patterns": payload.user_excluded_patterns,
            "difficulty": payload.difficulty,
            "chunk_size": effective_chunk_size,
        },
        **returned_roadmap_payload,
    }
