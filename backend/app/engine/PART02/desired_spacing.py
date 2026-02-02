from __future__ import annotations

from typing import Dict, List, Tuple

PatternCount = Dict[int, int]  # pattern_id -> count of chunks in this tier
DesiredPosition = Tuple[int, int]  # (pattern_id, desired_index)


def get_patterns_with_spacing(
    total_chunks_in_tier: int,
    pattern_counts: PatternCount,
) -> List[Tuple[int, int, float]]:
    """
    For each pattern, compute an ideal spacing between its chunks across the tier.
    """
    spacing_stats: List[Tuple[int, int, float]] = []
    for pattern, count in pattern_counts.items():
        spacing = (total_chunks_in_tier / count) if count else float("inf")
        spacing_stats.append((pattern, count, spacing))
    return spacing_stats


def get_desired_chunk_positions(
    total_chunks_in_tier: int,
    pattern_counts: PatternCount,
) -> List[DesiredPosition]:
    """
    Generate desired positions for each chunk instance of each pattern within a tier.
    """
    spacing_stats = get_patterns_with_spacing(total_chunks_in_tier, pattern_counts)
    desired: List[DesiredPosition] = []

    for pattern, count, spacing in spacing_stats:
        if count <= 0:
            continue
        for i in range(count):
            pos = int(round((i + 0.5) * spacing))
            pos = max(0, min(total_chunks_in_tier - 1, pos))
            desired.append((pattern, pos))

    desired.sort(key=lambda x: x[1])
    return desired


__all__ = ["get_patterns_with_spacing", "get_desired_chunk_positions"]
