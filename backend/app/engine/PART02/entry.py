from __future__ import annotations

from collections import defaultdict
from typing import DefaultDict, Dict, List

from .desired_spacing import get_desired_chunk_positions
from .materialize_nodes import materialize_tier_nodes
from .union_find_allocator import allocate_chunks_with_union_find
from ..models import Chunk, Node

PatternCount = Dict[int, int]  # pattern_id -> count of chunks in this tier


def run_spacing_and_allocation(
    chunks_by_tier: Dict[float, List[Chunk]],
    start_offset: int = 0,
) -> List[Node]:
    """
    Orchestrate spacing + allocation tier by tier.
    - chunks_by_tier: dict[tier, list[Chunk]] (each list already sorted in desired order)
    - start_offset: starting index for roadmap positions
    Returns a flat list of Nodes with chunk pointers and roadmap indices.
    """
    roadmap_nodes: List[Node] = []
    offset = start_offset

    for tier in sorted(chunks_by_tier.keys()):
        tier_chunks = chunks_by_tier[tier]
        total_slots = len(tier_chunks)

        # Count how many chunks per pattern in this tier
        pattern_counts: PatternCount = defaultdict(int)
        for chunk in tier_chunks:
            pattern_counts[chunk.subpattern_id] += 1

        desired_positions = get_desired_chunk_positions(total_slots, pattern_counts)
        placements = allocate_chunks_with_union_find(total_slots, desired_positions)
        tier_nodes = materialize_tier_nodes(placements, tier_chunks, offset)

        roadmap_nodes.extend(tier_nodes)
        offset += total_slots

    return roadmap_nodes


__all__ = ["run_spacing_and_allocation"]
