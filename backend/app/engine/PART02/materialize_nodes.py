from __future__ import annotations

from collections import defaultdict, deque
from typing import DefaultDict, List, Optional, Tuple

from ..models import Chunk, Node

Placement = Tuple[int, int]  # (slot_index, pattern_id)


def materialize_tier_nodes(
    placements: List[Placement],
    tier_chunks: List[Chunk],
    tier_offset: int,
) -> List[Node]:
    """
    Turn placement slots into Node objects, consuming chunks per-pattern in deterministic order.
    """
    pattern_to_queue: DefaultDict[int, deque] = defaultdict(deque)
    for chunk in tier_chunks:
        pattern_to_queue[chunk.subpattern_id].append(chunk)

    tier_nodes: List[Optional[Node]] = [None] * len(tier_chunks)

    for slot, pattern_id in placements:
        chunk = pattern_to_queue[pattern_id].popleft()
        node = Node(roadmap_index=tier_offset + slot)
        node.chunk_pointer = chunk
        tier_nodes[slot] = node

    # NOTE: We currently drop None slots. If placements ever return fewer than total_slots,
    # this would compress indices and break the offset invariant. Keep/update if you add
    # cases where some slots stay empty.
    return [n for n in tier_nodes if n is not None]


__all__ = ["materialize_tier_nodes"]
