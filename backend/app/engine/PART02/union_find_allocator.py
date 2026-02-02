from __future__ import annotations

from typing import List, Optional, Tuple

DesiredPosition = Tuple[int, int]  # (pattern_id, desired_index)
Placement = Tuple[int, int]  # (slot_index, pattern_id)


def allocate_chunks_with_union_find(
    total_slots: int,
    desired_positions: List[DesiredPosition],
) -> List[Placement]:
    """
    Assign each desired pattern chunk to the nearest available slot at or after its target.
    If that fails (no slots to the right), fall back to the closest free slot before the target.
    Returns placements as (slot_index, pattern_id).
    """
    parent = list(range(total_slots + 1))  # sentinel at total_slots means "no slot"
    occupied_slots: List[Optional[int]] = [None] * total_slots
    placements: List[Placement] = []

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def occupy(start: int) -> Optional[int]:
        root = find(start)
        if root >= total_slots:
            return None
        parent[root] = find(root + 1)
        return root

    for pattern, desired in desired_positions:
        desired = max(0, min(total_slots - 1, desired))

        slot = occupy(desired)
        if slot is None:
            for back in range(desired, -1, -1):
                if occupied_slots[back] is None:
                    slot = back
                    parent[back] = find(back + 1)
                    break

        if slot is None:
            continue

        occupied_slots[slot] = pattern
        placements.append((slot, pattern))

    return placements


__all__ = ["allocate_chunks_with_union_find", "Placement"]
