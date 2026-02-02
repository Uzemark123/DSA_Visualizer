from __future__ import annotations

import random
from typing import List, Optional, Sequence, Tuple

from .node_eleigibility import node_chunk_tier
from ..models import RevisionProblem

Placement = Tuple[int, RevisionProblem]  # (node_index, revision_problem)


def assign_revision_problems_random(
    roadmap_nodes: Sequence[object],
    revision_problems: Sequence[RevisionProblem],
    rng: Optional[random.Random] = None,
) -> Tuple[List[Placement], List[RevisionProblem]]:
    """
    Randomly assign revision problems to roadmap nodes with constraints:
    - chunk tier must exist
    - revision tier must be >= chunk tier
    - prefer strictly-lower-tier chunk placements; allow same-tier only if no lower-tier options exist

    Returns (placements, unplaced_revision_problems).
    Mutates roadmap nodes in-place to store the assigned revision problem.
    """
    if rng is None:
        rng = random.Random()

    # Precompute eligible indices once, keep tier lookups cheap.
    chunk_tier_by_index = {i: node_chunk_tier(node) for i, node in enumerate(roadmap_nodes)}
    available = {
        i
        for i, t in chunk_tier_by_index.items()
        if t is not None and getattr(roadmap_nodes[i], "revision_pointer", None) is None
    }

    placements: List[Placement] = []
    unplaced: List[RevisionProblem] = []

    for rp in revision_problems:
        eligible = [
            i
            for i in available
            if chunk_tier_by_index[i] is not None and chunk_tier_by_index[i] <= rp.tier
        ]

        if not eligible:
            unplaced.append(rp)
            continue

        lower_tier = [i for i in eligible if chunk_tier_by_index[i] < rp.tier]
        candidates = lower_tier if lower_tier else [i for i in eligible if chunk_tier_by_index[i] == rp.tier]

        if not candidates:
            unplaced.append(rp)
            continue

        chosen = rng.choice(candidates)
        node = roadmap_nodes[chosen]

        # Primary pointer (matches your Node model).
        setattr(node, "revision_pointer", rp)

        # Compatibility pointer (if some UI/serialization still expects this name).
        setattr(node, "revisionproblempointer", rp)

        rp.last_position = chosen
        placements.append((chosen, rp))
        available.discard(chosen)

    return placements, unplaced


__all__ = ["assign_revision_problems_random", "Placement"]
