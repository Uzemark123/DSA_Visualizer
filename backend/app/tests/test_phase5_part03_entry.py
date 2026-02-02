# app/tests/test_phase5_part03_entry.py

import random

from backend.app.engine.PART03.entry import assign_revision_problems_random
from backend.app.engine.models import Chunk, Node, RevisionProblem


def _node_with_tier(tier: float) -> Node:
    node = Node(roadmap_index=0)
    node.chunk_pointer = Chunk(
        subpattern_id=10.1,
        final_frequency_sum=10.0,
        tier=tier,
        roadmap_index_allocation=None,
        problems=[],
        chunk_number=0,
    )
    return node


def test_entry_assign_revision_passes_through():
    nodes = [_node_with_tier(1.0)]
    rp = RevisionProblem(subpattern_id=44.1, final_frequency_sum=8.0, tier=2.0, problem=("A", "http://a"))

    placements, unplaced = assign_revision_problems_random(nodes, [rp], rng=random.Random(0))

    assert placements
    assert not unplaced
