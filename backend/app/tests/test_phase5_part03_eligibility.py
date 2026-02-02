# app/tests/test_phase5_part03_eligibility.py

from backend.app.engine.PART03.node_eleigibility import node_chunk_tier
from backend.app.engine.models import Chunk, Node


def test_node_chunk_tier_from_chunk_pointer():
    node = Node(roadmap_index=0)
    node.chunk_pointer = Chunk(
        subpattern_id=10.1,
        final_frequency_sum=10.0,
        tier=2.5,
        roadmap_index_allocation=None,
        problems=[],
        chunk_number=0,
    )

    assert node_chunk_tier(node) == 2.5


def test_node_chunk_tier_missing_returns_none():
    node = Node(roadmap_index=1)
    assert node_chunk_tier(node) is None
