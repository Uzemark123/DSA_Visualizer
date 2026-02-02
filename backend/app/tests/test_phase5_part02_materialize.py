# app/tests/test_phase5_part02_materialize.py

from backend.app.engine.PART02.materialize_nodes import materialize_tier_nodes
from backend.app.engine.models import Chunk


def test_materialize_tier_nodes_assigns_indices_and_chunks():
    chunks = [
        Chunk(subpattern_id=10.1, final_frequency_sum=10.0, tier=1.0, roadmap_index_allocation=None, problems=[], chunk_number=0),
        Chunk(subpattern_id=10.1, final_frequency_sum=9.0, tier=1.0, roadmap_index_allocation=None, problems=[], chunk_number=1),
        Chunk(subpattern_id=44.1, final_frequency_sum=8.0, tier=1.0, roadmap_index_allocation=None, problems=[], chunk_number=0),
    ]
    placements = [(0, 10.1), (1, 44.1), (2, 10.1)]

    nodes = materialize_tier_nodes(placements, chunks, tier_offset=5)

    assert [node.roadmap_index for node in nodes] == [5, 6, 7]
    assert nodes[0].chunk_pointer.subpattern_id == 10.1
    assert nodes[1].chunk_pointer.subpattern_id == 44.1
    assert nodes[2].chunk_pointer.subpattern_id == 10.1
