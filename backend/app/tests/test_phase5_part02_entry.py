# app/tests/test_phase5_part02_entry.py

from backend.app.engine.PART02.entry import run_spacing_and_allocation
from backend.app.engine.models import Chunk


def test_run_spacing_and_allocation_offsets_by_tier_length():
    chunks_by_tier = {
        1.0: [
            Chunk(subpattern_id=10.1, final_frequency_sum=10.0, tier=1.0, roadmap_index_allocation=None, problems=[], chunk_number=0),
            Chunk(subpattern_id=10.1, final_frequency_sum=9.0, tier=1.0, roadmap_index_allocation=None, problems=[], chunk_number=1),
        ],
        2.0: [
            Chunk(subpattern_id=44.1, final_frequency_sum=8.0, tier=2.0, roadmap_index_allocation=None, problems=[], chunk_number=0),
        ],
    }

    nodes = run_spacing_and_allocation(chunks_by_tier, start_offset=0)
    indices = [node.roadmap_index for node in nodes]

    assert indices == [0, 1, 2]
