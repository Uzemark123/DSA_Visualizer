# app/tests/test_phase5_part02_union_find.py

from backend.app.engine.PART02.union_find_allocator import allocate_chunks_with_union_find


def test_allocate_chunks_with_union_find_assigns_slots():
    placements = allocate_chunks_with_union_find(
        total_slots=4,
        desired_positions=[(10, 1), (44, 1), (10, 1)],
    )

    slots = [slot for slot, _pattern in placements]
    assert len(slots) == 3
    assert len(set(slots)) == 3
    assert all(0 <= slot < 4 for slot in slots)
