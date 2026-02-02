# Backend Testing Plan (Comprehensive)

Goal: verify the full production path (route → controller → engine → serializer → controller response), plus unit-level validation of every engine module.

## Phase 0 — Test data setup (shared fixtures)
- Create a canonical payload fixture matching the frontend output.
- Create variants:
  - empty include list
  - include list with unknown pattern IDs
  - priority/exclude overlap
  - invalid types (string instead of number)
- Create DB fixtures:
  - small subset of `problems` and `pattern_stats`
  - or use the existing sqlite file in a read-only way for smoke tests.

## Phase 1 — Routes → Controller (API-level)
- Use a test client to hit the route with the payload fixture.
- Assert:
  - 200 response
  - response shape contains `request` and `roadmap`
  - request echo includes expected values
- Validate error cases:
  - missing required fields returns 422/400
  - invalid types return 422/400

## Phase 2 — BaseModel validation (models.py)
- Directly instantiate `generateRoadMapRequest` with:
  - valid payload
  - invalid payloads (missing keys, wrong types)
- Assert defaults:
  - `chunk_size` default applied
- Assert `difficulty` parses into the `DifficultySelection` model.

## Phase 3 — Controller logic (controllers/roadmap_controller.py)
- Call `generate_roadmap(payload)` directly (no HTTP).
- Assert:
  - difficulty tuple conversion happens correctly
  - `user_included_patterns` passes through unchanged
  - priority/exclude passed through
  - `returned_roadmap_payload` merged into response

## Phase 4 — Engine pipeline (end-to-end)
- Call `build_and_serialize_roadmap(...)` directly.
- Validate:
  - no crash
  - `roadmap` list exists
  - item shape matches serializer output
  - roadmap length is <= requested length

## Phase 5 — Engine unit tests (module by module)
- `engine/PART01/roadmap_pattern_allocation.py`
  - verify quotas match known `pattern_stats` + priority multipliers
  - verify renormalization sums to 100%
  - verify exclusions win
- `engine/PART01/database_query_builders.py`
  - verify SQL compiles
  - verify query filters out deselected ids
  - verify per-subpattern top-K logic works (using a tiny fixture DB)
- `engine/PART01/assemble_chunks_and_revisons.py`
  - verify structured/recap split is consistent with `structured_ratio`
  - verify chunk numbering increments per subpattern
- `engine/PART01/part01_entry.py`
  - verify quotas drive selection and assembly correctly
- `engine/PART02/desired_spacing.py`
  - verify spacing positions deterministic for known counts
- `engine/PART02/union_find_allocator.py`
  - verify placements are within bounds and no duplicates
- `engine/PART02/materialize_nodes.py`
  - verify nodes created with correct roadmap_index and chunk pointer
- `engine/PART02/entry.py`
  - verify tier ordering and offset progression
- `engine/PART03/node_eleigibility.py`
  - verify tier extraction works for chunk_pointer and legacy chunk
- `engine/PART03/revision_problem_placement.py`
  - verify placement respects tier constraints
  - verify unplaced list when no eligible nodes
- `engine/PART03/entry.py`
  - verify wrapper passes through unchanged
- `engine/serialize_roadmap.py`
  - verify output shape and required keys

## Phase 6 — Regression scenarios
- empty include list returns empty roadmap
- include list with unknown IDs returns empty or minimal roadmap
- priority + exclude overlap → exclude wins
- request length larger than available problems → capped safely
  (no crash, no duplicates)

## Notes
- Use direct function calls for most unit tests.
- Use API tests for full integration.
- Keep fixtures small to make tests fast and deterministic.
