# Backend Revisions Plan (engine)

Goal: separate request parsing, pattern expansion, DB querying, allocation, and serialization so the roadmap pipeline is clean and testable.

## Phase 0 — Inventory & Inputs
- Payload now uses `user_requested_roadmap_length`, `user_difficulty_selection`, `user_priority_patterns`, `user_excluded_patterns`, and `difficulty`.
- `problems` is keyed by `subpattern_id`.
- `pattern_stats.total_weight` is used as the base percentage share per subpattern.
- Mapping source is `PATTERN_ID_TO_SUBPATTERN_ID` in `config.py`.

## Phase 1 — Controller + Model Wiring (Implemented)
- `models.py` matches the frontend payload naming.
- `roadmap_controller.py` returns `returned_roadmap_payload`, includes `difficulty`, and adapts it to tuple-based difficulty selection.
- `pattern_stats_table` is wired through the controller and pipeline.

## Phase 2 — Pattern Expansion (Implemented)
- Expansion runs inside `build_roadmap_from_request` before any `problems` query:
  - `user_excluded_patterns` → `user_deselected_subpattern_ids` (subpattern ids)
  - `user_priority_patterns` → `user_priority_subpattern_multipliers`
- Uses `PATTERN_ID_TO_SUBPATTERN_ID` from `config.py`.
- Adds `DEFAULT_PRIORITY_MULTIPLIER` for priority-selected patterns.
- Existing `user_deselected_subpattern_ids` / `user_priority_subpattern_multipliers` are merged in.

## Phase 3 — Weighting & Allocation (Implemented)
- Stage A uses `pattern_stats.total_weight` from the DB.
- Priority multipliers apply only at the pattern (subpattern) level.
- Shares renormalized and quotas computed for `user_requested_roadmap_length`.
- `pattern_stats` returned as adjusted share percentages.

## Phase 4 — Query Structure (Implemented, Hybrid)
- Stage A (Python): quotas built from `pattern_stats.total_weight`.
- Stage B (SQL): difficulty-weighted ranking per subpattern and top-K selection via quota table.
- Priority never affects per-row scoring; difficulty does.

## Phase 5 — Serialization
- `serialize_roadmap.py` should only format output.
- No query logic or weighting math inside serializer.

## PART01 Split (Implemented)
New layout under `engine/PART01/`:
- `roadmap_pattern_allocation.py` — quotas from `pattern_stats`
- `database_query_builders.py` — SQL for difficulty scoring + top-K per subpattern
- `assemble_chunks_and_revisons.py` — bucket assignment + chunk/revision object assembly
- `types.py` — `WeightedProblemRow`
- `part01_entry.py` — orchestration wrapper

## PART02 Split (Implemented)
New layout under `engine/PART02/`:
- `desired_spacing.py` — desired positions per pattern
- `union_find_allocator.py` — union-find placement algorithm
- `materialize_nodes.py` — Node materialization
- `entry.py` — orchestration wrapper

## PART03 Split (Implemented)
New layout under `engine/PART03/`:
- `node_eleigibility.py` — node eligibility helpers
- `revision_problem_placement.py` — revision placement algorithm
- `entry.py` — orchestration wrapper
