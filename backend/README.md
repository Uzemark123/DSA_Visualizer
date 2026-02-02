# Backend (FastAPI) – DSA Visualizer

This backend exposes the roadmap generation API and translates a curated SQLite database into a structured sequence of problems. The API layer is thin; the core logic lives in `backend/app/engine/`.

## Database model (SQLite)
Defined in `backend/app/db/sqlalchemy_db.py` as SQLAlchemy table metadata. The backend assumes the database is already curated and normalized; it does **not** scrape or deduplicate at runtime.

### problems table
Primary source of roadmap problems.

Columns (as stored in SQLite):
- `Difficulty` (String) ? exposed as `difficulty` in queries.
- `Title` (String) ? used in API output.
- `combindedFrequency` (Float) ? exposed as `combinedFrequency`; used as the base weight for ranking.
- `Category` (String) ? category label (not used directly in the roadmap engine yet).
- `Pattern` (String) ? pattern label (not used directly in selection logic).
- `pattern_id` (Integer) ? parent pattern identifier.
- `subpattern` (String) ? subpattern label.
- `subpattern_id` (Float) ? **primary routing key** for quotas and selection.
- `url` (String) ? included in API output.
- `curatedListFreq` (Integer) ? provenance signal (not used directly in the current engine).
- `Notes` (String) ? optional notes (not used by engine).
- `Frequency` (Integer) ? raw frequency signal (not used directly in the current engine).
- `tier` (Float) ? used to assign problems to tiered roadmap layers.

What the engine actually reads:
- `combinedFrequency` is the base ranking signal (see `build_weighted_stmt`).
- `difficulty` and `tier` are used for filtering/constraints.
- `subpattern_id` is the unit of allocation (quotas, chunking, priority/exclusion).
- `Title` and `url` are returned to the frontend.

### pattern_stats table
Used to compute quotas per subpattern (how many problems to draw from each subpattern).

Columns:
- `pattern` (String)
- `subpattern` (String)
- `subpattern_id` (Float)
- `tier` (Float)
- `problem_count` (Integer)
- `total_weight` (Float) ? **quota driver** in `build_pattern_quotas`

## Data provenance and curation (intent + implementation)
The roadmap system assumes the database is **pattern-first** and **de-duplicated** before it reaches the backend. In practice, this means:
- Each problem is assigned a single “home” subpattern (`subpattern_id`).
- Composite weight signals (e.g., list frequency, curated prominence) are pre-folded into `combinedFrequency` and `total_weight`.
- The backend does not currently merge or reconcile sources at runtime; it treats the SQLite DB as the canonical, curated source.

If you want provenance-aware weighting, the fields already exist (`curatedListFreq`, `Frequency`, `combinedFrequency`) but the assembly pipeline that computes them lives **outside** the FastAPI runtime. The backend simply consumes the results.

## Pattern / tier configuration
`backend/app/config.py` contains the pattern graph metadata used by the generator:
- `PATTERN_ID_TO_SUBPATTERN_ID`: expands user-selected patterns into subpattern IDs.
- `PATTERN_TO_TIER` and `TIERS_TO_PATTERN`: tier definitions for spacing/placement.
- `STRUCTURED_RATIO`: ratio of structured vs recap problems.
- `DEFAULT_PRIORITY_MULTIPLIER`: applied when a pattern is marked priority.

## Roadmap generation pipeline (conceptual ? code)
The generation pipeline corresponds to distinct engine stages:

1) **Quota allocation (PART01)**
   - `build_pattern_quotas` reads `pattern_stats.total_weight` per subpattern and allocates per-subpattern quotas.
   - Priority multipliers are applied **here**, which increases quota allocation for selected subpatterns.

2) **Weighted selection (PART01)**
   - `build_weighted_stmt` ranks problems per subpattern using:
     - `combinedFrequency` (base weight)
     - difficulty multipliers (`difficulty` selection)
   - Top-K per subpattern is selected based on the quota.

3) **Structured vs recap split + chunking (PART01)**
   - Selected rows are split into:
     - **structured** bucket (early, ordered learning)
     - **recap** bucket (revision problems)
   - Structured rows are chunked by subpattern into short runs (`chunk_size`).

4) **Tiered spacing + allocation (PART02)**
   - `get_desired_chunk_positions` computes ideal spacing between chunks in each tier.
   - `allocate_chunks_with_union_find` assigns chunks to the nearest available slots.

5) **Revision placement (PART03)**
   - Recap problems are placed into existing nodes with tier constraints:
     - Revision tier must be **>=** chunk tier.
     - Prefer lower-tier placement when possible.
   - This creates the **spaced interleaving** effect (revisiting patterns later rather than only once).

6) **Serialization**
   - `serialize_roadmap.py` emits a list of roadmap nodes, each with optional `chunk` and `revisionProblem` payloads.

## User preferences ? engine behavior
- **Difficulty selection**: applied as multipliers during SQL ranking.
- **Included patterns**: expanded via `PATTERN_ID_TO_SUBPATTERN_ID` and used to form quotas.
- **Excluded patterns**: filtered out at the SQL stage.
- **Priority patterns**: multipliers applied during quota allocation only.
- **Chunk size**: defaults to 2 for requests <= 50, otherwise 3.

## Key assumptions / heuristics
- Quotas are proportional to `pattern_stats.total_weight`.
- Priority only affects quota allocation (not spacing or revision assignment).
- Difficulty acts as a ranking multiplier rather than a hard filter (unless a multiplier is 0).
- Revision placement is randomized within tier constraints.

## Legacy notes
`backend/app/db/db.py` exposes a direct sqlite3 helper and is largely legacy; current roadmap generation uses SQLAlchemy table metadata via `sqlalchemy_db.py`.
