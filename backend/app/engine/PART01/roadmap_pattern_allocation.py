from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from sqlalchemy import select


def build_pattern_quotas(
    *,
    pattern_stats_table,
    execute_stmt,
    included_subpattern_ids: Sequence[float],
    excluded_subpattern_ids: Sequence[float],
    priority_multipliers: Sequence[Tuple[float, float]],
    requested_total: int,
) -> tuple[List[Tuple[float, int]], Dict[float, float]]:
    """
    Stage A: compute per-pattern quotas from pattern_stats.total_weight.
    Priority multipliers apply only at this stage.
    """
    rows = execute_stmt(
        select(
            pattern_stats_table.c.subpattern_id,
            pattern_stats_table.c.total_weight,
        )
    )

    included_set = None if included_subpattern_ids is None else set(included_subpattern_ids)
    excluded_set = set(excluded_subpattern_ids)

    base_weights: Dict[float, float] = {}
    for row in rows:
        subpattern_id = float(row.subpattern_id)
        if included_set is not None and subpattern_id not in included_set:
            continue
        if subpattern_id in excluded_set:
            continue
        base_weights[subpattern_id] = float(row.total_weight)

    if not base_weights or requested_total <= 0:
        return [], {}

    priority_map: Dict[float, float] = {float(pid): float(mult) for pid, mult in priority_multipliers}
    adjusted_weights: Dict[float, float] = {
        subpattern_id: weight * priority_map.get(subpattern_id, 1.0)
        for subpattern_id, weight in base_weights.items()
    }

    sum_adjusted = sum(adjusted_weights.values())
    if sum_adjusted <= 0:
        return [], {}

    desired: List[tuple[float, float, float]] = []
    quotas: Dict[float, int] = {}
    for subpattern_id, adj_weight in adjusted_weights.items():
        share = adj_weight / sum_adjusted
        exact = share * requested_total
        floor_val = int(exact)
        quotas[subpattern_id] = floor_val
        desired.append((subpattern_id, exact, exact - floor_val))

    allocated = sum(quotas.values())
    remainder = max(0, requested_total - allocated)
    desired.sort(key=lambda item: item[2], reverse=True)
    for subpattern_id, _exact, _frac in desired[:remainder]:
        quotas[subpattern_id] += 1

    adjusted_share_percent: Dict[float, float] = {
        subpattern_id: round((adjusted_weights[subpattern_id] / sum_adjusted) * 100.0, 2)
        for subpattern_id in adjusted_weights
    }

    quota_rows = [(subpattern_id, quota) for subpattern_id, quota in quotas.items() if quota > 0]
    return quota_rows, adjusted_share_percent


__all__ = ["build_pattern_quotas"]
