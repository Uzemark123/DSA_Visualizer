from __future__ import annotations

from typing import Iterable, Sequence, Tuple

from sqlalchemy import Column, Float, Integer, String, func, literal, select, union_all


def build_weighted_stmt(
    *,
    problems,
    user_difficulty_selection: Sequence[Tuple[str, float]],
    user_deselected_subpattern_ids: Sequence[float],
    quota_rows: Sequence[Tuple[float, int]],
):
    """
    Stage B: difficulty-weighted ranking per subpattern + top-K selection via quota.
    """
    def _inline_rows(columns: Sequence[Column], rows: Iterable[Sequence[object]], name: str):
        row_list = list(rows)
        if not row_list:
            row_list = [tuple(None for _ in columns)]
        selects = [
            select(
                *[
                    literal(value).label(column.name)
                    for value, column in zip(row, columns)
                ]
            )
            for row in row_list
        ]
        return union_all(*selects).subquery(name)

    # Inline "deselected" table by subpattern_id
    deselected_tbl = _inline_rows(
        [Column("subpattern_id", Float)],
        [(x,) for x in user_deselected_subpattern_ids],
        "deselected",
    )

    # Inline difficulty multiplier table by difficulty label
    difficulty_tbl = _inline_rows(
        [Column("difficulty", String), Column("multiplier", Float)],
        list(user_difficulty_selection),
        "difficulty_modifiers",
    )

    # Inline quota table by subpattern_id
    quota_tbl = _inline_rows(
        [Column("subpattern_id", Float), Column("quota", Integer)],
        list(quota_rows),
        "quota",
    )

    base_freq = problems.c.combinedFrequency
    difficulty_multiplier = func.coalesce(difficulty_tbl.c.multiplier, 1.0)
    final_frequency = (base_freq * difficulty_multiplier).label("final_frequency")

    row_num = func.row_number().over(
        partition_by=problems.c.subpattern_id,
        order_by=final_frequency.desc(),
    ).label("row_num")

    scored = (
        select(
            problems.c.subpattern_id,
            problems.c.tier,
            final_frequency,
            problems.c.Title.label("title"),
            problems.c.url.label("url"),
            quota_tbl.c.quota.label("quota"),
            row_num,
        )
        .select_from(
            problems.join(quota_tbl, problems.c.subpattern_id == quota_tbl.c.subpattern_id)
            .outerjoin(difficulty_tbl, problems.c.difficulty == difficulty_tbl.c.difficulty)
            .outerjoin(deselected_tbl, problems.c.subpattern_id == deselected_tbl.c.subpattern_id)
        )
        .where(deselected_tbl.c.subpattern_id.is_(None))
    ).subquery()

    stmt = (
        select(
            scored.c.subpattern_id,
            scored.c.tier,
            scored.c.final_frequency,
            scored.c.title,
            scored.c.url,
        )
        .where(scored.c.row_num <= scored.c.quota)
        .order_by(scored.c.final_frequency.desc())
    )

    return stmt


__all__ = ["build_weighted_stmt"]
