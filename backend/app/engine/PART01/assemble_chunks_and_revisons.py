from __future__ import annotations

from collections import defaultdict
from typing import DefaultDict, Iterable, List, Tuple

from ..models import Chunk, RevisionProblem, ProblemTuple
from .types import WeightedProblemRow


def assign_buckets_and_chunks(
    rows: List[WeightedProblemRow],
    structured_ratio: float,
    chunk_size: int,
) -> List[WeightedProblemRow]:
    if not rows:
        return []

    ordered = sorted(rows, key=lambda row: row.final_frequency, reverse=True)
    structured_cutoff = len(ordered) * structured_ratio
    structured_counts: DefaultDict[float, int] = defaultdict(int)
    enriched: List[WeightedProblemRow] = []

    for index, row in enumerate(ordered, start=1):
        bucket = "structured" if index <= structured_cutoff else "recap"
        if bucket == "structured":
            subpattern_key = float(row.subpattern_id)
            chunk_number = structured_counts[subpattern_key] // chunk_size
            structured_counts[subpattern_key] += 1
        else:
            chunk_number = 0

        enriched.append(
            WeightedProblemRow(
                subpattern_id=row.subpattern_id,
                tier=row.tier,
                final_frequency=row.final_frequency,
                chunk_number=chunk_number,
                bucket=bucket,
                title=row.title,
                url=row.url,
            )
        )

    return enriched


def convert_rows_to_objects(
    rows: Iterable[WeightedProblemRow],
    pattern_stats: dict[float, float],
) -> Tuple[DefaultDict[float, List[Chunk]], List[Chunk], List[RevisionProblem], dict, dict]:
    grouped: DefaultDict[Tuple[float, float, int], List[ProblemTuple]] = defaultdict(list)
    revisions: List[RevisionProblem] = []
    tier_stats = defaultdict(int)

    for r in rows:
        if r.bucket == "structured":
            grouped[(r.tier, r.subpattern_id, r.chunk_number)].append((r.final_frequency, r.title, r.url))
        else:
            revisions.append(
                RevisionProblem(
                    subpattern_id=r.subpattern_id,
                    final_frequency_sum=r.final_frequency,
                    tier=r.tier,
                    problem=(r.title, r.url),
                )
            )

    chunks: List[Chunk] = []
    chunks_sorted_in_their_tiers: DefaultDict[float, List[Chunk]] = defaultdict(list)

    for (tier, subpattern_id, chunk_number), problems in grouped.items():
        tier_stats[tier] += 1
        final_sum = sum(freq for (freq, _title, _url) in problems)
        chunk = Chunk(
            subpattern_id=subpattern_id,
            final_frequency_sum=final_sum,
            tier=tier,
            roadmap_index_allocation=None,
            problems=problems,
            chunk_number=chunk_number,
        )
        chunks.append(chunk)
        chunks_sorted_in_their_tiers[tier].append(chunk)

    sorted_chunks = sorted(chunks, key=lambda c: (c.tier, -c.final_frequency_sum))

    for tier, chunk_list in chunks_sorted_in_their_tiers.items():
        chunk_list.sort(key=lambda c: -c.final_frequency_sum)

    return chunks_sorted_in_their_tiers, sorted_chunks, revisions, pattern_stats, tier_stats


__all__ = ["assign_buckets_and_chunks", "convert_rows_to_objects"]
