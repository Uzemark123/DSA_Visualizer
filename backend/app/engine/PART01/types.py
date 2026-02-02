from dataclasses import dataclass


@dataclass(frozen=True)
class WeightedProblemRow:
    subpattern_id: float
    tier: float
    final_frequency: float
    chunk_number: int
    bucket: str
    title: str
    url: str
