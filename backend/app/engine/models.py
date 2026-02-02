# Step 01 — domain models (chunks, revision problems, roadmap).
# Called by: later pipeline steps when constructing/attaching objects.
# Calls: no external modules; self-contained definitions.

from __future__ import annotations
from ..config import PATTERN_TO_TIER
from typing import Dict

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple, Union

ProblemTuple = Tuple[float, str, str]  # (final_frequency, title, url)

'''
main roadmap obkect that will be mutated over course of program function calls
'''

class RoadMap:


    pattern_to_tier: Dict[float, float] = PATTERN_TO_TIER #taken from config - static dictionary also create ll_problems_and_subpattern_ids_in_DB and tiers_to_pattern all caps cvariables

    def __init__(self, user_requested_roadmap_length, user_difficulty_selection, user_priority_patterns, user_excluded_patterns):
        self._user_requested_roadmap_length = None
        self._pattern_stats = None
        self._tier_stats = None

        # user-provided inputs
        self.user_requested_roadmap_length = user_requested_roadmap_length
        self.user_difficulty_selection = user_difficulty_selection
        self.user_priority_patterns = user_priority_patterns
        self.user_excluded_patterns = user_excluded_patterns

        self.roadmap_array_length = 0
        self.roadmap_array = []
    
    @property
    def user_requested_roadmap_length(self):
        if self._user_requested_roadmap_length is None:
            raise RuntimeError("user_requested_roadmap_length accessed before being set")
        return self._user_requested_roadmap_length

    @user_requested_roadmap_length.setter
    def user_requested_roadmap_length(self, value):
        if self._user_requested_roadmap_length is not None:
            raise RuntimeError("user_requested_roadmap_length can only be set once")
        self._user_requested_roadmap_length = value

    @property
    def pattern_stats(self):
        if self._pattern_stats is None:
            raise RuntimeError("pattern_stats accessed before being set")
        return self._pattern_stats

    @pattern_stats.setter
    def pattern_stats(self, value):
        if self._pattern_stats is not None:
            raise RuntimeError("pattern_stats can only be set once")
        self._pattern_stats = value

    @property
    def tier_stats(self):
        if self._tier_stats is None:
            raise RuntimeError("pattern_stats accessed before being set")
        return self._tier_stats

    @tier_stats.setter
    def tier_stats(self, value):
        if self._tier_stats is not None:
            raise RuntimeError("pattern_stats can only be set once")
        self._tier_stats = value
    

@dataclass
class Chunk:
    subpattern_id: int
    final_frequency_sum: float  # used for ordering / heap priority
    tier: int
    roadmap_index_allocation: Optional[int]
    problems: List[ProblemTuple]
    chunk_number: Optional[int] = None


@dataclass
class RevisionProblem:
    subpattern_id: int
    final_frequency_sum: float  # used for ordering / heap priority
    tier: int
    problem: Tuple[str, str]  # (title, url)
    last_position: Optional[int] = None


class Node:

    def __init__(self, roadmap_index=None):
        self.roadmap_index = roadmap_index
        self.chunk_pointer = None
        self.revision_pointer = None

    def set_node_index(self, index):
        self.roadmap_index = index

    def set_node_chunk_pointer(self, chunk: Chunk) -> None:
        self.chunk_pointer = chunk

    def set_node_revision_pointer(self, revision_problem: RevisionProblem) -> None:
        self.revision_pointer = revision_problem

        



__all__ = ["Chunk", "Node", "RevisionProblem", "RoadMap", "ProblemTuple"]
