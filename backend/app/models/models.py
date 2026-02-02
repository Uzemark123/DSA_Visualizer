from pydantic import BaseModel
from typing import List, Optional

class DifficultySelection(BaseModel):
    easy: bool
    medium: bool
    hard: bool

class generateRoadMapRequest(BaseModel):
    user_requested_roadmap_length: int
    user_difficulty_selection: List[int]
    user_priority_patterns: List[int]
    user_excluded_patterns: List[int]
    difficulty: DifficultySelection
    chunk_size: Optional[int] = None
    #TO ADD: roadMapType: str

'''
Payload matches frontend GenerateRoadmapRequest:
  - user_requested_roadmap_length
  - user_difficulty_selection
  - user_priority_patterns
  - user_excluded_patterns
  - difficulty (easy/medium/hard)
'''
