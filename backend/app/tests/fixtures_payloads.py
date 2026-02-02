# app/tests/fixtures_payloads.py

VALID_PAYLOAD = {
    "user_requested_roadmap_length": 6,
    "user_difficulty_selection": [10, 44],
    "user_priority_patterns": [10],
    "user_excluded_patterns": [],
    "difficulty": {"easy": True, "medium": True, "hard": False},
    "chunk_size": 3,
}

VALID_PAYLOAD_WITH_CHUNK = {
    **VALID_PAYLOAD,
    "chunk_size": 2,
}

PAYLOAD_EMPTY_INCLUDE = {
    "user_requested_roadmap_length": 6,
    "user_difficulty_selection": [],
    "user_priority_patterns": [],
    "user_excluded_patterns": [],
    "difficulty": {"easy": True, "medium": True, "hard": False},
    "chunk_size": 3,
}

PAYLOAD_OVERLAP_PRIORITY_EXCLUDE = {
    "user_requested_roadmap_length": 6,
    "user_difficulty_selection": [10, 44],
    "user_priority_patterns": [10, 44],
    "user_excluded_patterns": [10],
    "difficulty": {"easy": True, "medium": True, "hard": False},
    "chunk_size": 3,
}

PAYLOAD_INVALID_TYPES = {
    "user_requested_roadmap_length": "six",
    "user_difficulty_selection": ["10"],
    "user_priority_patterns": "10",
    "user_excluded_patterns": None,
    "difficulty": {"easy": "yes", "medium": 1, "hard": 0},
    "chunk_size": "three",
}
