# app/tests/test_phase2_models.py

import pytest
from pydantic import ValidationError

from backend.app.models.models import generateRoadMapRequest

from .fixtures_payloads import (
    PAYLOAD_EMPTY_INCLUDE,
    PAYLOAD_INVALID_TYPES,
    VALID_PAYLOAD,
    VALID_PAYLOAD_WITH_CHUNK,
)

def test_generateRoadMapRequest_valid_payload():
    payload = generateRoadMapRequest(**VALID_PAYLOAD)
    assert payload.user_requested_roadmap_length == VALID_PAYLOAD["user_requested_roadmap_length"]
    assert payload.user_difficulty_selection == VALID_PAYLOAD["user_difficulty_selection"]

def test_generateRoadMapRequest_custom_chunk_size():
    payload = generateRoadMapRequest(**VALID_PAYLOAD_WITH_CHUNK)
    assert payload.chunk_size == VALID_PAYLOAD_WITH_CHUNK["chunk_size"]

def test_generateRoadMapRequest_allows_empty_include():
    payload = generateRoadMapRequest(**PAYLOAD_EMPTY_INCLUDE)
    assert payload.user_difficulty_selection == []

def test_generateRoadMapRequest_invalid_types():
    with pytest.raises(ValidationError):
        generateRoadMapRequest(**PAYLOAD_INVALID_TYPES)
