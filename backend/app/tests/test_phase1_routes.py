# app/tests/test_phase1_routes.py

import importlib
import os

import pytest
from fastapi.testclient import TestClient

from .fixtures_payloads import VALID_PAYLOAD
from .fixtures_db import build_test_db

def make_client(monkeypatch, tmp_path) -> TestClient:
    db_path = build_test_db(tmp_path)

    # Must be set before importing backend modules that create the engine
    monkeypatch.setenv("DB_FILE", str(db_path))

    import backend.app.config as config
    import backend.app.db.sqlalchemy_db as sqlalchemy_db
    import backend.app.db as db_pkg
    import backend.app.controllers.roadmap_controller as controller
    import backend.app.main as main

    # Force reload so new DB_FILE is respected even if imports happened earlier
    importlib.reload(config)
    importlib.reload(sqlalchemy_db)
    importlib.reload(db_pkg)
    importlib.reload(controller)
    importlib.reload(main)

    return TestClient(main.app)

def test_generate_roadmap_request_success(monkeypatch, tmp_path):
    client = make_client(monkeypatch, tmp_path)
    resp = client.post("/api/generateRoadmapRequest", json=VALID_PAYLOAD)
    assert resp.status_code == 200

    data = resp.json()
    assert "roadmap" in data
    assert "request" in data
    assert data["request"]["user_requested_roadmap_length"] == VALID_PAYLOAD["user_requested_roadmap_length"]

def test_generate_roadmap_request_invalid_payload(monkeypatch, tmp_path):
    client = make_client(monkeypatch, tmp_path)
    resp = client.post("/api/generateRoadmapRequest", json={"user_requested_roadmap_length": 10})
    # FastAPI/Pydantic usually returns 422 for schema mismatch
    assert resp.status_code in (400, 422)
