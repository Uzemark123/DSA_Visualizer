# app/tests/test_phase3_controller.py

import importlib
import json

from backend.app.models.models import generateRoadMapRequest

from .fixtures_db import build_test_db
from .fixtures_payloads import VALID_PAYLOAD

def reload_for_db(monkeypatch, db_path: str):
    monkeypatch.setenv("DB_FILE", db_path)

    import backend.app.config as config
    import backend.app.db.sqlalchemy_db as sqlalchemy_db
    import backend.app.db as db_pkg
    import backend.app.controllers.roadmap_controller as controller

    importlib.reload(config)
    importlib.reload(sqlalchemy_db)
    importlib.reload(db_pkg)
    importlib.reload(controller)

    return controller

def test_controller_generate_roadmap(monkeypatch, tmp_path):
    db_path = str(build_test_db(tmp_path))
    controller = reload_for_db(monkeypatch, db_path)

    payload = generateRoadMapRequest(**VALID_PAYLOAD)
    resp = controller.generate_roadmap(payload)

    print("payload:", json.dumps(VALID_PAYLOAD, indent=2))
    print("response:", json.dumps(resp, default=lambda o: o.model_dump(), indent=2))

    assert "roadmap" in resp
    assert "request" in resp
    assert resp["request"]["user_requested_roadmap_length"] == VALID_PAYLOAD["user_requested_roadmap_length"]
    assert resp["request"]["chunk_size"] == 2
