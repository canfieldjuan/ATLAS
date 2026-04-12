from fastapi import FastAPI
from fastapi.testclient import TestClient

import atlas_brain.api.b2b_crm_events as crm_events_api


def test_ingest_crm_events_batch_rejects_invalid_json_before_db_touch(monkeypatch):
    app = FastAPI()
    app.include_router(crm_events_api.router)

    monkeypatch.setattr(crm_events_api.settings.crm_event, "enabled", True)

    def _boom():
        raise AssertionError("DB pool should not be acquired")

    monkeypatch.setattr(crm_events_api, "get_db_pool", _boom)

    with TestClient(app) as client:
        response = client.post(
            "/b2b/crm/events/batch",
            data="{",
            headers={"content-type": "application/json"},
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "Request body must be valid JSON"


def test_ingest_crm_events_batch_rejects_non_object_json_before_db_touch(monkeypatch):
    app = FastAPI()
    app.include_router(crm_events_api.router)

    monkeypatch.setattr(crm_events_api.settings.crm_event, "enabled", True)

    def _boom():
        raise AssertionError("DB pool should not be acquired")

    monkeypatch.setattr(crm_events_api, "get_db_pool", _boom)

    with TestClient(app) as client:
        response = client.post("/b2b/crm/events/batch", json=[])

    assert response.status_code == 400
    assert response.json()["detail"] == "Body must be an object containing an 'events' array"
