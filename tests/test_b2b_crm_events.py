from types import SimpleNamespace

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


from unittest.mock import AsyncMock, MagicMock


def test_ingest_crm_event_trims_text_fields_before_persistence(monkeypatch):
    app = FastAPI()
    app.include_router(crm_events_api.router)

    monkeypatch.setattr(crm_events_api.settings.crm_event, "enabled", True)
    pool = MagicMock()
    pool.is_initialized = True
    pool.fetchval = AsyncMock(return_value="11111111-1111-1111-1111-111111111111")
    monkeypatch.setattr(crm_events_api, "get_db_pool", lambda: pool)

    with TestClient(app) as client:
        response = client.post(
            "/b2b/crm/events",
            json={
                "crm_provider": " hubspot ",
                "event_type": " deal_won ",
                "crm_event_id": " evt-1 ",
                "company_name": " Acme Corp ",
                "contact_email": "   ",
                "contact_name": " Jane Doe ",
                "deal_id": " deal-1 ",
                "deal_name": " Expansion ",
                "deal_stage": " closedwon ",
                "event_timestamp": " 2026-04-12T10:00:00Z ",
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["crm_provider"] == "hubspot"
    assert body["event_type"] == "deal_won"

    args = pool.fetchval.await_args.args
    assert "account_id = COALESCE(b2b_crm_events.account_id, EXCLUDED.account_id)" in args[0]
    assert args[1] == "hubspot"
    assert args[2] == "evt-1"
    assert args[3] == "deal_won"
    assert args[4] == "Acme Corp"
    assert args[5] is None
    assert args[6] == "Jane Doe"
    assert args[7] == "deal-1"
    assert args[8] == "Expansion"
    assert args[9] == "closedwon"


def test_ingest_crm_events_batch_trims_text_fields_before_persistence(monkeypatch):
    app = FastAPI()
    app.include_router(crm_events_api.router)

    monkeypatch.setattr(crm_events_api.settings.crm_event, "enabled", True)
    pool = MagicMock()
    pool.is_initialized = True
    pool.fetchval = AsyncMock(return_value="22222222-2222-2222-2222-222222222222")
    monkeypatch.setattr(crm_events_api, "get_db_pool", lambda: pool)

    with TestClient(app) as client:
        response = client.post(
            "/b2b/crm/events/batch",
            json={
                "events": [
                    {
                        "crm_provider": " salesforce ",
                        "event_type": " meeting_booked ",
                        "crm_event_id": " batch-1 ",
                        "company_name": " Beta Corp ",
                        "contact_email": "   ",
                        "contact_name": " Sam Buyer ",
                        "deal_id": " deal-2 ",
                        "deal_name": " Renewal ",
                        "deal_stage": " evaluation ",
                        "event_timestamp": " 2026-04-12T11:00:00Z ",
                    }
                ]
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["ingested"] == 1
    assert body["errors"] == []
    assert body["created_ids"] == ["22222222-2222-2222-2222-222222222222"]

    args = pool.fetchval.await_args.args
    assert "account_id = COALESCE(b2b_crm_events.account_id, EXCLUDED.account_id)" in args[0]
    assert args[1] == "salesforce"
    assert args[2] == "batch-1"
    assert args[3] == "meeting_booked"
    assert args[4] == "Beta Corp"
    assert args[5] is None
    assert args[6] == "Sam Buyer"
    assert args[7] == "deal-2"
    assert args[8] == "Renewal"
    assert args[9] == "evaluation"



def test_list_crm_events_validates_filters_before_db_touch(monkeypatch):
    app = FastAPI()
    app.include_router(crm_events_api.router)

    monkeypatch.setattr(crm_events_api.settings.crm_event, "enabled", True)

    def _boom():
        raise AssertionError("DB pool should not be acquired")

    monkeypatch.setattr(crm_events_api, "get_db_pool", _boom)

    with TestClient(app) as client:
        status_response = client.get("/b2b/crm/events", params={"status": "not_a_status"})
        provider_response = client.get("/b2b/crm/events", params={"crm_provider": "not_a_provider"})
        start_response = client.get("/b2b/crm/events", params={"start_date": "not-a-date"})

    assert status_response.status_code == 400
    assert status_response.json()["detail"].startswith("Invalid status.")
    assert provider_response.status_code == 400
    assert provider_response.json()["detail"].startswith("Invalid crm_provider.")
    assert start_response.status_code == 400
    assert start_response.json()["detail"] == "Invalid start_date (ISO 8601 expected)"


def test_list_crm_events_normalizes_blank_optional_filters(monkeypatch):
    app = FastAPI()
    app.include_router(crm_events_api.router)

    pool = MagicMock()
    pool.is_initialized = True
    pool.fetch = AsyncMock(return_value=[])
    monkeypatch.setattr(crm_events_api, "get_db_pool", lambda: pool)

    with TestClient(app) as client:
        response = client.get(
            "/b2b/crm/events",
            params={
                "status": "   ",
                "crm_provider": "  ",
                "company_name": "	",
                "start_date": "   ",
                "end_date": "  ",
                "limit": 50,
                "offset": 0,
            },
        )

    assert response.status_code == 200
    assert response.json()["events"] == []
    query, *params = pool.fetch.await_args.args
    assert "status = $" not in query
    assert "crm_provider = $" not in query
    assert "LOWER(company_name) LIKE" not in query
    assert "received_at >= $" not in query
    assert "received_at < $" not in query
    assert params == [50, 0]


def test_native_crm_webhooks_reject_invalid_json_before_db_touch(monkeypatch):
    app = FastAPI()
    app.include_router(crm_events_api.router)
    app.dependency_overrides[crm_events_api.optional_auth] = lambda: SimpleNamespace(
        account_id="11111111-1111-1111-1111-111111111111"
    )

    monkeypatch.setattr(crm_events_api.settings.crm_event, "enabled", True)

    def _boom():
        raise AssertionError("DB pool should not be acquired")

    monkeypatch.setattr(crm_events_api, "get_db_pool", _boom)

    with TestClient(app) as client:
        for path in (
            "/b2b/crm/events/hubspot",
            "/b2b/crm/events/salesforce",
            "/b2b/crm/events/pipedrive",
        ):
            response = client.post(
                path,
                data="{",
                headers={"content-type": "application/json"},
            )
            assert response.status_code == 400
            assert response.json()["detail"] == "Invalid JSON"


def test_get_enrichment_stats_scopes_to_authenticated_account(monkeypatch):
    app = FastAPI()
    app.include_router(crm_events_api.router)
    app.dependency_overrides[crm_events_api.optional_auth] = lambda: SimpleNamespace(
        account_id="11111111-1111-1111-1111-111111111111"
    )

    pool = MagicMock()
    pool.is_initialized = True
    pool.fetchrow = AsyncMock(
        return_value={
            "total_events": 4,
            "matched": 2,
            "unmatched": 1,
            "skipped": 0,
            "pending": 1,
            "errored": 0,
            "has_company": 3,
            "has_email": 2,
            "missing_both": 1,
            "enriched_count": 1,
            "enriched_matched": 1,
        }
    )
    monkeypatch.setattr(crm_events_api, "get_db_pool", lambda: pool)

    with TestClient(app) as client:
        response = client.get("/b2b/crm/events/enrichment-stats")

    assert response.status_code == 200
    query, *params = pool.fetchrow.await_args.args
    assert "WHERE account_id = $1::uuid" in query
    assert params == ["11111111-1111-1111-1111-111111111111"]


def test_get_enrichment_stats_keeps_global_query_without_auth(monkeypatch):
    app = FastAPI()
    app.include_router(crm_events_api.router)

    pool = MagicMock()
    pool.is_initialized = True
    pool.fetchrow = AsyncMock(
        return_value={
            "total_events": 0,
            "matched": 0,
            "unmatched": 0,
            "skipped": 0,
            "pending": 0,
            "errored": 0,
            "has_company": 0,
            "has_email": 0,
            "missing_both": 0,
            "enriched_count": 0,
            "enriched_matched": 0,
        }
    )
    monkeypatch.setattr(crm_events_api, "get_db_pool", lambda: pool)

    with TestClient(app) as client:
        response = client.get("/b2b/crm/events/enrichment-stats")

    assert response.status_code == 200
    query, *params = pool.fetchrow.await_args.args
    assert "WHERE account_id = $1::uuid" not in query
    assert params == []


def test_ingest_hubspot_webhook_persists_account_id(monkeypatch):
    app = FastAPI()
    app.include_router(crm_events_api.router)
    app.dependency_overrides[crm_events_api.optional_auth] = lambda: SimpleNamespace(
        account_id="11111111-1111-1111-1111-111111111111"
    )

    monkeypatch.setattr(crm_events_api.settings.crm_event, "enabled", True)
    pool = MagicMock()
    pool.is_initialized = True
    pool.execute = AsyncMock(return_value="INSERT 0 1")
    monkeypatch.setattr(crm_events_api, "get_db_pool", lambda: pool)

    with TestClient(app) as client:
        response = client.post(
            "/b2b/crm/events/hubspot",
            json={
                "subscriptionType": "deal.propertyChange",
                "objectId": 123,
                "propertyName": "dealstage",
                "propertyValue": "closedwon",
                "occurredAt": 1712916000000,
                "properties": {
                    "company": "Acme Corp",
                    "email": "owner@acme.test",
                    "dealname": "Renewal",
                    "amount": "1200",
                },
            },
        )

    assert response.status_code == 200
    query, *params = pool.execute.await_args.args
    assert "processing_notes, account_id" in query
    assert "$12::uuid" in query
    assert "COALESCE(b2b_crm_events.account_id, EXCLUDED.account_id)" in query
    assert params[-1] == "11111111-1111-1111-1111-111111111111"


def test_salesforce_and_pipedrive_webhooks_preserve_account_id_on_upsert(monkeypatch):
    app = FastAPI()
    app.include_router(crm_events_api.router)
    app.dependency_overrides[crm_events_api.optional_auth] = lambda: SimpleNamespace(
        account_id="11111111-1111-1111-1111-111111111111"
    )

    monkeypatch.setattr(crm_events_api.settings.crm_event, "enabled", True)

    cases = [
        (
            "/b2b/crm/events/salesforce",
            {
                "sobject": "Opportunity",
                "record": {
                    "Id": "sf-1",
                    "StageName": "Closed Won",
                    "AccountName": "Acme Corp",
                    "ContactEmail": "owner@acme.test",
                    "Name": "Renewal",
                    "Amount": "1200",
                    "LastModifiedDate": "2026-04-12T10:00:00Z",
                },
            },
        ),
        (
            "/b2b/crm/events/pipedrive",
            {
                "event": "updated.deal",
                "current": {
                    "id": 123,
                    "status": "won",
                    "title": "Renewal",
                    "org_name": "Acme Corp",
                    "email": "owner@acme.test",
                    "value": "1200",
                    "update_time": "2026-04-12T10:00:00Z",
                },
            },
        ),
    ]

    with TestClient(app) as client:
        for path_name, payload in cases:
            pool = MagicMock()
            pool.is_initialized = True
            pool.execute = AsyncMock(return_value="INSERT 0 1")
            monkeypatch.setattr(crm_events_api, "get_db_pool", lambda pool=pool: pool)

            response = client.post(path_name, json=payload)

            assert response.status_code == 200
            query, *params = pool.execute.await_args.args
            assert "account_id = COALESCE(b2b_crm_events.account_id, EXCLUDED.account_id)" in query
            assert params[-1] == "11111111-1111-1111-1111-111111111111"
