from datetime import datetime, timezone
from uuid import uuid4
import csv
import io

import pytest
from unittest.mock import AsyncMock

from atlas_brain.api import prospects as mod


class _ProspectPool:
    def __init__(self, prospect_rows, sequence_rows):
        self._prospect_rows = prospect_rows
        self._sequence_rows = sequence_rows
        self.sequence_query = ""
        self.sequence_args = ()

    async def fetch(self, query, *args):
        if "FROM prospects" in query:
            return self._prospect_rows
        if "FROM campaign_sequences" in query:
            self.sequence_query = query
            self.sequence_args = args
            return self._sequence_rows
        return []

    async def fetchrow(self, query, *args):
        if "COUNT(*) AS total FROM prospects" in query:
            return {"total": len(self._prospect_rows)}
        return None


@pytest.mark.asyncio
async def test_list_prospects_normalizes_blank_and_trimmed_filters(monkeypatch):
    from atlas_brain.api import prospects as mod

    class Pool:
        async def fetch(self, query, *args):
            assert "LOWER(company_name) LIKE $1" not in query
            assert "status = $1" in query
            assert "seniority = $2" not in query
            assert args == ("active", 50, 0)
            return []

        async def fetchrow(self, query, *args):
            assert args == ("active",)
            return {"total": 0}

    monkeypatch.setattr(mod, "_pool_or_503", lambda: Pool())

    result = await mod.list_prospects(
        company="   ",
        status="  active  ",
        seniority="	",
        limit=50,
        offset=0,
        _user=object(),
    )

    assert result == {"prospects": [], "count": 0}


@pytest.mark.asyncio
async def test_list_manual_prospect_queue_normalizes_blank_company(monkeypatch):
    from atlas_brain.api import prospects as mod

    class Pool:
        async def fetch(self, query, *args):
            assert "LOWER(company_name_raw) LIKE $1" not in query
            assert args == (25, 0)
            return []

        async def fetchrow(self, query, *args):
            assert args == ()
            return {"total": 0}

    monkeypatch.setattr(mod, "_pool_or_503", lambda: Pool())

    result = await mod.list_manual_prospect_queue(
        company="   ",
        limit=25,
        offset=0,
        _user=object(),
    )

    assert result == {"queue": [], "count": 0}


@pytest.mark.asyncio
async def test_list_company_overrides_normalizes_blank_company(monkeypatch):
    from atlas_brain.api import prospects as mod

    monkeypatch.setattr(mod, "_pool_or_503", lambda: object())
    monkeypatch.setattr(
        mod,
        "fetch_company_override_map",
        AsyncMock(return_value={
            "acme": {"company_name_raw": "Acme Inc", "company_name_norm": "acme"},
            "globex": {"company_name_raw": "Globex Corp", "company_name_norm": "globex"},
        }),
    )

    result = await mod.list_company_overrides(company="   ", _user=object())

    assert result["count"] == 2
    assert {row["company_name_norm"] for row in result["overrides"]} == {"acme", "globex"}


@pytest.mark.asyncio
async def test_export_prospects_normalizes_blank_and_trimmed_filters(monkeypatch):
    from atlas_brain.api import prospects as mod

    class Pool:
        async def fetch(self, query, *args):
            assert "LOWER(p.company_name) LIKE $1" not in query
            assert "p.status = $1" in query
            assert "p.seniority = $2" not in query
            assert args == ("active",)
            return []

    monkeypatch.setattr(mod, "_pool_or_503", lambda: Pool())

    response = await mod.export_prospects(
        company="   ",
        status="  active  ",
        seniority="  ",
        _user=object(),
    )

    chunks = [chunk async for chunk in response.body_iterator]
    body = "".join(
        chunk.decode("utf-8") if isinstance(chunk, (bytes, bytearray)) else str(chunk)
        for chunk in chunks
    )
    assert body == ""



@pytest.mark.asyncio
async def test_list_prospects_enriches_from_latest_sequence_by_email(monkeypatch):
    created_at = datetime(2026, 4, 7, 12, 0, tzinfo=timezone.utc)
    prospect_id = uuid4()
    pool = _ProspectPool(
        prospect_rows=[{
            "id": prospect_id,
            "first_name": "Alex",
            "last_name": "Kim",
            "email": "alex@acme.com",
            "email_status": "valid",
            "title": "RevOps Lead",
            "seniority": "manager",
            "department": "operations",
            "company_name": "Acme Co",
            "company_domain": "acme.com",
            "linkedin_url": None,
            "city": "Austin",
            "state": "TX",
            "country": "US",
            "company_name_norm": "acme",
            "status": "active",
            "created_at": created_at,
            "updated_at": created_at,
        }],
        sequence_rows=[{
            "id": uuid4(),
            "company_name": "Different Display Name",
            "recipient_email": "alex@acme.com",
            "status": "active",
            "current_step": 2,
            "max_steps": 4,
            "last_sent_at": created_at,
            "updated_at": created_at,
            "created_at": created_at,
            "company_context": {
                "company": "Acme Co",
                "churning_from": "Salesforce",
                "target_persona": "revops",
                "reasoning_scope_summary": {
                    "selection_strategy": "vendor_facet_packet_v1",
                    "witnesses_in_scope": 8,
                },
                "reasoning_atom_context": {
                    "top_theses": [
                        {
                            "wedge": "pricing",
                            "summary": "Pricing pressure is driving re-evaluation.",
                            "why_now": "Budget owners are active.",
                            "confidence": "high",
                        },
                    ],
                    "account_signals": [
                        {
                            "company": "Acme Co",
                            "buying_stage": "evaluation",
                            "competitor_context": "HubSpot",
                            "primary_pain": "pricing",
                            "contract_end": "2026-06-30",
                            "quote": "We need better renewal controls.",
                        },
                    ],
                },
                "reasoning_delta_summary": {
                    "changed": True,
                    "new_account_signals": ["Acme Co"],
                },
            },
        }],
    )

    monkeypatch.setattr(mod, "_pool_or_503", lambda: pool)

    result = await mod.list_prospects(
        company=None,
        status=None,
        seniority=None,
        limit=50,
        offset=0,
        _user=object(),
    )

    prospect = result["prospects"][0]
    assert result["count"] == 1
    assert prospect["churning_from"] == "Salesforce"
    assert prospect["target_persona"] == "revops"
    assert prospect["related_sequence_status"] == "active"
    assert prospect["related_sequence_current_step"] == 2
    assert prospect["reasoning_scope_summary"]["selection_strategy"] == "vendor_facet_packet_v1"
    assert prospect["reasoning_atom_context"]["account_signals"][0]["quote"] == "We need better renewal controls."
    assert prospect["reasoning_delta_summary"]["new_account_signals"] == ["Acme Co"]


@pytest.mark.asyncio
async def test_list_prospects_falls_back_to_company_match_when_email_missing(monkeypatch):
    created_at = datetime(2026, 4, 7, 12, 0, tzinfo=timezone.utc)
    pool = _ProspectPool(
        prospect_rows=[{
            "id": uuid4(),
            "first_name": "Morgan",
            "last_name": "Lee",
            "email": None,
            "email_status": None,
            "title": "IT Director",
            "seniority": "director",
            "department": "technology",
            "company_name": "Acme Co.",
            "company_domain": "acme.com",
            "linkedin_url": None,
            "city": None,
            "state": None,
            "country": None,
            "company_name_norm": "acme",
            "status": "active",
            "created_at": created_at,
            "updated_at": created_at,
        }],
        sequence_rows=[{
            "id": uuid4(),
            "company_name": "Acme Company",
            "recipient_email": "owner@acme.com",
            "status": "completed",
            "current_step": 4,
            "max_steps": 4,
            "last_sent_at": created_at,
            "updated_at": created_at,
            "created_at": created_at,
            "company_context": {
                "company": "Acme Co",
                "churning_from": "Zendesk",
                "reasoning_atom_context": {
                    "account_signals": [
                        {
                            "company": "Acme Co",
                            "buying_stage": "renewal_decision",
                            "competitor_context": "Freshdesk",
                            "primary_pain": "support quality",
                        },
                    ],
                },
                "reasoning_delta_summary": {"changed": False},
            },
        }],
    )

    monkeypatch.setattr(mod, "_pool_or_503", lambda: pool)

    result = await mod.list_prospects(
        company=None,
        status=None,
        seniority=None,
        limit=50,
        offset=0,
        _user=object(),
    )

    prospect = result["prospects"][0]
    assert prospect["churning_from"] == "Zendesk"
    assert prospect["related_sequence_status"] == "completed"
    assert prospect["reasoning_atom_context"]["account_signals"][0]["buying_stage"] == "renewal_decision"
    assert pool.sequence_args[1] == ["acme"]
    assert "company_context ->> 'company'" in pool.sequence_query


@pytest.mark.asyncio
async def test_export_prospects_returns_filtered_csv(monkeypatch):
    class Pool:
        async def fetch(self, query, *args):
            assert "FROM prospects p" in query
            assert "p.status = $2" in query
            return [{
                "first_name": "Alex",
                "last_name": "Kim",
                "email": "alex@acme.com",
                "email_status": "valid",
                "title": "RevOps Lead",
                "seniority": "manager",
                "department": "operations",
                "company_name": "Acme Co",
                "company_domain": "acme.com",
                "linkedin_url": None,
                "city": "Austin",
                "state": "TX",
                "country": "US",
                "status": "active",
                "seq_status": "active",
                "seq_step": 2,
                "seq_max_steps": 4,
                "seq_last_sent": datetime(2026, 4, 7, 12, 0, tzinfo=timezone.utc),
                "created_at": datetime(2026, 4, 6, 12, 0, tzinfo=timezone.utc),
            }]

    monkeypatch.setattr(mod, "_pool_or_503", lambda: Pool())

    response = await mod.export_prospects(
        company="Acme",
        status="active",
        seniority=None,
        _user=object(),
    )

    chunks = [chunk async for chunk in response.body_iterator]
    body = "".join(
        chunk.decode("utf-8") if isinstance(chunk, (bytes, bytearray)) else str(chunk)
        for chunk in chunks
    )
    rows = list(csv.DictReader(io.StringIO(body)))

    assert rows[0]["company_name"] == "Acme Co"
    assert rows[0]["seq_status"] == "active"
