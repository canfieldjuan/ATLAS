from datetime import datetime, timezone
from uuid import uuid4

import pytest

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
