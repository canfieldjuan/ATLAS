import pytest


class _FakePool:
    def __init__(self):
        self.fetchrow_calls = []
        self.execute_calls = []

    async def fetch(self, query, *args):
        if "SELECT DISTINCT vendor_name" in query:
            return [{"vendor_name": "ACME CRM"}]
        return []

    async def fetchrow(self, query, *args):
        self.fetchrow_calls.append((query, args))
        candidate_names = args[0]
        candidate_domains = args[1]
        if "acme holdings" in candidate_names and "acme.com" in candidate_domains:
            return {
                "id": "org-1",
                "company_name_raw": "Acme Holdings Inc.",
                "company_name_norm": "acme holdings",
                "domain": "acme.com",
                "industry": "Software",
                "employee_count": 250,
                "annual_revenue_range": "$10M-$50M",
            }
        return None

    async def execute(self, query, *args):
        self.execute_calls.append((query, args))
        return "OK"


@pytest.mark.asyncio
async def test_sync_vendor_firmographics_reuses_apollo_override_alias_and_domain(monkeypatch):
    from atlas_brain.autonomous.tasks import _b2b_shared

    pool = _FakePool()

    async def _fake_overrides(_pool):
        return {
            "acme crm": {
                "company_name_raw": "ACME CRM",
                "company_name_norm": "acme crm",
                "search_names": ["Acme Holdings"],
                "domains": ["acme.com"],
            }
        }

    monkeypatch.setattr(_b2b_shared, "fetch_company_override_map", _fake_overrides)
    monkeypatch.setattr(_b2b_shared, "_canonicalize_vendor", lambda raw: raw)

    synced = await _b2b_shared._sync_vendor_firmographics(pool, as_of=__import__("datetime").date(2026, 3, 17))

    assert synced == 1
    assert pool.fetchrow_calls
    _, args = pool.fetchrow_calls[0]
    assert "acme crm" in args[0]
    assert "acme holdings" in args[0]
    assert "acme.com" in args[1]
    assert len(pool.execute_calls) == 2
