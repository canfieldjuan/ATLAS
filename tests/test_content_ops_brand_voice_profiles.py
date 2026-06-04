from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
import uuid

import pytest

from atlas_brain import _content_ops_brand_voice_profiles as service


MIGRATION = (
    Path(__file__).resolve().parent.parent
    / "atlas_brain"
    / "storage"
    / "migrations"
    / "333_content_ops_brand_voice_profiles.sql"
)


class _Pool:
    def __init__(self, *, fetchrow_result=None, fetch_rows=None) -> None:
        self.fetchrow_result = fetchrow_result
        self.fetch_rows = list(fetch_rows or [])
        self.fetchrow_calls: list[dict] = []
        self.fetch_calls: list[dict] = []

    async def fetchrow(self, query, *args):
        self.fetchrow_calls.append({"query": str(query), "args": args})
        return self.fetchrow_result

    async def fetch(self, query, *args):
        self.fetch_calls.append({"query": str(query), "args": args})
        return self.fetch_rows


class _FailingFetchrowPool(_Pool):
    async def fetchrow(self, query, *args):
        self.fetchrow_calls.append({"query": str(query), "args": args})
        raise RuntimeError("database unavailable")


def _row(**overrides):
    row = {
        "id": uuid.uuid4(),
        "account_id": uuid.uuid4(),
        "name": "Acme editorial",
        "descriptors": json.dumps(["plainspoken", "operator-led"]),
        "exemplars": json.dumps(["This is how Acme writes."]),
        "banned_terms": json.dumps(["synergy"]),
        "preferred_pov": "second_person",
        "reading_level": "plain",
        "metadata": json.dumps({"source": "operator"}),
        "created_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
        "updated_at": datetime(2026, 1, 2, tzinfo=timezone.utc),
        "archived_at": None,
    }
    row.update(overrides)
    return row


def test_content_ops_brand_voice_profiles_migration_is_tenant_scoped() -> None:
    sql = MIGRATION.read_text()

    assert "CREATE TABLE IF NOT EXISTS content_ops_brand_voice_profiles" in sql
    assert "account_id     UUID NOT NULL REFERENCES saas_accounts(id)" in sql
    assert "descriptors    JSONB NOT NULL DEFAULT '[]'::jsonb" in sql
    assert "archived_at    TIMESTAMPTZ" in sql
    assert "uq_content_ops_brand_voice_profiles_account_name_active" in sql
    assert "WHERE archived_at IS NULL" in sql


@pytest.mark.asyncio
async def test_create_brand_voice_profile_normalizes_and_inserts_profile() -> None:
    account_id = uuid.uuid4()
    pool = _Pool(fetchrow_result=_row(account_id=account_id))

    record = await service.create_brand_voice_profile(
        pool,
        account_id=account_id,
        payload={
            "name": " Acme editorial ",
            "descriptors": ["plainspoken", "operator-led"],
            "exemplars": [{"text": "Write like this."}],
            "banned_terms": ["synergy"],
            "preferred_pov": "you",
            "reading_level": "plain",
            "metadata": {"source": "operator"},
        },
    )

    call = pool.fetchrow_calls[0]
    assert "INSERT INTO content_ops_brand_voice_profiles" in call["query"]
    args = call["args"]
    assert args[0] == account_id
    assert args[1] == "Acme editorial"
    assert json.loads(args[2]) == ["plainspoken", "operator-led"]
    assert json.loads(args[3]) == ["Write like this."]
    assert json.loads(args[4]) == ["synergy"]
    assert args[5] == "second_person"
    assert args[6] == "plain"
    assert json.loads(args[7]) == {"source": "operator"}
    assert record.account_id == account_id
    assert record.descriptors == ("plainspoken", "operator-led")
    assert record.as_profile().banned_terms == ("synergy",)


@pytest.mark.asyncio
async def test_update_brand_voice_profile_is_tenant_scoped() -> None:
    account_id = uuid.uuid4()
    profile_id = uuid.uuid4()
    pool = _Pool(fetchrow_result=_row(id=profile_id, account_id=account_id))

    record = await service.update_brand_voice_profile(
        pool,
        account_id=account_id,
        profile_id=profile_id,
        payload={"name": "Acme v2", "descriptors": ["direct"]},
    )

    assert record is not None
    call = pool.fetchrow_calls[0]
    compact_sql = " ".join(call["query"].split())
    assert "UPDATE content_ops_brand_voice_profiles" in compact_sql
    assert "WHERE id = $1 AND account_id = $2 AND archived_at IS NULL" in compact_sql
    assert call["args"][:3] == (profile_id, account_id, "Acme v2")


@pytest.mark.asyncio
async def test_list_brand_voice_profiles_returns_display_records() -> None:
    account_id = uuid.uuid4()
    pool = _Pool(fetch_rows=[_row(account_id=account_id)])

    records = await service.list_brand_voice_profiles(pool, account_id=account_id)

    assert records[0].account_id == account_id
    assert records[0].metadata == {"source": "operator"}
    assert records[0].exemplars == ("This is how Acme writes.",)
    call = pool.fetch_calls[0]
    assert "WHERE account_id = $1" in call["query"]
    assert "archived_at IS NULL" in call["query"]
    assert call["args"] == (account_id,)


@pytest.mark.asyncio
async def test_lookup_brand_voice_profile_returns_full_profile() -> None:
    account_id = uuid.uuid4()
    profile_id = uuid.uuid4()
    pool = _Pool(fetchrow_result=_row(id=profile_id, account_id=account_id))

    profile = await service.lookup_brand_voice_profile(
        pool,
        account_id=str(account_id),
        profile_id=str(profile_id),
    )

    assert profile is not None
    assert profile.id == str(profile_id)
    assert profile.account_id == str(account_id)
    assert profile.exemplars == ("This is how Acme writes.",)
    assert pool.fetchrow_calls[0]["args"] == (profile_id, account_id)


@pytest.mark.asyncio
async def test_lookup_brand_voice_profile_fails_closed_on_bad_ids() -> None:
    pool = _Pool(fetchrow_result=_row())

    profile = await service.lookup_brand_voice_profile(
        pool,
        account_id="not-a-uuid",
        profile_id=str(uuid.uuid4()),
    )

    assert profile is None
    assert pool.fetchrow_calls == []


@pytest.mark.asyncio
async def test_lookup_brand_voice_profile_propagates_database_unavailable() -> None:
    with pytest.raises(RuntimeError, match="database unavailable"):
        await service.lookup_brand_voice_profile(
            _FailingFetchrowPool(),
            account_id=str(uuid.uuid4()),
            profile_id=str(uuid.uuid4()),
        )


@pytest.mark.asyncio
async def test_archive_brand_voice_profile_is_tenant_scoped() -> None:
    account_id = uuid.uuid4()
    profile_id = uuid.uuid4()
    pool = _Pool(fetchrow_result={"id": profile_id})

    archived = await service.archive_brand_voice_profile(
        pool,
        account_id=account_id,
        profile_id=profile_id,
    )

    assert archived is True
    call = pool.fetchrow_calls[0]
    compact_sql = " ".join(call["query"].split())
    assert "WHERE id = $1 AND account_id = $2 AND archived_at IS NULL" in compact_sql
    assert call["args"] == (profile_id, account_id)


@pytest.mark.asyncio
async def test_brand_voice_profile_requires_guidance() -> None:
    with pytest.raises(ValueError, match="requires at least one"):
        await service.create_brand_voice_profile(
            _Pool(fetchrow_result=_row()),
            account_id=uuid.uuid4(),
            payload={"name": "Empty profile"},
        )
