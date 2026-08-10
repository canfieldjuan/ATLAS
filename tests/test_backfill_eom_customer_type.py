"""Proof for the customer_type backfill (Slice 1 / Req A, website #174).

The backfill decides what type a real account gets, and billing shape follows
from that. So every refusal path is asserted here, not just the happy one: a
backfill that quietly skipped a bad row would leave the operator believing a
customer was classified when it was not.
"""

from __future__ import annotations

import importlib.util
import uuid
from pathlib import Path

import pytest

asyncpg = pytest.importorskip("asyncpg")

ROOT = Path(__file__).resolve().parent.parent
MIGRATIONS = ROOT / "atlas_brain" / "storage" / "migrations"
DATABASE_URL_ENV = "ATLAS_MIGRATION_TEST_DATABASE_URL"
TENANT = "effingham_maids"
FOREIGN_TENANT = "churnsignals"

_SPEC = importlib.util.spec_from_file_location(
    "backfill_eom_customer_type", ROOT / "scripts" / "backfill_eom_customer_type.py"
)
backfill = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(backfill)


class _PoolAdapter:
    def __init__(self, pool):
        self._pool = pool
        self.is_initialized = True

    async def initialize(self):
        return None

    async def fetch(self, query, *args):
        return await self._pool.fetch(query, *args)

    async def fetchrow(self, query, *args):
        return await self._pool.fetchrow(query, *args)

    async def fetchval(self, query, *args):
        return await self._pool.fetchval(query, *args)

    async def execute(self, query, *args):
        return await self._pool.execute(query, *args)


def _write_mapping(tmp_path: Path, rows: list[tuple[str, str, str]]) -> Path:
    path = tmp_path / "mapping.csv"
    lines = ["atlas_contact_id,customer_type,evidence"]
    lines += [f"{cid},{ctype},{evidence}" for cid, ctype, evidence in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


async def _seed(pool, *, name: str, tenant: str = TENANT) -> str:
    row = await pool.fetchrow(
        """
        INSERT INTO contacts (full_name, business_context_id, contact_type)
        VALUES ($1, $2, 'customer') RETURNING id
        """,
        name,
        tenant,
    )
    return str(row["id"])


@pytest.fixture
async def db(monkeypatch):
    import os

    database_url = os.environ.get(DATABASE_URL_ENV)
    if not database_url:
        pytest.skip(f"{DATABASE_URL_ENV} is not configured")

    schema = f"atlas_backfill_ctype_{uuid.uuid4().hex}"
    admin = await asyncpg.connect(database_url)
    pool = None
    try:
        await admin.execute(f'CREATE SCHEMA "{schema}"')
        await admin.execute(f'SET search_path TO "{schema}", public')
        # Stub the tables 035 only references, exactly as
        # tests/test_eom_lead_conversion_integration.py::_prepare_schema does.
        # 001_initial_schema.sql is deliberately not run: it needs
        # uuid_generate_v4(), and this suite only needs a real contacts table.
        await admin.execute("CREATE TABLE appointments (id UUID PRIMARY KEY)")
        await admin.execute("CREATE TABLE call_transcripts (id UUID PRIMARY KEY)")
        for name in ("035_contacts.sql", "366_contacts_customer_type.sql"):
            await admin.execute((MIGRATIONS / name).read_text())

        async def set_search_path(connection):
            await connection.execute(f'SET search_path TO "{schema}", public')

        pool = await asyncpg.create_pool(
            database_url, min_size=1, max_size=2, setup=set_search_path
        )
        adapter = _PoolAdapter(pool)
        import atlas_brain.storage.database as db_mod

        monkeypatch.setattr(db_mod, "get_db_pool", lambda: adapter)
        yield adapter
    finally:
        if pool is not None:
            await pool.close()
        await admin.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await admin.close()


@pytest.mark.asyncio
async def test_dry_run_reports_without_writing(db, tmp_path):
    """The default must not touch the database."""
    contact_id = await _seed(db, name="Dry Run Co")
    mapping = _write_mapping(tmp_path, [(contact_id, "commercial", "sites=2")])

    exit_code = await backfill.run(mapping_path=mapping, apply=False)

    assert exit_code == 0
    stored = await db.fetchrow("SELECT customer_type FROM contacts WHERE id = $1::uuid", contact_id)
    assert stored["customer_type"] == "unknown", "dry run must not write"


@pytest.mark.asyncio
async def test_apply_writes_both_types(db, tmp_path):
    """Both directions -- a backfill proven only on commercial proves half of it."""
    commercial = await _seed(db, name="Menards")
    residential = await _seed(db, name="Anna McClellan")
    mapping = _write_mapping(
        tmp_path,
        [(commercial, "commercial", "sites=1"), (residential, "residential", "sites=1")],
    )

    exit_code = await backfill.run(mapping_path=mapping, apply=True)

    assert exit_code == 0
    rows = {
        str(r["id"]): r["customer_type"]
        for r in await db.fetch("SELECT id, customer_type FROM contacts")
    }
    assert rows[commercial] == "commercial"
    assert rows[residential] == "residential"


@pytest.mark.asyncio
async def test_rerunning_is_a_no_op_and_never_overwrites_a_decision(db, tmp_path):
    """Idempotent, and an operator's later correction outlives a stale mapping."""
    contact_id = await _seed(db, name="Corrected Later")
    mapping = _write_mapping(tmp_path, [(contact_id, "commercial", "sites=1")])
    await backfill.run(mapping_path=mapping, apply=True)

    # The operator fixes it in the CRM afterwards.
    await db.execute(
        "UPDATE contacts SET customer_type = 'residential' WHERE id = $1::uuid",
        contact_id,
    )

    exit_code = await backfill.run(mapping_path=mapping, apply=True)

    stored = await db.fetchrow(
        "SELECT customer_type FROM contacts WHERE id = $1::uuid", contact_id
    )
    assert stored["customer_type"] == "residential", (
        "a stale mapping must not stomp a later correction"
    )
    assert exit_code == 1, "the disagreement must be surfaced, not silent"


@pytest.mark.asyncio
async def test_a_contact_in_another_tenant_is_refused(db, tmp_path):
    """Tenant scope is enforced by the backfill, not assumed from the mapping."""
    foreign = await _seed(db, name="Churnsignals Account", tenant=FOREIGN_TENANT)
    mapping = _write_mapping(tmp_path, [(foreign, "commercial", "sites=1")])

    exit_code = await backfill.run(mapping_path=mapping, apply=True)

    stored = await db.fetchrow(
        "SELECT customer_type FROM contacts WHERE id = $1::uuid", foreign
    )
    assert stored["customer_type"] == "unknown"
    assert exit_code == 1


@pytest.mark.asyncio
async def test_an_unknown_contact_id_is_reported_not_skipped(db, tmp_path):
    missing = str(uuid.uuid4())
    mapping = _write_mapping(tmp_path, [(missing, "residential", "sites=1")])

    exit_code = await backfill.run(mapping_path=mapping, apply=True)

    assert exit_code == 1


@pytest.mark.asyncio
async def test_a_value_outside_the_set_is_refused(db, tmp_path):
    """Including 'unknown': writing it changes nothing and hides a failed mapping."""
    contact_id = await _seed(db, name="Bad Value Co")
    mapping = _write_mapping(
        tmp_path, [(contact_id, "bogus", "sites=1"), (contact_id, "unknown", "sites=1")]
    )

    exit_code = await backfill.run(mapping_path=mapping, apply=True)

    stored = await db.fetchrow(
        "SELECT customer_type FROM contacts WHERE id = $1::uuid", contact_id
    )
    assert stored["customer_type"] == "unknown"
    assert exit_code == 1


def test_a_mapping_missing_its_columns_is_rejected(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("contact,type\nabc,commercial\n", encoding="utf-8")

    with pytest.raises(SystemExit):
        backfill.read_mapping(path)


@pytest.mark.asyncio
async def test_the_apply_statement_itself_refuses_wrong_tenant_and_already_set(db):
    """The UPDATE's own WHERE clause, exercised directly.

    The Python loop screens both cases before it ever issues the UPDATE, so
    every test above passes whether or not SQL_APPLY carries its guards -- I
    removed them and the suite stayed green, which is why this test exists. The
    clause is not redundant: the loop reads the row and then writes it, and
    between those two statements another writer can retenant or classify it.
    Only the WHERE makes the write conditional atomically.
    """
    foreign = await _seed(db, name="Foreign Tenant Co", tenant=FOREIGN_TENANT)
    decided = await _seed(db, name="Already Decided Co")
    await db.execute(
        "UPDATE contacts SET customer_type = 'residential' WHERE id = $1::uuid", decided
    )

    await db.execute(backfill.SQL_APPLY, foreign, "commercial", backfill.EOM_CONTEXT)
    await db.execute(backfill.SQL_APPLY, decided, "commercial", backfill.EOM_CONTEXT)

    rows = {
        str(r["id"]): r["customer_type"]
        for r in await db.fetch("SELECT id, customer_type FROM contacts")
    }
    assert rows[foreign] == "unknown", "the statement must not cross tenants"
    assert rows[decided] == "residential", "the statement must not overwrite a decision"

    # And it does apply when both conditions hold, so this is not a statement
    # that simply never writes.
    fresh = await _seed(db, name="Fresh Co")
    await db.execute(backfill.SQL_APPLY, fresh, "commercial", backfill.EOM_CONTEXT)
    stored = await db.fetchrow(
        "SELECT customer_type FROM contacts WHERE id = $1::uuid", fresh
    )
    assert stored["customer_type"] == "commercial"
