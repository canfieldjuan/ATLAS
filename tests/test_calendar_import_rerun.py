"""Pin the legacy ICS importer's rerun-duplicate defect before retiring it.

D5 / website #128. `scripts/import_calendar_contacts.py` writes through
`crm_provider.create_contact`, which resolves an existing contact on PHONE then
EMAIL only -- never address (see the provider docstring: "returning an existing
one if phone or email already matches. Dedup order: phone first, then email").
So an address-only record (no phone, no email) can never match an existing
contact and is re-created on every run. The replacement
`scripts/import_eom_customers_live.py` added an address pre-resolver; the legacy
script did not.

Retiring the runnable script (its `__main__` now raises) makes this defect
unobservable by execution, so this test documents that it existed and pins the
exact class it affected. The stub CRM is a faithful model of the real provider's
contract -- match on phone/email, not address -- so the duplication here is the
script's behavior, not the stub's.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

# crm_provider imports asyncpg at module load; stub it so monkeypatching the real
# get_crm_provider attribute does not require a live driver (mirrors
# tests/test_crm_read_scoping.py).
_asyncpg = MagicMock()
_asyncpg_exc = MagicMock()
_asyncpg_exc.UndefinedTableError = type("UndefinedTableError", (Exception,), {})
_asyncpg.exceptions = _asyncpg_exc
sys.modules.setdefault("asyncpg", _asyncpg)
sys.modules.setdefault("asyncpg.exceptions", _asyncpg_exc)

import import_calendar_contacts as ics  # noqa: E402


class _PhoneEmailOnlyCRM:
    """Faithful model of crm_provider.create_contact: resolve on phone then email,
    never address. A match returns created_at != updated_at (an update); a miss
    creates a fresh id with created_at == updated_at (import_records reads exactly
    that to count created-vs-updated)."""

    def __init__(self):
        self._by_phone: dict[str, str] = {}
        self._by_email: dict[str, str] = {}
        self.created: list[tuple[str, str]] = []  # (id, address)
        self._n = 0
        self.log_interaction = AsyncMock(return_value={"id": "i"})

    async def create_contact(self, data):
        phone = data.get("phone")
        email = (data.get("email") or "").lower() or None
        existing = None
        if phone and phone in self._by_phone:
            existing = self._by_phone[phone]
        elif email and email in self._by_email:
            existing = self._by_email[email]
        if existing is not None:
            return {"id": existing, "created_at": "t0", "updated_at": "t1"}
        self._n += 1
        cid = f"c{self._n}"
        if phone:
            self._by_phone[phone] = cid
        if email:
            self._by_email[email] = cid
        self.created.append((cid, data.get("address")))
        return {"id": cid, "created_at": "t0", "updated_at": "t0"}


def _record(**kw):
    base = dict(
        name="Test Cust",
        address="123 Main St",
        contact_type="customer",
        source_calendar="residential",
        event_count=1,
    )
    base.update(kw)
    return ics.CustomerRecord(**base)


@pytest.fixture
def stub_crm(monkeypatch):
    crm = _PhoneEmailOnlyCRM()
    monkeypatch.setattr(
        "atlas_brain.services.crm_provider.get_crm_provider", lambda: crm
    )
    return crm


@pytest.mark.asyncio
async def test_address_only_record_duplicates_on_rerun(stub_crm):
    """DEFECT PIN: an address-only record is re-created on every run."""
    rec = _record(name="Address Only", address="500 Oak Ave")  # no phone, no email

    await ics.import_records([rec], dry_run=False)
    await ics.import_records([rec], dry_run=False)  # re-run, same record

    same_address = [cid for cid, addr in stub_crm.created if addr == "500 Oak Ave"]
    assert len(same_address) == 2, (
        "the legacy importer duplicates address-only records on re-run because the "
        "provider resolves only on phone/email; two runs produced "
        f"{len(same_address)} contacts for one address"
    )


@pytest.mark.asyncio
async def test_phone_bearing_record_does_not_duplicate_on_rerun(stub_crm):
    """Contrast: a record carrying a phone resolves on re-run, so it does NOT
    duplicate. This proves the defect is specific to the address-only class, not
    `import_records` in general -- otherwise retiring it would be over-claiming."""
    rec = _record(name="Has Phone", address="600 Elm St", phone="618-555-0100")

    await ics.import_records([rec], dry_run=False)
    await ics.import_records([rec], dry_run=False)

    same_address = [cid for cid, addr in stub_crm.created if addr == "600 Elm St"]
    assert len(same_address) == 1, "a phone-bearing record must resolve on re-run"


def test_module_stays_importable_as_a_library():
    """The retirement is CLI-only: import_eom_customers_live.py does
    `import import_calendar_contacts as ics` and reuses this extraction core, so
    the module must remain importable even though its `__main__` now refuses."""
    assert hasattr(ics, "parse_ics")
    assert hasattr(ics, "CustomerRecord")
    assert hasattr(ics, "import_records")


def test_running_the_script_as_a_command_is_retired():
    """Review Contract criterion 1, executed: running the script as a command must
    exit nonzero with a deprecation message that names the replacement.

    This is the ONLY test that exercises the changed behavior end-to-end -- the
    others call import_records directly or inspect symbols, so they would all still
    pass if the guard were removed, moved to import time, returned success, or
    stopped naming the replacement. Run it as a real subprocess so the assertion is
    bound to the actual `__main__` exit, not an in-process re-import.
    """
    script = REPO / "scripts" / "import_calendar_contacts.py"
    result = subprocess.run(
        [sys.executable, str(script), "--dry-run"],
        capture_output=True,
        text=True,
        cwd=str(REPO),
    )
    assert result.returncode != 0, (
        f"the retired CLI must exit nonzero; got {result.returncode}\n"
        f"stdout={result.stdout!r}\nstderr={result.stderr!r}"
    )
    combined = (result.stdout + result.stderr).lower()
    assert "retired" in combined, f"missing deprecation wording; got {combined!r}"
    assert "import_eom_customers_live" in combined, (
        "the deprecation must name scripts/import_eom_customers_live.py as the "
        f"replacement; got {combined!r}"
    )
