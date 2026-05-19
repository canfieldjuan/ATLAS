from __future__ import annotations

import pytest

from atlas_brain.storage.repositories import invoice as invoice_repo_mod


class _InvoiceLookupPool:
    is_initialized = True

    def __init__(self) -> None:
        self.fetchrow_calls: list[tuple[str, tuple[object, ...]]] = []

    async def fetchrow(self, query: str, *args: object) -> dict[str, object] | None:
        self.fetchrow_calls.append((query, args))
        invoice_number = args[0]
        if (
            "lower(invoice_number) = lower($1)" in query
            and invoice_number == "INV-2026-May-0185"
        ):
            return {
                "id": "invoice-1",
                "invoice_number": "INV-2026-May-0185",
                "line_items": [],
                "metadata": {},
            }
        return None


@pytest.mark.asyncio
async def test_get_by_number_finds_mixed_case_generated_invoice_numbers(monkeypatch):
    pool = _InvoiceLookupPool()
    monkeypatch.setattr(invoice_repo_mod, "get_db_pool", lambda: pool)

    invoice = await invoice_repo_mod.InvoiceRepository().get_by_number(
        " INV-2026-May-0185 "
    )

    assert invoice is not None
    assert invoice["invoice_number"] == "INV-2026-May-0185"
    assert len(pool.fetchrow_calls) == 1
    query, args = pool.fetchrow_calls[0]
    assert "lower(invoice_number) = lower($1)" in query
    assert args == ("INV-2026-May-0185",)
