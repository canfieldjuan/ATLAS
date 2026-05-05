from __future__ import annotations

import sys
from importlib import import_module


_COMPETITIVE_VENDOR_TARGET_SELECTION = (
    "extracted_competitive_intelligence.services.vendor_target_selection"
)
_ATLAS_VENDOR_TARGET_SELECTION = "atlas_brain.services.vendor_target_selection"


def _load_vendor_target_selection_module():
    sys.modules.pop(_COMPETITIVE_VENDOR_TARGET_SELECTION, None)
    sys.modules.pop(_ATLAS_VENDOR_TARGET_SELECTION, None)
    return import_module(_COMPETITIVE_VENDOR_TARGET_SELECTION)


def test_dedupe_vendor_target_rows_prefers_account_row() -> None:
    vendor_target_selection = _load_vendor_target_selection_module()

    rows = [
        {
            "id": "global-1",
            "company_name": "HubSpot",
            "target_mode": "challenger_intel",
            "contact_email": "legacy@example.com",
            "account_id": None,
            "updated_at": "2026-03-18T10:00:00+00:00",
        },
        {
            "id": "owned-1",
            "company_name": " hubspot ",
            "target_mode": "challenger_intel",
            "contact_email": None,
            "account_id": "acct-1",
            "updated_at": "2026-03-19T10:00:00+00:00",
        },
    ]

    deduped = vendor_target_selection.dedupe_vendor_target_rows(rows)

    assert len(deduped) == 1
    assert deduped[0]["id"] == "owned-1"


def test_dedupe_vendor_target_rows_sorts_by_company_and_mode() -> None:
    vendor_target_selection = _load_vendor_target_selection_module()

    rows = [
        {
            "id": "beta-retention",
            "company_name": "Beta",
            "target_mode": "vendor_retention",
            "created_at": "2026-03-20T10:00:00+00:00",
        },
        {
            "id": "acme-challenger",
            "company_name": "Acme",
            "target_mode": "challenger_intel",
            "created_at": "2026-03-20T10:00:00+00:00",
        },
        {
            "id": "acme-retention",
            "company_name": "Acme",
            "target_mode": "vendor_retention",
            "created_at": "2026-03-20T10:00:00+00:00",
        },
    ]

    deduped = vendor_target_selection.dedupe_vendor_target_rows(rows)

    assert [row["id"] for row in deduped] == [
        "acme-challenger",
        "acme-retention",
        "beta-retention",
    ]


def test_competitive_vendor_target_selection_does_not_import_atlas() -> None:
    _load_vendor_target_selection_module()

    assert _ATLAS_VENDOR_TARGET_SELECTION not in sys.modules
