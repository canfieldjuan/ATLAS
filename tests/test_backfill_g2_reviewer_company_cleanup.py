import sys
from pathlib import Path
from uuid import uuid4

from tests._module_stub import stub_missing_module

# Stub asyncpg only when it is genuinely not importable, so CI (where asyncpg
# is installed) imports the real module and no sibling test is poisoned.
stub_missing_module(
    "asyncpg",
    attributes={"connect": object, "Connection": object, "Record": dict},
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from backfill_g2_reviewer_company_cleanup import (  # noqa: E402
    _plan_row_cleanup,
    _same_text,
)


def _row(**overrides):
    row = {
        "id": uuid4(),
        "source": "g2",
        "vendor_name": "Slack",
        "source_review_id": "slack-review-1",
        "reviewer_company": "Mid-Market (51-1000 emp.)",
        "reviewer_company_norm": "mid-market (51-1000 emp.)",
        "reviewer_industry": None,
        "company_size_raw": "Mid-Market (51-1000 emp.)",
        "raw_metadata": {"extraction_method": "html"},
        "enrichment": {"reviewer_context": {"company_name": "Mid-Market (51-1000 emp.)"}},
        "account_resolution_id": uuid4(),
    }
    row.update(overrides)
    return row


def test_same_text_is_case_insensitive_and_trimmed():
    assert _same_text(" Mid-Market (51-1000 emp.) ", "mid-market (51-1000 emp.)")
    assert not _same_text("Mid-Market (51-1000 emp.)", "Northwind")


def test_plan_row_cleanup_targets_company_size_labels():
    planned = _plan_row_cleanup(_row())

    assert planned is not None
    assert planned["cleanup_reason"] == "company_size_label"
    assert planned["clear_enrichment_company"] is True
    assert planned["metadata"]["g2_reviewer_company_cleanup"]["scope"] == "g2_reviewer_company_cleanup"


def test_plan_row_cleanup_targets_industry_labels():
    planned = _plan_row_cleanup(
        _row(
            reviewer_company="Newspapers",
            reviewer_company_norm="newspapers",
            reviewer_industry="Newspapers",
            company_size_raw="Small-Business (50 or fewer emp.)",
            enrichment={"reviewer_context": {"company_name": "Newspapers"}},
        ),
    )

    assert planned is not None
    assert planned["cleanup_reason"] == "industry_label"
    assert planned["clear_enrichment_company"] is True


def test_plan_row_cleanup_skips_non_g2_rows():
    assert _plan_row_cleanup(_row(source="reddit")) is None

