import sys
import types
from pathlib import Path
from unittest.mock import MagicMock
from uuid import uuid4


if "asyncpg" not in sys.modules:
    asyncpg_module = types.ModuleType("asyncpg")
    asyncpg_module.connect = MagicMock()
    asyncpg_module.Connection = object
    asyncpg_module.Record = dict
    sys.modules["asyncpg"] = asyncpg_module

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from backfill_trustpilot_jsonld_company_cleanup import (  # noqa: E402
    _plan_row_cleanup,
    _resolution_evidence_patch,
    _same_text,
)


def _row(**overrides):
    row = {
        "id": uuid4(),
        "source": "trustpilot",
        "vendor_name": "Slack",
        "reviewer_company": "Publisher Org",
        "reviewer_company_norm": "publisher org",
        "raw_metadata": {"extraction_method": "json_ld"},
        "enrichment": {"reviewer_context": {"company_name": "Publisher Org"}},
        "account_resolution_id": uuid4(),
    }
    row.update(overrides)
    return row


def test_same_text_is_case_insensitive_and_trimmed():
    assert _same_text(" Publisher Org ", "publisher org")
    assert not _same_text("Publisher Org", "Different Org")


def test_plan_row_cleanup_targets_only_trustpilot_jsonld_rows():
    planned = _plan_row_cleanup(_row())

    assert planned is not None
    assert planned["clear_enrichment_company"] is True
    assert planned["polluted_company"] == "Publisher Org"
    assert planned["account_resolution_id"] is not None
    assert planned["metadata"]["trustpilot_jsonld_company_cleanup"]["scope"] == (
        "trustpilot_jsonld_reviewer_company_cleanup"
    )


def test_plan_row_cleanup_preserves_different_enrichment_company():
    planned = _plan_row_cleanup(
        _row(enrichment={"reviewer_context": {"company_name": "Northwind"}}),
    )

    assert planned is not None
    assert planned["clear_enrichment_company"] is False


def test_plan_row_cleanup_skips_non_jsonld_rows():
    assert _plan_row_cleanup(_row(raw_metadata={"extraction_method": "html"})) is None
    assert _plan_row_cleanup(_row(source="reddit")) is None


def test_resolution_evidence_patch_marks_cleanup_scope():
    evidence = _resolution_evidence_patch(polluted_company="Publisher Org")

    assert evidence["reason"] == "trustpilot_jsonld_reviewer_company_cleanup"
    assert evidence["source"] == "trustpilot"
    assert evidence["polluted_company"] == "Publisher Org"
