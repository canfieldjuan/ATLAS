from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/smoke_content_ops_faq_lifecycle.py"
SPEC = importlib.util.spec_from_file_location(
    "smoke_content_ops_faq_lifecycle",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
smoke = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(smoke)
SUPPORT_TICKET_CSV = ROOT / "extracted_content_pipeline/examples/support_ticket_sources.csv"


class _Pool:
    def __init__(
        self,
        *,
        existing_relations=None,
        update_hits: bool = True,
    ) -> None:
        self.existing_relations = (
            set(existing_relations)
            if existing_relations is not None
            else {"ticket_faq_markdown"}
        )
        self.update_hits = update_hits
        self.rows: list[dict] = []
        self.fetchval_calls: list[dict] = []
        self.fetch_calls: list[dict] = []
        self.execute_calls: list[dict] = []
        self.closed = False

    async def fetchval(self, query, *args):
        self.fetchval_calls.append({"query": str(query), "args": args})
        if "to_regclass" in str(query):
            return args[0] if args and args[0] in self.existing_relations else None
        faq_id = f"faq-uuid-{len(self.rows) + 1}"
        self.rows.append({
            "id": faq_id,
            "account_id": args[0],
            "target_id": args[1],
            "target_mode": args[2],
            "title": args[3],
            "markdown": args[4],
            "items": args[5],
            "source_count": args[6],
            "ticket_source_count": args[7],
            "output_checks": args[8],
            "warnings": args[9],
            "metadata": args[10],
            "status": "draft",
        })
        return faq_id

    async def fetch(self, query, *args):
        self.fetch_calls.append({"query": str(query), "args": args})
        account_id = args[0] if args else ""
        status = args[1] if "status = $2" in str(query) and len(args) > 1 else None
        target_mode_index = 2 if "target_mode = $3" in str(query) else None
        target_mode = args[target_mode_index] if target_mode_index is not None and len(args) > target_mode_index else None
        rows = [
            row
            for row in self.rows
            if row["account_id"] == account_id
            and (status is None or row["status"] == status)
            and (target_mode is None or row["target_mode"] == target_mode)
        ]
        return rows

    async def execute(self, query, *args):
        self.execute_calls.append({"query": str(query), "args": args})
        if not self.update_hits:
            return "UPDATE 0"
        updated = 0
        for row in self.rows:
            if row["id"] == args[0] and row["account_id"] == args[2]:
                row["status"] = args[1]
                updated += 1
        return f"UPDATE {updated}"

    async def close(self):
        self.closed = True


async def _return_pool(pool):
    return pool


def _args(**overrides):
    values = {
        "path": SUPPORT_TICKET_CSV,
        "source_format": "csv",
        "target_mode": "vendor_retention",
        "title": "Customer Ticket FAQ",
        "account_id": "acct-smoke",
        "user_id": None,
        "min_source_rows": 2,
        "min_saved_faqs": 1,
        "review_status": "published",
        "export_limit": 20,
        "max_text_chars": 1200,
        "allow_ingestion_warnings": False,
        "default_field": [],
        "output_result": None,
        "json": False,
        "database_url": "postgres://example",
    }
    values.update(overrides)
    return argparse.Namespace(**values)


@pytest.mark.asyncio
async def test_faq_lifecycle_smoke_generates_exports_reviews_and_reexports(monkeypatch):
    pool = _Pool()
    monkeypatch.setattr(smoke, "_create_pool", lambda *_args, **_kwargs: _return_pool(pool))

    code, payload = await smoke.run_faq_lifecycle_smoke(_args())

    assert code == 0
    assert payload["ok"] is True
    assert payload["source_rows"] == 4
    assert payload["saved_ids"] == ["faq-uuid-1"]
    assert payload["generation"]["saved_ids"] == ["faq-uuid-1"]
    assert payload["generation"]["output_checks"] == {
        "uses_user_vocabulary": True,
        "condensed": True,
        "has_action_items": True,
    }
    assert payload["draft_export"]["rows"][0]["id"] == "faq-uuid-1"
    assert payload["draft_export"]["rows"][0]["status"] == "draft"
    assert payload["reviewed_export"]["rows"][0]["id"] == "faq-uuid-1"
    assert payload["reviewed_export"]["rows"][0]["status"] == "published"
    assert "# Customer Ticket FAQ" in payload["reviewed_export"]["rows"][0]["markdown"]
    assert pool.closed is True
    assert pool.execute_calls


@pytest.mark.asyncio
async def test_faq_lifecycle_smoke_persists_1000_row_json_bundle(monkeypatch, tmp_path):
    pool = _Pool()
    monkeypatch.setattr(smoke, "_create_pool", lambda *_args, **_kwargs: _return_pool(pool))
    source = tmp_path / "support_ticket_bundle.json"
    source.write_text(
        json.dumps({
            "support_tickets": [
                {
                    "ticket_id": f"ticket-lifecycle-{index}",
                    "source_type": "support_ticket",
                    "subject": "Billing renewal question",
                    "message": "How do I confirm my renewal invoice before payment?",
                    "pain_category": "billing",
                }
                for index in range(1000)
            ],
        })
        + "\n",
        encoding="utf-8",
    )

    code, payload = await smoke.run_faq_lifecycle_smoke(
        _args(
            path=source,
            source_format="json",
            title="Customer Ticket FAQ Lifecycle Scale Smoke",
            min_source_rows=1000,
            export_limit=5,
            default_field=[
                "company_name=Acme Billing",
                "contact_email=billing@example.com",
                "vendor_name=Atlas Billing",
            ],
        )
    )

    assert code == 0
    assert payload["ok"] is True
    assert payload["source_rows"] == 1000
    assert payload["saved_ids"] == ["faq-uuid-1"]
    assert payload["generation"]["source_count"] == 1000
    assert payload["generation"]["ticket_source_count"] == 1000
    assert payload["generation"]["items"][0]["ticket_count"] == 1000
    assert len(payload["generation"]["items"][0]["source_ids"]) == 1000
    assert payload["generation"]["items"][0]["source_ids"][0] == "ticket-lifecycle-0"
    assert payload["generation"]["items"][0]["source_ids"][-1] == "ticket-lifecycle-999"

    draft = payload["draft_export"]["rows"][0]
    reviewed = payload["reviewed_export"]["rows"][0]
    assert draft["source_count"] == 1000
    assert draft["ticket_source_count"] == 1000
    assert draft["status"] == "draft"
    assert draft["items"][0]["ticket_count"] == 1000
    assert len(draft["items"][0]["source_ids"]) == 1000
    assert reviewed["source_count"] == 1000
    assert reviewed["ticket_source_count"] == 1000
    assert reviewed["status"] == "published"
    assert "# Customer Ticket FAQ Lifecycle Scale Smoke" in reviewed["markdown"]
    assert pool.closed is True


@pytest.mark.asyncio
async def test_faq_lifecycle_smoke_fails_closed_when_table_missing(monkeypatch):
    pool = _Pool(existing_relations=())
    monkeypatch.setattr(smoke, "_create_pool", lambda *_args, **_kwargs: _return_pool(pool))

    code, payload = await smoke.run_faq_lifecycle_smoke(_args())

    assert code == 1
    assert any("ticket_faq_markdown" in error for error in payload["errors"])
    assert payload["generation"] is None
    assert len(pool.fetchval_calls) == 1
    assert pool.execute_calls == []
    assert pool.closed is True


@pytest.mark.asyncio
async def test_faq_lifecycle_smoke_reports_review_status_miss(monkeypatch):
    pool = _Pool(update_hits=False)
    monkeypatch.setattr(smoke, "_create_pool", lambda *_args, **_kwargs: _return_pool(pool))

    code, payload = await smoke.run_faq_lifecycle_smoke(_args())

    assert code == 1
    assert payload["saved_ids"] == ["faq-uuid-1"]
    assert any("review status update missed saved FAQ id" in error for error in payload["errors"])
    assert payload["reviewed_export"] is None
    assert pool.closed is True


def test_faq_lifecycle_smoke_rejects_invalid_args() -> None:
    args = _args(min_saved_faqs=0)

    with pytest.raises(SystemExit, match="--min-saved-faqs must be positive"):
        smoke._validate_args(args)
