from __future__ import annotations

import json
import importlib.util
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/content_ops_source_postgres_smoke_helpers.py"
SPEC = importlib.util.spec_from_file_location(
    "content_ops_source_postgres_smoke_helpers",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
helpers = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(helpers)


class _Pool:
    def __init__(self, *, existing_relations=None, rows=None):
        self.existing_relations = set(
            existing_relations or ("campaign_opportunities", "b2b_campaigns")
        )
        self.rows = list(rows or [])
        self.fetch_calls = []
        self.fetchval_calls = []

    async def fetchval(self, query, *args):
        self.fetchval_calls.append({"query": str(query), "args": args})
        if "to_regclass" in str(query):
            return args[0] if args and args[0] in self.existing_relations else None
        return None

    async def fetch(self, query, *args):
        self.fetch_calls.append({"query": str(query), "args": args})
        return self.rows


class _GenerationResult:
    def __init__(self, data: dict[str, Any]):
        self._data = dict(data)

    def as_dict(self):
        return dict(self._data)


@pytest.mark.asyncio
async def test_schema_readiness_errors_reports_missing_tables():
    pool = _Pool(existing_relations={"b2b_campaigns"})

    errors = await helpers.schema_readiness_errors(
        pool,
        opportunity_table="campaign_opportunities",
    )

    assert len(errors) == 1
    assert "campaign_opportunities" in errors[0]
    assert "run_extracted_content_pipeline_migrations.py" in errors[0]


@pytest.mark.asyncio
async def test_fetch_saved_drafts_reads_target_id_from_metadata_json():
    pool = _Pool(rows=[
        {
            "id": "campaign-1",
            "subject": "Subject",
            "body": "Body",
            "target_mode": "vendor_retention",
            "channel": "email_cold",
            "metadata": json.dumps({
                "source_opportunity": {
                    "target_id": "source-1",
                }
            }),
        }
    ])

    drafts = await helpers.fetch_saved_drafts(pool, ["campaign-1"])

    assert drafts == [{
        "id": "campaign-1",
        "target_id": "source-1",
        "subject": "Subject",
        "body": "Body",
        "target_mode": "vendor_retention",
        "channel": "email_cold",
    }]
    assert "FROM b2b_campaigns" in pool.fetch_calls[0]["query"]


def test_draft_errors_enforces_minimum_required_fields_and_forbidden_phrases():
    errors = helpers.draft_errors(
        {
            "drafts": [
                {
                    "subject": "Subject",
                    "body": "Acme appears to be weighing a switch.",
                    "target_id": "",
                    "channel": "email_cold",
                }
            ]
        },
        min_drafts=1,
        forbidden_phrases=["appears to be weighing"],
    )

    assert "draft 1 missing target_id" in errors
    assert "draft 1 contains forbidden phrase: appears to be weighing" in errors


def test_saved_draft_target_errors_require_imported_target_id():
    errors = helpers.saved_draft_target_errors(
        [
            {"id": "campaign-1", "target_id": ""},
            {"id": "campaign-2", "target_id": "other-source"},
        ],
        ["source-1"],
    )

    assert "persisted draft missing target_id metadata: campaign-1" in errors
    assert "persisted draft target_id was not imported: other-source" in errors


@pytest.mark.asyncio
async def test_generate_imported_target_drafts_aggregates_multiple_targets(monkeypatch):
    calls = []

    async def fake_generate(*_args, **kwargs):
        calls.append(kwargs)
        index = len(calls)
        return _GenerationResult({
            "requested": 1,
            "generated": 1,
            "skipped": index - 1,
            "reasoning_contexts_used": index,
            "saved_ids": [f"campaign-{index}"],
            "errors": [{"target_id": "source-2", "reason": "bad"}] if index == 2 else [],
        })

    monkeypatch.setattr(helpers, "generate_campaign_drafts_from_postgres", fake_generate)

    result = await helpers.generate_imported_target_drafts(
        pool=object(),
        account_id="acct",
        user_id=None,
        target_mode="vendor_retention",
        channels=("email_cold", "email_followup"),
        target_ids=("source-1", "source-2"),
        opportunity_table="campaign_opportunities",
    )

    assert result["requested"] == 2
    assert result["generated"] == 2
    assert result["skipped"] == 1
    assert result["reasoning_contexts_used"] == 3
    assert result["saved_ids"] == ["campaign-1", "campaign-2"]
    assert result["errors"] == [{"target_id": "source-2", "reason": "bad"}]
    assert calls[0]["filters"] == {"target_id": "source-1"}
    assert calls[0]["channels"] == ("email_cold", "email_followup")
