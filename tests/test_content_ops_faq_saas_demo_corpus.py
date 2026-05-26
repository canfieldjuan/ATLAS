import csv
from dataclasses import replace
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from extracted_content_pipeline.campaign_ports import TenantScope
from extracted_content_pipeline.campaign_source_adapters import (
    source_rows_to_campaign_opportunities,
)
from extracted_content_pipeline.ticket_faq_markdown import build_ticket_faq_markdown
from extracted_content_pipeline.ticket_faq_search import (
    build_ticket_faq_search_documents,
    search_ticket_faq_documents,
)


ROOT = Path(__file__).resolve().parents[1]
DEMO_PATH = ROOT / "extracted_content_pipeline/examples/support_ticket_saas_demo_sources.csv"
DEMO_FAQ_PATH = ROOT / "extracted_content_pipeline/examples/support_ticket_saas_demo_faq.md"
SEED_SCRIPT = ROOT / "scripts/seed_content_ops_faq_saas_demo.py"
SEED_SPEC = importlib.util.spec_from_file_location(
    "seed_content_ops_faq_saas_demo",
    SEED_SCRIPT,
)
assert SEED_SPEC is not None and SEED_SPEC.loader is not None
seeder = importlib.util.module_from_spec(SEED_SPEC)
sys.modules["seed_content_ops_faq_saas_demo"] = seeder
SEED_SPEC.loader.exec_module(seeder)

EXPECTED_LABEL = "synthetic_b2b_saas_demo"
MIN_ROWS = 36
REQUIRED_PAIN_CATEGORIES = {
    "api and webhooks",
    "billing and plan management",
    "dashboard freshness",
    "data import",
    "integration sync",
    "permissions and seats",
    "reporting export",
    "sso setup",
    "workflow automation",
}
BLOCKED_CONSUMER_FINANCE_TERMS = {
    "bankruptcy",
    "cfpb",
    "credit report",
    "debt collection",
    "escrow",
    "foreclosure",
    "mortgage",
    "payday loan",
}


def _rows() -> list[dict[str, str]]:
    with DEMO_PATH.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _haystack(rows: list[dict[str, str]]) -> str:
    return "\n".join(" ".join(row.values()) for row in rows).lower()


def _generated_demo_faq():
    rows = _rows()
    normalized = source_rows_to_campaign_opportunities(
        rows,
        target_mode="support_account",
    )
    result = build_ticket_faq_markdown(
        normalized.opportunities,
        title="Synthetic B2B SaaS Support FAQ Demo",
        max_items=12,
        max_evidence_per_item=3,
        support_contact="https://example.com/support",
    )
    return rows, normalized, result


def test_saas_demo_corpus_is_labeled_and_domain_clean() -> None:
    rows = _rows()

    assert len(rows) >= MIN_ROWS
    assert {row["Dataset Label"] for row in rows} == {EXPECTED_LABEL}
    assert {row["Source Type"] for row in rows} == {"support_ticket"}
    assert {row["Pain Category"] for row in rows} == REQUIRED_PAIN_CATEGORIES
    assert all(row["Ticket ID"].startswith("saas-demo-") for row in rows)
    assert all(row["Description"].endswith("?") for row in rows)

    corpus_text = _haystack(rows)
    leaked_terms = [
        term for term in sorted(BLOCKED_CONSUMER_FINANCE_TERMS)
        if term in corpus_text
    ]
    assert leaked_terms == []


def test_saas_demo_corpus_generates_valid_faq_output() -> None:
    rows, normalized, result = _generated_demo_faq()

    assert normalized.warnings == ()
    assert len(normalized.opportunities) == len(rows)

    assert result.source_count == len(rows)
    assert result.ticket_source_count == len(rows)
    assert result.output_checks == {
        "condensed": True,
        "has_action_items": True,
        "uses_user_vocabulary": True,
    }
    assert result.items
    assert all(item["source_ids"] for item in result.items)

    rendered_topics = {item["topic"] for item in result.items}
    assert "reporting friction" in rendered_topics
    assert "billing and payments" in rendered_topics
    assert "manual follow-up" in rendered_topics

    markdown = result.markdown.lower()
    leaked_terms = [
        term for term in sorted(BLOCKED_CONSUMER_FINANCE_TERMS)
        if term in markdown
    ]
    assert leaked_terms == []


def test_saas_demo_faq_artifact_matches_real_generator() -> None:
    _rows, _normalized, result = _generated_demo_faq()

    assert DEMO_FAQ_PATH.read_text(encoding="utf-8") == result.markdown


def test_saas_demo_faq_draft_projects_to_search_documents() -> None:
    draft = seeder.build_saas_demo_faq_draft()
    searchable = replace(draft, id="faq-demo-1", status="approved")

    documents = build_ticket_faq_search_documents(
        searchable,
        account_id="acct-demo",
        corpus_id="synthetic-b2b-saas-demo",
    )
    response = search_ticket_faq_documents(
        documents,
        query="export attribution reports",
        account_id="acct-demo",
        corpus_id="synthetic-b2b-saas-demo",
        status="approved",
    )

    assert len(documents) == len(draft.items)
    assert response.as_dict()["count"] >= 1
    first = response.as_dict()["results"][0]
    assert first["account_id"] == "acct-demo"
    assert first["corpus_id"] == "synthetic-b2b-saas-demo"
    assert "export" in first["question"].lower()


def test_saas_demo_seed_args_fail_closed_for_missing_required_values() -> None:
    errors = seeder._validate_args(
        SimpleNamespace(
            database_url="",
            account_id="",
            corpus_id="",
            target_id="",
            status="",
            query="",
            limit=0,
        )
    )

    assert errors == [
        "Missing --database-url, EXTRACTED_DATABASE_URL, or DATABASE_URL",
        "ATLAS_FAQ_SEARCH_ACCOUNT_ID, ATLAS_ACCOUNT_ID, or --account-id is required",
        "--corpus-id is required",
        "--target-id is required",
        "--status is required",
        "--query is required",
        "--limit must be positive",
    ]


def test_saas_demo_cleanup_args_skip_seed_only_required_values() -> None:
    errors = seeder._validate_args(
        SimpleNamespace(
            database_url="postgresql://example",
            account_id="acct-demo",
            cleanup_faq_id="",
            corpus_id="",
            target_id="",
            status="",
            query="",
            limit=0,
        )
    )

    assert errors == ["--cleanup-faq-id must be a non-empty FAQ id"]


def test_saas_demo_cleanup_delete_status_parser() -> None:
    assert seeder._deleted_row_count("DELETE 1") == 1
    assert seeder._deleted_row_count("DELETE 0") == 0
    assert seeder._deleted_row_count("UPDATE 1") is None
    assert seeder._deleted_row_count("DELETE nope") is None
    assert seeder._deleted_row_count(None) is None


@pytest.mark.asyncio
async def test_saas_demo_seeder_saves_approves_projects_and_searches(monkeypatch) -> None:
    class _Pool:
        draft = None
        scope = None
        documents = ()

    class _FAQRepo:
        def __init__(self, pool):
            self.pool = pool

        async def save_drafts(self, drafts, *, scope: TenantScope):
            self.pool.draft = drafts[0]
            self.pool.scope = scope
            return ("11111111-1111-1111-1111-111111111111",)

        async def update_status(self, faq_id, status, *, scope: TenantScope):
            assert faq_id == "11111111-1111-1111-1111-111111111111"
            assert status == "approved"
            assert scope.account_id == "acct-demo"
            self.pool.documents = build_ticket_faq_search_documents(
                replace(self.pool.draft, id=faq_id, status=status),
                account_id=scope.account_id,
                corpus_id=self.pool.draft.metadata["corpus_id"],
            )
            return True

    class _SearchRepo:
        def __init__(self, pool):
            self.pool = pool

        async def search(self, **kwargs):
            return search_ticket_faq_documents(self.pool.documents, **kwargs)

    monkeypatch.setattr(seeder, "PostgresTicketFAQRepository", _FAQRepo)
    monkeypatch.setattr(seeder, "PostgresTicketFAQSearchRepository", _SearchRepo)

    payload = await seeder.seed_saas_demo_faq(
        _Pool(),
        account_id="acct-demo",
        corpus_id="synthetic-b2b-saas-demo",
    )

    assert payload["ok"] is True
    assert payload["errors"] == []
    assert payload["faq_id"] == "11111111-1111-1111-1111-111111111111"
    assert payload["source_count"] >= MIN_ROWS
    assert payload["projected_documents"] == payload["generated_items"]
    assert payload["search"]["count"] >= 1
    assert payload["search"]["matched_seeded_faq"] is True
    assert payload["search"]["first_result"]["faq_id"] == payload["faq_id"]


@pytest.mark.asyncio
async def test_saas_demo_seeder_rejects_search_results_for_different_faq(monkeypatch) -> None:
    class _Response:
        def as_dict(self):
            return {
                "query": "export attribution reports",
                "count": 1,
                "results": [{"faq_id": "different-faq-id"}],
            }

    class _FAQRepo:
        def __init__(self, _pool):
            pass

        async def save_drafts(self, _drafts, *, scope: TenantScope):
            assert scope.account_id == "acct-demo"
            return ("11111111-1111-1111-1111-111111111111",)

        async def update_status(self, _faq_id, _status, *, scope: TenantScope):
            assert scope.account_id == "acct-demo"
            return True

    class _SearchRepo:
        def __init__(self, _pool):
            pass

        async def search(self, **_kwargs):
            return _Response()

    monkeypatch.setattr(seeder, "PostgresTicketFAQRepository", _FAQRepo)
    monkeypatch.setattr(seeder, "PostgresTicketFAQSearchRepository", _SearchRepo)

    payload = await seeder.seed_saas_demo_faq(object(), account_id="acct-demo")

    assert payload["ok"] is False
    assert payload["errors"] == [
        "Seeded SaaS FAQ id was not present in verification search results"
    ]
    assert payload["search"]["matched_seeded_faq"] is False


@pytest.mark.asyncio
async def test_saas_demo_cleanup_deletes_single_faq_for_account() -> None:
    class _Pool:
        def __init__(self) -> None:
            self.calls = []

        async def execute(self, query, *args):
            self.calls.append({"query": query, "args": args})
            return "DELETE 1"

    pool = _Pool()

    payload = await seeder.cleanup_saas_demo_faq(
        pool,
        account_id="acct-demo",
        faq_id="11111111-1111-1111-1111-111111111111",
    )

    assert payload == {
        "phase": "cleanup",
        "ok": True,
        "account_id": "acct-demo",
        "faq_id": "11111111-1111-1111-1111-111111111111",
        "deleted_faq_ids": 1,
        "delete_status": "DELETE 1",
        "error": None,
    }
    assert "WHERE id = $1::uuid" in pool.calls[0]["query"]
    assert "AND account_id = $2" in pool.calls[0]["query"]
    assert pool.calls[0]["args"] == (
        "11111111-1111-1111-1111-111111111111",
        "acct-demo",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("delete_status", "expected"),
    [
        (
            "DELETE 0",
            {
                "deleted_faq_ids": 0,
                "error": "cleanup deleted 0 FAQ rows but expected 1",
            },
        ),
        (
            "UPDATE 1",
            {
                "deleted_faq_ids": None,
                "error": "cleanup delete status is not parseable: 'UPDATE 1'",
            },
        ),
    ],
)
async def test_saas_demo_cleanup_fails_on_bad_delete_result(delete_status, expected) -> None:
    class _Pool:
        async def execute(self, _query, *_args):
            return delete_status

    payload = await seeder.cleanup_saas_demo_faq(
        _Pool(),
        account_id="acct-demo",
        faq_id="11111111-1111-1111-1111-111111111111",
    )

    assert payload["ok"] is False
    assert payload["deleted_faq_ids"] == expected["deleted_faq_ids"]
    assert payload["error"] == expected["error"]
