import csv
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import shlex
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
RUNBOOK_PATH = ROOT / "docs/extraction/validation/content_ops_faq_saas_demo_route_case_runbook.md"
SEED_SCRIPT = ROOT / "scripts/seed_content_ops_faq_saas_demo.py"
ROUTE_SCRIPT = ROOT / "scripts/smoke_content_ops_faq_search_route_concurrency.py"
E2E_SCRIPT = ROOT / "scripts/smoke_content_ops_faq_saas_demo_route_e2e.py"
MIGRATION_SCRIPT = ROOT / "scripts/run_extracted_content_pipeline_migrations.py"
SEED_SPEC = importlib.util.spec_from_file_location(
    "seed_content_ops_faq_saas_demo",
    SEED_SCRIPT,
)
assert SEED_SPEC is not None and SEED_SPEC.loader is not None
seeder = importlib.util.module_from_spec(SEED_SPEC)
sys.modules["seed_content_ops_faq_saas_demo"] = seeder
SEED_SPEC.loader.exec_module(seeder)
ROUTE_SPEC = importlib.util.spec_from_file_location(
    "smoke_content_ops_faq_search_route_concurrency_for_saas_demo_test",
    ROUTE_SCRIPT,
)
assert ROUTE_SPEC is not None and ROUTE_SPEC.loader is not None
route_smoke = importlib.util.module_from_spec(ROUTE_SPEC)
ROUTE_SPEC.loader.exec_module(route_smoke)
E2E_SPEC = importlib.util.spec_from_file_location(
    "smoke_content_ops_faq_saas_demo_route_e2e_for_saas_demo_test",
    E2E_SCRIPT,
)
assert E2E_SPEC is not None and E2E_SPEC.loader is not None
e2e_smoke = importlib.util.module_from_spec(E2E_SPEC)
E2E_SPEC.loader.exec_module(e2e_smoke)
MIGRATION_SPEC = importlib.util.spec_from_file_location(
    "run_extracted_content_pipeline_migrations_for_saas_demo_test",
    MIGRATION_SCRIPT,
)
assert MIGRATION_SPEC is not None and MIGRATION_SPEC.loader is not None
migration_cli = importlib.util.module_from_spec(MIGRATION_SPEC)
MIGRATION_SPEC.loader.exec_module(migration_cli)

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


def _clear_atlas_db_env(monkeypatch) -> None:
    for name in (
        "ATLAS_DB_HOST",
        "ATLAS_DB_PORT",
        "ATLAS_DB_DATABASE",
        "ATLAS_DB_USER",
        "ATLAS_DB_PASSWORD",
        "ATLAS_DB_SOCKET_PATH",
    ):
        monkeypatch.delenv(name, raising=False)


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


def _runbook_command_args(marker: str, *, section: str | None = None) -> list[str]:
    doc = RUNBOOK_PATH.read_text(encoding="utf-8")
    search_from = doc.index(section) if section is not None else 0
    start = doc.index(marker, search_from)
    end = doc.index("```", start)
    command = doc[start:end].replace("\\\n", " ")
    parts = shlex.split(command)
    assert parts[:2] == ["python", marker.removeprefix("python ")]
    return parts[2:]


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


def test_saas_demo_route_case_runbook_migration_command_matches_parser(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _runbook_command_args(
        "python scripts/run_extracted_content_pipeline_migrations.py"
    )
    monkeypatch.delenv("EXTRACTED_DATABASE_URL", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)

    parsed = migration_cli._parse_args(args)
    doc = RUNBOOK_PATH.read_text(encoding="utf-8")

    assert parsed.database_url is None
    assert parsed.dry_run is False
    assert parsed.json is False
    assert "extracted_content_pipeline/storage/migration_runner.py --apply" not in doc
    assert MIGRATION_SCRIPT.exists()


def test_saas_demo_route_case_runbook_e2e_command_matches_parser() -> None:
    args = _runbook_command_args(
        "python scripts/smoke_content_ops_faq_saas_demo_route_e2e.py",
        section="## Recommended One-Command Smoke",
    )

    parsed = e2e_smoke._build_parser().parse_args(args)

    assert parsed.database_url == "${EXTRACTED_DATABASE_URL:-$DATABASE_URL}"
    assert parsed.base_url == "$ATLAS_API_BASE_URL"
    assert parsed.token == "${ATLAS_B2B_JWT:-$ATLAS_TOKEN}"
    assert parsed.account_id == "${ATLAS_FAQ_SEARCH_ACCOUNT_ID:-$ATLAS_ACCOUNT_ID}"
    assert parsed.route_requests == 40
    assert parsed.concurrency == 8
    assert parsed.max_error_rate == 0
    assert parsed.max_case_error_rate == 0
    assert parsed.max_detail_ms == 2500
    assert parsed.artifact_dir == Path("/tmp/faq-saas-demo-route-e2e-artifacts")
    assert parsed.output_result == Path("/tmp/faq-saas-demo-route-e2e-result.json")
    parsed.base_url = "https://atlas.example.com"
    assert e2e_smoke._validate_args(parsed) == []


def test_saas_demo_route_case_runbook_preflight_command_matches_parser() -> None:
    args = _runbook_command_args(
        "python scripts/smoke_content_ops_faq_saas_demo_route_e2e.py",
        section="## Hosted Run Blocker Preflight",
    )

    parsed = e2e_smoke._build_parser().parse_args(args)

    assert parsed.database_url == "${EXTRACTED_DATABASE_URL:-$DATABASE_URL}"
    assert parsed.base_url == "$ATLAS_API_BASE_URL"
    assert parsed.token == "${ATLAS_B2B_JWT:-$ATLAS_TOKEN}"
    assert parsed.account_id == "${ATLAS_FAQ_SEARCH_ACCOUNT_ID:-$ATLAS_ACCOUNT_ID}"
    assert parsed.preflight_only is True
    assert parsed.json is True
    assert parsed.artifact_dir is None
    assert parsed.output_result == Path("/tmp/faq-saas-demo-route-e2e-preflight.json")
    parsed.base_url = "https://atlas.example.com"
    assert e2e_smoke._validate_args(parsed) == []


def test_saas_demo_route_case_runbook_seed_command_matches_parser() -> None:
    args = _runbook_command_args("python scripts/seed_content_ops_faq_saas_demo.py")

    parsed = seeder._parse_args(args)

    assert parsed.database_url == "${EXTRACTED_DATABASE_URL:-$DATABASE_URL}"
    assert parsed.account_id == "${ATLAS_FAQ_SEARCH_ACCOUNT_ID:-$ATLAS_ACCOUNT_ID}"
    assert parsed.route_case_file_output == Path("/tmp/faq-saas-demo-route-cases.json")
    assert parsed.output_result == Path("/tmp/faq-saas-demo-seed-result.json")
    assert parsed.cleanup_faq_id is None
    assert seeder._validate_args(parsed) == []


def test_saas_demo_route_case_runbook_route_command_matches_parser() -> None:
    args = _runbook_command_args(
        "python scripts/smoke_content_ops_faq_search_route_concurrency.py"
    )

    parsed = route_smoke._build_parser().parse_args(args)

    assert parsed.base_url == "$ATLAS_API_BASE_URL"
    assert parsed.token == "${ATLAS_B2B_JWT:-$ATLAS_TOKEN}"
    assert parsed.case_file == Path("/tmp/faq-saas-demo-route-cases.json")
    assert parsed.require_detail is True
    assert parsed.max_error_rate == 0
    assert parsed.max_case_error_rate == 0
    assert parsed.max_detail_ms == 2500
    assert parsed.output_result == Path("/tmp/faq-saas-demo-route-result.json")
    assert route_smoke._validate_args(parsed) == []


def test_saas_demo_route_case_runbook_commands_share_case_file() -> None:
    seed_args = seeder._parse_args(
        _runbook_command_args("python scripts/seed_content_ops_faq_saas_demo.py")
    )
    route_args = route_smoke._build_parser().parse_args(
        _runbook_command_args(
            "python scripts/smoke_content_ops_faq_search_route_concurrency.py"
        )
    )

    assert seed_args.route_case_file_output == route_args.case_file
    assert route_args.require_detail is True
    assert route_args.max_case_single_request_ms == 3000


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


def test_saas_demo_seed_default_database_url_prefers_url_env(monkeypatch) -> None:
    monkeypatch.setenv("EXTRACTED_DATABASE_URL", "postgresql://env/atlas")
    monkeypatch.setenv("DATABASE_URL", "postgresql://database-url/atlas")

    assert seeder._default_database_url() == "postgresql://env/atlas"


def test_saas_demo_seed_default_database_url_falls_back_to_atlas_db_settings(monkeypatch) -> None:
    monkeypatch.delenv("EXTRACTED_DATABASE_URL", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("ATLAS_DB_HOST", "settings-host")
    monkeypatch.setenv("ATLAS_DB_PORT", "6543")
    monkeypatch.setenv("ATLAS_DB_DATABASE", "atlas_settings")
    monkeypatch.setenv("ATLAS_DB_USER", "atlas_user")
    monkeypatch.setenv("ATLAS_DB_PASSWORD", "atlas_pass")
    monkeypatch.delenv("ATLAS_DB_SOCKET_PATH", raising=False)

    assert (
        seeder._default_database_url()
        == "postgresql://atlas_user:atlas_pass@settings-host:6543/atlas_settings"
    )


def test_saas_demo_seed_default_database_url_ignores_implicit_atlas_db_defaults(monkeypatch) -> None:
    monkeypatch.delenv("EXTRACTED_DATABASE_URL", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    _clear_atlas_db_env(monkeypatch)

    assert seeder._default_database_url() == ""


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


def test_saas_demo_route_case_output_is_seed_only() -> None:
    errors = seeder._validate_args(
        SimpleNamespace(
            database_url="postgresql://example",
            account_id="acct-demo",
            cleanup_faq_id="11111111-1111-1111-1111-111111111111",
            route_case_file_output=Path("route-cases.json"),
            corpus_id="",
            target_id="",
            status="",
            query="",
            limit=0,
        )
    )

    assert errors == ["--route-case-file-output is only available in seed mode"]


def test_saas_demo_seed_preflight_writes_result_before_exit(tmp_path) -> None:
    result_path = tmp_path / "seed-preflight.json"

    with pytest.raises(SystemExit) as exc:
        seeder.main([
            "--database-url",
            "",
            "--account-id",
            "",
            "--output-result",
            str(result_path),
            "--json",
        ])

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert str(exc.value) == (
        "Missing --database-url, EXTRACTED_DATABASE_URL, or DATABASE_URL; "
        "ATLAS_FAQ_SEARCH_ACCOUNT_ID, ATLAS_ACCOUNT_ID, or --account-id is required"
    )
    assert payload == {
        "phase": "preflight",
        "ok": False,
        "mode": "seed",
        "errors": [
            "Missing --database-url, EXTRACTED_DATABASE_URL, or DATABASE_URL",
            "ATLAS_FAQ_SEARCH_ACCOUNT_ID, ATLAS_ACCOUNT_ID, or --account-id is required",
        ],
    }


def test_saas_demo_cleanup_preflight_writes_result_before_exit(tmp_path) -> None:
    result_path = tmp_path / "cleanup-preflight.json"

    with pytest.raises(SystemExit) as exc:
        seeder.main([
            "--database-url",
            "postgresql://example",
            "--account-id",
            "acct-demo",
            "--cleanup-faq-id",
            "",
            "--output-result",
            str(result_path),
            "--json",
        ])

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert str(exc.value) == "--cleanup-faq-id must be a non-empty FAQ id"
    assert payload == {
        "phase": "preflight",
        "ok": False,
        "mode": "cleanup",
        "errors": ["--cleanup-faq-id must be a non-empty FAQ id"],
    }


def test_saas_demo_seed_runtime_failure_writes_sanitized_result(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    database_url = "postgresql://user:secret@example.invalid/atlas"
    result_path = tmp_path / "seed-runtime.json"

    async def _raise_pool(database_url_arg):
        raise RuntimeError(f"could not connect to {database_url_arg}")

    monkeypatch.setattr(seeder, "_create_pool", _raise_pool)

    code = seeder.main([
        "--database-url",
        database_url,
        "--account-id",
        "acct-demo",
        "--output-result",
        str(result_path),
        "--json",
    ])

    output = capsys.readouterr().out
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert code == 1
    assert database_url not in output
    assert database_url not in result_path.read_text(encoding="utf-8")
    assert payload == {
        "phase": "runtime",
        "ok": False,
        "mode": "seed",
        "errors": [
            "RuntimeError: could not connect to [redacted-database-url]"
        ],
        "error": {
            "type": "RuntimeError",
            "message": "could not connect to [redacted-database-url]",
        },
    }


def test_saas_demo_cleanup_runtime_failure_writes_result(
    tmp_path,
    monkeypatch,
) -> None:
    class _Pool:
        async def close(self):
            return None

    async def _fake_pool(_database_url):
        return _Pool()

    async def _raise_cleanup(*_args, **_kwargs):
        raise RuntimeError("cleanup failed")

    monkeypatch.setattr(seeder, "_create_pool", _fake_pool)
    monkeypatch.setattr(seeder, "cleanup_saas_demo_faq", _raise_cleanup)
    result_path = tmp_path / "cleanup-runtime.json"

    code = seeder.main([
        "--database-url",
        "postgresql://example/atlas",
        "--account-id",
        "acct-demo",
        "--cleanup-faq-id",
        "11111111-1111-1111-1111-111111111111",
        "--output-result",
        str(result_path),
        "--json",
    ])

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert code == 1
    assert payload == {
        "phase": "runtime",
        "ok": False,
        "mode": "cleanup",
        "errors": ["RuntimeError: cleanup failed"],
        "error": {
            "type": "RuntimeError",
            "message": "cleanup failed",
        },
    }


def test_saas_demo_seed_preserves_payload_when_pool_close_fails(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    database_url = "postgresql://user:secret@example.invalid/atlas"

    class _Pool:
        async def close(self):
            raise OSError(f"pool close failed for {database_url}")

    async def _fake_pool(_database_url):
        return _Pool()

    async def _seed_success(*_args, **_kwargs):
        return {
            "phase": "seed",
            "ok": True,
            "errors": [],
            "account_id": "acct-demo",
            "corpus_id": "synthetic-b2b-saas-demo",
            "faq_id": "11111111-1111-1111-1111-111111111111",
            "status": "approved",
            "source_count": 36,
            "ticket_source_count": 36,
            "generated_items": 7,
            "projected_documents": 7,
            "search": {"query": "export attribution reports", "count": 1},
        }

    monkeypatch.setattr(seeder, "_create_pool", _fake_pool)
    monkeypatch.setattr(seeder, "seed_saas_demo_faq", _seed_success)
    result_path = tmp_path / "seed-close.json"

    code = seeder.main([
        "--database-url",
        database_url,
        "--account-id",
        "acct-demo",
        "--output-result",
        str(result_path),
        "--json",
    ])

    output = capsys.readouterr().out
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert code == 1
    assert database_url not in output
    assert database_url not in result_path.read_text(encoding="utf-8")
    assert payload["phase"] == "seed"
    assert payload["faq_id"] == "11111111-1111-1111-1111-111111111111"
    assert payload["search"] == {"query": "export attribution reports", "count": 1}
    assert payload["ok"] is False
    assert payload["errors"] == []
    assert payload["pool_close"] == {
        "ok": False,
        "attempted": True,
        "error": {
            "type": "OSError",
            "message": "pool close failed for [redacted-database-url]",
        },
    }


def test_saas_demo_cleanup_preserves_payload_when_pool_close_fails(
    tmp_path,
    monkeypatch,
) -> None:
    class _Pool:
        async def close(self):
            raise OSError("pool close failed")

    async def _fake_pool(_database_url):
        return _Pool()

    async def _cleanup_success(*_args, **_kwargs):
        return {
            "phase": "cleanup",
            "ok": True,
            "errors": [],
            "account_id": "acct-demo",
            "faq_id": "11111111-1111-1111-1111-111111111111",
            "deleted_faq_ids": 1,
            "delete_status": "DELETE 1",
        }

    monkeypatch.setattr(seeder, "_create_pool", _fake_pool)
    monkeypatch.setattr(seeder, "cleanup_saas_demo_faq", _cleanup_success)
    result_path = tmp_path / "cleanup-close.json"

    code = seeder.main([
        "--database-url",
        "postgresql://example/atlas",
        "--account-id",
        "acct-demo",
        "--cleanup-faq-id",
        "11111111-1111-1111-1111-111111111111",
        "--output-result",
        str(result_path),
        "--json",
    ])

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert code == 1
    assert payload["phase"] == "cleanup"
    assert payload["faq_id"] == "11111111-1111-1111-1111-111111111111"
    assert payload["deleted_faq_ids"] == 1
    assert payload["ok"] is False
    assert payload["errors"] == []
    assert payload["pool_close"] == {
        "ok": False,
        "attempted": True,
        "error": {
            "type": "OSError",
            "message": "pool close failed",
        },
    }


def test_saas_demo_cleanup_delete_status_parser() -> None:
    assert seeder._deleted_row_count("DELETE 1") == 1
    assert seeder._deleted_row_count("DELETE 0") == 0
    assert seeder._deleted_row_count("UPDATE 1") is None
    assert seeder._deleted_row_count("DELETE nope") is None
    assert seeder._deleted_row_count(None) is None


@pytest.mark.asyncio
async def test_saas_demo_seeder_saves_approves_projects_and_searches(
    monkeypatch,
    tmp_path,
) -> None:
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
        limit=7,
        route_case_file_output=tmp_path / "route-cases.json",
    )

    assert payload["ok"] is True
    assert payload["errors"] == []
    assert payload["faq_id"] == "11111111-1111-1111-1111-111111111111"
    assert payload["target_id"] == "support-synthetic-b2b-saas-demo"
    assert payload["target_mode"] == "support_account"
    assert payload["limit"] == 7
    assert payload["source_count"] >= MIN_ROWS
    assert payload["projected_documents"] == payload["generated_items"]
    assert payload["search"]["count"] >= 1
    assert payload["search"]["matched_seeded_faq"] is True
    assert payload["search"]["first_result"]["faq_id"] == payload["faq_id"]
    assert payload["route_case_file"] == {
        "ok": True,
        "path": str(tmp_path / "route-cases.json"),
        "cases": 1,
    }
    assert json.loads((tmp_path / "route-cases.json").read_text(encoding="utf-8")) == [
        {
            "corpus_id": "synthetic-b2b-saas-demo",
            "expected_detail_account_id": "acct-demo",
            "expected_detail_status": "approved",
            "expected_detail_target_id": "support-synthetic-b2b-saas-demo",
            "expected_detail_target_mode": "support_account",
            "expected_detail_title": "Synthetic B2B SaaS Support FAQ Demo",
            "expected_first_account_id": "acct-demo",
            "expected_first_corpus_id": "synthetic-b2b-saas-demo",
            "expected_first_faq_id": "11111111-1111-1111-1111-111111111111",
            "limit": 7,
            "query": "export attribution reports",
            "require_results": True,
            "status": "approved",
        }
    ]


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
async def test_saas_demo_seeder_reports_route_case_write_failure(monkeypatch, tmp_path) -> None:
    class _Response:
        def as_dict(self):
            return {
                "query": "export attribution reports",
                "count": 1,
                "results": [{"faq_id": "11111111-1111-1111-1111-111111111111"}],
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

    def _raise_write(_path, _payload):
        raise OSError("disk full")

    monkeypatch.setattr(seeder, "PostgresTicketFAQRepository", _FAQRepo)
    monkeypatch.setattr(seeder, "PostgresTicketFAQSearchRepository", _SearchRepo)
    monkeypatch.setattr(seeder, "_write_route_case_file", _raise_write)

    route_case_file = tmp_path / "route-cases.json"
    payload = await seeder.seed_saas_demo_faq(
        object(),
        account_id="acct-demo",
        route_case_file_output=route_case_file,
    )

    assert payload["ok"] is False
    assert payload["faq_id"] == "11111111-1111-1111-1111-111111111111"
    assert payload["errors"] == ["route case file could not be written: disk full"]
    assert payload["route_case_file"] == {
        "ok": False,
        "path": str(route_case_file),
        "error": "disk full",
    }


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
        "errors": [],
        "account_id": "acct-demo",
        "faq_id": "11111111-1111-1111-1111-111111111111",
        "deleted_faq_ids": 1,
        "delete_status": "DELETE 1",
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
                "errors": ["cleanup deleted 0 FAQ rows but expected 1"],
            },
        ),
        (
            "UPDATE 1",
            {
                "deleted_faq_ids": None,
                "errors": ["cleanup delete status is not parseable: 'UPDATE 1'"],
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
    assert payload["errors"] == expected["errors"]
