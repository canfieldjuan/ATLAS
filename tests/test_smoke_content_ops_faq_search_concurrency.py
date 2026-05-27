from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/smoke_content_ops_faq_search_concurrency.py"
SPEC = importlib.util.spec_from_file_location("smoke_content_ops_faq_search_concurrency", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
smoke = importlib.util.module_from_spec(SPEC)
sys.modules["smoke_content_ops_faq_search_concurrency"] = smoke
SPEC.loader.exec_module(smoke)


def _case(*, query: str = "export attribution report", expected_hit: bool = True):
    return smoke.SearchCase(
        account_id="acct-1",
        corpus_id="corpus-1",
        faq_id="11111111-1111-1111-1111-111111111111",
        query=query,
        expected_hit=expected_hit,
    )


def test_build_cases_creates_hit_and_miss_for_each_corpus() -> None:
    cases = smoke._build_cases(
        run_id="abc123",
        account_count=2,
        corpora_per_account=3,
    )

    assert len(cases) == 12
    assert sum(1 for case in cases if case.expected_hit) == 6
    assert sum(1 for case in cases if not case.expected_hit) == 6
    assert len({case.account_id for case in cases}) == 2
    assert len({case.corpus_id for case in cases}) == 3
    assert all(case.account_id.startswith("faq-search-abc123-acct-") for case in cases)
    assert {case.query for case in cases} == {
        "export attribution report",
        "saml domain verification",
    }
    assert not any("escrow" in case.query for case in cases)
    for corpus_id in {case.corpus_id for case in cases}:
        assert len({case.account_id for case in cases if case.corpus_id == corpus_id}) == 2


def test_build_cases_accepts_single_account_override() -> None:
    cases = smoke._build_cases(
        run_id="abc123",
        account_count=1,
        corpora_per_account=2,
        account_id="hosted-token-account",
    )

    assert {case.account_id for case in cases} == {"hosted-token-account"}
    assert len(cases) == 4


def test_documents_for_case_are_ranked_and_scoped() -> None:
    documents = smoke._documents_for_case(_case(), documents_per_corpus=3)

    assert [document.rank for document in documents] == [1, 2, 3]
    assert {document.account_id for document in documents} == {"acct-1"}
    assert {document.corpus_id for document in documents} == {"corpus-1"}
    assert {document.status for document in documents} == {"approved"}
    assert {document.target_mode for document in documents} == {"support_account"}
    assert all("export attribution report" in document.search_text for document in documents)


def test_route_case_payload_carries_seeded_hit_and_miss_expectations() -> None:
    payload = smoke._route_case_payload(
        [_case(), _case(query="saml domain verification", expected_hit=False)],
        documents_per_corpus=7,
        limit=5,
    )

    assert payload[0]["expected_count"] == 5
    assert payload[0]["expected_first_account_id"] == "acct-1"
    assert payload[0]["expected_first_corpus_id"] == "corpus-1"
    assert payload[0]["expected_first_faq_id"] == "11111111-1111-1111-1111-111111111111"
    assert payload[0]["expected_detail_account_id"] == "acct-1"
    assert payload[0]["expected_detail_target_id"] == "support-corpus-1"
    assert payload[0]["expected_detail_target_mode"] == "support_account"
    assert payload[0]["expected_detail_title"] == "FAQ Search Smoke"
    assert payload[0]["expected_detail_status"] == "approved"
    assert payload[1] == {
        "query": "saml domain verification",
        "corpus_id": "corpus-1",
        "status": "approved",
        "limit": 5,
        "require_results": False,
        "expected_count": 0,
    }


def test_write_route_case_file_writes_deterministic_json(tmp_path) -> None:
    path = tmp_path / "route-cases.json"
    smoke._write_route_case_file(path, [_case()], documents_per_corpus=1)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload[0]["expected_count"] == 1
    assert payload[0]["expected_first_faq_id"] == "11111111-1111-1111-1111-111111111111"


def test_cleanup_manifest_payload_carries_all_seeded_ids() -> None:
    cases = [
        _case(),
        _case(query="saml domain verification", expected_hit=False),
        smoke.SearchCase(
            account_id="acct-1",
            corpus_id="corpus-2",
            faq_id="22222222-2222-2222-2222-222222222222",
            query="export attribution report",
            expected_hit=True,
        ),
    ]

    assert smoke._cleanup_manifest_payload(cases) == {
        "account_ids": ["acct-1"],
        "corpus_ids": ["corpus-1", "corpus-2"],
        "faq_ids": [
            "11111111-1111-1111-1111-111111111111",
            "22222222-2222-2222-2222-222222222222",
        ],
        "search_cases": 3,
    }


def test_write_cleanup_manifest_writes_deterministic_json(tmp_path) -> None:
    path = tmp_path / "cleanup-manifest.json"

    smoke._write_cleanup_manifest(path, [_case()])

    assert json.loads(path.read_text(encoding="utf-8")) == {
        "account_ids": ["acct-1"],
        "corpus_ids": ["corpus-1"],
        "faq_ids": ["11111111-1111-1111-1111-111111111111"],
        "search_cases": 1,
    }


@pytest.mark.asyncio
async def test_run_case_records_failures_for_leaked_rows_and_hit_miss_mismatches() -> None:
    class _Response:
        def __init__(self, rows):
            self.rows = rows

        def as_dict(self):
            return {"results": self.rows}

    class _Repo:
        def __init__(self, rows):
            self.rows = rows

        async def search(self, **_kwargs):
            return _Response(self.rows)

    hit_case = smoke.SearchCase(
        account_id="acct-1",
        corpus_id="shared-corpus",
        faq_id="faq-1",
        query="export attribution report",
        expected_hit=True,
    )
    miss_case = smoke.SearchCase(
        account_id="acct-1",
        corpus_id="shared-corpus",
        faq_id="faq-1",
        query="saml domain verification",
        expected_hit=False,
    )

    leaked = await smoke._run_case(
        _Repo([{"account_id": "acct-2", "corpus_id": "other-corpus"}]),
        hit_case,
        limit=5,
    )
    empty_hit = await smoke._run_case(_Repo([]), hit_case, limit=5)
    unexpected_miss = await smoke._run_case(
        _Repo([{"account_id": "acct-1", "corpus_id": "shared-corpus"}]),
        miss_case,
        limit=5,
    )

    assert leaked["failures"] == [
        "wrong account_id returned",
        "wrong corpus_id returned",
    ]
    assert empty_hit["failures"] == ["expected hit returned no rows"]
    assert unexpected_miss["failures"] == ["expected miss returned rows"]


def test_latency_summary_reports_empty_and_percentiles() -> None:
    assert smoke._latency_summary([]) == {
        "count": 0,
        "p50_ms": 0.0,
        "p95_ms": 0.0,
        "max_ms": 0.0,
    }

    summary = smoke._latency_summary([
        {"elapsed_ms": 4.0},
        {"elapsed_ms": 1.0},
        {"elapsed_ms": 3.0},
        {"elapsed_ms": 2.0},
    ])

    assert summary == {
        "count": 4,
        "p50_ms": 2.5,
        "p95_ms": 4.0,
        "max_ms": 4.0,
    }


def test_latency_budget_summary_reports_passes_and_failures() -> None:
    latency = {"p95_ms": 4.0, "max_ms": 5.0}

    assert smoke._latency_budget_summary(
        latency,
        max_p95_ms=None,
        max_single_request_ms=None,
    ) == {"ok": True, "checks": [], "failures": []}

    passing = smoke._latency_budget_summary(
        latency,
        max_p95_ms=4.0,
        max_single_request_ms=5.0,
    )
    failing = smoke._latency_budget_summary(
        latency,
        max_p95_ms=3.5,
        max_single_request_ms=4.5,
    )

    assert passing == {
        "ok": True,
        "checks": [
            {"metric": "p95_ms", "actual_ms": 4.0, "max_ms": 4.0, "ok": True},
            {"metric": "max_ms", "actual_ms": 5.0, "max_ms": 5.0, "ok": True},
        ],
        "failures": [],
    }
    assert failing["ok"] is False
    assert failing["failures"] == [
        "p95_ms exceeded 3.5 ms",
        "max_ms exceeded 4.5 ms",
    ]


def test_validate_args_rejects_nonpositive_latency_budgets() -> None:
    args = SimpleNamespace(
        database_url="postgresql://example",
        account_count=1,
        corpora_per_account=1,
        documents_per_corpus=1,
        iterations=1,
        concurrency=1,
        pool_size=1,
        max_p95_ms=0,
        max_single_request_ms=None,
    )

    with pytest.raises(SystemExit, match="--max-p95-ms must be positive"):
        smoke._validate_args(args)


def test_validate_args_rejects_route_case_output_without_kept_data() -> None:
    args = SimpleNamespace(
        database_url="postgresql://example",
        account_count=1,
        account_id="acct-1",
        corpora_per_account=1,
        documents_per_corpus=1,
        iterations=1,
        concurrency=1,
        pool_size=1,
        max_p95_ms=None,
        max_single_request_ms=None,
        route_case_file_output=Path("cases.json"),
        cleanup_manifest_output=None,
        keep_data=False,
    )

    with pytest.raises(SystemExit, match="--route-case-file-output requires --keep-data"):
        smoke._validate_args(args)


def test_validate_args_rejects_account_override_for_multiple_accounts() -> None:
    args = SimpleNamespace(
        database_url="postgresql://example",
        account_count=2,
        account_id="acct-1",
        corpora_per_account=1,
        documents_per_corpus=1,
        iterations=1,
        concurrency=1,
        pool_size=1,
        max_p95_ms=None,
        max_single_request_ms=None,
        route_case_file_output=None,
        cleanup_manifest_output=None,
        keep_data=True,
    )

    with pytest.raises(SystemExit, match="--account-id requires --account-count 1"):
        smoke._validate_args(args)


def test_validate_args_rejects_cleanup_manifest_without_kept_data() -> None:
    args = SimpleNamespace(
        database_url="postgresql://example",
        account_count=1,
        account_id="acct-1",
        corpora_per_account=1,
        documents_per_corpus=1,
        iterations=1,
        concurrency=1,
        pool_size=1,
        max_p95_ms=None,
        max_single_request_ms=None,
        route_case_file_output=None,
        cleanup_manifest_output=Path("cleanup.json"),
        keep_data=False,
    )

    with pytest.raises(SystemExit, match="--cleanup-manifest-output requires --keep-data"):
        smoke._validate_args(args)


def test_main_writes_result_for_preflight_failure(tmp_path) -> None:
    result_path = tmp_path / "faq-search-preflight.json"

    code = smoke.main([
        "--database-url",
        "",
        "--account-count",
        "1",
        "--corpora-per-account",
        "1",
        "--documents-per-corpus",
        "1",
        "--iterations",
        "1",
        "--concurrency",
        "1",
        "--pool-size",
        "1",
        "--output-result",
        str(result_path),
        "--json",
    ])

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert code == 2
    assert payload["ok"] is False
    assert payload["run_id"] == "preflight"
    assert payload["requests"]["total"] == 0
    assert payload["setup"] == {
        "ok": False,
        "phase": "preflight",
        "error": {
            "type": "SystemExit",
            "message": "Missing --database-url, EXTRACTED_DATABASE_URL, or DATABASE_URL",
        },
    }
    assert payload["latency"] == {
        "count": 0,
        "p50_ms": 0.0,
        "p95_ms": 0.0,
        "max_ms": 0.0,
    }


def test_main_prints_preflight_error_in_human_summary(capsys) -> None:
    code = smoke.main([
        "--database-url",
        "",
        "--account-count",
        "1",
        "--corpora-per-account",
        "1",
        "--documents-per-corpus",
        "1",
        "--iterations",
        "1",
        "--concurrency",
        "1",
        "--pool-size",
        "1",
    ])

    assert code == 2
    output = capsys.readouterr().out
    assert "FAQ search concurrency smoke: ok=False" in output
    assert "setup_error=Missing --database-url, EXTRACTED_DATABASE_URL, or DATABASE_URL" in output


def test_main_writes_result_for_pool_creation_failure(tmp_path, monkeypatch) -> None:
    async def _raise_pool(*_args, **_kwargs):
        raise RuntimeError("resolver failed")

    monkeypatch.setattr(smoke, "_create_pool", _raise_pool)
    result_path = tmp_path / "faq-search-result.json"

    code = smoke.main([
        "--database-url",
        "postgresql://example.invalid/atlas",
        "--account-count",
        "1",
        "--corpora-per-account",
        "1",
        "--documents-per-corpus",
        "1",
        "--iterations",
        "1",
        "--concurrency",
        "1",
        "--pool-size",
        "1",
        "--output-result",
        str(result_path),
        "--json",
    ])

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert code == 1
    assert payload["ok"] is False
    assert payload["requests"]["total"] == 0
    assert payload["setup"] == {
        "ok": False,
        "phase": "pool_create",
        "error": {
            "type": "RuntimeError",
            "message": "resolver failed",
        },
    }
    assert payload["latency"] == {
        "count": 0,
        "p50_ms": 0.0,
        "p95_ms": 0.0,
        "max_ms": 0.0,
    }


def test_main_writes_result_for_migration_failure(tmp_path, monkeypatch) -> None:
    class _Pool:
        def __init__(self) -> None:
            self.closed = False
            self.cleanup_calls = 0

        async def execute(self, *_args):
            self.cleanup_calls += 1
            raise AssertionError("cleanup should not run before migrations succeed")

        async def close(self):
            self.closed = True

    pool = _Pool()

    async def _fake_pool(*_args, **_kwargs):
        return pool

    async def _raise_migrations(_pool):
        raise RuntimeError("migration failed")

    monkeypatch.setattr(smoke, "_create_pool", _fake_pool)
    monkeypatch.setattr(smoke, "_apply_migrations", _raise_migrations)
    result_path = tmp_path / "faq-search-migrations.json"

    code = smoke.main([
        "--database-url",
        "postgresql://example.invalid/atlas",
        "--account-count",
        "1",
        "--corpora-per-account",
        "1",
        "--documents-per-corpus",
        "1",
        "--iterations",
        "1",
        "--concurrency",
        "1",
        "--pool-size",
        "1",
        "--output-result",
        str(result_path),
        "--json",
    ])

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert code == 1
    assert pool.closed is True
    assert pool.cleanup_calls == 0
    assert payload["ok"] is False
    assert payload["requests"]["total"] == 0
    assert payload["setup"] == {
        "ok": False,
        "phase": "migrations",
        "error": {
            "type": "RuntimeError",
            "message": "migration failed",
        },
    }


def test_main_writes_result_for_seed_failure(tmp_path, monkeypatch) -> None:
    class _Pool:
        def __init__(self) -> None:
            self.closed = False
            self.cleanup_calls = 0

        async def execute(self, *_args):
            self.cleanup_calls += 1
            return "DELETE 0"

        async def close(self):
            self.closed = True

    pool = _Pool()

    async def _fake_pool(*_args, **_kwargs):
        return pool

    async def _apply_noop(_pool):
        return None

    async def _raise_seed(*_args, **_kwargs):
        raise RuntimeError("seed failed")

    monkeypatch.setattr(smoke, "_create_pool", _fake_pool)
    monkeypatch.setattr(smoke, "_apply_migrations", _apply_noop)
    monkeypatch.setattr(smoke, "_seed", _raise_seed)
    result_path = tmp_path / "faq-search-seed.json"

    code = smoke.main([
        "--database-url",
        "postgresql://example.invalid/atlas",
        "--account-count",
        "1",
        "--corpora-per-account",
        "1",
        "--documents-per-corpus",
        "1",
        "--iterations",
        "1",
        "--concurrency",
        "1",
        "--pool-size",
        "1",
        "--output-result",
        str(result_path),
        "--json",
    ])

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert code == 1
    assert pool.closed is True
    assert pool.cleanup_calls == 1
    assert payload["ok"] is False
    assert payload["requests"]["total"] == 0
    assert payload["setup"] == {
        "ok": False,
        "phase": "seed",
        "error": {
            "type": "RuntimeError",
            "message": "seed failed",
        },
    }


def test_failure_summary_limits_output() -> None:
    results = [
        {
            "account_id": f"acct-{index}",
            "corpus_id": "corpus",
            "query": "export",
            "failures": ["wrong account_id returned"],
        }
        for index in range(25)
    ]

    summary = smoke._failure_summary(results)

    assert summary["count"] == 25
    assert len(summary["items"]) == 20
    assert summary["truncated"] is True
