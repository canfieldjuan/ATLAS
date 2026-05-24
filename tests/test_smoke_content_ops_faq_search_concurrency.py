from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/smoke_content_ops_faq_search_concurrency.py"
SPEC = importlib.util.spec_from_file_location("smoke_content_ops_faq_search_concurrency", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
smoke = importlib.util.module_from_spec(SPEC)
sys.modules["smoke_content_ops_faq_search_concurrency"] = smoke
SPEC.loader.exec_module(smoke)


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
    for corpus_id in {case.corpus_id for case in cases}:
        assert len({case.account_id for case in cases if case.corpus_id == corpus_id}) == 2


def test_documents_for_case_are_ranked_and_scoped() -> None:
    case = smoke.SearchCase(
        account_id="acct-1",
        corpus_id="corpus-1",
        faq_id="11111111-1111-1111-1111-111111111111",
        query="password reset",
        expected_hit=True,
    )

    documents = smoke._documents_for_case(case, documents_per_corpus=3)

    assert [document.rank for document in documents] == [1, 2, 3]
    assert {document.account_id for document in documents} == {"acct-1"}
    assert {document.corpus_id for document in documents} == {"corpus-1"}
    assert {document.status for document in documents} == {"approved"}
    assert {document.target_mode for document in documents} == {"support_account"}
    assert all("password reset" in document.search_text for document in documents)


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
        query="password reset",
        expected_hit=True,
    )
    miss_case = smoke.SearchCase(
        account_id="acct-1",
        corpus_id="shared-corpus",
        faq_id="faq-1",
        query="escrow shortage",
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
        "p95_ms": 3.0,
        "max_ms": 4.0,
    }


def test_failure_summary_limits_output() -> None:
    results = [
        {
            "account_id": f"acct-{index}",
            "corpus_id": "corpus",
            "query": "password",
            "failures": ["wrong account_id returned"],
        }
        for index in range(25)
    ]

    summary = smoke._failure_summary(results)

    assert summary["count"] == 25
    assert len(summary["items"]) == 20
    assert summary["truncated"] is True
