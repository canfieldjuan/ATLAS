from __future__ import annotations

import json
from pathlib import Path

from extracted_content_pipeline.support_ticket_generated_content_eval import (
    evaluate_support_ticket_generated_content,
)


FIXTURE_DIR = (
    Path(__file__).resolve().parents[1]
    / "docs/extraction/validation/fixtures/"
    "support_ticket_saas_demo_generated_content_acceptance_2026-05-28"
)


def _load_fixture(name: str) -> dict:
    return json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))


def test_current_saas_demo_blog_fixture_passes_generated_content_eval() -> None:
    result = evaluate_support_ticket_generated_content(
        _load_fixture("current_saas_demo_blog_post.json"),
        output="blog_post",
    )

    assert result["ok"] is True
    assert result["errors"] == []
    assert result["source_context_summary"] == {
        "cluster_count": 9,
        "included_ticket_row_count": 36,
        "question_like_ticket_count": 35,
        "source_period": "Uploaded support tickets",
        "source_row_count": 36,
    }


def test_known_bad_saas_demo_blog_fixture_still_fails() -> None:
    result = evaluate_support_ticket_generated_content(
        _load_fixture("known_bad_saas_demo_blog_post.json"),
        output="blog_post",
    )

    assert result["ok"] is False
    assert any(
        check["name"] == "support_ticket_outcome_claims_grounded"
        and check["passed"] is False
        for check in result["checks"]
    )
