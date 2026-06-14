from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/smoke_content_ops_cfpb_faq_markdown.py"
SPEC = importlib.util.spec_from_file_location("smoke_content_ops_cfpb_faq_markdown", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
smoke = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(smoke)


def _row(row_id: str, text: str) -> dict[str, str]:
    return {
        "id": f"cfpb:{row_id}",
        "source_id": f"cfpb:{row_id}",
        "source": "cfpb",
        "source_system": "cfpb",
        "source_type": "support_ticket",
        "vendor_name": "Example Bank",
        "text": text,
        "pain_category": "Fees",
        "source_title": "Checking account - Fees",
    }


def _profile(*, source_count: int, scanned_count: int | None = None) -> dict[str, object]:
    scanned = scanned_count if scanned_count is not None else source_count
    return {
        "status": "ok",
        "raw_row_count": scanned,
        "raw_row_count_source": "cfpb_csv_rows_scanned",
        "usable_source_count": source_count,
        "skipped_row_count": scanned - source_count,
        "missing_complaint_id_count": 0,
        "missing_narrative_count": scanned - source_count,
        "skipped_other_count": 0,
        "usable_source_ratio": round(source_count / scanned, 6) if scanned else None,
        "requested_source_count": 2,
        "max_rows_scanned": 5,
        "stop_reason": "limit" if source_count >= 2 else "exhausted",
        "require_narrative": True,
    }


def _args(**overrides):
    values = {
        "company": "Example Bank",
        "product": None,
        "issue": "Fees",
        "search_term": "fees",
        "api_url": "https://example.test/cfpb",
        "limit": 2,
        "max_rows_scanned": 5,
        "timeout": 3.5,
        "source_system": "cfpb",
        "source_type": "support_ticket",
        "user_agent": "HostFetcher/2.0",
        "referer": "https://host.example/source-export",
        "title": "CFPB FAQ",
        "max_items": 8,
        "max_evidence_per_item": 3,
        "max_text_chars": 1200,
        "support_contact": "https://example.com/support",
        "compare_embedding_booster": False,
        "output_source_rows": None,
        "output_markdown": None,
        "json": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_cfpb_faq_smoke_builds_grounded_markdown(monkeypatch, tmp_path: Path) -> None:
    calls = []
    monkeypatch.setattr(
        smoke,
        "fetch_cfpb_source_rows_with_profile",
        lambda **kwargs: calls.append(kwargs)
        or (
            # #1460: both complaints repeat the same overdraft-fee question
            # so the smoke still yields a billable FAQ cluster.
            [
                _row("1", "I was charged overdraft fees after I closed the account."),
                _row("2", "I was charged overdraft fees after closing my account."),
            ],
            _profile(source_count=2, scanned_count=4),
        ),
    )

    code, payload = smoke.run_cfpb_faq_markdown_smoke(
        _args(output_markdown=tmp_path / "cfpb_faq.md"),
        source_rows_path=tmp_path / "cfpb_sources.jsonl",
    )

    assert code == 0
    assert payload["faq"]["output_checks"] == {
        "uses_user_vocabulary": True,
        "condensed": True,
        "has_action_items": True,
        "resolution_evidence_scoped": True,
    }
    assert calls[0]["require_narrative"] is True
    assert payload["source_profile"]["raw_row_count"] == 4
    assert payload["source_profile"]["usable_source_count"] == 2
    assert payload["source_profile"]["missing_narrative_count"] == 2
    markdown = (tmp_path / "cfpb_faq.md").read_text(encoding="utf-8")
    assert "What should I do if I was charged overdraft fees" in markdown
    assert "Review the cited ticket evidence and confirm the policy-approved answer" in markdown


def test_cfpb_faq_smoke_fails_when_fetch_returns_too_few_rows(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        smoke,
        "fetch_cfpb_source_rows_with_profile",
        lambda **_kwargs: (
            [_row("1", "I was charged a fee.")],
            _profile(source_count=1, scanned_count=3),
        ),
    )

    code, payload = smoke.run_cfpb_faq_markdown_smoke(
        _args(limit=2),
        source_rows_path=tmp_path / "rows.jsonl",
    )

    assert code == 1
    assert payload["errors"] == ["expected 2 CFPB source row(s), got 1"]
    assert payload["source_profile"]["raw_row_count"] == 3
    assert payload["source_profile"]["usable_source_count"] == 1


def test_cfpb_faq_smoke_accepts_source_policy_questions_for_weak_rows(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        smoke,
        "fetch_cfpb_source_rows_with_profile",
        lambda **_kwargs: (
            # #1460: the weak rows repeat the same gist so the
            # source-policy question path still yields an item.
            [
                _row("1", "Fee appeared after closure."),
                _row("2", "Fee appeared again after closure."),
            ],
            _profile(source_count=2),
        ),
    )

    code, payload = smoke.run_cfpb_faq_markdown_smoke(
        _args(),
        source_rows_path=tmp_path / "rows.jsonl",
    )

    assert code == 0
    assert payload["faq"]["output_checks"] == {
        "uses_user_vocabulary": True,
        "condensed": True,
        "has_action_items": True,
        "resolution_evidence_scoped": True,
    }
    assert payload["faq"]["items"][0]["question_source"] == "source_policy"
    assert payload["errors"] == []


def test_json_output_and_invalid_args(capsys) -> None:
    payload = {"ok": False, "errors": ["broken"]}
    smoke._print_payload(payload, as_json=True)
    assert json.loads(capsys.readouterr().out) == payload
    with pytest.raises(SystemExit, match="--limit must be positive"):
        smoke._validate_args(_args(limit=0))
    with pytest.raises(SystemExit, match="--max-rows-scanned must be >= --limit"):
        smoke._validate_args(_args(limit=2, max_rows_scanned=1))


def test_cfpb_faq_smoke_compares_embedding_booster(
    monkeypatch,
    tmp_path: Path,
) -> None:
    fetch_calls = []
    factory_calls = []
    embedded_batches = []

    class FakeEmbeddingPort:
        def embed_texts(self, texts):
            embedded_batches.append(tuple(texts))
            return ((1.0, 0.0), (0.99, 0.01))

    monkeypatch.setattr(
        smoke,
        "fetch_cfpb_source_rows_with_profile",
        lambda **kwargs: fetch_calls.append(kwargs)
        or (
            [
                _row("1", "How do I get my money back after an overdraft charge?"),
                _row("2", "What is the process for a refund on an overdraft fee?"),
            ],
            _profile(source_count=2),
        ),
    )

    code, payload = smoke.run_cfpb_faq_markdown_smoke(
        _args(compare_embedding_booster=True),
        source_rows_path=tmp_path / "rows.jsonl",
        embedding_port_factory=lambda: factory_calls.append(True) or FakeEmbeddingPort(),
    )

    assert code == 0
    assert len(fetch_calls) == 1
    assert factory_calls == [True]
    assert embedded_batches == [(
        "How do I get my money back after an overdraft charge?",
        "What is the process for a refund on an overdraft fee?",
    )]
    assert payload["faq"]["generated"] == 1
    comparison = payload["embedding_comparison"]
    assert comparison["enabled"] is True
    assert comparison["primary"] == "boosted"
    assert comparison["baseline"]["generated"] == 0
    assert comparison["baseline"]["non_repeat_ticket_count"] == 2
    assert comparison["boosted"]["generated"] == 1
    assert comparison["boosted"]["non_repeat_ticket_count"] == 0
    assert comparison["delta"]["generated"] == 1
    assert comparison["delta"]["non_repeat_ticket_count"] == -2
    assert comparison["delta"]["added_questions"] == [
        "How do I get my money back after an overdraft charge?"
    ]
    assert payload["errors"] == []


def test_cfpb_faq_smoke_compare_mode_fails_closed_when_port_unavailable(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        smoke,
        "fetch_cfpb_source_rows_with_profile",
        lambda **_kwargs: (
            [
                _row("1", "I was charged overdraft fees after I closed the account."),
                _row("2", "I was charged overdraft fees after closing my account."),
            ],
            _profile(source_count=2),
        ),
    )

    def unavailable_port():
        raise RuntimeError("model files missing")

    code, payload = smoke.run_cfpb_faq_markdown_smoke(
        _args(
            compare_embedding_booster=True,
            output_markdown=tmp_path / "should_not_write.md",
        ),
        source_rows_path=tmp_path / "rows.jsonl",
        embedding_port_factory=unavailable_port,
    )

    assert code == 1
    assert payload["embedding_comparison"] == {
        "enabled": True,
        "primary": "baseline",
        "error": "RuntimeError: model files missing",
    }
    assert payload["errors"] == [
        "embedding booster unavailable: RuntimeError: model files missing"
    ]
    assert not (tmp_path / "should_not_write.md").exists()


def test_cfpb_faq_smoke_compare_mode_fails_when_embedding_call_is_swallowed(
    monkeypatch,
    tmp_path: Path,
) -> None:
    class BrokenEmbeddingPort:
        def embed_texts(self, _texts):
            raise RuntimeError("inference crashed")

    monkeypatch.setattr(
        smoke,
        "fetch_cfpb_source_rows_with_profile",
        lambda **_kwargs: (
            [
                _row("1", "How do I get my money back after an overdraft charge?"),
                _row("2", "What is the process for a refund on an overdraft fee?"),
            ],
            _profile(source_count=2),
        ),
    )

    code, payload = smoke.run_cfpb_faq_markdown_smoke(
        _args(compare_embedding_booster=True),
        source_rows_path=tmp_path / "rows.jsonl",
        embedding_port_factory=BrokenEmbeddingPort,
    )

    assert code == 1
    assert payload["faq"]["generated"] == 0
    assert payload["embedding_comparison"] == {
        "enabled": True,
        "primary": "baseline",
        "error": "RuntimeError: inference crashed",
    }
    assert payload["errors"] == [
        "embedding booster unavailable: RuntimeError: inference crashed",
        "FAQ Markdown generated no items",
        "FAQ output checks failed: condensed, has_action_items, resolution_evidence_scoped, uses_user_vocabulary",
    ]
