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
            [
                _row("1", "I was charged overdraft fees after I closed the account."),
                _row("2", "My payment was applied to the wrong loan balance."),
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
            [_row("1", "Fee appeared after closure."), _row("2", "Fees changed.")],
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
