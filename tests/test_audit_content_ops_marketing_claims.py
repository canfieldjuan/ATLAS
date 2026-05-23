from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "audit_content_ops_marketing_claims.py"
)


def load_auditor():
    spec = importlib.util.spec_from_file_location(
        "audit_content_ops_marketing_claims",
        SCRIPT,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write(root: Path, relative: str, text: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_happy_path_allows_defensible_support_ticket_copy(tmp_path):
    auditor = load_auditor()
    _write(
        tmp_path,
        "docs/products/ticket-deflection.md",
        "# Ticket Deflection\n"
        "Atlas groups repeated issue language from support-ticket exports, "
        "ranks FAQ opportunities by ticket volume and failure-risk signals, "
        "and drafts review-ready answers.\n",
    )

    findings = auditor.audit_paths(
        ["docs/products/ticket-deflection.md"],
        root=tmp_path,
    )

    assert findings == ()


def test_audit_surfaces_unsupported_claim_families(tmp_path):
    auditor = load_auditor()
    _write(
        tmp_path,
        "docs/products/ticket-deflection.md",
        "Connect Zendesk and automatically publish answers to your help center.\n"
        "Native Shopify integration publishes directly to your docs site.\n"
        "Self-serve hosted uploads handle 50,000 tickets and reduce ticket "
        "volume by 30%.\n"
        "Semantic clustering ranks by cost for every ticket export.\n",
    )

    findings = auditor.audit_paths(
        ["docs/products"],
        root=tmp_path,
    )

    assert {
        (finding.code, finding.line_no, finding.message)
        for finding in findings
    } == {
        (
            "AUTO_PUBLISH",
            1,
            "Do not claim automatic help-center publishing.",
        ),
        (
            "LIVE_HELPDESK_INTEGRATION",
            1,
            "Do not claim live help-desk integrations until built.",
        ),
        (
            "AUTO_PUBLISH",
            2,
            "Do not claim automatic help-center publishing.",
        ),
        (
            "LIVE_HELPDESK_INTEGRATION",
            2,
            "Do not claim live help-desk integrations until built.",
        ),
        (
            "GUARANTEED_DEFLECTION",
            3,
            "Do not claim guaranteed ticket-volume outcomes.",
        ),
        (
            "UNBOUNDED_HOSTED_UPLOADS",
            3,
            "Do not claim unbounded or 50k hosted synchronous uploads.",
        ),
        (
            "COST_RANKING",
            4,
            "Do not claim cost ranking without imported cost/handle-time data.",
        ),
        (
            "SEMANTIC_CLUSTERING",
            4,
            "Use intent/repeated-issue grouping unless semantic clustering lands.",
        ),
    }
    assert all(finding.path == Path("docs/products/ticket-deflection.md") for finding in findings)


def test_audit_catches_named_platform_and_publish_targets(tmp_path):
    auditor = load_auditor()
    _write(
        tmp_path,
        "docs/products/ticket-deflection.md",
        "Connects to HubSpot Service Hub.\n"
        "Native Shopify integration.\n"
        "Front integration included.\n"
        "Connect to Jira Service Management.\n"
        "Publishes directly to your knowledge base.\n",
    )

    findings = auditor.audit_paths(
        ["docs/products/ticket-deflection.md"],
        root=tmp_path,
    )

    assert [finding.code for finding in findings] == [
        "LIVE_HELPDESK_INTEGRATION",
        "LIVE_HELPDESK_INTEGRATION",
        "LIVE_HELPDESK_INTEGRATION",
        "LIVE_HELPDESK_INTEGRATION",
        "AUTO_PUBLISH",
    ]


def test_audit_catches_percent_ticket_reduction_phrase(tmp_path):
    auditor = load_auditor()
    _write(
        tmp_path,
        "docs/products/ticket-deflection.md",
        "Reduce tickets by 30%.\n"
        "Cut support tickets by 50%.\n"
        "Deflect ticket volume by 20%.\n",
    )

    findings = auditor.audit_paths(
        ["docs/products/ticket-deflection.md"],
        root=tmp_path,
    )

    assert [finding.code for finding in findings] == [
        "GUARANTEED_DEFLECTION",
        "GUARANTEED_DEFLECTION",
        "GUARANTEED_DEFLECTION",
    ]


def test_allow_marker_requires_reason_and_skips_documented_antipattern(tmp_path):
    auditor = load_auditor()
    _write(
        tmp_path,
        "docs/products/ticket-deflection.md",
        "Do not say auto-publish. <!-- claim-audit: allow quoted anti-pattern -->\n"
        "Do not say connect Zendesk. <!-- claim-audit: allow -->\n",
    )

    findings = auditor.audit_paths(
        ["docs/products/ticket-deflection.md"],
        root=tmp_path,
    )

    assert [finding.code for finding in findings] == ["LIVE_HELPDESK_INTEGRATION"]


def test_audit_rejects_unsafe_paths(tmp_path):
    auditor = load_auditor()

    with pytest.raises(ValueError, match="absolute paths are not allowed"):
        auditor.audit_paths([str(tmp_path / "docs")], root=tmp_path)

    with pytest.raises(ValueError, match="path traversal is not allowed"):
        auditor.audit_paths(["../outside.md"], root=tmp_path)

    outside = tmp_path.parent / "outside-marketing-doc.md"
    outside.write_text("Safe copy.\n", encoding="utf-8")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "escape.md").symlink_to(outside)
    with pytest.raises(ValueError, match="path escapes repository root"):
        auditor.audit_paths(["docs/escape.md"], root=tmp_path)


def test_audit_rejects_empty_scan_scope(tmp_path):
    auditor = load_auditor()
    (tmp_path / "docs" / "products").mkdir(parents=True)

    with pytest.raises(ValueError, match="no Markdown/text files found"):
        auditor.audit_paths(["docs/products"], root=tmp_path)


def test_main_returns_failure_for_unsafe_claims(tmp_path, monkeypatch, capsys):
    auditor = load_auditor()
    _write(
        tmp_path,
        "docs/products/ticket-deflection.md",
        "Unlimited tickets with semantic clustering.\n",
    )
    monkeypatch.setattr(auditor, "REPO_ROOT", tmp_path)

    code = auditor.main(["docs/products"])

    captured = capsys.readouterr()
    assert code == 1
    assert "UNBOUNDED_HOSTED_UPLOADS" in captured.out
    assert "SEMANTIC_CLUSTERING" in captured.out


def test_main_returns_zero_for_clean_copy(tmp_path, monkeypatch, capsys):
    auditor = load_auditor()
    _write(
        tmp_path,
        "docs/products/ticket-deflection.md",
        "Atlas groups repeated issue language and drafts review-ready answers.\n",
    )
    monkeypatch.setattr(auditor, "REPO_ROOT", tmp_path)

    code = auditor.main(["docs/products"])

    captured = capsys.readouterr()
    assert code == 0
    assert "1 files scanned" in captured.out


def test_main_returns_error_for_bad_path(tmp_path, monkeypatch, capsys):
    auditor = load_auditor()
    monkeypatch.setattr(auditor, "REPO_ROOT", tmp_path)

    code = auditor.main(["docs/products/missing.md"])

    captured = capsys.readouterr()
    assert code == 2
    assert "path does not exist" in captured.err
