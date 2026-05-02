"""Minimal standalone vendor briefing email renderer."""

from __future__ import annotations

from html import escape
from typing import Any, Iterable


def _text(value: Any, fallback: str = "") -> str:
    text = str(value or "").strip()
    return text or fallback


def _iter_dicts(value: Any) -> Iterable[dict[str, Any]]:
    if not isinstance(value, list):
        return ()
    return (item for item in value if isinstance(item, dict))


def _section(title: str, body: str) -> str:
    if not body:
        return ""
    return f"<section><h2>{escape(title)}</h2>{body}</section>"


def _render_pain_breakdown(briefing: dict[str, Any]) -> str:
    rows: list[str] = []
    for item in _iter_dicts(briefing.get("pain_breakdown")):
        label = _text(item.get("label") or item.get("category") or item.get("pain_category"))
        count = _text(item.get("mention_count") or item.get("count"))
        if not label:
            continue
        suffix = f" ({escape(count)} mentions)" if count else ""
        rows.append(f"<li>{escape(label)}{suffix}</li>")
    return _section("Pain Signals", f"<ul>{''.join(rows)}</ul>" if rows else "")


def _render_evidence(briefing: dict[str, Any]) -> str:
    quotes: list[str] = []
    for item in _iter_dicts(briefing.get("evidence")):
        if item.get("phrase_verbatim") is not True:
            continue
        quote = _text(item.get("quote") or item.get("text") or item.get("excerpt_text"))
        if quote:
            quotes.append(f"<blockquote>&ldquo;{escape(quote)}&rdquo;</blockquote>")
    return _section("What Customers Are Saying", "".join(quotes))


def _render_named_accounts(briefing: dict[str, Any]) -> str:
    rows: list[str] = []
    for item in _iter_dicts(briefing.get("named_accounts")):
        company = _text(item.get("company") or item.get("company_name"))
        signal = _text(item.get("signal") or item.get("reason"))
        if not company:
            continue
        suffix = f" - {escape(signal)}" if signal else ""
        rows.append(f"<li><strong>{escape(company)}</strong>{suffix}</li>")
    return _section("Named Accounts", f"<ul>{''.join(rows)}</ul>" if rows else "")


def render_vendor_briefing_html(briefing: dict[str, Any]) -> str:
    """Render a compact customer-facing vendor briefing HTML document."""
    vendor = _text(
        briefing.get("vendor_name")
        or briefing.get("target_vendor")
        or briefing.get("company_name"),
        "Vendor",
    )
    challenger_mode = bool(briefing.get("challenger_mode"))
    title = (
        f"Sales Intelligence Briefing: {vendor}"
        if challenger_mode
        else f"Churn Intelligence Briefing: {vendor}"
    )
    summary = _text(
        briefing.get("analyst_summary")
        or briefing.get("headline")
        or briefing.get("summary")
    )
    sections = [
        _section("Summary", f"<p>{escape(summary)}</p>" if summary else ""),
        _render_pain_breakdown(briefing),
        _render_evidence(briefing),
        _render_named_accounts(briefing),
    ]
    body = "".join(section for section in sections if section)
    return (
        "<!doctype html>"
        "<html><head><meta charset=\"utf-8\">"
        f"<title>{escape(title)}</title>"
        "</head><body>"
        f"<h1>{escape(title)}</h1>"
        f"{body}"
        "</body></html>"
    )
