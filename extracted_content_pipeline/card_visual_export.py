"""Static HTML visual exports for generated quote/stat card drafts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from html import escape as _html_escape
from typing import Any


JsonDict = Mapping[str, Any]

_SUPPORTED_ASSETS = frozenset({"quote_card", "stat_card"})


def supports_card_visual_export(asset: str) -> bool:
    """Return True when an asset can be exported as static visual cards."""

    return str(asset or "").strip() in _SUPPORTED_ASSETS


def render_card_visual_html(asset: str, rows: Sequence[JsonDict]) -> str:
    """Render quote/stat rows as a static, downloadable HTML document."""

    asset_name = str(asset or "").strip()
    if asset_name not in _SUPPORTED_ASSETS:
        raise ValueError("visual card export supports quote_card and stat_card")

    title = "Quote Cards" if asset_name == "quote_card" else "Stat Cards"
    cards = "\n".join(_render_card(asset_name, row) for row in rows)
    if not cards:
        cards = '<p class="empty-state">No cards matched the selected filters.</p>'
    return "\n".join((
        "<!doctype html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>{_text(title)} Visual Export</title>",
        f"<style>{_styles()}</style>",
        "</head>",
        "<body>",
        '<header class="export-header">',
        '<p class="kicker">Content Ops visual export</p>',
        f"<h1>{_text(title)}</h1>",
        f'<p class="summary">{len(rows)} {_plural("card", len(rows))}</p>',
        "</header>",
        '<main class="card-grid">',
        cards,
        "</main>",
        "</body>",
        "</html>",
        "",
    ))


def _render_card(asset: str, row: JsonDict) -> str:
    if asset == "quote_card":
        return _render_quote_card(row)
    return _render_stat_card(row)


def _render_quote_card(row: JsonDict) -> str:
    source = _source_line(row)
    pain_points = _pain_points(row)
    return "\n".join((
        f'<article class="visual-card quote-card" data-card-id="{_attr(row.get("id"))}">',
        f'<p class="eyebrow">{_text(row.get("theme") or "customer_proof")}</p>',
        f"<h2>{_text(row.get('headline') or 'Customer proof')}</h2>",
        f"<blockquote>&quot;{_text(row.get('quote'))}&quot;</blockquote>",
        f'<p class="attribution">{_text(row.get("attribution") or source)}</p>',
        f'<p class="supporting-text">{_text(row.get("supporting_text"))}</p>',
        f'<p class="source-line">{_text(source)}</p>',
        pain_points,
        "</article>",
    ))


def _render_stat_card(row: JsonDict) -> str:
    source = _source_line(row)
    pain_points = _pain_points(row)
    metric_display = row.get("metric_display") or row.get("metric_value")
    return "\n".join((
        f'<article class="visual-card stat-card" data-card-id="{_attr(row.get("id"))}">',
        f'<p class="eyebrow">{_text(row.get("metric_label") or row.get("theme") or "metric")}</p>',
        f'<p class="metric">{_text(metric_display)}</p>',
        f"<h2>{_text(row.get('claim') or row.get('headline') or 'Customer metric')}</h2>",
        f'<p class="supporting-text">{_text(row.get("supporting_text"))}</p>',
        f'<p class="evidence">{_text(row.get("evidence"))}</p>',
        f'<p class="source-line">{_text(source)}</p>',
        pain_points,
        "</article>",
    ))


def _source_line(row: JsonDict) -> str:
    parts = [
        str(row.get("company_name") or "").strip(),
        str(row.get("vendor_name") or "").strip(),
        str(row.get("source_type") or "").strip(),
    ]
    return " / ".join(part for part in parts if part) or "Source evidence"


def _pain_points(row: JsonDict) -> str:
    raw = row.get("pain_points")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return ""
    items = [
        f"<li>{_text(value)}</li>"
        for value in raw
        if str(value or "").strip()
    ]
    if not items:
        return ""
    return '<ul class="pain-points">' + "".join(items) + "</ul>"


def _text(value: Any) -> str:
    if value is None:
        return ""
    return _html_escape(str(value).strip())


def _attr(value: Any) -> str:
    if value is None:
        return ""
    return _html_escape(str(value).strip(), quote=True)


def _plural(noun: str, count: int) -> str:
    return noun if count == 1 else f"{noun}s"


def _styles() -> str:
    return """
:root {
  color: #172026;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  background: #f5f7f8;
}
* { box-sizing: border-box; }
body { margin: 0; padding: 32px; }
.export-header { max-width: 1120px; margin: 0 auto 24px; }
.kicker { margin: 0 0 8px; color: #53616a; font-size: 13px; text-transform: uppercase; }
h1 { margin: 0; font-size: 34px; line-height: 1.1; }
.summary { margin: 8px 0 0; color: #53616a; }
.card-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
  gap: 20px;
  max-width: 1120px;
  margin: 0 auto;
}
.visual-card {
  min-height: 420px;
  border: 1px solid #d8e0e3;
  border-radius: 8px;
  padding: 28px;
  background: #ffffff;
  box-shadow: 0 18px 40px rgba(15, 23, 42, 0.08);
}
.quote-card { border-top: 8px solid #087f8c; }
.stat-card { border-top: 8px solid #bd5d38; }
.eyebrow {
  margin: 0 0 16px;
  color: #53616a;
  font-size: 12px;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}
h2 { margin: 0 0 18px; font-size: 24px; line-height: 1.2; }
blockquote { margin: 0 0 18px; font-size: 30px; line-height: 1.16; font-weight: 700; }
.metric { margin: 0 0 12px; font-size: 72px; line-height: 0.95; font-weight: 800; }
.attribution { margin: 0 0 20px; font-weight: 700; }
.supporting-text, .evidence, .source-line { line-height: 1.5; }
.supporting-text { color: #33424a; }
.evidence { color: #53616a; font-size: 14px; }
.source-line { margin-top: 22px; color: #53616a; font-size: 13px; }
.pain-points { display: flex; flex-wrap: wrap; gap: 8px; margin: 18px 0 0; padding: 0; }
.pain-points li {
  display: block;
  border-radius: 999px;
  padding: 6px 10px;
  background: #eef3f4;
  color: #33424a;
  font-size: 12px;
}
.empty-state {
  grid-column: 1 / -1;
  border: 1px dashed #b7c3c7;
  border-radius: 8px;
  padding: 32px;
  color: #53616a;
  text-align: center;
}
""".strip()


__all__ = [
    "render_card_visual_html",
    "supports_card_visual_export",
]
