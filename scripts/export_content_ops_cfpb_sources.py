#!/usr/bin/env python3
"""Export CFPB complaint narratives as Content Ops source-row JSONL."""

from __future__ import annotations

import argparse
import csv
import html
import io
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence
from urllib.parse import urlencode
from urllib.request import Request, urlopen


DEFAULT_API_URL = (
    "https://www.consumerfinance.gov/data-research/consumer-complaints/"
    "search/api/v1/"
)
DEFAULT_DETAIL_URL_BASE = (
    "https://www.consumerfinance.gov/data-research/consumer-complaints/"
    "search/detail"
)
DEFAULT_LIMIT = 25
DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_MAX_ROWS_SCANNED = 500
DEFAULT_SOURCE_SYSTEM = "cfpb"
DEFAULT_SOURCE_TYPE = "support_ticket"


FIELD_COMPLAINT_ID = "Complaint ID"
FIELD_NARRATIVE = "Consumer complaint narrative"
FIELD_COMPANY = "Company"
FIELD_PRODUCT = "Product"
FIELD_SUB_PRODUCT = "Sub-product"
FIELD_ISSUE = "Issue"
FIELD_SUB_ISSUE = "Sub-issue"
FIELD_DATE_RECEIVED = "Date received"
FIELD_DATE_SENT = "Date sent to company"
FIELD_PUBLIC_RESPONSE = "Company public response"
FIELD_COMPANY_RESPONSE = "Company response to consumer"
FIELD_TIMELY = "Timely response?"
FIELD_CONSUMER_DISPUTED = "Consumer disputed?"
FIELD_CONSENT = "Consumer consent provided?"
FIELD_SUBMITTED_VIA = "Submitted via"
FIELD_STATE = "State"
FIELD_TAGS = "Tags"


def _clean_text(value: Any) -> str:
    return html.unescape(str(value or "").strip())


def _optional_text(value: Any) -> str | None:
    text = _clean_text(value)
    return text or None


def cfpb_row_to_source_row(
    row: Mapping[str, Any],
    *,
    source_system: str = DEFAULT_SOURCE_SYSTEM,
    source_type: str = DEFAULT_SOURCE_TYPE,
    detail_url_base: str = DEFAULT_DETAIL_URL_BASE,
) -> dict[str, Any]:
    """Convert one CFPB CSV row into a Content Ops source row."""

    narrative = _clean_text(row.get(FIELD_NARRATIVE))
    complaint_id = _clean_text(row.get(FIELD_COMPLAINT_ID))
    if not narrative or not complaint_id:
        return {}

    product = _clean_text(row.get(FIELD_PRODUCT))
    issue = _clean_text(row.get(FIELD_ISSUE))
    source_title = " - ".join(part for part in (product, issue) if part)
    out: dict[str, Any] = {
        "id": complaint_id,
        "source_id": complaint_id,
        "source": source_system,
        "source_system": source_system,
        "source_type": source_type,
        "vendor_name": _clean_text(row.get(FIELD_COMPANY)),
        "text": narrative,
        "pain_category": issue,
    }
    optional = {
        "source_title": source_title,
        "source_url": _detail_url(detail_url_base, complaint_id),
        "product": product,
        "category": product,
        "sub_product": row.get(FIELD_SUB_PRODUCT),
        "issue": issue,
        "sub_issue": row.get(FIELD_SUB_ISSUE),
        "date_received": row.get(FIELD_DATE_RECEIVED),
        "date_sent_to_company": row.get(FIELD_DATE_SENT),
        "company_public_response": row.get(FIELD_PUBLIC_RESPONSE),
        "company_response": row.get(FIELD_COMPANY_RESPONSE),
        "timely_response": row.get(FIELD_TIMELY),
        "consumer_disputed": row.get(FIELD_CONSUMER_DISPUTED),
        "consumer_consent_provided": row.get(FIELD_CONSENT),
        "submitted_via": row.get(FIELD_SUBMITTED_VIA),
        "state": row.get(FIELD_STATE),
        "tags": row.get(FIELD_TAGS),
    }
    for key, value in optional.items():
        text = _optional_text(value)
        if text:
            out[key] = text
    return out


def _detail_url(base_url: str, complaint_id: str) -> str:
    base = str(base_url or "").rstrip("/")
    if not base or not complaint_id:
        return ""
    return f"{base}/{complaint_id}"


def build_cfpb_query(
    *,
    company: str | None = None,
    product: str | None = None,
    issue: str | None = None,
    search_term: str | None = None,
    date_received_min: str | None = None,
    date_received_max: str | None = None,
    limit: int = DEFAULT_LIMIT,
) -> dict[str, Any]:
    """Build CFPB complaint API query params."""

    params: dict[str, Any] = {
        "format": "csv",
        "field": "all",
        "no_aggs": "true",
        "sort": "created_date_desc",
        "size": max(int(limit), 1),
    }
    for key, value in (
        ("company", company),
        ("product", product),
        ("issue", issue),
        ("search_term", search_term),
        ("date_received_min", date_received_min),
        ("date_received_max", date_received_max),
    ):
        text = _optional_text(value)
        if text:
            params[key] = text
    return params


def build_cfpb_url(api_url: str, params: Mapping[str, Any]) -> str:
    separator = "&" if "?" in api_url else "?"
    return f"{api_url}{separator}{urlencode(params, doseq=True)}"


def fetch_cfpb_source_rows(
    *,
    api_url: str = DEFAULT_API_URL,
    company: str | None = None,
    product: str | None = None,
    issue: str | None = None,
    search_term: str | None = None,
    date_received_min: str | None = None,
    date_received_max: str | None = None,
    limit: int = DEFAULT_LIMIT,
    max_rows_scanned: int = DEFAULT_MAX_ROWS_SCANNED,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    source_system: str = DEFAULT_SOURCE_SYSTEM,
    source_type: str = DEFAULT_SOURCE_TYPE,
    detail_url_base: str = DEFAULT_DETAIL_URL_BASE,
) -> list[dict[str, Any]]:
    """Fetch CFPB complaint rows and return Content Ops source rows."""

    params = build_cfpb_query(
        company=company,
        product=product,
        issue=issue,
        search_term=search_term,
        date_received_min=date_received_min,
        date_received_max=date_received_max,
        limit=max(limit, max_rows_scanned),
    )
    request = Request(
        build_cfpb_url(api_url, params),
        headers={"User-Agent": "Atlas-Content-Ops-CFPB-Source/1.0"},
    )
    rows: list[dict[str, Any]] = []
    with urlopen(request, timeout=timeout) as response:
        stream = io.TextIOWrapper(response, encoding="utf-8-sig", newline="")
        reader = csv.DictReader(stream)
        for scanned, row in enumerate(reader, start=1):
            source_row = cfpb_row_to_source_row(
                row,
                source_system=source_system,
                source_type=source_type,
                detail_url_base=detail_url_base,
            )
            if source_row:
                rows.append(source_row)
            if len(rows) >= limit or scanned >= max_rows_scanned:
                break
    return rows


def render_jsonl(rows: Sequence[Mapping[str, Any]]) -> str:
    return "\n".join(json.dumps(dict(row), sort_keys=True) for row in rows)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export CFPB complaints as Content Ops source-row JSONL."
    )
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    parser.add_argument("--company", default=None)
    parser.add_argument("--product", default=None)
    parser.add_argument("--issue", default=None)
    parser.add_argument("--search-term", default=None)
    parser.add_argument("--date-received-min", default=None)
    parser.add_argument("--date-received-max", default=None)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--max-rows-scanned", type=int, default=DEFAULT_MAX_ROWS_SCANNED)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--source-system", default=DEFAULT_SOURCE_SYSTEM)
    parser.add_argument("--source-type", default=DEFAULT_SOURCE_TYPE)
    parser.add_argument("--detail-url-base", default=DEFAULT_DETAIL_URL_BASE)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    if int(args.limit) < 1:
        raise SystemExit("--limit must be positive")
    if int(args.max_rows_scanned) < 1:
        raise SystemExit("--max-rows-scanned must be positive")
    if int(args.max_rows_scanned) < int(args.limit):
        raise SystemExit("--max-rows-scanned must be >= --limit")
    if float(args.timeout) <= 0:
        raise SystemExit("--timeout must be positive")


def _main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _validate_args(args)
    rows = fetch_cfpb_source_rows(
        api_url=str(args.api_url),
        company=args.company,
        product=args.product,
        issue=args.issue,
        search_term=args.search_term,
        date_received_min=args.date_received_min,
        date_received_max=args.date_received_max,
        limit=int(args.limit),
        max_rows_scanned=int(args.max_rows_scanned),
        timeout=float(args.timeout),
        source_system=str(args.source_system),
        source_type=str(args.source_type),
        detail_url_base=str(args.detail_url_base),
    )
    payload = render_jsonl(rows)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + ("\n" if payload else ""), encoding="utf-8")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
