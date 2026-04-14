"""Targeted Trustpilot raw-page capture and field-presence audit helpers."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

import httpx
from bs4 import BeautifulSoup

from ...config import settings
from .client import get_scrape_client
from .parsers import ScrapeTarget
from .parsers.trustpilot import _BASE_URL, _DOMAIN
from .web_unlocker_http import get_with_web_unlocker


_EMPLOYER_KEYWORDS = (
    "worksFor",
    "jobTitle",
    "companyName",
    "employer",
    "employment",
)

_HTML_KEYWORD_PATTERNS: dict[str, re.Pattern[str]] = {
    key: re.compile(re.escape(key), re.I)
    for key in _EMPLOYER_KEYWORDS
}

_HTML_SELECTOR_HINTS: dict[str, str] = {
    "review_card": "[data-service-review-card-paper]",
    "reviewer_company": (
        '[data-service-review-card-paper] [class*="reviewer-company"], '
        '[data-service-review-card-paper] [data-reviewer-company], '
        '[data-service-review-card-paper] [data-company]'
    ),
    "reviewer_title": (
        '[data-service-review-card-paper] [class*="reviewer-title"], '
        '[data-service-review-card-paper] [class*="job-title"], '
        '[data-service-review-card-paper] [data-title], '
        '[data-service-review-card-paper] [data-job-title]'
    ),
    "author": '[data-service-review-card-paper] [itemprop="author"], [data-service-review-card-paper] [class*="author"]',
}


def build_trustpilot_page_url(product_slug: str, page: int) -> str:
    base = f"{_BASE_URL}/{product_slug}"
    if page <= 1:
        return base
    return f"{base}?page={page}"


def build_trustpilot_referer(vendor_name: str, product_slug: str, page: int) -> str:
    if page <= 1:
        return f"https://www.google.com/search?q={quote_plus(vendor_name)}+trustpilot+reviews"
    if page == 2:
        return f"{_BASE_URL}/{product_slug}"
    return f"{_BASE_URL}/{product_slug}?page={page - 1}"


def _coerce_json(value: str) -> Any:
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return None


def _iter_review_objects(obj: Any) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    stack = [obj]
    while stack:
        current = stack.pop()
        if isinstance(current, dict):
            current_type = str(current.get("@type") or "")
            if current_type in {"Review", "http://schema.org/Review"}:
                found.append(current)
            for value in current.values():
                if isinstance(value, (dict, list)):
                    stack.append(value)
        elif isinstance(current, list):
            stack.extend(current)
    return found


def analyze_trustpilot_jsonld_fields(html: str) -> dict[str, Any]:
    soup = BeautifulSoup(html, "html.parser")
    scripts = soup.select('script[type="application/ld+json"]')
    matched_keys: set[str] = set()
    review_count = 0
    author_count = 0
    author_job_title_count = 0
    author_company_count = 0
    review_level_job_title_count = 0
    review_level_company_count = 0

    for script in scripts:
        payload = _coerce_json(script.string or "")
        if payload is None:
            continue
        for review in _iter_review_objects(payload):
            review_count += 1
            author = review.get("author")
            if isinstance(author, dict):
                author_count += 1
                if str(author.get("jobTitle") or "").strip():
                    author_job_title_count += 1
                    matched_keys.add("author.jobTitle")
                works_for = author.get("worksFor")
                if isinstance(works_for, dict):
                    if str(works_for.get("name") or "").strip():
                        author_company_count += 1
                        matched_keys.add("author.worksFor.name")
                elif str(works_for or "").strip():
                    author_company_count += 1
                    matched_keys.add("author.worksFor")
                if str(author.get("companyName") or "").strip():
                    author_company_count += 1
                    matched_keys.add("author.companyName")
                if str(author.get("employer") or "").strip():
                    author_company_count += 1
                    matched_keys.add("author.employer")
            if str(review.get("jobTitle") or "").strip():
                review_level_job_title_count += 1
                matched_keys.add("review.jobTitle")
            if str(review.get("companyName") or "").strip():
                review_level_company_count += 1
                matched_keys.add("review.companyName")
            if str(review.get("employer") or "").strip():
                review_level_company_count += 1
                matched_keys.add("review.employer")

    return {
        "script_count": len(scripts),
        "review_object_count": review_count,
        "author_object_count": author_count,
        "author_job_title_count": author_job_title_count,
        "author_company_count": author_company_count,
        "review_level_job_title_count": review_level_job_title_count,
        "review_level_company_count": review_level_company_count,
        "matched_keys": sorted(matched_keys),
    }


def analyze_trustpilot_html_field_hints(html: str) -> dict[str, Any]:
    soup = BeautifulSoup(html, "html.parser")
    keyword_hits = {
        name: len(pattern.findall(html))
        for name, pattern in _HTML_KEYWORD_PATTERNS.items()
    }
    selector_hits = {
        name: len(soup.select(selector))
        for name, selector in _HTML_SELECTOR_HINTS.items()
    }
    review_cards = soup.select(_HTML_SELECTOR_HINTS["review_card"])
    review_card_meta_samples: list[str] = []
    for card in review_cards[:5]:
        container = (
            card.select_one('[class*="consumerDetails"]')
            or card.select_one('[class*="consumerInfo"]')
            or card
        )
        text = " ".join(container.get_text(" ", strip=True).split())
        if text:
            review_card_meta_samples.append(text[:200])
    return {
        "keyword_hits": keyword_hits,
        "selector_hits": selector_hits,
        "review_card_count": len(review_cards),
        "review_card_meta_samples": review_card_meta_samples,
    }


def analyze_trustpilot_raw_html(html: str) -> dict[str, Any]:
    jsonld = analyze_trustpilot_jsonld_fields(html)
    markup = analyze_trustpilot_html_field_hints(html)
    employer_fields_present = bool(
        jsonld["author_company_count"]
        or jsonld["review_level_company_count"]
        or markup["selector_hits"].get("reviewer_company")
    )
    title_fields_present = bool(
        jsonld["author_job_title_count"]
        or jsonld["review_level_job_title_count"]
        or markup["selector_hits"].get("reviewer_title")
    )
    return {
        "jsonld": jsonld,
        "markup": markup,
        "employer_fields_present": employer_fields_present,
        "title_fields_present": title_fields_present,
    }


async def fetch_trustpilot_page_http(
    target: ScrapeTarget,
    *,
    page: int,
) -> dict[str, Any]:
    url = build_trustpilot_page_url(target.product_slug, page)
    referer = build_trustpilot_referer(target.vendor_name, target.product_slug, page)
    client = get_scrape_client()
    response = await client.get(
        url,
        domain=_DOMAIN,
        referer=referer,
        sticky_session=True,
        prefer_residential=True,
        timeout_seconds=60,
    )
    return {
        "method": "http",
        "url": url,
        "status_code": response.status_code,
        "final_url": str(response.url),
        "headers": dict(response.headers),
        "body": response.text or "",
    }


async def fetch_trustpilot_page_web_unlocker(
    target: ScrapeTarget,
    *,
    page: int,
) -> dict[str, Any]:
    url = build_trustpilot_page_url(target.product_slug, page)
    referer = build_trustpilot_referer(target.vendor_name, target.product_slug, page)
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": referer,
    }
    response = await get_with_web_unlocker(url, headers=headers, domain=_DOMAIN)
    return {
        "method": "web_unlocker",
        "url": url,
        "status_code": response.status_code,
        "final_url": str(response.url),
        "headers": dict(response.headers),
        "body": response.text or "",
    }


async def capture_trustpilot_page(
    target: ScrapeTarget,
    *,
    page: int,
    method: str,
) -> dict[str, Any]:
    if method == "http":
        capture = await fetch_trustpilot_page_http(target, page=page)
    elif method == "web_unlocker":
        capture = await fetch_trustpilot_page_web_unlocker(target, page=page)
    else:
        raise ValueError(f"Unsupported method: {method}")
    capture["analysis"] = analyze_trustpilot_raw_html(capture.get("body") or "")
    capture["body_bytes"] = len((capture.get("body") or "").encode("utf-8", errors="replace"))
    return capture


def write_trustpilot_capture_artifacts(
    output_dir: Path,
    *,
    vendor_name: str,
    page: int,
    capture: dict[str, Any],
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_vendor = re.sub(r"[^a-z0-9_-]+", "_", vendor_name.strip().lower()).strip("_") or "target"
    stem = f"{safe_vendor}_page{page}_{capture['method']}_{capture['status_code']}"
    html_path = output_dir / f"{stem}.html"
    meta_path = output_dir / f"{stem}.json"
    html_path.write_text(capture.get("body") or "", encoding="utf-8")
    metadata = dict(capture)
    metadata.pop("body", None)
    meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "html_path": str(html_path),
        "metadata_path": str(meta_path),
    }
