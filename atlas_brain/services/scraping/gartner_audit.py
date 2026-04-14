"""Targeted Gartner raw-page capture and reviewer-field audit helpers."""

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
from .parsers.gartner import _DOMAIN, _build_reviews_url, _extract_gartner_company_name
from .web_unlocker_http import get_with_web_unlocker

_CARD_SELECTOR = (
    '[data-testid="review-card"], '
    '[class*="review-card"], '
    '[class*="ReviewCard"], '
    '[itemprop="review"], '
    'div[class*="review-listing"], '
    'div[class*="peer-review"], '
    'div[class*="reviewContent"]'
)


def build_gartner_page_url(product_slug: str, page: int) -> str:
    offset = max(0, page - 1) * 10
    return _build_reviews_url(product_slug, offset)


def build_gartner_referer(vendor_name: str, product_slug: str, page: int) -> str:
    if page <= 1:
        return f"https://www.google.com/search?q={quote_plus(vendor_name)}+gartner+peer+insights+reviews"
    return build_gartner_page_url(product_slug, page - 1)


def _coerce_json(value: str) -> Any:
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return None


def _iter_json_ld_items(data: object) -> list[dict[str, Any]]:
    items = data if isinstance(data, list) else [data]
    expanded: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        expanded.append(item)
        graph = item.get("@graph")
        if isinstance(graph, list):
            expanded.extend(node for node in graph if isinstance(node, dict))
        elif isinstance(graph, dict):
            expanded.append(graph)
    return expanded


def analyze_gartner_jsonld_fields(html: str) -> dict[str, Any]:
    soup = BeautifulSoup(html, "html.parser")
    scripts = soup.select('script[type="application/ld+json"]')
    matched_keys: set[str] = set()
    review_count = 0
    author_count = 0
    author_job_title_count = 0
    author_company_count = 0

    for script in scripts:
        payload = _coerce_json(script.string or "")
        if payload is None:
            continue
        for item in _iter_json_ld_items(payload):
            review_list = item.get("review", [])
            if not isinstance(review_list, list):
                review_list = [review_list]
            for review in review_list:
                if not isinstance(review, dict):
                    continue
                review_count += 1
                author = review.get("author")
                if isinstance(author, dict):
                    author_count += 1
                    if str(author.get("jobTitle") or "").strip():
                        author_job_title_count += 1
                        matched_keys.add("author.jobTitle")
                    company_name = _extract_gartner_company_name(author)
                    if str(company_name or "").strip():
                        author_company_count += 1
                        matched_keys.add("author.company")

    return {
        "script_count": len(scripts),
        "review_object_count": review_count,
        "author_object_count": author_count,
        "author_job_title_count": author_job_title_count,
        "author_company_count": author_company_count,
        "matched_keys": sorted(matched_keys),
    }


def _build_snippet_map(snippets: object) -> dict[str, dict[str, Any]]:
    if not isinstance(snippets, dict):
        return {}
    indexed: dict[str, dict[str, Any]] = {}
    for value in snippets.values():
        if not isinstance(value, list):
            continue
        for entry in value:
            if not isinstance(entry, dict):
                continue
            review_id = entry.get("reviewId")
            if review_id is not None:
                indexed[str(review_id)] = entry
    return indexed


def analyze_gartner_next_data_fields(html: str) -> dict[str, Any]:
    soup = BeautifulSoup(html, "html.parser")
    script = soup.find("script", id="__NEXT_DATA__")
    if not script or not script.string:
        return {
            "review_object_count": 0,
            "reviewer_title_count": 0,
            "reviewer_company_count": 0,
            "company_size_count": 0,
            "reviewer_industry_count": 0,
            "samples": [],
        }
    payload = _coerce_json(script.string)
    if not isinstance(payload, dict):
        return {
            "review_object_count": 0,
            "reviewer_title_count": 0,
            "reviewer_company_count": 0,
            "company_size_count": 0,
            "reviewer_industry_count": 0,
            "samples": [],
        }

    server_data = payload.get("props", {}).get("pageProps", {}).get("serverSideXHRData", {})
    review_data = server_data.get("user-reviews-by-market-vendor-product", {})
    raw_reviews = review_data.get("userReviews", [])
    snippet_map = _build_snippet_map(server_data.get("vendor-snippets", {}))
    if not isinstance(raw_reviews, list):
        raw_reviews = []

    title_count = 0
    company_count = 0
    size_count = 0
    industry_count = 0
    samples: list[dict[str, Any]] = []
    for entry in raw_reviews:
        if not isinstance(entry, dict):
            continue
        snippet = snippet_map.get(str(entry.get("reviewId")), {})
        reviewer_title = entry.get("jobTitle") or None
        reviewer_company = _extract_gartner_company_name(entry, snippet)
        company_size = entry.get("companySize") or snippet.get("companySize")
        reviewer_industry = entry.get("industryName") or snippet.get("industryName")
        if reviewer_title:
            title_count += 1
        if reviewer_company:
            company_count += 1
        if company_size:
            size_count += 1
        if reviewer_industry:
            industry_count += 1
        if len(samples) < 5 and any((reviewer_title, reviewer_company, company_size, reviewer_industry)):
            samples.append(
                {
                    "reviewer_title": reviewer_title,
                    "reviewer_company": reviewer_company,
                    "company_size_raw": company_size,
                    "reviewer_industry": reviewer_industry,
                }
            )
    return {
        "review_object_count": len(raw_reviews),
        "reviewer_title_count": title_count,
        "reviewer_company_count": company_count,
        "company_size_count": size_count,
        "reviewer_industry_count": industry_count,
        "samples": samples,
    }


def _get_text(card: Any, selector: str) -> str | None:
    node = card.select_one(selector)
    if node is None:
        return None
    text = node.get_text(strip=True)
    return text or None


def analyze_gartner_review_cards(html: str) -> dict[str, Any]:
    soup = BeautifulSoup(html, "html.parser")
    cards = soup.select(_CARD_SELECTOR)
    title_count = 0
    company_count = 0
    size_count = 0
    industry_count = 0
    samples: list[dict[str, Any]] = []

    for card in cards:
        reviewer_title = _get_text(
            card,
            '[class*="reviewer-title"], [class*="ReviewerTitle"], [class*="job-title"], [class*="jobTitle"]',
        )
        reviewer_company = _get_text(
            card,
            '[class*="reviewer-company"], [class*="ReviewerCompany"], [class*="organization"], [class*="companyName"]',
        )
        company_size = _get_text(
            card,
            '[class*="company-size"], [class*="CompanySize"], [class*="employees"], [class*="revenue"]',
        )
        reviewer_industry = _get_text(card, '[class*="industry"], [class*="Industry"]')

        if reviewer_title:
            title_count += 1
        if reviewer_company:
            company_count += 1
        if company_size:
            size_count += 1
        if reviewer_industry:
            industry_count += 1
        if len(samples) < 5 and any((reviewer_title, reviewer_company, company_size, reviewer_industry)):
            samples.append(
                {
                    "reviewer_title": reviewer_title,
                    "reviewer_company": reviewer_company,
                    "company_size_raw": company_size,
                    "reviewer_industry": reviewer_industry,
                }
            )

    return {
        "review_card_count": len(cards),
        "reviewer_title_count": title_count,
        "reviewer_company_count": company_count,
        "company_size_count": size_count,
        "reviewer_industry_count": industry_count,
        "samples": samples,
    }


def analyze_gartner_raw_html(html: str) -> dict[str, Any]:
    jsonld = analyze_gartner_jsonld_fields(html)
    next_data = analyze_gartner_next_data_fields(html)
    review_cards = analyze_gartner_review_cards(html)
    return {
        "jsonld": jsonld,
        "next_data": next_data,
        "review_cards": review_cards,
        "title_fields_present": bool(
            jsonld["author_job_title_count"]
            or next_data["reviewer_title_count"]
            or review_cards["reviewer_title_count"]
        ),
        "employer_fields_present": bool(
            jsonld["author_company_count"]
            or next_data["reviewer_company_count"]
            or review_cards["reviewer_company_count"]
        ),
        "industry_fields_present": bool(
            next_data["reviewer_industry_count"] or review_cards["reviewer_industry_count"]
        ),
        "company_size_fields_present": bool(
            next_data["company_size_count"] or review_cards["company_size_count"]
        ),
    }


async def fetch_gartner_page_http(target: ScrapeTarget, *, page: int) -> dict[str, Any]:
    url = build_gartner_page_url(target.product_slug, page)
    referer = build_gartner_referer(target.vendor_name, target.product_slug, page)
    client = get_scrape_client()
    response = await client.get(
        url,
        domain=_DOMAIN,
        referer=referer,
        sticky_session=True,
        prefer_residential=True,
        timeout_seconds=90,
    )
    return {
        "method": "http",
        "url": url,
        "status_code": response.status_code,
        "final_url": str(response.url),
        "headers": dict(response.headers),
        "body": response.text or "",
    }


async def fetch_gartner_page_web_unlocker(target: ScrapeTarget, *, page: int) -> dict[str, Any]:
    url = build_gartner_page_url(target.product_slug, page)
    referer = build_gartner_referer(target.vendor_name, target.product_slug, page)
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/131.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
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


async def capture_gartner_page(target: ScrapeTarget, *, page: int, method: str) -> dict[str, Any]:
    if method == "http":
        capture = await fetch_gartner_page_http(target, page=page)
    elif method == "web_unlocker":
        capture = await fetch_gartner_page_web_unlocker(target, page=page)
    else:
        raise ValueError(f"Unsupported method: {method}")
    capture["analysis"] = analyze_gartner_raw_html(capture.get("body") or "")
    capture["body_bytes"] = len((capture.get("body") or "").encode("utf-8", errors="replace"))
    return capture


def write_gartner_capture_artifacts(
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
    return {"html_path": str(html_path), "metadata_path": str(meta_path)}
