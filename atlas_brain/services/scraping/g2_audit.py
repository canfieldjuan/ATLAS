"""Targeted G2 raw-page capture and reviewer-field audit helpers."""

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
from .parsers.g2 import _BASE_URL, _DOMAIN, _merge_tailwind_reviewer_info
from .web_unlocker_http import get_with_web_unlocker

_CARD_SELECTOR = '[itemprop="review"], [data-testid*="review"], [class*="review-card"], article'

_JSONLD_REVIEWER_KEYS = (
    "author.jobTitle",
    "author.worksFor.name",
    "author.worksFor",
    "author.companyName",
    "author.employer",
)


def build_g2_page_url(product_slug: str, page: int) -> str:
    base = f"{_BASE_URL}/{product_slug}/reviews"
    if page <= 1:
        return base
    return f"{base}?page={page}"


def build_g2_referer(vendor_name: str, product_slug: str, page: int) -> str:
    if page <= 1:
        return f"https://www.google.com/search?q={quote_plus(vendor_name)}+g2+reviews"
    if page == 2:
        return f"{_BASE_URL}/{product_slug}/reviews"
    return f"{_BASE_URL}/{product_slug}/reviews?page={page - 1}"


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


def analyze_g2_jsonld_fields(html: str) -> dict[str, Any]:
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


def _get_text(card: Any, selector: str) -> str | None:
    node = card.select_one(selector)
    if node is None:
        return None
    text = node.get_text(strip=True)
    return text or None


def analyze_g2_review_cards(html: str) -> dict[str, Any]:
    soup = BeautifulSoup(html, "html.parser")
    cards = soup.select(_CARD_SELECTOR)
    samples: list[dict[str, Any]] = []
    title_count = 0
    company_count = 0
    industry_count = 0
    size_count = 0

    for card in cards:
        reviewer_name = _get_text(card, '[itemprop="author"], [class*="reviewer-name"]')
        reviewer_title = _get_text(card, '[class*="reviewer-title"], [class*="job-title"]')
        reviewer_company = _get_text(card, '[class*="reviewer-company"], [class*="organization"]')
        reviewer_industry = _get_text(card, '[class*="industry"]')
        company_size = _get_text(card, '[class*="company-size"], [class*="employees"]')

        name_el = card.select_one('div[class*="elv-font-bold"]:not([class*="elv-text-lg"])')
        if name_el:
            if not reviewer_name:
                reviewer_name = name_el.get_text(strip=True) or None
            info_parent = name_el.parent.parent if name_el.parent else None
            if info_parent:
                subtle = [
                    node.get_text(strip=True)
                    for node in info_parent.select('div[class*="elv-text-subtle"]')
                    if node.get_text(strip=True)
                ]
                reviewer_title, reviewer_company, reviewer_industry, company_size = _merge_tailwind_reviewer_info(
                    subtle,
                    reviewer_title=reviewer_title,
                    reviewer_company=reviewer_company,
                    reviewer_industry=reviewer_industry,
                    company_size=company_size,
                )

        if reviewer_title:
            title_count += 1
        if reviewer_company:
            company_count += 1
        if reviewer_industry:
            industry_count += 1
        if company_size:
            size_count += 1
        if len(samples) < 5 and any((reviewer_name, reviewer_title, reviewer_company, reviewer_industry, company_size)):
            samples.append(
                {
                    "reviewer_name": reviewer_name,
                    "reviewer_title": reviewer_title,
                    "reviewer_company": reviewer_company,
                    "reviewer_industry": reviewer_industry,
                    "company_size_raw": company_size,
                }
            )

    return {
        "review_card_count": len(cards),
        "reviewer_title_count": title_count,
        "reviewer_company_count": company_count,
        "reviewer_industry_count": industry_count,
        "company_size_count": size_count,
        "samples": samples,
    }


def analyze_g2_raw_html(html: str) -> dict[str, Any]:
    jsonld = analyze_g2_jsonld_fields(html)
    cards = analyze_g2_review_cards(html)
    return {
        "jsonld": jsonld,
        "review_cards": cards,
        "title_fields_present": bool(
            jsonld["author_job_title_count"]
            or jsonld["review_level_job_title_count"]
            or cards["reviewer_title_count"]
        ),
        "employer_fields_present": bool(
            jsonld["author_company_count"]
            or jsonld["review_level_company_count"]
            or cards["reviewer_company_count"]
        ),
        "industry_fields_present": bool(cards["reviewer_industry_count"]),
        "company_size_fields_present": bool(cards["company_size_count"]),
    }


async def fetch_g2_page_http(target: ScrapeTarget, *, page: int) -> dict[str, Any]:
    url = build_g2_page_url(target.product_slug, page)
    referer = build_g2_referer(target.vendor_name, target.product_slug, page)
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


async def fetch_g2_page_web_unlocker(target: ScrapeTarget, *, page: int) -> dict[str, Any]:
    url = build_g2_page_url(target.product_slug, page)
    referer = build_g2_referer(target.vendor_name, target.product_slug, page)
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


async def capture_g2_page(target: ScrapeTarget, *, page: int, method: str) -> dict[str, Any]:
    if method == "http":
        capture = await fetch_g2_page_http(target, page=page)
    elif method == "web_unlocker":
        capture = await fetch_g2_page_web_unlocker(target, page=page)
    else:
        raise ValueError(f"Unsupported method: {method}")
    capture["analysis"] = analyze_g2_raw_html(capture.get("body") or "")
    capture["body_bytes"] = len((capture.get("body") or "").encode("utf-8", errors="replace"))
    return capture


def write_g2_capture_artifacts(
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
