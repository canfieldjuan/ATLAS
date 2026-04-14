"""Targeted GetApp raw-page capture and reviewer-field audit helpers."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

from bs4 import BeautifulSoup

from ...config import settings
from .client import get_scrape_client
from .parsers import ScrapeTarget
from .parsers.getapp import _DOMAIN, _build_url
from .web_unlocker_http import get_with_web_unlocker


def build_getapp_page_url(product_slug: str, page: int) -> str:
    return _build_url(product_slug, page)


def build_getapp_referer(vendor_name: str, product_slug: str, page: int) -> str:
    if page <= 1:
        return f"https://www.google.com/search?q={quote_plus(vendor_name)}+getapp+reviews"
    if page == 2:
        return _build_url(product_slug, 1)
    return _build_url(product_slug, page - 1)


def _coerce_json(value: str) -> Any:
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return None


def _iter_review_objects(obj: Any) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    stack = [obj]
    seen_review_nodes: set[int] = set()
    while stack:
        current = stack.pop()
        if isinstance(current, dict):
            current_type = str(current.get("@type") or "")
            if current_type in {"Review", "http://schema.org/Review"}:
                node_id = id(current)
                if node_id not in seen_review_nodes:
                    found.append(current)
                    seen_review_nodes.add(node_id)
            review_items = current.get("review")
            if isinstance(review_items, list):
                stack.extend(review_items)
            elif isinstance(review_items, dict):
                stack.append(review_items)
            for value in current.values():
                if isinstance(value, (dict, list)):
                    stack.append(value)
        elif isinstance(current, list):
            stack.extend(current)
    return found


def analyze_getapp_jsonld_fields(html: str) -> dict[str, Any]:
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


def analyze_getapp_review_cards(html: str) -> dict[str, Any]:
    soup = BeautifulSoup(html, "html.parser")
    cards = soup.select(
        '[data-testid="review-card"], '
        '[data-review-id], '
        '.review-card, '
        '[class*="ReviewCard"], '
        'div[class*="review-"][class*="card"], '
        'div[id^="review-"], '
        'div[class*="review-content"], '
        'div[class*="review-listing"], '
        '[itemtype*="Review"]'
    )
    title_count = 0
    company_count = 0
    industry_count = 0
    size_count = 0
    samples: list[dict[str, Any]] = []

    for card in cards:
        reviewer_title = _get_text(card, '[class*="title"], [class*="job"], [class*="role"]')
        reviewer_company = _get_text(card, '[class*="company"], [class*="org"]')
        reviewer_industry = _get_text(card, '[class*="industry"], [class*="sector"]')
        company_size = _get_text(card, '[class*="size"], [class*="employees"]')

        if reviewer_title:
            title_count += 1
        if reviewer_company:
            company_count += 1
        if reviewer_industry:
            industry_count += 1
        if company_size:
            size_count += 1
        if len(samples) < 5 and any((reviewer_title, reviewer_company, reviewer_industry, company_size)):
            samples.append(
                {
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


def _get_text(node: Any, selector: str) -> str | None:
    match = node.select_one(selector)
    if not match:
        return None
    text = match.get_text(" ", strip=True)
    return text or None


def analyze_getapp_raw_html(html: str) -> dict[str, Any]:
    jsonld = analyze_getapp_jsonld_fields(html)
    cards = analyze_getapp_review_cards(html)
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


async def fetch_getapp_page_http(target: ScrapeTarget, *, page: int) -> dict[str, Any]:
    url = build_getapp_page_url(target.product_slug, page)
    referer = build_getapp_referer(target.vendor_name, target.product_slug, page)
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


async def fetch_getapp_page_web_unlocker(target: ScrapeTarget, *, page: int) -> dict[str, Any]:
    url = build_getapp_page_url(target.product_slug, page)
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/131.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": build_getapp_referer(target.vendor_name, target.product_slug, page),
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


async def fetch_getapp_page_browser(target: ScrapeTarget, *, page: int) -> dict[str, Any]:
    from playwright.async_api import async_playwright

    ws_url = settings.b2b_scrape.scraping_browser_ws_url.strip()
    if not ws_url:
        raise RuntimeError("Bright Data Scraping Browser is not configured")
    url = build_getapp_page_url(target.product_slug, page)
    referer = build_getapp_referer(target.vendor_name, target.product_slug, page)
    timeout_ms = settings.b2b_scrape.playwright_timeout_ms

    async with async_playwright() as pw:
        browser = await pw.chromium.connect_over_cdp(ws_url, timeout=timeout_ms)
        context = browser.contexts[0] if browser.contexts else await browser.new_context()
        page_obj = await context.new_page()
        try:
            resp = await page_obj.goto(
                url,
                wait_until="commit",
                timeout=timeout_ms,
                referer=referer,
            )
            status = resp.status if resp else 0
            try:
                await page_obj.wait_for_load_state(
                    "domcontentloaded",
                    timeout=min(timeout_ms, 15000),
                )
            except Exception:
                pass
            try:
                await page_obj.wait_for_selector(
                    '[data-testid="review-card"], [data-review-id], [itemtype*="Review"]',
                    timeout=min(timeout_ms, 15000),
                )
            except Exception:
                pass
            html = await page_obj.content()
            final_url = page_obj.url
        finally:
            await page_obj.close()
            await browser.close()

    return {
        "method": "browser",
        "url": url,
        "status_code": status,
        "final_url": final_url,
        "headers": {},
        "body": html or "",
    }


async def capture_getapp_page(target: ScrapeTarget, *, page: int, method: str) -> dict[str, Any]:
    if method == "http":
        capture = await fetch_getapp_page_http(target, page=page)
    elif method == "browser":
        capture = await fetch_getapp_page_browser(target, page=page)
    elif method == "web_unlocker":
        capture = await fetch_getapp_page_web_unlocker(target, page=page)
    else:
        raise ValueError(f"Unsupported method: {method}")
    capture["analysis"] = analyze_getapp_raw_html(capture.get("body") or "")
    capture["body_bytes"] = len((capture.get("body") or "").encode("utf-8", errors="replace"))
    return capture


def write_getapp_capture_artifacts(
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
