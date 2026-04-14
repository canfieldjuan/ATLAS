"""Targeted Capterra raw-page capture and reviewer-field audit helpers."""

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
from .parsers.capterra import _BASE_URL, _DOMAIN, _extract_live_profile_metadata, _select_review_cards
from .web_unlocker_http import get_with_web_unlocker


def build_capterra_page_url(product_slug: str, page: int) -> str:
    base = f"{_BASE_URL}/{product_slug}/reviews/"
    if page <= 1:
        return base
    return f"{base}?page={page}"


def build_capterra_referer(vendor_name: str, product_slug: str, page: int) -> str:
    base = f"{_BASE_URL}/{product_slug}/reviews/"
    if page <= 1:
        return f"https://www.google.com/search?q={quote_plus(vendor_name)}+capterra+reviews"
    if page == 2:
        return base
    return f"{base}?page={page - 1}"


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


def analyze_capterra_jsonld_fields(html: str) -> dict[str, Any]:
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


def analyze_capterra_review_cards(html: str) -> dict[str, Any]:
    soup = BeautifulSoup(html, "html.parser")
    cards = _select_review_cards(soup)
    title_count = 0
    company_count = 0
    industry_count = 0
    size_count = 0
    samples: list[dict[str, Any]] = []

    for card in cards:
        profile_meta = _extract_live_profile_metadata(card)
        reviewer_title = str(profile_meta.get("reviewer_title") or "").strip() or None
        reviewer_company = str(profile_meta.get("reviewer_company") or "").strip() or None
        reviewer_industry = str(profile_meta.get("reviewer_industry") or "").strip() or None
        company_size = str(profile_meta.get("company_size_raw") or "").strip() or None

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


def analyze_capterra_raw_html(html: str) -> dict[str, Any]:
    jsonld = analyze_capterra_jsonld_fields(html)
    cards = analyze_capterra_review_cards(html)
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


async def fetch_capterra_page_http(target: ScrapeTarget, *, page: int) -> dict[str, Any]:
    url = build_capterra_page_url(target.product_slug, page)
    referer = build_capterra_referer(target.vendor_name, target.product_slug, page)
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


async def fetch_capterra_page_web_unlocker(target: ScrapeTarget, *, page: int) -> dict[str, Any]:
    url = build_capterra_page_url(target.product_slug, page)
    referer = build_capterra_referer(target.vendor_name, target.product_slug, page)
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


async def fetch_capterra_page_browser(target: ScrapeTarget, *, page: int) -> dict[str, Any]:
    from playwright.async_api import async_playwright
    import logging

    logger = logging.getLogger("atlas.services.scraping.capterra_audit")
    ws_url = settings.b2b_scrape.scraping_browser_ws_url.strip()
    if not ws_url:
        raise RuntimeError("Bright Data Scraping Browser is not configured")
    url = build_capterra_page_url(target.product_slug, page)
    referer = build_capterra_referer(target.vendor_name, target.product_slug, page)
    timeout_ms = settings.b2b_scrape.playwright_timeout_ms
    async with async_playwright() as pw:
        logger.info(
            "Capterra audit connecting Scraping Browser vendor=%s slug=%s page=%d ws_host=%s",
            target.vendor_name,
            target.product_slug,
            page,
            ws_url.split("@")[-1],
        )
        browser = await pw.chromium.connect_over_cdp(ws_url, timeout=timeout_ms)
        context = browser.contexts[0] if browser.contexts else await browser.new_context()
        page_obj = await context.new_page()
        logger.info(
            "Capterra audit navigating vendor=%s page=%d url=%s",
            target.vendor_name,
            page,
            url,
        )
        resp = await page_obj.goto(
            url,
            wait_until="commit",
            timeout=timeout_ms,
            referer=referer,
        )
        status = resp.status if resp else 0
        try:
            await page_obj.wait_for_load_state("domcontentloaded", timeout=min(timeout_ms, 15000))
        except Exception:
            pass
        try:
            await page_obj.wait_for_selector(
                '[data-testid="review-card"], [class*="review-card"], [itemprop="review"]',
                timeout=min(timeout_ms, 15000),
            )
        except Exception:
            pass
        html = await page_obj.content()
        logger.info(
            "Capterra audit browser response vendor=%s page=%d status=%s final_url=%s body_bytes=%d",
            target.vendor_name,
            page,
            status,
            page_obj.url,
            len((html or "").encode("utf-8", errors="replace")),
        )
    return {
        "method": "browser",
        "url": url,
        "status_code": status,
        "final_url": page_obj.url,
        "headers": {},
        "body": html or "",
    }


async def capture_capterra_page(target: ScrapeTarget, *, page: int, method: str) -> dict[str, Any]:
    if method == "http":
        capture = await fetch_capterra_page_http(target, page=page)
    elif method == "browser":
        capture = await fetch_capterra_page_browser(target, page=page)
    elif method == "web_unlocker":
        capture = await fetch_capterra_page_web_unlocker(target, page=page)
    else:
        raise ValueError(f"Unsupported method: {method}")
    capture["analysis"] = analyze_capterra_raw_html(capture.get("body") or "")
    capture["body_bytes"] = len((capture.get("body") or "").encode("utf-8", errors="replace"))
    return capture


def write_capterra_capture_artifacts(
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
