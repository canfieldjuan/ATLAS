"""
Convert raw HTML to clean text suitable for LLM extraction.

Uses trafilatura (already a project dependency) with BeautifulSoup fallback.
Strips noise, preserves structural hints, truncates to a token budget.
"""

from __future__ import annotations

import re
from urllib.parse import urljoin

from bs4 import BeautifulSoup


def html_to_text(html: str, max_chars: int = 30_000) -> str:
    """Extract readable text from HTML, preserving table/list structure.

    Tries trafilatura first (better at isolating main content), falls back
    to BeautifulSoup raw text extraction.
    """
    # Try trafilatura (fast, high-quality main-content extraction)
    try:
        import trafilatura

        text = trafilatura.extract(
            html, include_tables=True, include_links=True
        )
        if text and len(text.strip()) > 100:
            return text[:max_chars]
    except Exception:
        pass

    # Fallback: BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")

    # Remove noise elements
    for tag in soup(
        ["script", "style", "nav", "footer", "header", "aside", "noscript", "svg"]
    ):
        tag.decompose()

    text = soup.get_text(separator="\n", strip=True)
    # Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text[:max_chars]


def extract_next_page_url(
    html: str, css_selector: str, base_url: str
) -> str | None:
    """Find the next-page URL by matching a CSS selector on an <a> tag.

    Returns an absolute URL or None if not found.
    """
    soup = BeautifulSoup(html, "html.parser")
    el = soup.select_one(css_selector)
    if el and el.get("href"):
        href = el["href"]
        if href.startswith("http"):
            return href
        return urljoin(base_url, href)
    return None
