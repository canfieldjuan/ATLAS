"""
Universal web scraper — data-agnostic extraction from any website.

Uses the existing anti-detection HTTP client + stealth browser infrastructure
for fetching, and the LLM for intelligent data extraction based on a
user-defined schema (natural language or JSON field definitions).

Supports multi-site scrape jobs with parallel execution.
"""

from .orchestrator import UniversalScraper, get_universal_scraper, load_config_file
from .schemas import (
    ExtractionSchema,
    PaginationConfig,
    PaginationStrategy,
    ScrapeJobConfig,
    ScrapeTarget,
)

__all__ = [
    "ExtractionSchema",
    "PaginationConfig",
    "PaginationStrategy",
    "ScrapeJobConfig",
    "ScrapeTarget",
    "UniversalScraper",
    "get_universal_scraper",
    "load_config_file",
]
