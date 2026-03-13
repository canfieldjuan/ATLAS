"""
Universal web scraper — data-agnostic extraction from any website.

Uses the existing anti-detection HTTP client + stealth browser infrastructure
for fetching, and the LLM for intelligent data extraction based on a
user-defined schema (natural language or JSON field definitions).

Supports multi-site scrape jobs with parallel execution.
"""

from .orchestrator import (
    UniversalScraper,
    get_universal_scraper,
    load_config_file,
    reconcile_orphaned_jobs,
)
from .schemas import (
    ExtractionSchema,
    PaginationConfig,
    PaginationStrategy,
    ScrapeJobConfig,
    ScrapeTarget,
)

from .b2b_adapter import UniversalReviewAdapter, get_universal_adapter
from .b2b_mode import ScrapeMode, get_scrape_mode
from .source_configs import SourceAdapterConfig, get_source_adapter_config

__all__ = [
    "ExtractionSchema",
    "PaginationConfig",
    "PaginationStrategy",
    "ScrapeJobConfig",
    "ScrapeTarget",
    "UniversalScraper",
    "get_universal_scraper",
    "load_config_file",
    "reconcile_orphaned_jobs",
    # B2B adapter layer
    "UniversalReviewAdapter",
    "get_universal_adapter",
    "ScrapeMode",
    "get_scrape_mode",
    "SourceAdapterConfig",
    "get_source_adapter_config",
]
