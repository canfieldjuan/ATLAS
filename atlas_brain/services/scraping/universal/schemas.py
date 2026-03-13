"""
Pydantic models for universal scraper job configuration and results.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


class PaginationStrategy(str, Enum):
    NONE = "none"
    URL_PATTERN = "url_pattern"
    CSS_SELECTOR = "css_selector"


class PaginationConfig(BaseModel):
    """How to follow pages for a single target."""

    strategy: PaginationStrategy = PaginationStrategy.NONE
    url_pattern: Optional[str] = None
    css_selector: Optional[str] = None
    max_pages: int = Field(default=5, ge=1, le=100)


class ExtractionSchema(BaseModel):
    """What to extract — natural language OR structured field definitions."""

    description: Optional[str] = None
    fields: Optional[dict[str, str]] = None

    def to_prompt_fragment(self) -> str:
        """Convert to an LLM prompt fragment describing the desired output."""
        if self.fields:
            lines = [f'  "{name}": <{typ}>' for name, typ in self.fields.items()]
            return (
                "Extract items matching this JSON schema:\n{\n"
                + ",\n".join(lines)
                + "\n}"
            )
        if self.description:
            return (
                f"Extract the following fields from each item on the page: "
                f"{self.description}\n"
                f"Return each item as a JSON object with descriptive field names."
            )
        return "Extract the main structured data from this page as JSON objects."


class ScrapeTarget(BaseModel):
    """A single URL (or URL pattern) to scrape."""

    url: str
    use_browser: bool = Field(
        default=False,
        description="True = Playwright stealth browser (JS-rendered); False = HTTP client",
    )
    wait_for_selector: Optional[str] = Field(
        default=None,
        description="CSS selector to wait for before extracting (browser mode only)",
    )
    pagination: PaginationConfig = Field(default_factory=PaginationConfig)
    extra_headers: Optional[dict[str, str]] = None
    prefer_residential: bool = False
    sticky_session: bool = False

    @field_validator("url")
    @classmethod
    def _validate_url(cls, v: str) -> str:
        from .url_validation import validate_url

        return validate_url(v)


class ScrapeJobConfig(BaseModel):
    """Top-level job definition — submitted via API or loaded from JSON file."""

    name: str
    schema_def: ExtractionSchema = Field(alias="schema")
    targets: list[ScrapeTarget]
    concurrency: int = Field(default=3, ge=1, le=10)
    llm_workload: str = Field(
        default="triage",
        description="LLM workload tier: triage, draft, synthesis, local_fast, vllm",
    )
    llm_max_tokens: int = Field(default=4096, ge=256, le=16384)
    store_raw_llm: bool = Field(
        default=False,
        description="Persist raw LLM output for debugging (may contain sensitive page content)",
    )

    model_config = {"populate_by_name": True}


# ── Response models ──────────────────────────────────────────────────


class JobStatus(BaseModel):
    id: UUID
    name: str
    status: str
    total_targets: int
    completed_targets: int
    failed_targets: int
    total_records: int
    error: Optional[str]
    started_at: Optional[datetime]
    finished_at: Optional[datetime]
    created_at: datetime


class ScrapeResultItem(BaseModel):
    id: UUID
    job_id: UUID
    target_url: str
    page_number: int
    page_title: Optional[str]
    extracted_data: list[dict[str, Any]]
    item_count: int
    duration_ms: Optional[int]
    error: Optional[str]
    created_at: datetime
