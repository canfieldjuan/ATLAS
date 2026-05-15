"""Standalone ports for AI Content Ops blog-post drafts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence

from .campaign_ports import JsonDict, TenantScope


@dataclass(frozen=True)
class BlogPostDraft:
    """Generated, not-yet-published blog post draft."""

    slug: str
    title: str
    content: str
    topic_type: str
    description: str = ""
    tags: Sequence[str] = field(default_factory=tuple)
    charts: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    data_context: JsonDict = field(default_factory=dict)
    metadata: JsonDict = field(default_factory=dict)
    id: str = ""
    status: str = ""

    def as_dict(self) -> JsonDict:
        return {
            "slug": self.slug,
            "title": self.title,
            "description": self.description,
            "topic_type": self.topic_type,
            "tags": list(self.tags),
            "content": self.content,
            "charts": [dict(chart) for chart in self.charts],
            "data_context": dict(self.data_context),
            "metadata": dict(self.metadata),
            "id": self.id,
            "status": self.status,
        }


@dataclass(frozen=True)
class BlogBlueprint:
    """In-memory representation of a stored blog blueprint row."""

    target_mode: str
    topic_type: str
    slug: str
    suggested_title: str = ""
    payload: Mapping[str, Any] = field(default_factory=dict)


class BlogBlueprintRepository(Protocol):
    """Read blog blueprints from host-owned intelligence storage."""

    async def read_blog_blueprints(
        self,
        *,
        scope: TenantScope,
        target_mode: str,
        limit: int,
        filters: Mapping[str, Any] | None = None,
    ) -> Sequence[Mapping[str, Any]]:
        """Return blueprint rows ready for the blog generation skill."""


class BlogPostRepository(Protocol):
    """Persistence contract for generated blog drafts."""

    async def save_drafts(
        self,
        drafts: Sequence[BlogPostDraft],
        *,
        scope: TenantScope,
    ) -> Sequence[str]:
        """Persist drafts and return assigned blog post ids."""

    async def list_drafts(
        self,
        *,
        scope: TenantScope,
        status: str | None = None,
        topic_type: str | None = None,
        limit: int | None = None,
    ) -> Sequence[BlogPostDraft]:
        """Return drafts filtered by tenant scope and optional facets."""

    async def update_status(
        self,
        blog_post_id: str,
        status: str,
        *,
        scope: TenantScope,
    ) -> bool:
        """Update a draft status and return True on hit."""

    async def update_statuses(
        self,
        blog_post_ids: Sequence[str],
        status: str,
        *,
        scope: TenantScope,
    ) -> Sequence[str]:
        """Bulk-update draft statuses and return ids that matched the tenant scope."""


__all__ = [
    "BlogBlueprint",
    "BlogBlueprintRepository",
    "BlogPostDraft",
    "BlogPostRepository",
]
