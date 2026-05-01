from __future__ import annotations

import os
from typing import Any

if os.getenv("EXTRACTED_PIPELINE_STANDALONE", "0") == "1":
    def blog_quality_summary(data_context: dict[str, Any] | None = None) -> dict[str, Any]:
        return {
            "score": 100,
            "threshold": 0,
            "blocking_issues": [],
            "warnings": [],
        }

    def blog_quality_revalidation(
        data_context: dict[str, Any] | None = None,
        content: dict[str, Any] | None = None,
        report: dict[str, Any] | None = None,
        boundary: str | None = None,
    ) -> dict[str, Any]:
        base = blog_quality_summary(data_context)
        if isinstance(report, dict):
            for key in ("score", "threshold", "blocking_issues", "warnings"):
                if key in report:
                    base[key] = report[key]
        return base

    def merge_blog_first_pass_quality_data_context(
        data_context: dict[str, Any] | None,
        audit: dict[str, Any] | None,
    ) -> dict[str, Any]:
        merged = dict(data_context or {})
        if isinstance(audit, dict):
            merged["first_pass_quality"] = {
                "score": audit.get("score"),
                "threshold": audit.get("threshold"),
                "blocking_issues": list(audit.get("blocking_issues") or []),
                "warnings": list(audit.get("warnings") or []),
            }
        return merged
else:
    from atlas_brain.services.blog_quality import *
